"""Losses for omega-regression training."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

def compute_pattern_weights(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    pattern_target = batch["pattern_target"].float()
    return torch.isfinite(pattern_target).to(dtype=torch.float32)


def compute_omega_regression_weights(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    pattern_weights = compute_pattern_weights(batch)
    omega_target = batch["omega_target"].float()
    omega_valid = torch.isfinite(omega_target).to(dtype=torch.float32)
    omega_mask = batch["pattern_target"].float().detach().clamp(0.0, 1.0)
    return pattern_weights * omega_valid * omega_mask


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, delta: float) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target) & (weights > 0.0)
    if not bool(valid.any()):
        return pred.sum() * 0.0
    # Use smooth_l1_loss (beta param) instead of huber_loss (delta param).
    # PyTorch huber_loss scales the linear regime by delta, making loss ≈ delta*|err|.
    # smooth_l1_loss does NOT scale: linear regime is |err| - 0.5*beta.
    # With omega's tiny delta (~5.5e-4), huber gave loss ~1e-6 vs pattern BCE ~0.02.
    loss = F.smooth_l1_loss(pred[valid], target[valid], reduction="none", beta=float(delta))
    return (loss * weights[valid]).sum() / torch.clamp(weights[valid].sum(), min=1e-6)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    valid = torch.isfinite(logits) & torch.isfinite(target) & (weights > 0.0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    pw = logits.new_tensor([float(pos_weight)]) if pos_weight != 1.0 else None
    loss = F.binary_cross_entropy_with_logits(
        logits[valid], target[valid], reduction="none", pos_weight=pw,
    )
    return (loss * weights[valid]).sum() / torch.clamp(weights[valid].sum(), min=1e-6)


def observability_regression_loss(
    pred_score: torch.Tensor,
    target_score: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 between predicted and ground-truth observability score.

    Targets are normalized to [0,1] by dividing by 10 (observed max ~10.3) to keep
    loss magnitude comparable to other components without needing extreme lambda tuning.
    """
    valid = torch.isfinite(pred_score) & torch.isfinite(target_score) & (weights > 0.0)
    if not bool(valid.any()):
        return pred_score.sum() * 0.0
    # Normalize both pred and target to ~[0,1] range
    norm_pred = pred_score[valid] / 10.0
    norm_target = target_score[valid] / 10.0
    loss = F.smooth_l1_loss(norm_pred, norm_target, reduction="none", beta=0.05)
    return (loss * weights[valid]).sum() / torch.clamp(weights[valid].sum(), min=1e-6)


def uncertainty_aware_omega_loss(
    omega_pred: torch.Tensor,
    omega_target: torch.Tensor,
    log_variance: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic loss: L = 0.5 * exp(-s) * (y - ŷ)^2 + 0.5 * s.

    This learns to predict uncertainty: high s → low confidence, low s → high confidence.
    The 0.5*s term prevents the model from making s → ∞ to avoid all loss.
    """
    valid = (
        torch.isfinite(omega_pred)
        & torch.isfinite(omega_target)
        & torch.isfinite(log_variance)
        & (weights > 0.0)
    )
    if not bool(valid.any()):
        return omega_pred.sum() * 0.0
    precision = torch.exp(-log_variance[valid])
    sq_err = (omega_pred[valid] - omega_target[valid]).square()
    loss = 0.5 * precision * sq_err + 0.5 * log_variance[valid]
    # Use softplus to ensure non-negative loss while preserving gradient flow.
    # Previous clamp(min=0) killed gradients when loss < 0 (which happens when
    # log_variance is negative and sq_err is small), making this branch untrainable.
    loss = F.softplus(loss)
    return (loss * weights[valid]).sum() / torch.clamp(weights[valid].sum(), min=1e-6)


def temporal_delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    recording_ids: list[str],
    sequence_index: torch.Tensor,
    chunk_id: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    same_recording = pred.new_tensor([1.0 if recording_ids[i] == recording_ids[i - 1] else 0.0 for i in range(1, len(recording_ids))])
    same_chunk = (chunk_id[1:] == chunk_id[:-1]).to(dtype=pred.dtype)
    consecutive = (sequence_index[1:] - sequence_index[:-1] == 1).to(dtype=pred.dtype)
    pair_mask = same_recording * same_chunk * consecutive
    pair_weights = torch.minimum(weights[1:], weights[:-1]) * pair_mask
    return weighted_huber(pred[1:] - pred[:-1], target[1:] - target[:-1], pair_weights, delta)


def temporal_acceleration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    recording_ids: list[str],
    sequence_index: torch.Tensor,
    chunk_id: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    if pred.numel() < 3:
        return pred.new_tensor(0.0)
    same_recording = pred.new_tensor([1.0 if recording_ids[i] == recording_ids[i - 1] == recording_ids[i - 2] else 0.0 for i in range(2, len(recording_ids))])
    same_chunk = ((chunk_id[2:] == chunk_id[1:-1]) & (chunk_id[1:-1] == chunk_id[:-2])).to(dtype=pred.dtype)
    consecutive = (
        (sequence_index[1:-1] - sequence_index[:-2] == 1)
        & (sequence_index[2:] - sequence_index[1:-1] == 1)
    ).to(dtype=pred.dtype)
    triplet_mask = same_recording * same_chunk * consecutive
    triplet_weights = torch.minimum(torch.minimum(weights[2:], weights[1:-1]), weights[:-2]) * triplet_mask
    pred_acc = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    target_acc = target[2:] - 2.0 * target[1:-1] + target[:-2]
    return weighted_huber(pred_acc, target_acc, triplet_weights, delta)


def combined_omega_loss(model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor | list[str]], cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    pattern_target = batch["pattern_target"].float().clamp(0.0, 1.0)
    pattern_weights = compute_pattern_weights(batch)
    omega_weights = compute_omega_regression_weights(batch)
    omega_pred = model_out["omega_pred"]
    pattern_logit = model_out["pattern_logit"]
    omega_target = batch["omega_target"]
    omega_loss = weighted_huber(
        omega_pred,
        omega_target,
        omega_weights,
        float(cfg["training"].get("omega_huber_delta", 1.099099e-03)),
    )
    delta_omega_loss = temporal_delta_loss(
        omega_pred,
        omega_target,
        omega_weights,
        batch["recording_id"],
        batch["sequence_index"],
        batch["chunk_id"],
        float(cfg["training"].get("delta_omega_huber_delta", 1.8318324510727655e-04)),
    )
    acc_omega_loss = temporal_acceleration_loss(
        omega_pred,
        omega_target,
        omega_weights,
        batch["recording_id"],
        batch["sequence_index"],
        batch["chunk_id"],
        float(cfg["training"].get("acc_omega_huber_delta", 9.159162255363828e-05)),
    )
    omega_lambda = float(cfg["training"].get("lambda_omega", 1.0))
    delta_lambda = float(cfg["training"].get("lambda_delta", 0.2))
    acc_lambda = float(cfg["training"].get("lambda_acc", 0.05))
    pattern_lambda = float(cfg["training"].get("lambda_pattern", 1.0))
    pattern_pos_weight = float(cfg["training"].get("pattern_pos_weight", 1.0))
    dynamic_loss = delta_lambda * delta_omega_loss + acc_lambda * acc_omega_loss
    pattern_loss = weighted_bce_with_logits(pattern_logit, pattern_target, pattern_weights, pos_weight=pattern_pos_weight)

    # Phase 2: Cross-frequency consistency loss (computed in model forward)
    cross_freq_loss = model_out.get("cross_freq_consistency", omega_pred.new_tensor(0.0))
    cross_freq_lambda = float(cfg["training"].get("lambda_cross_freq", 0.1))

    # Phase 2: Observability regression loss
    obs_lambda = float(cfg["training"].get("lambda_observability", 0.5))
    obs_target = batch.get("observability_score")
    if obs_target is not None and "observability_pred" in model_out:
        obs_loss = observability_regression_loss(
            model_out["observability_pred"], obs_target, pattern_weights,
        )
    else:
        obs_loss = omega_pred.new_tensor(0.0)

    # Phase 3: Uncertainty-aware omega loss
    uncertainty_lambda = float(cfg["training"].get("lambda_uncertainty", 0.01))
    if "log_variance" in model_out:
        uncertainty_loss = uncertainty_aware_omega_loss(
            omega_pred, omega_target, model_out["log_variance"], omega_weights,
        )
    else:
        uncertainty_loss = omega_pred.new_tensor(0.0)

    total = (
        pattern_lambda * pattern_loss
        + omega_lambda * omega_loss
        + dynamic_loss
        + cross_freq_lambda * cross_freq_loss
        + obs_lambda * obs_loss
        + uncertainty_lambda * uncertainty_loss
    )
    return {
        "loss_total": total,
        "loss_distance": omega_loss.detach(),
        "loss_omega": omega_loss.detach(),
        "loss_pattern": pattern_loss.detach(),
        "loss_velocity": delta_omega_loss.detach(),
        "loss_acceleration": acc_omega_loss.detach(),
        "loss_dynamic": dynamic_loss.detach(),
        "loss_smooth": dynamic_loss.detach(),
        "loss_cross_freq": cross_freq_loss.detach(),
        "loss_observability": obs_loss.detach(),
        "loss_uncertainty": uncertainty_loss.detach(),
        "sample_weights": omega_weights.detach(),
        "pattern_weights": pattern_weights.detach(),
    }
