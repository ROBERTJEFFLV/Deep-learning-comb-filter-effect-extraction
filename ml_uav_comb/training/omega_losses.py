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
    loss = F.huber_loss(pred[valid], target[valid], reduction="none", delta=float(delta))
    return (loss * weights[valid]).sum() / torch.clamp(weights[valid].sum(), min=1e-6)


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(logits) & torch.isfinite(target) & (weights > 0.0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits[valid], target[valid], reduction="none")
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
        float(cfg["training"].get("omega_huber_delta", 5.495497353218297e-04)),
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
    dynamic_loss = delta_lambda * delta_omega_loss + acc_lambda * acc_omega_loss
    pattern_loss = weighted_bce_with_logits(pattern_logit, pattern_target, pattern_weights)
    total = pattern_lambda * pattern_loss + omega_lambda * omega_loss + dynamic_loss
    return {
        "loss_total": total,
        "loss_distance": omega_loss.detach(),
        "loss_omega": omega_loss.detach(),
        "loss_pattern": pattern_loss.detach(),
        "loss_velocity": delta_omega_loss.detach(),
        "loss_acceleration": acc_omega_loss.detach(),
        "loss_dynamic": dynamic_loss.detach(),
        "loss_smooth": dynamic_loss.detach(),
        "sample_weights": omega_weights.detach(),
        "pattern_weights": pattern_weights.detach(),
    }
