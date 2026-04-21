"""Metrics for K+1 cepstral-bin classification."""
from __future__ import annotations

from typing import Dict, List

import torch


def compute_cepstral_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bin_centers_cm: torch.Tensor,
    distance_cm: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-batch classification and distance metrics.

    Args:
        logits: [B, K+1]
        targets: [B] integer class labels
        bin_centers_cm: [K] distance center for each bin
        distance_cm: [B] ground-truth distance (may contain nan)

    Returns:
        dict of metric name → value
    """
    B = logits.shape[0]
    K = bin_centers_cm.shape[0]
    pred_class = logits.argmax(dim=-1)

    # Overall accuracy
    correct = (pred_class == targets).float()
    accuracy = correct.mean().item()

    # Pattern detection: class 0 = no-pattern, 1+ = pattern
    gt_has_pattern = (targets > 0)
    gt_no_pattern = (targets == 0)
    pred_has_pattern = (pred_class > 0)
    pred_no_pattern = (pred_class == 0)

    # Pattern recall (sensitivity): among GT pattern frames, fraction detected
    n_gt_pattern = gt_has_pattern.sum().item()
    pattern_recall = (
        (gt_has_pattern & pred_has_pattern).float().sum().item() / max(n_gt_pattern, 1)
    )

    # No-pattern specificity: among GT no-pattern, fraction correctly classified
    n_gt_no_pattern = gt_no_pattern.sum().item()
    nopattern_specificity = (
        (gt_no_pattern & pred_no_pattern).float().sum().item() / max(n_gt_no_pattern, 1)
    )

    # False positive rate: among GT no-pattern, fraction predicted as pattern
    fp_rate = (
        (gt_no_pattern & pred_has_pattern).float().sum().item() / max(n_gt_no_pattern, 1)
    )

    # Distance MAE: among frames where both GT and pred have pattern
    both_pattern = gt_has_pattern & pred_has_pattern
    distance_mae = 0.0
    distance_rmse_sq = 0.0
    n_distance = 0
    if both_pattern.any():
        pred_bin_idx = pred_class[both_pattern] - 1  # 0-indexed
        pred_bin_idx = pred_bin_idx.clamp(0, K - 1)
        pred_d = bin_centers_cm[pred_bin_idx]
        gt_d = distance_cm[both_pattern]
        valid = torch.isfinite(gt_d)
        if valid.any():
            errors = torch.abs(pred_d[valid] - gt_d[valid])
            distance_mae = errors.mean().item()
            distance_rmse_sq = (errors ** 2).mean().item()
            n_distance = int(valid.sum().item())

    # Soft distance (expected value decoding)
    probs = torch.softmax(logits, dim=-1)
    pattern_probs = probs[:, 1:]  # [B, K]
    pattern_sum = pattern_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    soft_d = (pattern_probs / pattern_sum * bin_centers_cm.unsqueeze(0)).sum(dim=-1)

    soft_mae = 0.0
    n_soft = 0
    if gt_has_pattern.any():
        gt_d_pattern = distance_cm[gt_has_pattern]
        soft_d_pattern = soft_d[gt_has_pattern]
        valid = torch.isfinite(gt_d_pattern)
        if valid.any():
            soft_mae = torch.abs(soft_d_pattern[valid] - gt_d_pattern[valid]).mean().item()
            n_soft = int(valid.sum().item())

    # Bin accuracy (among pattern frames, exact bin match)
    bin_accuracy = 0.0
    if n_gt_pattern > 0:
        bin_accuracy = (
            (gt_has_pattern & (pred_class == targets)).float().sum().item()
            / n_gt_pattern
        )

    return {
        "accuracy": accuracy,
        "pattern_recall": pattern_recall,
        "nopattern_specificity": nopattern_specificity,
        "fp_rate": fp_rate,
        "bin_accuracy": bin_accuracy,
        "distance_mae_cm": distance_mae,
        "distance_rmse_sq_cm": distance_rmse_sq,
        "soft_distance_mae_cm": soft_mae,
        "_n_total": float(B),
        "_n_gt_pattern": float(n_gt_pattern),
        "_n_gt_nopattern": float(n_gt_no_pattern),
        "_n_distance_valid": float(n_distance),
        "_n_soft_valid": float(n_soft),
    }


def reduce_cepstral_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Weighted reduction of per-batch metrics."""
    if not metrics:
        return {}

    result: Dict[str, float] = {}
    total_n = sum(m.get("_n_total", 0) for m in metrics)
    total_pattern = sum(m.get("_n_gt_pattern", 0) for m in metrics)
    total_nopattern = sum(m.get("_n_gt_nopattern", 0) for m in metrics)
    total_dist = sum(m.get("_n_distance_valid", 0) for m in metrics)
    total_soft = sum(m.get("_n_soft_valid", 0) for m in metrics)

    # Weighted averages
    for key, weight_key, total in [
        ("accuracy", "_n_total", total_n),
        ("loss_total", "_n_total", total_n),
        ("pattern_recall", "_n_gt_pattern", total_pattern),
        ("nopattern_specificity", "_n_gt_nopattern", total_nopattern),
        ("fp_rate", "_n_gt_nopattern", total_nopattern),
        ("bin_accuracy", "_n_gt_pattern", total_pattern),
        ("distance_mae_cm", "_n_distance_valid", total_dist),
        ("distance_rmse_sq_cm", "_n_distance_valid", total_dist),
        ("soft_distance_mae_cm", "_n_soft_valid", total_soft),
    ]:
        if total > 0:
            weighted = sum(
                m.get(key, 0) * m.get(weight_key, 0) for m in metrics
            )
            result[key] = weighted / total
        else:
            result[key] = 0.0

    # Derive RMSE
    if "distance_rmse_sq_cm" in result:
        result["distance_rmse_cm"] = result["distance_rmse_sq_cm"] ** 0.5

    # Counts
    result["_n_total"] = total_n
    result["_n_gt_pattern"] = total_pattern
    result["_n_gt_nopattern"] = total_nopattern

    return result
