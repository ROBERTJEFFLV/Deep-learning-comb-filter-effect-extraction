"""Metrics for omega-regression training."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ml_uav_comb.data_pipeline.offline_omega_feature_extractor import omega_to_distance_cm


def compute_omega_metrics(
    pred_omega: torch.Tensor,
    target_omega: torch.Tensor,
    weights: torch.Tensor,
    *,
    pattern_logit: Optional[torch.Tensor] = None,
    pattern_target: Optional[torch.Tensor] = None,
    pattern_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    valid = torch.isfinite(pred_omega) & torch.isfinite(target_omega) & (weights > 0.0)
    metrics = {
        "omega_mae": 0.0,
        "omega_rmse": 0.0,
        "distance_mae_cm": 0.0,
        "distance_rmse_cm": 0.0,
        "valid_ratio": float(valid.float().mean().item()),
    }
    if bool(valid.any()):
        pred = pred_omega[valid]
        target = target_omega[valid]
        w = weights[valid]
        wsum = torch.clamp(w.sum(), min=1e-6)
        omega_mae = ((torch.abs(pred - target) * w).sum() / wsum).item()
        omega_rmse = torch.sqrt((torch.square(pred - target) * w).sum() / wsum).item()
        dist_pred = omega_to_distance_cm(pred)
        dist_target = omega_to_distance_cm(target)
        distance_mae = ((torch.abs(dist_pred - dist_target) * w).sum() / wsum).item()
        distance_rmse = torch.sqrt((torch.square(dist_pred - dist_target) * w).sum() / wsum).item()
        metrics.update(
            {
                "omega_mae": float(omega_mae),
                "omega_rmse": float(omega_rmse),
                "distance_mae_cm": float(distance_mae),
                "distance_rmse_cm": float(distance_rmse),
            }
        )

    if pattern_logit is not None and pattern_target is not None and pattern_weights is not None:
        pvalid = torch.isfinite(pattern_logit) & torch.isfinite(pattern_target) & (pattern_weights > 0.0)
        metrics["pattern_valid_ratio"] = float(pvalid.float().mean().item())
        metrics["pattern_acc"] = 0.0
        metrics["pattern_bce"] = 0.0
        metrics["pattern_positive_rate"] = 0.0
        if bool(pvalid.any()):
            logits = pattern_logit[pvalid]
            target_soft = pattern_target[pvalid]
            pw = pattern_weights[pvalid]
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            target_hard = (target_soft >= 0.5).float()
            pwsum = torch.clamp(pw.sum(), min=1e-6)
            acc = (((pred == target_hard).float()) * pw).sum() / pwsum
            bce = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target_soft, reduction="none") * pw).sum() / pwsum
            metrics["pattern_acc"] = float(acc.item())
            metrics["pattern_bce"] = float(bce.item())
            metrics["pattern_positive_rate"] = float(((target_hard * pw).sum() / pwsum).item())
    return metrics


def reduce_metric_list(metrics: list[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics[0].keys())
    return {key: float(sum(m.get(key, 0.0) for m in metrics) / len(metrics)) for key in keys}
