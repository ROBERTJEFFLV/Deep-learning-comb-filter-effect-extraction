"""Metrics for omega-regression training."""
from __future__ import annotations

from typing import Dict, List, Optional

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
    observability_score: Optional[torch.Tensor] = None,
    observability_pred: Optional[torch.Tensor] = None,
    distance_cm: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute metrics for a single batch.

    Returns a dict that includes ``_n_valid`` (number of weighted-valid samples)
    and ``_n_all_valid`` (number of finite-omega samples) to enable proper
    weighted reduction across batches via :func:`reduce_metric_list`.
    """
    valid = torch.isfinite(pred_omega) & torch.isfinite(target_omega) & (weights > 0.0)
    n_valid = int(valid.sum().item())

    # "all valid": finite omega regardless of pattern weight (for real test data)
    all_valid = torch.isfinite(pred_omega) & torch.isfinite(target_omega)
    n_all_valid = int(all_valid.sum().item())

    metrics: Dict[str, float] = {
        "omega_mae": 0.0,
        "omega_rmse_sq": 0.0,
        "distance_mae_cm": 0.0,
        "distance_rmse_sq_cm": 0.0,
        "valid_ratio": float(valid.float().mean().item()),
        "_n_valid": float(n_valid),
        "_n_all_valid": float(n_all_valid),
        # Unweighted metrics on ALL finite-omega frames
        "distance_mae_cm_all": 0.0,
        "distance_rmse_sq_cm_all": 0.0,
    }

    if n_valid > 0:
        pred = pred_omega[valid]
        target = target_omega[valid]
        w = weights[valid]
        wsum = torch.clamp(w.sum(), min=1e-6)
        omega_mae = ((torch.abs(pred - target) * w).sum() / wsum).item()
        omega_rmse_sq = ((torch.square(pred - target) * w).sum() / wsum).item()
        dist_pred = omega_to_distance_cm(pred)
        dist_target = omega_to_distance_cm(target)
        distance_mae = ((torch.abs(dist_pred - dist_target) * w).sum() / wsum).item()
        distance_rmse_sq = ((torch.square(dist_pred - dist_target) * w).sum() / wsum).item()
        metrics.update(
            {
                "omega_mae": float(omega_mae),
                "omega_rmse_sq": float(omega_rmse_sq),
                "distance_mae_cm": float(distance_mae),
                "distance_rmse_sq_cm": float(distance_rmse_sq),
            }
        )

        # Stratified metrics by distance bins
        for bin_name, lo, hi in [("near", 0.0, 15.0), ("mid", 15.0, 35.0), ("far", 35.0, 100.0)]:
            bin_mask = (dist_target >= lo) & (dist_target < hi)
            if bin_mask.any():
                bin_mae = torch.abs(dist_pred[bin_mask] - dist_target[bin_mask]).mean().item()
                bin_count = int(bin_mask.sum().item())
            else:
                bin_mae = 0.0
                bin_count = 0
            metrics[f"distance_mae_cm_{bin_name}"] = float(bin_mae)
            metrics[f"distance_count_{bin_name}"] = float(bin_count)

        # Phase 2: Stratified by observability score
        if observability_score is not None:
            obs_valid = observability_score[valid]
            obs_finite = torch.isfinite(obs_valid)
            for obs_name, lo, hi in [("obs_low", 0.0, 1.0), ("obs_mid", 1.0, 2.0), ("obs_high", 2.0, float("inf"))]:
                obs_mask = obs_finite & (obs_valid >= lo) & (obs_valid < hi)
                if obs_mask.any():
                    obs_mae = torch.abs(dist_pred[obs_mask] - dist_target[obs_mask]).mean().item()
                    obs_count = int(obs_mask.sum().item())
                else:
                    obs_mae = 0.0
                    obs_count = 0
                metrics[f"distance_mae_cm_{obs_name}"] = float(obs_mae)
                metrics[f"distance_count_{obs_name}"] = float(obs_count)

        # Phase 2: Observability prediction accuracy
        if observability_pred is not None and observability_score is not None:
            obs_gt = observability_score[valid]
            obs_pd = observability_pred[valid] if observability_pred.shape == pred_omega.shape else observability_pred
            obs_both_valid = torch.isfinite(obs_gt) & torch.isfinite(obs_pd)
            if obs_both_valid.any():
                metrics["observability_mae"] = float(torch.abs(obs_pd[obs_both_valid] - obs_gt[obs_both_valid]).mean().item())
            else:
                metrics["observability_mae"] = 0.0

    # Unweighted metrics on ALL finite-omega frames (critical for real test data
    # where pattern_target=0 for most frames but omega is still valid)
    if n_all_valid > 0:
        pred_all = pred_omega[all_valid]
        target_all = target_omega[all_valid]
        dist_pred_all = omega_to_distance_cm(pred_all)
        dist_target_all = omega_to_distance_cm(target_all)
        # Optionally filter to model operating range using GT distance
        if distance_cm is not None:
            dist_gt_all = distance_cm[all_valid]
            in_range = torch.isfinite(dist_gt_all) & (dist_gt_all >= 5.0) & (dist_gt_all <= 60.0)
            n_in_range = int(in_range.sum().item())
            if n_in_range > 0:
                metrics["distance_mae_cm_all"] = float(torch.abs(dist_pred_all[in_range] - dist_target_all[in_range]).mean().item())
                metrics["distance_rmse_sq_cm_all"] = float(torch.square(dist_pred_all[in_range] - dist_target_all[in_range]).mean().item())
                metrics["_n_all_valid"] = float(n_in_range)
            else:
                metrics["_n_all_valid"] = 0.0
        else:
            metrics["distance_mae_cm_all"] = float(torch.abs(dist_pred_all - dist_target_all).mean().item())
            metrics["distance_rmse_sq_cm_all"] = float(torch.square(dist_pred_all - dist_target_all).mean().item())

    if pattern_logit is not None and pattern_target is not None and pattern_weights is not None:
        pvalid = torch.isfinite(pattern_logit) & torch.isfinite(pattern_target) & (pattern_weights > 0.0)
        n_pvalid = int(pvalid.sum().item())
        metrics["pattern_valid_ratio"] = float(pvalid.float().mean().item())
        metrics["_n_pattern_valid"] = float(n_pvalid)
        metrics["pattern_acc"] = 0.0
        metrics["pattern_bce"] = 0.0
        metrics["pattern_positive_rate"] = 0.0
        if n_pvalid > 0:
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


# Keys that should be reduced as weighted sums using _n_valid or _n_all_valid
_OMEGA_WEIGHTED_KEYS = {
    "omega_mae", "omega_rmse_sq", "distance_mae_cm", "distance_rmse_sq_cm",
    "valid_ratio",
}
_ALL_WEIGHTED_KEYS = {"distance_mae_cm_all", "distance_rmse_sq_cm_all"}
_PATTERN_WEIGHTED_KEYS = {"pattern_acc", "pattern_bce", "pattern_positive_rate", "pattern_valid_ratio"}
_COUNT_SUM_KEYS = {
    "_n_valid", "_n_all_valid", "_n_pattern_valid",
    "distance_count_near", "distance_count_mid", "distance_count_far",
    "distance_count_obs_low", "distance_count_obs_mid", "distance_count_obs_high",
}


def reduce_metric_list(metrics: list[Dict[str, float]]) -> Dict[str, float]:
    """Properly reduce per-batch metrics using sample-count-weighted averaging.

    Metrics like distance_mae_cm are weighted by the number of valid samples
    in each batch (_n_valid), preventing dilution by empty batches.
    """
    if not metrics:
        return {}

    all_keys = set()
    for m in metrics:
        all_keys.update(m.keys())

    result: Dict[str, float] = {}

    # Compute total valid counts for weighting
    total_n_valid = sum(m.get("_n_valid", 0.0) for m in metrics)
    total_n_all = sum(m.get("_n_all_valid", 0.0) for m in metrics)
    total_n_pattern = sum(m.get("_n_pattern_valid", 0.0) for m in metrics)

    for key in sorted(all_keys):
        if key.startswith("_"):
            if key in _COUNT_SUM_KEYS:
                result[key] = sum(m.get(key, 0.0) for m in metrics)
            continue

        values = [m[key] for m in metrics if key in m]
        if not values:
            continue

        if key in _OMEGA_WEIGHTED_KEYS and total_n_valid > 0:
            # Weight by number of valid samples per batch
            weighted_sum = sum(m.get(key, 0.0) * m.get("_n_valid", 0.0) for m in metrics)
            result[key] = float(weighted_sum / total_n_valid)
        elif key in _ALL_WEIGHTED_KEYS and total_n_all > 0:
            weighted_sum = sum(m.get(key, 0.0) * m.get("_n_all_valid", 0.0) for m in metrics)
            result[key] = float(weighted_sum / total_n_all)
        elif key in _PATTERN_WEIGHTED_KEYS and total_n_pattern > 0:
            weighted_sum = sum(m.get(key, 0.0) * m.get("_n_pattern_valid", 0.0) for m in metrics)
            result[key] = float(weighted_sum / total_n_pattern)
        elif key in _COUNT_SUM_KEYS:
            result[key] = sum(m.get(key, 0.0) for m in metrics)
        else:
            # Default: simple average (for distance bins, observability bins, etc.)
            result[key] = float(sum(values) / len(values))

    # Compute final RMSE from mean squared error
    if "omega_rmse_sq" in result:
        result["omega_rmse"] = float(result["omega_rmse_sq"] ** 0.5)
    if "distance_rmse_sq_cm" in result:
        result["distance_rmse_cm"] = float(result["distance_rmse_sq_cm"] ** 0.5)
    if "distance_rmse_sq_cm_all" in result:
        result["distance_rmse_cm_all"] = float(result["distance_rmse_sq_cm_all"] ** 0.5)

    return result
