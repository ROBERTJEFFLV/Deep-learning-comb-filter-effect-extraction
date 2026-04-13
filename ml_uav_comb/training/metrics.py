"""Evaluation metrics for likelihood observer training."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _masked_distance_error(
    pred_distance_cm: torch.Tensor,
    target_distance_cm: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    valid = mask > 0.5
    if not torch.any(valid):
        return 0.0, 0.0
    dist_err = pred_distance_cm[valid] - target_distance_cm[valid]
    mae = float(torch.mean(torch.abs(dist_err)).item())
    rmse = float(torch.sqrt(torch.mean(dist_err**2)).item())
    return mae, rmse


def _masked_binary_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    valid = mask > 0.5
    if not torch.any(valid):
        return 0.0
    pred = (torch.sigmoid(logits[valid]) >= 0.5).float()
    return float((pred == target[valid]).float().mean().item())


def compute_batch_metrics(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    target_space: str = "raw",
) -> Dict[str, float]:
    del target_space
    measurement_mae_gt, measurement_rmse_gt = _masked_distance_error(
        pred["measurement_distance_cm"],
        batch["measurement_distance_target_cm"],
        batch["valid_dist_gt_mask"],
    )
    measurement_mae_train, measurement_rmse_train = _masked_distance_error(
        pred["measurement_distance_cm"],
        batch["measurement_distance_target_cm"],
        batch["measurement_distance_train_mask"],
    )
    # Batch-level posterior metrics are unavailable before sequential KF pass.
    return {
        "measurement_mae_cm_gt": measurement_mae_gt,
        "measurement_rmse_cm_gt": measurement_rmse_gt,
        "measurement_mae_cm_train": measurement_mae_train,
        "measurement_rmse_cm_train": measurement_rmse_train,
        "posterior_mae_cm_gt": 0.0,
        "posterior_rmse_cm_gt": 0.0,
        "measurement_validity_acc_train": _masked_binary_accuracy(
            pred["measurement_validity_logit"],
            batch["measurement_validity_target"],
            batch["measurement_validity_train_mask"],
        ),
        "valid_window_ratio": float(batch["measurement_distance_train_mask"].float().mean().item()),
    }


def reduce_metric_list(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {
            "measurement_mae_cm_gt": 0.0,
            "measurement_rmse_cm_gt": 0.0,
            "measurement_mae_cm_train": 0.0,
            "measurement_rmse_cm_train": 0.0,
            "posterior_mae_cm_gt": 0.0,
            "posterior_rmse_cm_gt": 0.0,
            "measurement_validity_acc_train": 0.0,
            "valid_window_ratio": 0.0,
        }
    keys = metric_list[0].keys()
    return {key: float(np.mean([item[key] for item in metric_list])) for key in keys}
