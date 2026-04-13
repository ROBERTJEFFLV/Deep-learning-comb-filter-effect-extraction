"""Evaluation helpers for cached-window datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from ml_uav_comb.filtering.observer_filter import DistanceGridRangeTracker
from ml_uav_comb.training.metrics import compute_batch_metrics, reduce_metric_list


def _maybe_mark_cudagraph_step_begin() -> None:
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        try:
            compiler_mod.cudagraph_mark_step_begin()
        except Exception:
            pass


def _masked_errors(values: List[Dict[str, float]], pred_key: str, gt_key: str, mask_key: str) -> tuple[float, float]:
    errs = []
    for row in values:
        if float(row.get(mask_key, 0.0)) <= 0.5:
            continue
        pred = float(row.get(pred_key, np.nan))
        gt = float(row.get(gt_key, np.nan))
        if np.isfinite(pred) and np.isfinite(gt):
            errs.append(pred - gt)
    if not errs:
        return 0.0, 0.0
    arr = np.asarray(errs, dtype=np.float64)
    mae = float(np.mean(np.abs(arr)))
    rmse = float(np.sqrt(np.mean(np.square(arr))))
    return mae, rmse


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    cfg: Dict[str, object],
    output_csv: str | Path | None = None,
) -> Dict[str, float]:
    model.eval()
    batch_metrics = []
    rows: List[Dict[str, float]] = []
    with torch.inference_mode():
        for batch in dataloader:
            _maybe_mark_cudagraph_step_begin()
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            pred = model(tensor_batch)
            batch_metrics.append(compute_batch_metrics(pred, tensor_batch))

            bs = int(pred["measurement_distance_cm"].shape[0])
            measurement_distance_cm = pred["measurement_distance_cm"].detach().cpu().numpy()
            measurement_logvar = pred["measurement_logvar"].detach().cpu().numpy()
            measurement_validity_prob = pred["measurement_validity_prob"].detach().cpu().numpy()
            measurement_entropy = pred["measurement_entropy"].detach().cpu().numpy()
            measurement_top1_cm = pred["measurement_top1_cm"].detach().cpu().numpy()
            measurement_top2_cm = pred["measurement_top2_cm"].detach().cpu().numpy()
            measurement_margin = pred["measurement_margin"].detach().cpu().numpy()
            target_time_sec = batch["target_time_sec"].detach().cpu().numpy()
            valid_dist_gt_mask = batch["valid_dist_gt_mask"].detach().cpu().numpy()
            distance_gt_cm = batch["measurement_distance_target_cm"].detach().cpu().numpy()
            baseline_distance_cm = batch["heuristic_distance_cm"].detach().cpu().numpy()
            baseline_available = batch["heuristic_distance_available"].detach().cpu().numpy()

            for idx in range(bs):
                rows.append(
                    {
                        "recording_id": str(batch["recording_id"][idx]),
                        "split": str(batch["split"][idx]),
                        "time_sec": float(target_time_sec[idx]),
                        "distance_gt_cm": float(distance_gt_cm[idx]) if float(valid_dist_gt_mask[idx]) > 0.5 else float("nan"),
                        "baseline_distance_cm": (
                            float(baseline_distance_cm[idx]) if float(baseline_available[idx]) > 0.5 else float("nan")
                        ),
                        "measurement_distance_cm": float(measurement_distance_cm[idx]),
                        "measurement_logvar": float(measurement_logvar[idx]),
                        "measurement_validity_prob": float(measurement_validity_prob[idx]),
                        "measurement_entropy": float(measurement_entropy[idx]),
                        "measurement_top1_cm": float(measurement_top1_cm[idx]),
                        "measurement_top2_cm": float(measurement_top2_cm[idx]),
                        "measurement_margin": float(measurement_margin[idx]),
                        "posterior_distance_cm": float("nan"),
                        "posterior_velocity_cm_s": float("nan"),
                        "posterior_covariance": float("nan"),
                        "measurement_used_flag": 0.0,
                        "R_eff": float("nan"),
                        "valid_dist_gt_mask": float(valid_dist_gt_mask[idx]),
                        "measurement_target_source": str(batch["measurement_target_source"][idx]),
                    }
                )

    rows.sort(key=lambda item: (item["recording_id"], float(item["time_sec"])))
    tracker_by_recording: Dict[str, DistanceGridRangeTracker] = {}
    for row in rows:
        recording_id = str(row["recording_id"])
        tracker = tracker_by_recording.get(recording_id)
        if tracker is None:
            tracker = DistanceGridRangeTracker(cfg)
            tracker_by_recording[recording_id] = tracker
        posterior = tracker.step(
            measurement_distance_cm=float(row["measurement_distance_cm"]),
            measurement_logvar=float(row["measurement_logvar"]),
            measurement_validity_prob=float(row["measurement_validity_prob"]),
            measurement_entropy=float(row["measurement_entropy"]),
            measurement_margin=float(row["measurement_margin"]),
            timestamp_sec=float(row["time_sec"]),
        )
        row.update(posterior)

    measurement_mae, measurement_rmse = _masked_errors(
        rows,
        pred_key="measurement_distance_cm",
        gt_key="distance_gt_cm",
        mask_key="valid_dist_gt_mask",
    )
    posterior_mae, posterior_rmse = _masked_errors(
        rows,
        pred_key="posterior_distance_cm",
        gt_key="distance_gt_cm",
        mask_key="valid_dist_gt_mask",
    )
    baseline_mae, baseline_rmse = _masked_errors(
        rows,
        pred_key="baseline_distance_cm",
        gt_key="distance_gt_cm",
        mask_key="valid_dist_gt_mask",
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "recording_id",
                    "split",
                    "time_sec",
                    "distance_gt_cm",
                    "baseline_distance_cm",
                    "measurement_distance_cm",
                    "measurement_logvar",
                    "measurement_validity_prob",
                    "measurement_entropy",
                    "measurement_top1_cm",
                    "measurement_top2_cm",
                    "measurement_margin",
                    "posterior_distance_cm",
                    "posterior_velocity_cm_s",
                    "posterior_covariance",
                    "measurement_used_flag",
                    "R_eff",
                    "valid_dist_gt_mask",
                    "measurement_target_source",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    reduced_batch_metrics = reduce_metric_list(batch_metrics)
    reduced_batch_metrics.update(
        {
            "measurement_mae_cm_gt": float(measurement_mae),
            "measurement_rmse_cm_gt": float(measurement_rmse),
            "posterior_mae_cm_gt": float(posterior_mae),
            "posterior_rmse_cm_gt": float(posterior_rmse),
            "baseline_mae_cm_gt": float(baseline_mae),
            "baseline_rmse_cm_gt": float(baseline_rmse),
        }
    )
    return reduced_batch_metrics
