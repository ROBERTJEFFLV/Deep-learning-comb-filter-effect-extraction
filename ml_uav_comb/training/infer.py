"""Single-file inference using the cached feature contract."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ml_uav_comb.data_pipeline.normalization import load_normalization_stats
from ml_uav_comb.data_pipeline.offline_feature_extractor import process_audio_array
from ml_uav_comb.filtering.observer_filter import DistanceGridRangeTracker
from ml_uav_comb.features.feature_utils import ensure_dir, load_audio_mono


def _maybe_mark_cudagraph_step_begin() -> None:
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        try:
            compiler_mod.cudagraph_mark_step_begin()
        except Exception:
            pass


def make_infer_windows(features: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    hop_sec = float(cfg["audio"]["hop_len"]) / float(cfg["audio"]["target_sr"])
    window_frames = int(round(float(cfg["dataset"]["window_sec"]) / hop_sec))
    stride_frames = int(round(float(cfg["dataset"]["stride_sec"]) / hop_sec))
    stride_frames = max(1, stride_frames)
    frame_time_sec = features["frame_time_sec"]
    windows = []
    for start in range(0, len(frame_time_sec) - window_frames + 1, stride_frames):
        end = start + window_frames
        target = end - 1
        center = start + (window_frames // 2)
        windows.append(
            {
                "start": int(start),
                "end": int(end),
                "center": int(center),
                "target": int(target),
                "time_sec": float(frame_time_sec[target]),
            }
        )
    return windows


def _normalize_phase(phase: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean = stats["phase_mean"].reshape(1, 1, -1)
    std = stats["phase_std"].reshape(1, 1, -1)
    return ((phase - mean) / std).astype(np.float32)


def _normalize_comb(comb: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean = stats["comb_mean"].reshape(1, 1, -1)
    std = stats["comb_std"].reshape(1, 1, -1)
    return ((comb - mean) / std).astype(np.float32)


def _normalize_scalar(scalar: np.ndarray, observed_mask: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    median = stats["scalar_median"].reshape(1, -1)
    iqr = stats["scalar_iqr"].reshape(1, -1)
    normalized = (scalar - median) / iqr
    normalized = np.where(observed_mask > 0.5, normalized, 0.0)
    return normalized.astype(np.float32)


def _normalize_stpacc(stpacc: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean = stats["stpacc_mean"].reshape(1, -1)
    std = stats["stpacc_std"].reshape(1, -1)
    return ((stpacc - mean) / std).astype(np.float32)


def _teacher_field_index(field_names: np.ndarray, key: str) -> int:
    names = [str(v) for v in field_names.tolist()]
    return names.index(key)


def infer_single_wav(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    wav_path: str | Path,
    output_csv: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    audio, sr = load_audio_mono(
        wav_path,
        target_sr=int(cfg["audio"]["target_sr"]),
        max_duration_sec=cfg["audio"].get("max_duration_sec"),
    )
    features = process_audio_array(audio, sr, cfg)
    windows = make_infer_windows(features, cfg)
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    stats = load_normalization_stats(cfg["dataset"]["normalization_path"])
    wav_stem = Path(wav_path).stem
    tracker = DistanceGridRangeTracker(cfg)

    kf_idx = _teacher_field_index(features["teacher_field_names"], "heuristic_distance_kf_cm")
    kf_avail_idx = _teacher_field_index(features["teacher_field_names"], "heuristic_distance_kf_available")
    raw_idx = _teacher_field_index(features["teacher_field_names"], "heuristic_distance_raw_cm")
    raw_avail_idx = _teacher_field_index(features["teacher_field_names"], "heuristic_distance_raw_available")

    rows = []
    model.eval()
    with torch.inference_mode():
        for win in windows:
            _maybe_mark_cudagraph_step_begin()
            scalar_raw = features["scalar_seq"][win["start"] : win["end"]].astype(np.float32)
            observed_mask = features["scalar_observed_mask"][win["start"] : win["end"]]
            reliable_mask = features["scalar_reliable_mask"][win["start"] : win["end"]]
            phase_np = _normalize_phase(features["phase_stft"][win["start"] : win["end"]], stats)
            comb_np = _normalize_comb(features["diff_comb"][win["start"] : win["end"]], stats)
            scalar_np = _normalize_scalar(scalar_raw, observed_mask, stats)

            phase = torch.from_numpy(phase_np.transpose(2, 0, 1))[None].float().to(device)
            comb = torch.from_numpy(comb_np.transpose(2, 0, 1))[None].float().to(device)
            scalar = torch.from_numpy(scalar_np)[None].float().to(device)
            scalar_observed_mask = torch.from_numpy(observed_mask)[None].float().to(device)
            scalar_reliable_mask = torch.from_numpy(reliable_mask)[None].float().to(device)
            frequencies_hz = torch.from_numpy(features["frequencies_hz"])[None].float().to(device)
            if "stpacc" in features:
                stpacc_np = _normalize_stpacc(features["stpacc"][win["start"] : win["end"]].astype(np.float32), stats)
                stpacc = torch.from_numpy(stpacc_np[None, None]).float().to(device)
            else:
                stpacc = torch.zeros((1, 1, phase.shape[2], 64), dtype=torch.float32, device=device)

            pred = model(
                {
                    "phase": phase,
                    "comb": comb,
                    "scalar": scalar,
                    "scalar_observed_mask": scalar_observed_mask,
                    "scalar_reliable_mask": scalar_reliable_mask,
                    "stpacc": stpacc,
                    "frequencies_hz": frequencies_hz,
                }
            )

            target_idx = int(win["target"])
            teacher_row = features["teacher_seq"][target_idx]
            if float(teacher_row[kf_avail_idx]) > 0.5:
                baseline_distance = float(teacher_row[kf_idx])
            elif float(teacher_row[raw_avail_idx]) > 0.5:
                baseline_distance = float(teacher_row[raw_idx])
            else:
                baseline_distance = float("nan")

            posterior = tracker.step(
                measurement_distance_cm=float(pred["measurement_distance_cm"].detach().cpu()[0].item()),
                measurement_logvar=float(pred["measurement_logvar"].detach().cpu()[0].item()),
                measurement_validity_prob=float(pred["measurement_validity_prob"].detach().cpu()[0].item()),
                measurement_entropy=float(pred["measurement_entropy"].detach().cpu()[0].item()),
                measurement_margin=float(pred["measurement_margin"].detach().cpu()[0].item()),
                timestamp_sec=float(win["time_sec"]),
            )

            rows.append(
                {
                    "recording_id": wav_stem,
                    "split": "infer",
                    "time_sec": float(win["time_sec"]),
                    "distance_gt_cm": float("nan"),
                    "baseline_distance_cm": baseline_distance,
                    "measurement_distance_cm": float(pred["measurement_distance_cm"].detach().cpu()[0].item()),
                    "measurement_logvar": float(pred["measurement_logvar"].detach().cpu()[0].item()),
                    "measurement_validity_prob": float(pred["measurement_validity_prob"].detach().cpu()[0].item()),
                    "measurement_entropy": float(pred["measurement_entropy"].detach().cpu()[0].item()),
                    "measurement_top1_cm": float(pred["measurement_top1_cm"].detach().cpu()[0].item()),
                    "measurement_top2_cm": float(pred["measurement_top2_cm"].detach().cpu()[0].item()),
                    "measurement_margin": float(pred["measurement_margin"].detach().cpu()[0].item()),
                    "posterior_distance_cm": float(posterior["posterior_distance_cm"]),
                    "posterior_velocity_cm_s": float(posterior["posterior_velocity_cm_s"]),
                    "posterior_covariance": float(posterior["posterior_covariance"]),
                    "measurement_used_flag": float(posterior["measurement_used_flag"]),
                    "R_eff": float(posterior["R_eff"]),
                    "valid_dist_gt_mask": 0.0,
                    "measurement_target_source": "infer_only",
                }
            )

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

    return {
        "num_windows": len(rows),
        "output_csv": str(output_csv),
        "last_prediction": rows[-1] if rows else None,
    }
