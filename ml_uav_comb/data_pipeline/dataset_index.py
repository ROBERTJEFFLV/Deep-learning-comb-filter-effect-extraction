"""Dataset index generation and label binding."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ml_uav_comb.features.feature_utils import (
    SIGN_TO_ID,
    encode_distance_or_nan,
    interpolate_confidence_valid,
    interpolate_distance,
    interpolate_distance_valid,
)


def build_distance_grid_cm(cfg: Dict[str, Any]) -> np.ndarray:
    model_cfg = cfg["model"]
    num_candidates = int(model_cfg.get("num_candidates", 64))
    min_cm = float(model_cfg.get("distance_grid_cm_min", 20.0))
    max_cm = float(model_cfg.get("distance_grid_cm_max", 300.0))
    mode = str(model_cfg.get("distance_grid_mode", "uniform")).strip().lower()
    min_cm = max(min_cm, 1e-3)
    max_cm = max(max_cm, min_cm + 1e-3)
    if num_candidates < 2:
        raise ValueError("model.num_candidates must be >= 2")
    if mode == "uniform":
        return np.linspace(min_cm, max_cm, num_candidates, dtype=np.float32)
    if mode == "log":
        return np.logspace(np.log10(min_cm), np.log10(max_cm), num_candidates, dtype=np.float32)
    raise ValueError(f"unsupported distance_grid_mode: {mode}")


def gaussian_soft_target_distance_cm(
    distance_grid_cm: np.ndarray,
    target_distance_cm: float,
    sigma_cm: float,
    eps: float = 1e-6,
) -> np.ndarray:
    distance_grid_cm = np.asarray(distance_grid_cm, dtype=np.float32)
    if not np.isfinite(float(target_distance_cm)):
        return np.zeros_like(distance_grid_cm, dtype=np.float32)
    sigma_cm = max(float(sigma_cm), 1e-6)
    diff = (distance_grid_cm - float(target_distance_cm)) / sigma_cm
    weights = np.exp(-0.5 * np.square(diff)).astype(np.float32) + float(eps)
    denom = float(np.sum(weights))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.zeros_like(distance_grid_cm, dtype=np.float32)
    return (weights / denom).astype(np.float32)


def _field_index(field_names: np.ndarray, field_name: str) -> int:
    names = [str(v) for v in field_names.tolist()]
    return names.index(field_name)


def compute_heuristic_confidence(
    target_scalar: np.ndarray,
    target_reliable_mask: np.ndarray,
    field_names: np.ndarray,
    cfg: Dict[str, Any],
) -> float:
    min_amplitude = float(cfg["front_end"]["min_amplitude"])
    lag_idx = _field_index(field_names, "comb_shift_lag")
    rho_idx = _field_index(field_names, "comb_shift_rho")
    is_sound_idx = _field_index(field_names, "is_sound_present")
    amp_idx = _field_index(field_names, "sum_abs_d1_smooth")
    lag_reliable = target_reliable_mask[lag_idx] > 0.5
    rho_reliable = target_reliable_mask[rho_idx] > 0.5
    return float(
        (target_scalar[is_sound_idx] > 0.5)
        and (target_scalar[amp_idx] >= min_amplitude)
        and lag_reliable
        and rho_reliable
    )


def _pick_heuristic_distance(target_teacher: np.ndarray, teacher_field_names: np.ndarray) -> float:
    names = [str(v) for v in teacher_field_names.tolist()]
    kf_idx = names.index("heuristic_distance_kf_cm")
    raw_idx = names.index("heuristic_distance_raw_cm")
    kf_avail_idx = names.index("heuristic_distance_kf_available")
    raw_avail_idx = names.index("heuristic_distance_raw_available")
    if target_teacher[kf_avail_idx] > 0.5:
        return float(target_teacher[kf_idx])
    if target_teacher[raw_avail_idx] > 0.5:
        return float(target_teacher[raw_idx])
    return float("nan")


def _dataset_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    return bool(cfg["dataset"].get(key, default))


def _dataset_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    return float(cfg["dataset"].get(key, default))


def _distance_loss_weight(
    train_mask: float,
    dist_reliable_mask: float,
    cfg: Dict[str, Any],
) -> float:
    if train_mask <= 0.5:
        return 0.0
    if not _dataset_bool(cfg, "use_physics_gate_as_distance_weight", True):
        return 1.0
    floor = _dataset_float(cfg, "distance_weight_floor", 0.25)
    reliable = _dataset_float(cfg, "distance_weight_reliable", 1.0)
    reliability = float(np.clip(dist_reliable_mask, 0.0, 1.0))
    return float(floor + reliability * (reliable - floor))


def infer_label_bundle(
    target_time_sec: float,
    target_scalar: np.ndarray,
    target_reliable_mask: np.ndarray,
    target_teacher: np.ndarray,
    scalar_field_names: np.ndarray,
    teacher_field_names: np.ndarray,
    label_dict: Optional[Dict[str, np.ndarray]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    target_space = str(cfg["dataset"]["target_space"])
    supervision_mode = str(cfg["dataset"].get("supervision_mode", "strict_gt")).strip().lower()
    distance_grid_cm = build_distance_grid_cm(cfg)
    sigma_cm = float(cfg["model"].get("distance_target_sigma_cm", 12.0))

    heuristic_distance = _pick_heuristic_distance(target_teacher, teacher_field_names)
    heuristic_confidence = compute_heuristic_confidence(
        target_scalar=target_scalar,
        target_reliable_mask=target_reliable_mask,
        field_names=scalar_field_names,
        cfg=cfg,
    )
    dist_reliable_mask = float(heuristic_confidence)

    empty_grid_text = ";".join(f"{float(v):.8f}" for v in np.zeros_like(distance_grid_cm, dtype=np.float32))
    bundle = {
        "measurement_distance_target_cm": np.nan,
        "measurement_distance_train_mask": 0.0,
        "measurement_validity_target": float(dist_reliable_mask),
        "measurement_validity_train_mask": 1.0,
        "measurement_target_source": "unavailable",
        "validity_target_source": "dist_reliable_fallback",
        "distance_target_grid": empty_grid_text,
        "valid_dist_gt_mask": 0.0,
        "external_distance_valid_mask": 0.0,
        "dist_reliable_mask": dist_reliable_mask,
        "dist_loss_weight": 0.0,
        "heuristic_distance_cm": heuristic_distance,
        "heuristic_distance_target": encode_distance_or_nan(heuristic_distance, target_space),
        "heuristic_distance_available": 0.0 if np.isnan(heuristic_distance) else 1.0,
        "heuristic_confidence": heuristic_confidence,
        "label_source": "no_external_labels",
        "supervision_type": supervision_mode,
        # Legacy aliases for compatibility
        "distance_cm": np.nan,
        "distance_target": np.nan,
        "distance_target_source": "unavailable",
        "dist_train_mask": 0.0,
        "sign_label": SIGN_TO_ID["unknown"],
        "sign_target_source": "disabled",
        "valid_sign_gt_mask": 0.0,
        "sign_train_mask": 0.0,
        "confidence_target": float(dist_reliable_mask),
        "confidence_target_source": "dist_reliable_fallback",
        "valid_conf_gt_mask": 0.0,
        "conf_train_mask": 1.0,
    }

    if label_dict is None:
        return bundle

    distance_cm = interpolate_distance(target_time_sec, label_dict)
    valid_dist_gt_mask = 0.0 if distance_cm is None else 1.0
    has_distance_valid = bool(label_dict.get("has_distance_valid", np.asarray([0.0], dtype=np.float32))[0] > 0.5)
    explicit_distance_valid = interpolate_distance_valid(target_time_sec, label_dict)
    if valid_dist_gt_mask <= 0.5:
        external_distance_valid_mask = 0.0
    elif has_distance_valid:
        external_distance_valid_mask = (
            0.0 if explicit_distance_valid is None else float(explicit_distance_valid >= 0.5)
        )
    else:
        external_distance_valid_mask = 1.0 if _dataset_bool(cfg, "default_distance_valid_when_missing", True) else 0.0

    distance_train_mask = float(valid_dist_gt_mask * external_distance_valid_mask)
    if _dataset_bool(cfg, "use_physics_gate_as_distance_mask", False):
        distance_train_mask *= float(dist_reliable_mask >= 0.5)

    if valid_dist_gt_mask <= 0.5:
        measurement_target_source = "unavailable"
        target_grid = np.zeros_like(distance_grid_cm, dtype=np.float32)
    elif external_distance_valid_mask > 0.5:
        measurement_target_source = "external_distance_gt"
        target_grid = gaussian_soft_target_distance_cm(
            distance_grid_cm=distance_grid_cm,
            target_distance_cm=float(distance_cm),
            sigma_cm=sigma_cm,
            eps=1e-6,
        )
    else:
        measurement_target_source = "external_distance_gt_invalid"
        target_grid = gaussian_soft_target_distance_cm(
            distance_grid_cm=distance_grid_cm,
            target_distance_cm=float(distance_cm),
            sigma_cm=sigma_cm,
            eps=1e-6,
        )

    confidence_gt = interpolate_confidence_valid(target_time_sec, label_dict)
    if confidence_gt is not None:
        measurement_validity_target = float(confidence_gt >= 0.5)
        measurement_validity_train_mask = 1.0
        validity_target_source = "explicit_valid_mask_gt"
        valid_conf_gt_mask = 1.0
    else:
        measurement_validity_target = float(dist_reliable_mask)
        measurement_validity_train_mask = 1.0
        validity_target_source = "dist_reliable_fallback"
        valid_conf_gt_mask = 0.0

    bundle.update(
        {
            "measurement_distance_target_cm": np.nan if distance_cm is None else float(distance_cm),
            "measurement_distance_train_mask": float(distance_train_mask),
            "measurement_validity_target": float(measurement_validity_target),
            "measurement_validity_train_mask": float(measurement_validity_train_mask),
            "measurement_target_source": measurement_target_source,
            "validity_target_source": validity_target_source,
            "distance_target_grid": ";".join(f"{float(v):.8f}" for v in target_grid.astype(np.float32)),
            "valid_dist_gt_mask": float(valid_dist_gt_mask),
            "external_distance_valid_mask": float(external_distance_valid_mask),
            "dist_loss_weight": _distance_loss_weight(
                train_mask=float(distance_train_mask),
                dist_reliable_mask=float(dist_reliable_mask),
                cfg=cfg,
            ),
            "label_source": f"measurement:{measurement_target_source}|validity:{validity_target_source}",
            # Legacy aliases
            "distance_cm": np.nan if distance_cm is None else float(distance_cm),
            "distance_target": encode_distance_or_nan(distance_cm, target_space),
            "distance_target_source": measurement_target_source,
            "dist_train_mask": float(distance_train_mask),
            "confidence_target": float(measurement_validity_target),
            "confidence_target_source": validity_target_source,
            "valid_conf_gt_mask": float(valid_conf_gt_mask),
            "conf_train_mask": float(measurement_validity_train_mask),
        }
    )
    return bundle


def _resolve_split_name(
    start_time_sec: float,
    end_time_sec: float,
    split_override: Optional[str],
    split_intervals: Optional[Dict[str, tuple[float, float]]],
) -> Optional[str]:
    if split_override is not None:
        return split_override
    if split_intervals is None:
        return "train"
    for split_name, (split_start, split_end) in split_intervals.items():
        if start_time_sec >= split_start and end_time_sec <= split_end:
            return split_name
    return None


def _resolve_target_frame(
    start_frame: int,
    end_frame: int,
    window_anchor: str,
    target_position: str,
) -> int:
    if window_anchor == "trailing" and target_position == "window_end":
        return int(end_frame - 1)
    if target_position == "window_end":
        return int(end_frame - 1)
    return int(start_frame + ((end_frame - start_frame) // 2))


def build_dataset_index_for_cache(
    recording_id: str,
    cache_path: str | Path,
    label_dict: Optional[Dict[str, np.ndarray]],
    cfg: Dict[str, Any],
    split_override: Optional[str] = None,
    split_intervals: Optional[Dict[str, tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    cache = np.load(cache_path, allow_pickle=True)
    version = int(np.asarray(cache["schema_version"]).reshape(-1)[0])
    if version != 2:
        raise ValueError(f"expected cache schema_version 2, got {version} for {cache_path}")

    frame_time_sec = cache["frame_time_sec"]
    scalar_seq = cache["scalar_seq"]
    scalar_field_names = cache["scalar_field_names"]
    scalar_reliable_mask = cache["scalar_reliable_mask"]
    teacher_seq = cache["teacher_seq"]
    teacher_field_names = cache["teacher_field_names"]

    hop_sec = float(cfg["audio"]["hop_len"]) / float(cfg["audio"]["target_sr"])
    window_frames = int(round(float(cfg["dataset"]["window_sec"]) / hop_sec))
    stride_frames = int(round(float(cfg["dataset"]["stride_sec"]) / hop_sec))
    stride_frames = max(1, stride_frames)

    window_anchor = str(cfg["dataset"].get("window_anchor", "trailing")).strip().lower()
    target_position = str(cfg["dataset"].get("target_position", "window_end")).strip().lower()

    rows: List[Dict[str, Any]] = []
    if len(frame_time_sec) < window_frames:
        return rows

    for start_frame in range(0, len(frame_time_sec) - window_frames + 1, stride_frames):
        end_frame = start_frame + window_frames
        center_frame = start_frame + (window_frames // 2)
        target_frame = _resolve_target_frame(
            start_frame=start_frame,
            end_frame=end_frame,
            window_anchor=window_anchor,
            target_position=target_position,
        )

        start_time_sec = float(frame_time_sec[start_frame])
        end_time_sec = float(frame_time_sec[end_frame - 1])
        split_name = _resolve_split_name(
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
            split_override=split_override,
            split_intervals=split_intervals,
        )
        if split_name is None:
            continue

        center_time_sec = float(frame_time_sec[center_frame])
        target_time_sec = float(frame_time_sec[target_frame])
        label_bundle = infer_label_bundle(
            target_time_sec=target_time_sec,
            target_scalar=scalar_seq[target_frame],
            target_reliable_mask=scalar_reliable_mask[target_frame],
            target_teacher=teacher_seq[target_frame],
            scalar_field_names=scalar_field_names,
            teacher_field_names=teacher_field_names,
            label_dict=label_dict,
            cfg=cfg,
        )
        rows.append(
            {
                "recording_id": recording_id,
                "split": split_name,
                "cache_path": str(cache_path),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "center_frame": center_frame,
                "center_time_sec": center_time_sec,
                "target_frame": target_frame,
                "target_time_sec": target_time_sec,
                **label_bundle,
            }
        )
    return rows


def write_dataset_index(rows: List[Dict[str, Any]], index_path: str | Path) -> None:
    if not rows:
        raise ValueError("dataset index rows are empty")
    fieldnames = [
        "recording_id",
        "split",
        "cache_path",
        "start_frame",
        "end_frame",
        "center_frame",
        "center_time_sec",
        "target_frame",
        "target_time_sec",
        "label_source",
        "supervision_type",
        "measurement_distance_target_cm",
        "measurement_distance_train_mask",
        "measurement_validity_target",
        "measurement_validity_train_mask",
        "measurement_target_source",
        "validity_target_source",
        "distance_target_grid",
        "distance_cm",
        "distance_target",
        "distance_target_source",
        "sign_label",
        "sign_target_source",
        "confidence_target",
        "confidence_target_source",
        "valid_dist_gt_mask",
        "external_distance_valid_mask",
        "dist_reliable_mask",
        "dist_train_mask",
        "dist_loss_weight",
        "valid_sign_gt_mask",
        "sign_train_mask",
        "valid_conf_gt_mask",
        "conf_train_mask",
        "heuristic_distance_cm",
        "heuristic_distance_target",
        "heuristic_distance_available",
        "heuristic_confidence",
    ]
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
