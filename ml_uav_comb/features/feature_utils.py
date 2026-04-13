"""Utility helpers shared across feature extraction, export, and training."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml
from scipy import signal

from ml_uav_comb.features.direction import SHIFT_TO_NUM
from ml_uav_comb.training.target_space import encode_distance_target


SIGN_TO_ID = {
    "approach": 0,
    "retreat": 1,
    "unknown": 2,
}

ID_TO_SIGN = {value: key for key, value in SIGN_TO_ID.items()}

CONFIDENCE_ALIASES = ("valid_mask", "valid", "reliable", "confidence_valid")
DISTANCE_VALID_ALIASES = ("distance_valid",)
SIGN_ANNOTATION_ALIASES = ("sign_annotated",)
V_PERP_MPS_ALIASES = ("v_perp_mps", "velocity_mps", "speed_mps", "radial_velocity_mps", "v_mps")
OBSERVABILITY_SCORE_ALIASES = ("observability_score_res",)
PATTERN_LABEL_ALIASES = ("pattern_label_res",)

DEFAULT_RESOLUTION_CENTER_FREQ_HZ = 3000.0
DEFAULT_RESOLUTION_BIN_WIDTH_HZ = 93.75
DEFAULT_RESOLUTION_DELTA_T_SEC = 0.048
DEFAULT_RESOLUTION_VELOCITY_COEFF_MPS_PER_M = (
    DEFAULT_RESOLUTION_BIN_WIDTH_HZ / (DEFAULT_RESOLUTION_CENTER_FREQ_HZ * DEFAULT_RESOLUTION_DELTA_T_SEC)
)
DEFAULT_PATTERN_SOFT_TARGET_LOWER = 0.8
DEFAULT_PATTERN_SOFT_TARGET_UPPER = 1.2


class TerminalProgressBar:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        width: int = 28,
        stream: Any = None,
        enabled: bool = True,
    ) -> None:
        self.label = str(label)
        self.total = max(0, int(total))
        self.width = max(10, int(width))
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = bool(enabled)
        self.start_time = time.monotonic()
        self._last_line = ""

    def update(self, done: int, *, extra: str = "") -> None:
        if not self.enabled:
            return
        clipped_done = min(max(0, int(done)), self.total if self.total > 0 else max(1, int(done)))
        total = self.total if self.total > 0 else max(1, clipped_done)
        ratio = float(clipped_done) / float(total)
        filled = min(self.width, int(round(self.width * ratio)))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(0.0, time.monotonic() - self.start_time)
        suffix = f" {extra}" if extra else ""
        line = f"{self.label:<20} [{bar}] {clipped_done:>4}/{total:<4} {ratio * 100:>6.2f}% {elapsed:>6.1f}s{suffix}"
        padding = ""
        if len(self._last_line) > len(line):
            padding = " " * (len(self._last_line) - len(line))
        self.stream.write("\r" + line + padding)
        self.stream.flush()
        self._last_line = line

    def finish(self, *, extra: str = "") -> None:
        self.update(self.total, extra=extra)
        if not self.enabled:
            return
        self.stream.write("\n")
        self.stream.flush()
        self._last_line = ""


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def metadata_path_for_index(index_path: str | Path) -> Path:
    index_path = Path(index_path)
    return index_path.with_name(index_path.stem + "_meta.json")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return None
        return result
    except Exception:
        return None


def load_audio_mono(
    path: str | Path,
    target_sr: int,
    max_duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    if sr != target_sr:
        gcd = int(np.gcd(int(sr), int(target_sr)))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        audio = signal.resample_poly(audio, up=up, down=down).astype(np.float32)
        sr = target_sr

    if max_duration_sec is not None:
        max_len = int(round(max_duration_sec * sr))
        audio = audio[:max_len]

    return audio.astype(np.float32), int(sr)


def _column_present(rows: Iterable[Dict[str, Any]], keys: Iterable[str]) -> bool:
    key_set = tuple(keys)
    for row in rows:
        for key in key_set:
            if key in row:
                return True
    return False


def _first_present_value(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _has_explicit_string(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def load_optional_labels(
    label_path: Optional[str | Path],
    recording_id: str,
) -> Optional[Dict[str, np.ndarray]]:
    if label_path is None:
        return None

    path = Path(label_path)
    if not path.exists():
        return None

    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded = loaded.get("labels", [])
        for item in loaded:
            if isinstance(item, dict):
                rows.append(dict(item))
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))

    if not rows:
        return None

    has_recording_id_column = _column_present(rows, ("recording_id",))
    if has_recording_id_column:
        filtered_rows = []
        for row in rows:
            row_recording_id = str(row.get("recording_id") or "").strip()
            if row_recording_id == str(recording_id):
                filtered_rows.append(dict(row))
        rows = filtered_rows
    else:
        for row in rows:
            row["recording_id"] = recording_id

    if not rows:
        return None

    has_conf_column = _column_present(rows, CONFIDENCE_ALIASES)
    has_distance_valid_column = _column_present(rows, DISTANCE_VALID_ALIASES)
    has_sign_annotation_column = _column_present(rows, SIGN_ANNOTATION_ALIASES)

    times: List[float] = []
    distance_cm: List[float] = []
    distance_valid: List[float] = []
    motion_sign: List[str] = []
    explicit_motion_sign_mask: List[float] = []
    row_has_distance_value: List[float] = []
    row_has_distance_valid: List[float] = []
    row_has_motion_sign_value: List[float] = []
    row_has_sign_annotation: List[float] = []
    confidence_values: List[float] = []
    row_has_conf_value: List[float] = []
    v_perp_mps: List[float] = []
    row_has_v_perp_value: List[float] = []
    observability_score_res: List[float] = []
    row_has_observability_score_value: List[float] = []
    pattern_label_res: List[float] = []
    row_has_pattern_label_value: List[float] = []

    for row in rows:
        t = safe_float(row.get("time_sec") or row.get("time") or row.get("t"))
        if t is None:
            continue
        d = safe_float(row.get("distance_cm"))
        if d is None and row.get("distance_mm") not in (None, ""):
            d_mm = safe_float(row.get("distance_mm"))
            if d_mm is not None:
                d = d_mm / 10.0

        times.append(float(t))
        row_has_distance_value.append(1.0 if d is not None else 0.0)
        distance_cm.append(np.nan if d is None else float(d))

        dist_valid = safe_float(_first_present_value(row, DISTANCE_VALID_ALIASES))
        row_has_distance_valid.append(1.0 if dist_valid is not None else 0.0)
        distance_valid.append(np.nan if dist_valid is None else float(dist_valid))

        raw_sign_value = row.get("motion_sign")
        sign_text = str(raw_sign_value or "unknown").strip().lower()
        if sign_text not in SIGN_TO_ID:
            sign_text = "unknown"
        motion_sign.append(sign_text)
        row_has_motion_sign_value.append(1.0 if _has_explicit_string(raw_sign_value) else 0.0)

        sign_annotated = safe_float(_first_present_value(row, SIGN_ANNOTATION_ALIASES))
        row_has_sign_annotation.append(1.0 if sign_annotated is not None else 0.0)
        explicit_motion_sign_mask.append(
            0.0 if sign_annotated is None else float(sign_annotated >= 0.5)
        )

        conf_val = None
        for alias in CONFIDENCE_ALIASES:
            if alias in row:
                conf_val = safe_float(row.get(alias))
                break
        row_has_conf_value.append(1.0 if conf_val is not None else 0.0)
        confidence_values.append(np.nan if conf_val is None else float(conf_val))

        velocity_val = safe_float(_first_present_value(row, V_PERP_MPS_ALIASES))
        row_has_v_perp_value.append(1.0 if velocity_val is not None else 0.0)
        v_perp_mps.append(np.nan if velocity_val is None else float(velocity_val))

        score_val = safe_float(_first_present_value(row, OBSERVABILITY_SCORE_ALIASES))
        row_has_observability_score_value.append(1.0 if score_val is not None else 0.0)
        observability_score_res.append(np.nan if score_val is None else float(score_val))

        pattern_val = safe_float(_first_present_value(row, PATTERN_LABEL_ALIASES))
        row_has_pattern_label_value.append(1.0 if pattern_val is not None else 0.0)
        pattern_label_res.append(np.nan if pattern_val is None else float(pattern_val))

    if not times:
        return None

    times_np = np.asarray(times, dtype=np.float32)
    distances_np = np.asarray(distance_cm, dtype=np.float32)
    distance_valid_np = np.asarray(distance_valid, dtype=np.float32)
    motion_sign_np = np.asarray(motion_sign, dtype=object)
    explicit_motion_sign_mask_np = np.asarray(explicit_motion_sign_mask, dtype=np.float32)
    row_has_distance_value_np = np.asarray(row_has_distance_value, dtype=np.float32)
    row_has_distance_valid_np = np.asarray(row_has_distance_valid, dtype=np.float32)
    row_has_motion_sign_value_np = np.asarray(row_has_motion_sign_value, dtype=np.float32)
    row_has_sign_annotation_np = np.asarray(row_has_sign_annotation, dtype=np.float32)
    confidence_np = np.asarray(confidence_values, dtype=np.float32)
    row_has_conf_value_np = np.asarray(row_has_conf_value, dtype=np.float32)
    v_perp_np = np.asarray(v_perp_mps, dtype=np.float32)
    row_has_v_perp_value_np = np.asarray(row_has_v_perp_value, dtype=np.float32)
    observability_score_np = np.asarray(observability_score_res, dtype=np.float32)
    row_has_observability_score_value_np = np.asarray(row_has_observability_score_value, dtype=np.float32)
    pattern_label_np = np.asarray(pattern_label_res, dtype=np.float32)
    row_has_pattern_label_value_np = np.asarray(row_has_pattern_label_value, dtype=np.float32)

    order = np.argsort(times_np)
    times_np = times_np[order]
    distances_np = distances_np[order]
    distance_valid_np = distance_valid_np[order]
    motion_sign_np = motion_sign_np[order]
    explicit_motion_sign_mask_np = explicit_motion_sign_mask_np[order]
    row_has_distance_value_np = row_has_distance_value_np[order]
    row_has_distance_valid_np = row_has_distance_valid_np[order]
    row_has_motion_sign_value_np = row_has_motion_sign_value_np[order]
    row_has_sign_annotation_np = row_has_sign_annotation_np[order]
    confidence_np = confidence_np[order]
    row_has_conf_value_np = row_has_conf_value_np[order]
    v_perp_np = v_perp_np[order]
    row_has_v_perp_value_np = row_has_v_perp_value_np[order]
    observability_score_np = observability_score_np[order]
    row_has_observability_score_value_np = row_has_observability_score_value_np[order]
    pattern_label_np = pattern_label_np[order]
    row_has_pattern_label_value_np = row_has_pattern_label_value_np[order]

    derived_observability_np = resolution_observability_score(
        distance_m=(distances_np.astype(np.float32) / 100.0),
        velocity_mps=v_perp_np.astype(np.float32),
    ).astype(np.float32)
    use_derived_score = (~np.isfinite(observability_score_np)) & np.isfinite(derived_observability_np)
    observability_score_np = np.where(use_derived_score, derived_observability_np, observability_score_np).astype(np.float32)
    row_has_observability_score_value_np = np.where(
        np.isfinite(observability_score_np),
        1.0,
        row_has_observability_score_value_np,
    ).astype(np.float32)

    derived_pattern_np = np.where(
        np.isfinite(observability_score_np),
        (observability_score_np >= 1.0).astype(np.float32),
        np.nan,
    ).astype(np.float32)
    use_derived_pattern = (~np.isfinite(pattern_label_np)) & np.isfinite(derived_pattern_np)
    pattern_label_np = np.where(use_derived_pattern, derived_pattern_np, pattern_label_np).astype(np.float32)
    row_has_pattern_label_value_np = np.where(
        np.isfinite(pattern_label_np),
        1.0,
        row_has_pattern_label_value_np,
    ).astype(np.float32)

    return {
        "time_sec": times_np,
        "distance_cm": distances_np,
        "distance_valid": distance_valid_np,
        "motion_sign": motion_sign_np,
        "explicit_motion_sign_mask": explicit_motion_sign_mask_np,
        "row_has_distance_value": row_has_distance_value_np,
        "row_has_distance_valid": row_has_distance_valid_np,
        "row_has_motion_sign_value": row_has_motion_sign_value_np,
        "row_has_sign_annotation": row_has_sign_annotation_np,
        "confidence_valid": confidence_np.astype(np.float32),
        "row_has_conf_value": row_has_conf_value_np,
        "v_perp_mps": v_perp_np.astype(np.float32),
        "row_has_v_perp_value": row_has_v_perp_value_np.astype(np.float32),
        "observability_score_res": observability_score_np.astype(np.float32),
        "row_has_observability_score_value": row_has_observability_score_value_np.astype(np.float32),
        "pattern_label_res": pattern_label_np.astype(np.float32),
        "row_has_pattern_label_value": row_has_pattern_label_value_np.astype(np.float32),
        "has_conf_gt": np.asarray([1.0 if has_conf_column else 0.0], dtype=np.float32),
        "has_distance_valid": np.asarray([1.0 if has_distance_valid_column else 0.0], dtype=np.float32),
        "has_sign_annotation": np.asarray([1.0 if has_sign_annotation_column else 0.0], dtype=np.float32),
    }


def _numeric_series(
    label_dict: Dict[str, np.ndarray],
    value_key: str,
    present_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(label_dict["time_sec"], dtype=np.float32)
    values = np.asarray(label_dict[value_key], dtype=np.float32)
    mask = np.isfinite(values)
    if present_key is not None and present_key in label_dict:
        mask &= np.asarray(label_dict[present_key], dtype=np.float32) > 0.5
    return times[mask], values[mask]


def _interp_numeric(time_sec: float, times: np.ndarray, values: np.ndarray) -> Optional[float]:
    if times.size == 0:
        return None
    if time_sec < float(times[0]) or time_sec > float(times[-1]):
        return None
    if times.size == 1:
        return float(values[0]) if abs(float(times[0]) - float(time_sec)) <= 1e-6 else None
    return float(np.interp(time_sec, times, values))


def _nearest_numeric(time_sec: float, times: np.ndarray, values: np.ndarray) -> Optional[float]:
    if times.size == 0:
        return None
    if time_sec < float(times[0]) or time_sec > float(times[-1]):
        return None
    idx = int(np.argmin(np.abs(times - time_sec)))
    return float(values[idx])


def resolution_velocity_floor_mps(distance_m: Any, coeff_mps_per_m: float = DEFAULT_RESOLUTION_VELOCITY_COEFF_MPS_PER_M) -> Any:
    distance_np = np.asarray(distance_m, dtype=np.float32)
    return np.maximum(distance_np, 0.0) * np.float32(coeff_mps_per_m)


def resolution_observability_score(
    distance_m: Any,
    velocity_mps: Any,
    coeff_mps_per_m: float = DEFAULT_RESOLUTION_VELOCITY_COEFF_MPS_PER_M,
) -> Any:
    distance_np = np.asarray(distance_m, dtype=np.float32)
    velocity_np = np.asarray(velocity_mps, dtype=np.float32)
    required = np.maximum(distance_np, 1e-6) * np.float32(coeff_mps_per_m)
    score = np.abs(velocity_np) / np.maximum(required, 1e-6)
    invalid = (~np.isfinite(distance_np)) | (~np.isfinite(velocity_np)) | (distance_np <= 0.0)
    return np.where(invalid, np.nan, score).astype(np.float32)


def resolution_pattern_soft_target(
    observability_score: Any,
    lower: float = DEFAULT_PATTERN_SOFT_TARGET_LOWER,
    upper: float = DEFAULT_PATTERN_SOFT_TARGET_UPPER,
) -> Any:
    score_np = np.asarray(observability_score, dtype=np.float32)
    denom = max(float(upper) - float(lower), 1e-6)
    scaled = (score_np - float(lower)) / float(denom)
    soft = np.clip(scaled, 0.0, 1.0)
    return np.where(np.isfinite(score_np), soft, np.nan).astype(np.float32)


def resolution_pattern_binary_target(
    observability_score: Any,
    threshold: float = 1.0,
) -> Any:
    score_np = np.asarray(observability_score, dtype=np.float32)
    target = (score_np >= float(threshold)).astype(np.float32)
    return np.where(np.isfinite(score_np), target, np.nan).astype(np.float32)


def interpolate_distance(time_sec: float, label_dict: Dict[str, np.ndarray]) -> Optional[float]:
    times, distances = _numeric_series(
        label_dict,
        "distance_cm",
        present_key="row_has_distance_value",
    )
    return _interp_numeric(time_sec, times, distances)


def interpolate_distance_valid(time_sec: float, label_dict: Dict[str, np.ndarray]) -> Optional[float]:
    has_distance_valid = bool(label_dict.get("has_distance_valid", np.asarray([0.0], dtype=np.float32))[0] > 0.5)
    if not has_distance_valid:
        return None
    times, values = _numeric_series(
        label_dict,
        "distance_valid",
        present_key="row_has_distance_valid",
    )
    return _interp_numeric(time_sec, times, values)


def interpolate_confidence_valid(time_sec: float, label_dict: Dict[str, np.ndarray]) -> Optional[float]:
    has_conf_gt = bool(label_dict["has_conf_gt"][0] > 0.5)
    if not has_conf_gt:
        return None
    times, confidence = _numeric_series(
        label_dict,
        "confidence_valid",
        present_key="row_has_conf_value",
    )
    return _interp_numeric(time_sec, times, confidence)


def interpolate_v_perp_mps(time_sec: float, label_dict: Dict[str, np.ndarray]) -> Optional[float]:
    times, velocity = _numeric_series(
        label_dict,
        "v_perp_mps",
        present_key="row_has_v_perp_value",
    )
    return _interp_numeric(time_sec, times, velocity)


def interpolate_observability_score_res(time_sec: float, label_dict: Dict[str, np.ndarray]) -> Optional[float]:
    times, values = _numeric_series(
        label_dict,
        "observability_score_res",
        present_key="row_has_observability_score_value",
    )
    return _interp_numeric(time_sec, times, values)


def interpolate_pattern_label_res(time_sec: float, label_dict: Dict[str, np.ndarray]) -> Optional[float]:
    times, values = _numeric_series(
        label_dict,
        "pattern_label_res",
        present_key="row_has_pattern_label_value",
    )
    return _nearest_numeric(time_sec, times, values)


def nearest_motion_sign(
    time_sec: float,
    label_dict: Dict[str, np.ndarray],
    require_annotation: bool = True,
) -> Optional[str]:
    times = np.asarray(label_dict["time_sec"], dtype=np.float32)
    signs = np.asarray(label_dict["motion_sign"], dtype=object)
    if require_annotation:
        mask = np.asarray(label_dict["explicit_motion_sign_mask"], dtype=np.float32) > 0.5
    else:
        mask = np.asarray(
            label_dict.get(
                "row_has_motion_sign_value",
                np.ones_like(times, dtype=np.float32),
            ),
            dtype=np.float32,
        ) > 0.5

    times = times[mask]
    signs = signs[mask]
    if times.size == 0:
        return None
    idx = int(np.argmin(np.abs(times - time_sec)))
    if time_sec < float(times[0]) or time_sec > float(times[-1]):
        return None
    sign_text = str(signs[idx]).lower()
    return sign_text if sign_text in SIGN_TO_ID else None


def derive_motion_sign(slope_cm_per_sec: Optional[float], eps_cm_per_sec: float) -> str:
    if slope_cm_per_sec is None:
        return "unknown"
    if slope_cm_per_sec < -eps_cm_per_sec:
        return "approach"
    if slope_cm_per_sec > eps_cm_per_sec:
        return "retreat"
    return "unknown"


def fit_local_motion_sign(
    center_time_sec: float,
    label_dict: Dict[str, np.ndarray],
    half_window_sec: float,
    eps_cm_per_sec: float,
    min_points: int,
    max_fit_rmse_cm: float,
) -> Dict[str, Any]:
    times, distances = _numeric_series(
        label_dict,
        "distance_cm",
        present_key="row_has_distance_value",
    )
    mask = (times >= (center_time_sec - half_window_sec)) & (times <= (center_time_sec + half_window_sec))
    if int(np.sum(mask)) < int(min_points):
        return {
            "sign_label": "unknown",
            "sign_train_mask": 0.0,
            "slope_cm_per_sec": 0.0,
            "fit_rmse_cm": float("inf"),
        }

    local_t = times[mask].astype(np.float64)
    local_d = distances[mask].astype(np.float64)
    t_centered = local_t - float(center_time_sec)
    slope, intercept = np.polyfit(t_centered, local_d, deg=1)
    fitted = slope * t_centered + intercept
    fit_rmse = float(np.sqrt(np.mean((fitted - local_d) ** 2)))
    if not np.isfinite(fit_rmse) or fit_rmse > float(max_fit_rmse_cm):
        return {
            "sign_label": "unknown",
            "sign_train_mask": 0.0,
            "slope_cm_per_sec": float(slope),
            "fit_rmse_cm": fit_rmse,
        }

    sign_label = derive_motion_sign(float(slope), eps_cm_per_sec=float(eps_cm_per_sec))
    return {
        "sign_label": sign_label,
        "sign_train_mask": 1.0,
        "slope_cm_per_sec": float(slope),
        "fit_rmse_cm": fit_rmse,
    }


def compute_single_recording_split_intervals(
    total_duration_sec: float,
    split_ratio: Iterable[float],
    split_names: Iterable[str],
    margin_sec: float,
) -> Dict[str, Tuple[float, float]]:
    ratios = np.asarray(list(split_ratio), dtype=np.float64)
    names = list(split_names)
    if len(ratios) != len(names):
        raise ValueError("split_ratio and split_names length mismatch")
    ratios = ratios / ratios.sum()
    boundaries = [0.0]
    cumulative = 0.0
    for ratio in ratios:
        cumulative += float(ratio)
        boundaries.append(float(total_duration_sec) * cumulative)

    intervals: Dict[str, Tuple[float, float]] = {}
    half_margin = float(margin_sec) / 2.0
    for idx, name in enumerate(names):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        inner_start = start if idx == 0 else start + half_margin
        inner_end = end if idx == len(names) - 1 else end - half_margin
        if inner_end <= inner_start:
            inner_start = start
            inner_end = end
        intervals[name] = (float(inner_start), float(inner_end))
    return intervals


def choose_recording_level_split(
    recording_id: str,
    recording_order: Iterable[str],
    split_ratio: Iterable[float],
    split_names: Iterable[str],
    seed: int = 0,
) -> str:
    names = list(split_names)
    ordered = list(recording_order)
    if recording_id not in ordered:
        raise ValueError(f"recording_id {recording_id} not found in recording_order")
    shuffled = list(ordered)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled)
    ratios = np.asarray(list(split_ratio), dtype=np.float64)
    ratios = ratios / ratios.sum()
    frac = float(shuffled.index(recording_id) + 0.5) / float(len(shuffled))
    cumulative = 0.0
    for ratio, name in zip(ratios, names):
        cumulative += float(ratio)
        if frac <= cumulative:
            return name
    return names[-1]


def encode_distance_or_nan(distance_cm: Optional[float], target_space: str) -> float:
    if distance_cm is None or not np.isfinite(distance_cm):
        return float("nan")
    return float(encode_distance_target(np.asarray(distance_cm, dtype=np.float32), target_space))


def shift_num(direction_text: str) -> float:
    return float(SHIFT_TO_NUM.get(direction_text, 0.0))
