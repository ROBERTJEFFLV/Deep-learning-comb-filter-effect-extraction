"""Dataset for omega-regression cached windows."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ml_uav_comb.data_pipeline.omega_dataset_index import open_omega_index_split
from ml_uav_comb.data_pipeline.omega_normalization import (
    DYNAMIC_CHANNEL_KEYS,
    load_omega_normalization_stats,
)
from ml_uav_comb.features.feature_utils import (
    DEFAULT_PATTERN_SOFT_TARGET_LOWER,
    DEFAULT_PATTERN_SOFT_TARGET_UPPER,
    DEFAULT_RESOLUTION_DELTA_T_SEC,
    metadata_path_for_index,
    resolution_observability_score,
    resolution_pattern_binary_target,
    resolution_pattern_soft_target,
)


def _build_contiguous_recording_segments(recording_code: np.ndarray) -> List[Tuple[int, int, int]]:
    codes = np.asarray(recording_code)
    if int(codes.shape[0]) == 0:
        return []
    segments: List[Tuple[int, int, int]] = []
    seg_start = 0
    current_code = int(codes[0])
    for idx in range(1, int(codes.shape[0])):
        code = int(codes[idx])
        if code != current_code:
            segments.append((int(seg_start), int(idx - seg_start), int(current_code)))
            seg_start = idx
            current_code = code
    segments.append((int(seg_start), int(codes.shape[0] - seg_start), int(current_code)))
    return segments


def _derive_v_perp_from_distance(
    distance_cm: np.ndarray,
    frame_time_sec: np.ndarray,
    *,
    delta_t_sec: float = DEFAULT_RESOLUTION_DELTA_T_SEC,
) -> np.ndarray:
    distance_cm = np.asarray(distance_cm, dtype=np.float32)
    frame_time_sec = np.asarray(frame_time_sec, dtype=np.float32)
    out = np.full(distance_cm.shape, np.nan, dtype=np.float32)
    if int(distance_cm.shape[0]) < 2 or int(frame_time_sec.shape[0]) != int(distance_cm.shape[0]):
        return out
    time_diff = np.diff(frame_time_sec)
    finite_dt = time_diff[np.isfinite(time_diff) & (time_diff > 0.0)]
    if finite_dt.size == 0:
        return out
    hop_sec = float(np.median(finite_dt))
    lag = max(1, int(round(float(delta_t_sec) / max(hop_sec, 1e-6))))
    num_frames = int(distance_cm.shape[0])
    for idx in range(num_frames):
        left = max(0, idx - lag)
        right = min(num_frames - 1, idx + lag)
        if right == left:
            continue
        d_left = float(distance_cm[left])
        d_right = float(distance_cm[right])
        t_left = float(frame_time_sec[left])
        t_right = float(frame_time_sec[right])
        if not (np.isfinite(d_left) and np.isfinite(d_right) and np.isfinite(t_left) and np.isfinite(t_right)):
            continue
        dt = t_right - t_left
        if dt <= 0.0:
            continue
        out[idx] = np.float32(((d_right - d_left) / 100.0) / dt)
    return out


def _ensure_pattern_targets(cache: Dict[str, Any]) -> None:
    if bool(cache.get("_pattern_targets_ready", False)):
        return
    pattern_target = np.asarray(cache["frame_pattern_target"], dtype=np.float32).copy()
    pattern_binary = np.asarray(cache.get("frame_pattern_binary_target", np.full_like(pattern_target, np.nan)), dtype=np.float32).copy()
    observability = np.asarray(cache.get("frame_observability_score_res", np.full_like(pattern_target, np.nan)), dtype=np.float32).copy()
    v_perp = np.asarray(cache.get("frame_v_perp_mps", np.full_like(pattern_target, np.nan)), dtype=np.float32).copy()
    distance_cm = np.asarray(cache.get("frame_distance_cm", np.full_like(pattern_target, np.nan)), dtype=np.float32)
    frame_time_sec = np.asarray(cache.get("frame_time_sec", np.full_like(pattern_target, np.nan)), dtype=np.float32)

    if np.any(~np.isfinite(v_perp)) and np.any(np.isfinite(distance_cm)):
        derived_v_perp = _derive_v_perp_from_distance(distance_cm, frame_time_sec)
        fill_mask = (~np.isfinite(v_perp)) & np.isfinite(derived_v_perp)
        v_perp = np.where(fill_mask, derived_v_perp, v_perp).astype(np.float32)

    if np.any(~np.isfinite(observability)) and np.any(np.isfinite(distance_cm)) and np.any(np.isfinite(v_perp)):
        derived_score = resolution_observability_score(distance_cm / 100.0, v_perp)
        fill_mask = (~np.isfinite(observability)) & np.isfinite(derived_score)
        observability = np.where(fill_mask, derived_score, observability).astype(np.float32)

    if np.any(~np.isfinite(pattern_target)) and np.any(np.isfinite(observability)):
        derived_pattern = resolution_pattern_soft_target(
            observability,
            lower=DEFAULT_PATTERN_SOFT_TARGET_LOWER,
            upper=DEFAULT_PATTERN_SOFT_TARGET_UPPER,
        )
        fill_mask = (~np.isfinite(pattern_target)) & np.isfinite(derived_pattern)
        pattern_target = np.where(fill_mask, derived_pattern, pattern_target).astype(np.float32)

    if np.any(~np.isfinite(pattern_binary)) and np.any(np.isfinite(observability)):
        derived_binary = resolution_pattern_binary_target(observability)
        fill_mask = (~np.isfinite(pattern_binary)) & np.isfinite(derived_binary)
        pattern_binary = np.where(fill_mask, derived_binary, pattern_binary).astype(np.float32)

    cache["frame_v_perp_mps"] = v_perp
    cache["frame_observability_score_res"] = observability
    cache["frame_pattern_target"] = pattern_target
    cache["frame_pattern_binary_target"] = pattern_binary
    cache["_pattern_targets_ready"] = True


class OmegaWindowDataset(Dataset):
    def __init__(self, index_path: str | Path, split: Optional[str] = None, max_cache_files: int = 2) -> None:
        if split is None:
            raise ValueError("OmegaWindowDataset requires an explicit split")
        self.index_path = Path(index_path)
        self.meta_path = metadata_path_for_index(self.index_path)
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if int(self.meta.get("schema_version", -1)) != 2:
            raise ValueError(f"expected omega dataset schema_version 2, got {self.meta.get('schema_version')}")

        self.split = str(split)
        self.index_manifest, self.split_info, arrays = open_omega_index_split(self.index_path, self.split, mmap_mode="r")
        self.recording_code = arrays["recording_code"]
        self.start_frame = arrays["start_frame"]
        self.sequence_index = arrays["sequence_index"]
        self.window_frames = int(self.index_manifest["window_frames"])
        self.center_offset = self.window_frames // 2

        self.max_cache_files = max(0, int(max_cache_files))
        self.cache_store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.norm_stats = load_omega_normalization_stats(self.meta["normalization_path"])

        # Build per-channel normalization: list of (key, mean[1,F], std[1,F])
        self.channel_norm: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for ch_key in DYNAMIC_CHANNEL_KEYS:
            mean_key = f"{ch_key}_mean"
            std_key = f"{ch_key}_std"
            if mean_key in self.norm_stats and std_key in self.norm_stats:
                m = np.asarray(self.norm_stats[mean_key], dtype=np.float32).reshape(1, -1)
                s = np.asarray(self.norm_stats[std_key], dtype=np.float32).reshape(1, -1)
                self.channel_norm.append((ch_key, m, s))
        if not self.channel_norm:
            raise RuntimeError("no normalized channels found in normalization stats")
        self.num_input_channels = len(self.channel_norm)

        self.recording_entries = {
            int(entry["recording_code"]): entry for entry in self.index_manifest.get("recordings", [])
        }
        self.recording_segments = self._build_recording_segments()

    def _build_recording_segments(self) -> List[Tuple[int, int, int]]:
        return _build_contiguous_recording_segments(self.recording_code)

    def __len__(self) -> int:
        return int(self.start_frame.shape[0])

    def _get_cache(self, cache_path: str) -> Dict[str, Any]:
        if cache_path in self.cache_store:
            self.cache_store.move_to_end(cache_path)
            return self.cache_store[cache_path]
        cache = dict(np.load(cache_path, allow_pickle=True))
        version = int(np.asarray(cache["schema_version"]).reshape(-1)[0])
        if version not in (6,):
            raise ValueError(f"expected omega cache schema_version 6, got {version} for {cache_path}")
        _ensure_pattern_targets(cache)
        if self.max_cache_files > 0:
            self.cache_store[cache_path] = cache
            self.cache_store.move_to_end(cache_path)
            while len(self.cache_store) > self.max_cache_files:
                self.cache_store.popitem(last=False)
        return cache

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start = int(self.start_frame[idx])
        recording_code = int(self.recording_code[idx])
        sequence_index = int(self.sequence_index[idx])
        recording_entry = self.recording_entries[recording_code]
        cache_path = str(recording_entry["cache_path"])
        cache = self._get_cache(cache_path)

        end = start + self.window_frames
        target_frame = end - 1

        # Build multi-channel input [C, T, F]
        channels = []
        for ch_key, ch_mean, ch_std in self.channel_norm:
            ch_data = np.asarray(cache[ch_key][start:end], dtype=np.float32)
            ch_data = ((ch_data - ch_mean) / ch_std).astype(np.float32, copy=False)
            channels.append(ch_data)
        x = np.stack(channels, axis=0)  # [C, T, F]

        frame_time_sec = np.asarray(cache["frame_time_sec"], dtype=np.float32)
        omega_target = np.asarray(cache["frame_omega_target"], dtype=np.float32)
        pattern_target = np.asarray(cache["frame_pattern_target"], dtype=np.float32)
        observability_arr = np.asarray(
            cache.get("frame_observability_score_res", np.full(omega_target.shape, np.nan, dtype=np.float32)),
            dtype=np.float32,
        )
        distance_cm_arr = np.asarray(
            cache.get("frame_distance_cm", np.full(omega_target.shape, np.nan, dtype=np.float32)),
            dtype=np.float32,
        )

        return {
            "x": torch.from_numpy(x).float(),
            "omega_target": torch.tensor(float(omega_target[target_frame]), dtype=torch.float32),
            "pattern_target": torch.tensor(float(pattern_target[target_frame]), dtype=torch.float32),
            "observability_score": torch.tensor(float(observability_arr[target_frame]), dtype=torch.float32),
            "distance_cm": torch.tensor(float(distance_cm_arr[target_frame]), dtype=torch.float32),
            "target_time_sec": torch.tensor(float(frame_time_sec[target_frame]), dtype=torch.float32),
            "sequence_index": torch.tensor(sequence_index, dtype=torch.long),
            "recording_id": str(recording_entry["recording_id"]),
        }

    def subset_by_recording_codes(
        self,
        recording_codes: np.ndarray | List[int],
        *,
        split: str,
    ) -> "OmegaWindowDatasetView":
        selected_codes = np.asarray(recording_codes, dtype=self.recording_code.dtype)
        if int(selected_codes.size) == 0:
            indices = np.empty((0,), dtype=np.int64)
        else:
            indices = np.flatnonzero(np.isin(self.recording_code, selected_codes)).astype(np.int64, copy=False)
        return OmegaWindowDatasetView(self, indices=indices, split=split)


class OmegaWindowDatasetView(Dataset):
    def __init__(self, base_dataset: OmegaWindowDataset, indices: np.ndarray, split: str) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.index_path = base_dataset.index_path
        self.meta_path = base_dataset.meta_path
        self.meta = base_dataset.meta
        self.index_manifest = base_dataset.index_manifest
        self.split_info = {"num_windows": int(self.indices.shape[0])}
        self.split = str(split)
        self.window_frames = int(base_dataset.window_frames)
        self.center_offset = int(base_dataset.center_offset)
        self.max_cache_files = int(base_dataset.max_cache_files)
        self.cache_store = base_dataset.cache_store
        self.norm_stats = base_dataset.norm_stats
        self.norm_mean = base_dataset.norm_mean
        self.norm_std = base_dataset.norm_std
        self.channel_norm = base_dataset.channel_norm
        self.num_input_channels = base_dataset.num_input_channels
        self.recording_entries = base_dataset.recording_entries
        self.recording_code = np.asarray(base_dataset.recording_code[self.indices])
        self.start_frame = np.asarray(base_dataset.start_frame[self.indices])
        self.sequence_index = np.asarray(base_dataset.sequence_index[self.indices])
        self.recording_segments = _build_contiguous_recording_segments(self.recording_code)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.base_dataset[int(self.indices[idx])]
