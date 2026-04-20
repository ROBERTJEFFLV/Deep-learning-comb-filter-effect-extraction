"""Normalization stats for omega datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ml_uav_comb.data_pipeline.omega_dataset_index import open_omega_index_split
from ml_uav_comb.features.feature_utils import TerminalProgressBar

NORM_SCHEMA_VERSION = 5
DYNAMIC_CHANNEL_KEYS = ("log_mag_band", "log_mag_preprocessed", "log_mag_preprocessed_dt1", "log_mag_preprocessed_abs")


def compute_omega_normalization_stats(
    index_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "train",
) -> Dict[str, Any]:
    manifest, split_info, arrays = open_omega_index_split(index_path, split, mmap_mode="r")
    num_train_windows = int(split_info.get("num_windows", 0))
    if num_train_windows <= 0:
        raise ValueError(f"cannot compute omega normalization without {split} rows")

    window_frames = int(manifest["window_frames"])
    # LRU-style cache with bounded size to prevent OOM when deriving dynamic channels
    MAX_NORM_CACHE_FILES = 32
    cache_store: Dict[str, Dict[str, Any]] = {}

    accumulators: Dict[str, Dict[str, np.ndarray | float]] = {}
    recording_code_all = np.asarray(arrays["recording_code"], dtype=np.int64)
    start_frame_all = np.asarray(arrays["start_frame"], dtype=np.int64)
    progress = TerminalProgressBar(f"Normalize {split}", len(manifest["recordings"]))
    processed_recordings = 0
    processed_windows = 0
    for recording_entry in manifest["recordings"]:
        processed_recordings += 1
        recording_code = int(recording_entry["recording_code"])
        code_mask = recording_code_all == recording_code
        split_count = int(np.sum(code_mask))
        if split_count <= 0:
            progress.update(
                processed_recordings,
                extra=f"windows={processed_windows}/{num_train_windows}",
            )
            continue
        cache_path = str(recording_entry["cache_path"])
        if cache_path not in cache_store:
            cache_store[cache_path] = dict(np.load(cache_path, allow_pickle=True))
            c = cache_store[cache_path]
            version = int(np.asarray(c.get("schema_version", [7])).reshape(-1)[0])
            if version not in (7,):
                raise ValueError(f"expected omega cache schema_version 7, got {version} for {cache_path}")
            # Evict oldest entries to prevent OOM
            while len(cache_store) > MAX_NORM_CACHE_FILES:
                oldest_key = next(iter(cache_store))
                del cache_store[oldest_key]
        cache = cache_store[cache_path]
        start_frames = start_frame_all[code_mask]

        for ch_key in DYNAMIC_CHANNEL_KEYS:
            if ch_key not in cache:
                continue
            ch_data = np.asarray(cache[ch_key], dtype=np.float32)
            prefix_sum = np.concatenate(
                [np.zeros((1, ch_data.shape[-1]), dtype=np.float64),
                 np.cumsum(ch_data, axis=0, dtype=np.float64)], axis=0,
            )
            prefix_sq = np.concatenate(
                [np.zeros((1, ch_data.shape[-1]), dtype=np.float64),
                 np.cumsum(np.square(ch_data, dtype=np.float64), axis=0, dtype=np.float64)], axis=0,
            )
            window_sum = prefix_sum[start_frames + window_frames] - prefix_sum[start_frames]
            window_sq = prefix_sq[start_frames + window_frames] - prefix_sq[start_frames]
            if ch_key not in accumulators:
                accumulators[ch_key] = {
                    "sum": np.zeros((ch_data.shape[-1],), dtype=np.float64),
                    "sq_sum": np.zeros((ch_data.shape[-1],), dtype=np.float64),
                    "count": 0.0,
                }
            accumulators[ch_key]["sum"] += window_sum.sum(axis=0)
            accumulators[ch_key]["sq_sum"] += window_sq.sum(axis=0)
            accumulators[ch_key]["count"] += float(split_count * window_frames)

        processed_windows += split_count
        progress.update(
            processed_recordings,
            extra=f"windows={processed_windows}/{num_train_windows}",
        )

    if int(recording_code_all.shape[0]) != num_train_windows:
        raise RuntimeError(
            f"{split} split window count mismatch: expected {num_train_windows}, aggregated {recording_code_all.shape[0]}"
        )

    stats: Dict[str, np.ndarray] = {
        "schema_version": np.asarray([NORM_SCHEMA_VERSION], dtype=np.int64),
    }
    for ch_key, acc in accumulators.items():
        count = acc["count"]
        if count <= 0:
            continue
        mean = acc["sum"] / count
        var = np.maximum((acc["sq_sum"] / count) - mean**2, 1e-6)
        stats[f"{ch_key}_mean"] = mean.astype(np.float32)
        stats[f"{ch_key}_std"] = np.sqrt(var).astype(np.float32)

    # Ensure at least log_mag_band stats exist (required)
    if "log_mag_band_mean" not in stats:
        raise RuntimeError("log_mag_band channel not found in any cache file")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **stats)
    progress.finish(extra=f"windows={processed_windows}/{num_train_windows}")
    return {
        "normalization_path": str(output_path),
        "num_normalization_windows": int(num_train_windows),
        "num_train_windows": int(num_train_windows),
        "channels_normalized": [k for k in DYNAMIC_CHANNEL_KEYS if f"{k}_mean" in stats],
    }


def load_omega_normalization_stats(path: str | Path) -> Dict[str, np.ndarray]:
    stats = dict(np.load(path, allow_pickle=True))
    version = int(np.asarray(stats["schema_version"]).reshape(-1)[0])
    if version not in (NORM_SCHEMA_VERSION,):
        raise ValueError(f"expected omega normalization schema_version {NORM_SCHEMA_VERSION}, got {version}")
    return stats
