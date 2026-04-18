"""Normalization stats for omega datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ml_uav_comb.data_pipeline.omega_dataset_index import open_omega_index_split
from ml_uav_comb.features.feature_utils import TerminalProgressBar


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
    cache_store: Dict[str, Dict[str, Any]] = {}
    smooth_sum = None
    smooth_sq_sum = None
    count = 0.0
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
        cache = cache_store[cache_path]
        start_frames = start_frame_all[code_mask]
        target_frames = start_frames + (window_frames - 1)

        smooth = np.asarray(cache["smooth_d1"], dtype=np.float32)
        smooth_prefix = np.concatenate(
            [
                np.zeros((1, smooth.shape[-1]), dtype=np.float64),
                np.cumsum(smooth, axis=0, dtype=np.float64),
            ],
            axis=0,
        )
        smooth_sq_prefix = np.concatenate(
            [
                np.zeros((1, smooth.shape[-1]), dtype=np.float64),
                np.cumsum(np.square(smooth, dtype=np.float64), axis=0, dtype=np.float64),
            ],
            axis=0,
        )
        window_sum = smooth_prefix[start_frames + window_frames] - smooth_prefix[start_frames]
        window_sq_sum = smooth_sq_prefix[start_frames + window_frames] - smooth_sq_prefix[start_frames]
        if smooth_sum is None:
            smooth_sum = np.zeros((window_sum.shape[-1],), dtype=np.float64)
            smooth_sq_sum = np.zeros((window_sq_sum.shape[-1],), dtype=np.float64)
        smooth_sum += window_sum.sum(axis=0)
        smooth_sq_sum += window_sq_sum.sum(axis=0)
        count += float(split_count * window_frames)
        processed_windows += split_count
        progress.update(
            processed_recordings,
            extra=f"windows={processed_windows}/{num_train_windows}",
        )

    if int(recording_code_all.shape[0]) != num_train_windows:
        raise RuntimeError(
            f"{split} split window count mismatch: expected {num_train_windows}, aggregated {recording_code_all.shape[0]}"
        )
    mean = smooth_sum / float(count)
    var = np.maximum((smooth_sq_sum / float(count)) - mean**2, 1e-6)
    stats = {
        "schema_version": np.asarray([2], dtype=np.int64),
        "smooth_d1_mean": mean.astype(np.float32),
        "smooth_d1_std": np.sqrt(var).astype(np.float32),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **stats)
    progress.finish(extra=f"windows={processed_windows}/{num_train_windows}")
    return {
        "normalization_path": str(output_path),
        "num_normalization_windows": int(num_train_windows),
        "num_train_windows": int(num_train_windows),
    }


def load_omega_normalization_stats(path: str | Path) -> Dict[str, np.ndarray]:
    stats = dict(np.load(path, allow_pickle=True))
    version = int(np.asarray(stats["schema_version"]).reshape(-1)[0])
    if version != 2:
        raise ValueError(f"expected omega normalization schema_version 2, got {version}")
    return stats
