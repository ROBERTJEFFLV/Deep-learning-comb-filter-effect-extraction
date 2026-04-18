"""Train-only normalization stats for v2 cached datasets."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _mean_std_from_sums(sum_arr: np.ndarray, sq_sum_arr: np.ndarray, count: float) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        mean = np.zeros_like(sum_arr, dtype=np.float32)
        std = np.ones_like(sum_arr, dtype=np.float32)
        return mean, std
    mean = sum_arr / float(count)
    var = np.maximum((sq_sum_arr / float(count)) - mean**2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def compute_normalization_stats(index_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    index_path = Path(index_path)
    output_path = Path(output_path)
    rows: List[Dict[str, Any]] = []
    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                rows.append(row)

    if not rows:
        raise ValueError("cannot compute normalization stats without train rows")

    cache_store: Dict[str, Dict[str, Any]] = {}
    phase_sum = np.zeros(3, dtype=np.float64)
    phase_sq_sum = np.zeros(3, dtype=np.float64)
    comb_sum = np.zeros(3, dtype=np.float64)
    comb_sq_sum = np.zeros(3, dtype=np.float64)
    stp_sum = None
    stp_sq_sum = None
    phase_count = 0.0
    comb_count = 0.0
    stp_count = 0.0
    scalar_values = defaultdict(list)
    acoustic_field_names = None

    for row in rows:
        cache_path = row["cache_path"]
        if cache_path not in cache_store:
            cache_store[cache_path] = dict(np.load(cache_path, allow_pickle=True))
        cache = cache_store[cache_path]
        start = int(row["start_frame"])
        end = int(row["end_frame"])

        phase = cache["phase_stft"][start:end]
        comb = cache["diff_comb"][start:end]
        scalar = cache["scalar_seq"][start:end]
        scalar_observed = cache["scalar_observed_mask"][start:end]
        phase_sum += phase.sum(axis=(0, 1))
        phase_sq_sum += np.square(phase).sum(axis=(0, 1))
        comb_sum += comb.sum(axis=(0, 1))
        comb_sq_sum += np.square(comb).sum(axis=(0, 1))
        phase_count += float(phase.shape[0] * phase.shape[1])
        comb_count += float(comb.shape[0] * comb.shape[1])

        acoustic_field_names = cache["scalar_field_names"]
        for dim in range(scalar.shape[-1]):
            valid = scalar_observed[:, dim] > 0.5
            if np.any(valid):
                scalar_values[dim].append(scalar[valid, dim])

        if "stpacc" in cache:
            stpacc = cache["stpacc"][start:end]
            if stp_sum is None:
                stp_sum = np.zeros(stpacc.shape[-1], dtype=np.float64)
                stp_sq_sum = np.zeros(stpacc.shape[-1], dtype=np.float64)
            stp_sum += stpacc.sum(axis=0)
            stp_sq_sum += np.square(stpacc).sum(axis=0)
            stp_count += float(stpacc.shape[0])

    phase_mean, phase_std = _mean_std_from_sums(phase_sum, phase_sq_sum, phase_count)
    comb_mean, comb_std = _mean_std_from_sums(comb_sum, comb_sq_sum, comb_count)
    if stp_sum is None or stp_sq_sum is None:
        stp_mean = np.zeros(64, dtype=np.float32)
        stp_std = np.ones(64, dtype=np.float32)
    else:
        stp_mean, stp_std = _mean_std_from_sums(stp_sum, stp_sq_sum, stp_count)

    scalar_median = []
    scalar_iqr = []
    scalar_dim = int(len(acoustic_field_names))
    for dim in range(scalar_dim):
        if dim not in scalar_values:
            scalar_median.append(0.0)
            scalar_iqr.append(1.0)
            continue
        stacked = np.concatenate(scalar_values[dim], axis=0)
        q25 = float(np.percentile(stacked, 25))
        q50 = float(np.percentile(stacked, 50))
        q75 = float(np.percentile(stacked, 75))
        scalar_median.append(q50)
        scalar_iqr.append(max(q75 - q25, 1e-3))

    stats = {
        "schema_version": np.asarray([2], dtype=np.int64),
        "phase_mean": phase_mean,
        "phase_std": phase_std,
        "comb_mean": comb_mean,
        "comb_std": comb_std,
        "stpacc_mean": stp_mean.astype(np.float32),
        "stpacc_std": stp_std.astype(np.float32),
        "scalar_median": np.asarray(scalar_median, dtype=np.float32),
        "scalar_iqr": np.asarray(scalar_iqr, dtype=np.float32),
        "scalar_field_names": np.asarray(acoustic_field_names, dtype=object),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **stats)
    return {
        "normalization_path": str(output_path),
        "num_train_windows": len(rows),
    }


def load_normalization_stats(path: str | Path) -> Dict[str, np.ndarray]:
    stats = dict(np.load(path, allow_pickle=True))
    version = int(np.asarray(stats["schema_version"]).reshape(-1)[0])
    if version != 2:
        raise ValueError(f"expected normalization schema_version 2, got {version}")
    return stats

