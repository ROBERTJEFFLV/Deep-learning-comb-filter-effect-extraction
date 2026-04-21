"""Dataset for K+1 cepstral-bin classification.

Reads existing omega caches (schema v7+) and computes cepstral patches on-the-fly
from the ``log_mag_preprocessed`` channel.  No cache rebuild needed.

Per-frame pipeline:
  log_mag_preprocessed[t]  →  mean-remove  →  FFT  →  |·|  →  slice [qmin:qmax]
  →  MAD normalization  →  cepstral_patch [Q]

Target: integer class in [0, K]:
  class 0 = no-pattern
  class k ∈ [1..K] = distance bin k (mapped from GT distance via quefrency geometry)
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ml_uav_comb.data_pipeline.omega_dataset_index import open_omega_index_split
from ml_uav_comb.data_pipeline.omega_dataset import (
    _build_contiguous_recording_segments,
    _ensure_pattern_targets,
)
from ml_uav_comb.models.comb_cepstral_net import compute_cepstral_geometry, C_SPEED


def _compute_cepstral_patches(
    log_mag_preprocessed: np.ndarray,
    cep_min_bin: int,
    cep_max_bin: int,
    use_normalization: bool = True,
    norm_eps: float = 1e-6,
) -> np.ndarray:
    """Compute normalized cepstral search-band patches from preprocessed log magnitude.

    Args:
        log_mag_preprocessed: [T, F] preprocessed log magnitude spectrum
        cep_min_bin, cep_max_bin: quefrency search range (absolute bins)
        use_normalization: if True, apply per-frame MAD normalization
        norm_eps: epsilon for MAD normalization

    Returns:
        patches: [T, Q] where Q = cep_max_bin - cep_min_bin
    """
    T, F = log_mag_preprocessed.shape

    # Vectorized cepstrum: center each frame, FFT, take magnitude
    centered = log_mag_preprocessed - log_mag_preprocessed.mean(axis=1, keepdims=True)
    cepstrum = np.abs(np.fft.fft(centered, axis=1))  # [T, F]

    # Slice to comb search band
    patches = cepstrum[:, cep_min_bin:cep_max_bin].astype(np.float32)  # [T, Q]

    if use_normalization:
        # Per-frame robust normalization: (x - median) / (MAD + eps)
        med = np.median(patches, axis=1, keepdims=True)
        mad = np.median(np.abs(patches - med), axis=1, keepdims=True)
        patches = (patches - med) / (mad + norm_eps)

    return patches


def distance_cm_to_bin(
    distance_cm: float,
    quef_tau_factor: float,
    cep_min_bin: int,
    Q: int,
) -> int:
    """Map ground-truth distance (cm) to quefrency bin index (0-based in search range).

    Returns bin in [0, Q-1].  Caller adds +1 to get class index (0=no-pattern).
    """
    tau = 2.0 * (distance_cm / 100.0) / C_SPEED  # delay in seconds
    cep_bin_float = tau / quef_tau_factor           # absolute quefrency bin
    local_bin = round(cep_bin_float) - cep_min_bin  # 0-indexed in search range
    return int(np.clip(local_bin, 0, Q - 1))


def distance_cm_to_soft_target(
    distance_cm: float,
    quef_tau_factor: float,
    cep_min_bin: int,
    Q: int,
    soft_bin_sigma: float = 1.0,
    no_pattern_mass: float = 0.0,
) -> np.ndarray:
    """Generate soft K+1 target distribution centred at the GT distance bin.

    Probability mass is spread with a Gaussian of width *soft_bin_sigma* bins
    over the K distance bins (indices 1..K in the output array).  Class 0
    (no-pattern) receives *no_pattern_mass* and the remaining mass is
    distributed across the Gaussian.

    Args:
        distance_cm: ground-truth distance
        quef_tau_factor, cep_min_bin, Q: cepstral geometry
        soft_bin_sigma: Gaussian std in bins (0 → hard one-hot)
        no_pattern_mass: probability mass reserved for the no-pattern class

    Returns:
        soft: [K+1] float32 probability vector (sums to 1)
    """
    K = Q
    local_bin = distance_cm_to_bin(distance_cm, quef_tau_factor, cep_min_bin, Q)
    ks = np.arange(K, dtype=np.float32)
    if soft_bin_sigma > 0:
        gauss = np.exp(-0.5 * ((ks - local_bin) / soft_bin_sigma) ** 2)
    else:
        gauss = (ks == local_bin).astype(np.float32)
    gauss_sum = gauss.sum()
    if gauss_sum > 0:
        gauss /= gauss_sum
    pattern_mass = 1.0 - no_pattern_mass
    soft = np.zeros(K + 1, dtype=np.float32)
    soft[0] = no_pattern_mass
    soft[1:] = gauss * pattern_mass
    return soft


class CepstralBinDataset(Dataset):
    """K+1 classification dataset from omega caches.

    Each item returns:
        x: [T, Q] normalized cepstral patch
        target: int in [0, K]  (0=no-pattern, 1..K=distance bin)
        distance_cm: float (GT distance, may be nan)
        target_time_sec: float
        sequence_index: int
        recording_id: str
    """

    def __init__(
        self,
        index_path: str | Path,
        split: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        max_cache_files: int = 2,
    ) -> None:
        if split is None:
            raise ValueError("CepstralBinDataset requires an explicit split")
        if cfg is None:
            raise ValueError("CepstralBinDataset requires a config dict")

        self.index_path = Path(index_path)
        from ml_uav_comb.features.feature_utils import metadata_path_for_index
        self.meta_path = metadata_path_for_index(self.index_path)
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if int(self.meta.get("schema_version", -1)) != 2:
            raise ValueError(f"Expected schema_version 2, got {self.meta.get('schema_version')}")

        self.split = str(split)
        self.cfg = cfg

        # Cepstral geometry
        geom = compute_cepstral_geometry(cfg)
        self.cep_min_bin = geom["cep_min_bin"]
        self.cep_max_bin = geom["cep_max_bin"]
        self.Q = geom["Q"]
        self.K = geom["K"]
        self.quef_tau_factor = geom["quef_tau_factor"]
        self.bin_centers_cm = geom["bin_centers_cm"]

        model_cfg = cfg.get("model", {})
        self.use_normalization = bool(model_cfg.get("use_normalization", True))
        self.norm_eps = float(model_cfg.get("norm_eps", 1e-6))
        self.soft_bin_sigma = float(model_cfg.get("soft_bin_sigma", 0.0))
        self.soft_no_pattern_mass = float(model_cfg.get("soft_no_pattern_mass", 0.0))

        # Load index
        self.index_manifest, self.split_info, arrays = open_omega_index_split(
            self.index_path, self.split, mmap_mode="r"
        )
        self.recording_code = arrays["recording_code"]
        self.start_frame = arrays["start_frame"]
        self.sequence_index = arrays["sequence_index"]
        self.window_frames = int(self.index_manifest["window_frames"])

        # Cache management
        self.max_cache_files = max(0, int(max_cache_files))
        self.cache_store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        self.recording_entries = {
            int(entry["recording_code"]): entry
            for entry in self.index_manifest.get("recordings", [])
        }
        self.recording_segments = _build_contiguous_recording_segments(self.recording_code)

    def __len__(self) -> int:
        return int(self.start_frame.shape[0])

    def _get_cache(self, cache_path: str) -> Dict[str, Any]:
        if cache_path in self.cache_store:
            self.cache_store.move_to_end(cache_path)
            return self.cache_store[cache_path]
        cache = dict(np.load(cache_path, allow_pickle=True))
        version = int(np.asarray(cache["schema_version"]).reshape(-1)[0])
        if version not in (7, 8):
            raise ValueError(f"Expected cache schema 7 or 8, got {version} for {cache_path}")
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

        # Compute cepstral patches on-the-fly
        log_mag_preprocessed = np.asarray(
            cache["log_mag_preprocessed"][start:end], dtype=np.float32
        )
        patches = _compute_cepstral_patches(
            log_mag_preprocessed,
            self.cep_min_bin,
            self.cep_max_bin,
            use_normalization=self.use_normalization,
            norm_eps=self.norm_eps,
        )  # [T, Q]

        # Target: bin classification
        pattern_binary = float(
            np.asarray(cache["frame_pattern_binary_target"], dtype=np.float32)[target_frame]
        )
        distance_cm_val = float(
            np.asarray(cache["frame_distance_cm"], dtype=np.float32)[target_frame]
        )
        frame_time = float(
            np.asarray(cache["frame_time_sec"], dtype=np.float32)[target_frame]
        )

        is_pattern = (
            np.isfinite(pattern_binary)
            and pattern_binary >= 0.5
            and np.isfinite(distance_cm_val)
        )

        if not is_pattern:
            target = 0  # no-pattern hard label
            if self.soft_bin_sigma > 0:
                # No-pattern: hard one-hot on class 0
                soft_target = np.zeros(self.K + 1, dtype=np.float32)
                soft_target[0] = 1.0
            else:
                soft_target = None
        else:
            bin_idx = distance_cm_to_bin(
                distance_cm_val, self.quef_tau_factor, self.cep_min_bin, self.Q
            )
            target = bin_idx + 1  # class 1..K
            if self.soft_bin_sigma > 0:
                soft_target = distance_cm_to_soft_target(
                    distance_cm_val,
                    self.quef_tau_factor,
                    self.cep_min_bin,
                    self.Q,
                    soft_bin_sigma=self.soft_bin_sigma,
                    no_pattern_mass=self.soft_no_pattern_mass,
                )
            else:
                soft_target = None

        out = {
            "x": torch.from_numpy(patches).float(),               # [T, Q]
            "target": torch.tensor(target, dtype=torch.long),      # scalar (hard, for metrics)
            "distance_cm": torch.tensor(distance_cm_val, dtype=torch.float32),
            "target_time_sec": torch.tensor(frame_time, dtype=torch.float32),
            "sequence_index": torch.tensor(sequence_index, dtype=torch.long),
            "recording_id": str(recording_entry["recording_id"]),
        }
        if soft_target is not None:
            out["soft_target"] = torch.from_numpy(soft_target).float()  # [K+1]
        return out

    def subset_by_recording_codes(
        self, recording_codes: np.ndarray | List[int], *, split: str
    ) -> "CepstralBinDatasetView":
        selected = np.asarray(recording_codes, dtype=self.recording_code.dtype)
        if int(selected.size) == 0:
            indices = np.empty((0,), dtype=np.int64)
        else:
            indices = np.flatnonzero(np.isin(self.recording_code, selected)).astype(np.int64)
        return CepstralBinDatasetView(self, indices=indices, split=split)


class CepstralBinDatasetView(Dataset):
    """View into a CepstralBinDataset for a subset of recording codes."""

    def __init__(self, base: CepstralBinDataset, indices: np.ndarray, split: str) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        # Expose attributes needed by ContiguousSequenceBatchSampler
        self.index_path = base.index_path
        self.meta_path = base.meta_path
        self.meta = base.meta
        self.index_manifest = base.index_manifest
        self.split_info = {"num_windows": int(self.indices.shape[0])}
        self.split = str(split)
        self.window_frames = base.window_frames
        self.recording_code = np.asarray(base.recording_code[self.indices])
        self.start_frame = np.asarray(base.start_frame[self.indices])
        self.sequence_index = np.asarray(base.sequence_index[self.indices])
        self.recording_entries = base.recording_entries
        self.recording_segments = _build_contiguous_recording_segments(self.recording_code)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.base[int(self.indices[idx])]
