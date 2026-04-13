"""PyTorch dataset for cached UAV comb-motion features."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ml_uav_comb.data_pipeline.normalization import load_normalization_stats
from ml_uav_comb.features.feature_utils import metadata_path_for_index


class CachedWindowDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        split: Optional[str] = None,
        max_cache_files: int = 2,
    ) -> None:
        self.index_path = Path(index_path)
        self.meta_path = metadata_path_for_index(self.index_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(f"dataset metadata not found: {self.meta_path}")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if int(self.meta.get("schema_version", -1)) != 2:
            raise ValueError(f"expected dataset schema_version 2, got {self.meta.get('schema_version')}")

        self.rows: List[Dict[str, Any]] = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is None or row["split"] == split:
                    self.rows.append(row)
        self.max_cache_files = max(0, int(max_cache_files))
        self.cache_store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.norm_stats = load_normalization_stats(self.meta["normalization_path"])
        self.default_num_candidates = 64
        self.indices_by_cache_path: Dict[str, List[int]] = {}
        self.indices_by_recording_id: Dict[str, List[int]] = {}
        for idx, row in enumerate(self.rows):
            self.indices_by_cache_path.setdefault(row["cache_path"], []).append(idx)
            self.indices_by_recording_id.setdefault(str(row["recording_id"]), []).append(idx)
        if self.rows:
            maybe_grid = self._row_vector(self.rows[0], "distance_target_grid", default_dim=0)
            if maybe_grid.size > 0:
                self.default_num_candidates = int(maybe_grid.size)

    def __len__(self) -> int:
        return len(self.rows)

    def _get_cache(self, cache_path: str) -> Dict[str, Any]:
        if cache_path in self.cache_store:
            self.cache_store.move_to_end(cache_path)
            return self.cache_store[cache_path]

        if self.max_cache_files == 0:
            cache = dict(np.load(cache_path, allow_pickle=True))
            version = int(np.asarray(cache["schema_version"]).reshape(-1)[0])
            if version != 2:
                raise ValueError(f"expected cache schema_version 2, got {version} for {cache_path}")
            return cache

        cache = dict(np.load(cache_path, allow_pickle=True))
        version = int(np.asarray(cache["schema_version"]).reshape(-1)[0])
        if version != 2:
            raise ValueError(f"expected cache schema_version 2, got {version} for {cache_path}")
        self.cache_store[cache_path] = cache
        self.cache_store.move_to_end(cache_path)
        while len(self.cache_store) > self.max_cache_files:
            self.cache_store.popitem(last=False)
        return self.cache_store[cache_path]

    def _normalize_phase(self, phase: np.ndarray) -> np.ndarray:
        mean = self.norm_stats["phase_mean"].reshape(1, 1, -1)
        std = self.norm_stats["phase_std"].reshape(1, 1, -1)
        return ((phase - mean) / std).astype(np.float32)

    def _normalize_comb(self, comb: np.ndarray) -> np.ndarray:
        mean = self.norm_stats["comb_mean"].reshape(1, 1, -1)
        std = self.norm_stats["comb_std"].reshape(1, 1, -1)
        return ((comb - mean) / std).astype(np.float32)

    def _normalize_scalar(self, scalar: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        median = self.norm_stats["scalar_median"].reshape(1, -1)
        iqr = self.norm_stats["scalar_iqr"].reshape(1, -1)
        normalized = (scalar - median) / iqr
        normalized = np.where(observed_mask > 0.5, normalized, 0.0)
        return normalized.astype(np.float32)

    def _normalize_stpacc(self, stpacc: np.ndarray) -> np.ndarray:
        mean = self.norm_stats["stpacc_mean"].reshape(1, -1)
        std = self.norm_stats["stpacc_std"].reshape(1, -1)
        return ((stpacc - mean) / std).astype(np.float32)

    @staticmethod
    def _row_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
        value = row.get(key, "")
        if value in ("", None):
            return float(default)
        try:
            parsed = float(value)
        except Exception:
            return float(default)
        if np.isnan(parsed) or np.isinf(parsed):
            return float(default)
        return float(parsed)

    @staticmethod
    def _row_str(row: Dict[str, Any], key: str, default: str = "") -> str:
        value = row.get(key, default)
        return default if value is None else str(value)

    @staticmethod
    def _row_vector(row: Dict[str, Any], key: str, default_dim: int) -> np.ndarray:
        value = row.get(key, "")
        if value in ("", None):
            return np.zeros((default_dim,), dtype=np.float32)
        try:
            text = str(value).strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            if ";" in text:
                parts = [p for p in text.split(";") if p.strip() != ""]
            else:
                parts = [p for p in text.split() if p.strip() != ""]
            arr = np.asarray([float(p) for p in parts], dtype=np.float32).reshape(-1)
        except Exception:
            return np.zeros((default_dim,), dtype=np.float32)
        if arr.size == 0:
            return np.zeros((default_dim,), dtype=np.float32)
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        cache = self._get_cache(row["cache_path"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        center = int(self._row_float(row, "center_frame", default=float(start + (end - start) // 2)))
        target = int(self._row_float(row, "target_frame", default=float(end - 1)))

        phase_np = self._normalize_phase(cache["phase_stft"][start:end])
        comb_np = self._normalize_comb(cache["diff_comb"][start:end])
        scalar_np = cache["scalar_seq"][start:end].astype(np.float32)
        scalar_observed_mask_np = cache["scalar_observed_mask"][start:end].astype(np.float32)
        scalar_reliable_mask_np = cache["scalar_reliable_mask"][start:end].astype(np.float32)
        scalar_norm_np = self._normalize_scalar(scalar_np, scalar_observed_mask_np)
        teacher_np = cache["teacher_seq"][start:end].astype(np.float32)
        frequencies_hz_np = cache["frequencies_hz"].astype(np.float32)
        if "stpacc" in cache:
            stpacc_np = self._normalize_stpacc(cache["stpacc"][start:end].astype(np.float32))
            stpacc = torch.from_numpy(stpacc_np[None, :, :]).float()
        else:
            stpacc = torch.zeros((1, end - start, 64), dtype=torch.float32)

        target_grid_np = self._row_vector(row, "distance_target_grid", default_dim=self.default_num_candidates)
        self.default_num_candidates = int(max(self.default_num_candidates, int(target_grid_np.size)))

        phase = torch.from_numpy(phase_np.transpose(2, 0, 1)).float()
        comb = torch.from_numpy(comb_np.transpose(2, 0, 1)).float()
        scalar = torch.from_numpy(scalar_norm_np).float()
        scalar_observed_mask = torch.from_numpy(scalar_observed_mask_np).float()
        scalar_reliable_mask = torch.from_numpy(scalar_reliable_mask_np).float()
        teacher = torch.from_numpy(teacher_np).float()
        frequencies_hz = torch.from_numpy(frequencies_hz_np).float()
        distance_target_grid = torch.from_numpy(target_grid_np.astype(np.float32)).float()

        measurement_distance_target_cm = self._row_float(
            row,
            "measurement_distance_target_cm",
            default=self._row_float(row, "distance_cm"),
        )
        measurement_distance_train_mask = self._row_float(
            row,
            "measurement_distance_train_mask",
            default=self._row_float(row, "dist_train_mask"),
        )
        measurement_validity_target = self._row_float(
            row,
            "measurement_validity_target",
            default=self._row_float(row, "confidence_target"),
        )
        measurement_validity_train_mask = self._row_float(
            row,
            "measurement_validity_train_mask",
            default=self._row_float(row, "conf_train_mask"),
        )

        sample = {
            "phase": phase,
            "comb": comb,
            "scalar": scalar,
            "scalar_observed_mask": scalar_observed_mask,
            "scalar_reliable_mask": scalar_reliable_mask,
            "teacher": teacher,
            "stpacc": stpacc,
            "frequencies_hz": frequencies_hz,
            "distance_target_grid": distance_target_grid,
            "measurement_distance_target_cm": torch.tensor(measurement_distance_target_cm, dtype=torch.float32),
            "measurement_distance_train_mask": torch.tensor(measurement_distance_train_mask, dtype=torch.float32),
            "measurement_validity_target": torch.tensor(measurement_validity_target, dtype=torch.float32),
            "measurement_validity_train_mask": torch.tensor(measurement_validity_train_mask, dtype=torch.float32),
            "valid_dist_gt_mask": torch.tensor(self._row_float(row, "valid_dist_gt_mask"), dtype=torch.float32),
            "external_distance_valid_mask": torch.tensor(
                self._row_float(row, "external_distance_valid_mask"),
                dtype=torch.float32,
            ),
            "dist_reliable_mask": torch.tensor(self._row_float(row, "dist_reliable_mask"), dtype=torch.float32),
            "dist_loss_weight": torch.tensor(self._row_float(row, "dist_loss_weight"), dtype=torch.float32),
            "heuristic_distance_cm": torch.tensor(self._row_float(row, "heuristic_distance_cm"), dtype=torch.float32),
            "heuristic_distance_available": torch.tensor(
                self._row_float(row, "heuristic_distance_available"),
                dtype=torch.float32,
            ),
            "target_time_sec": torch.tensor(
                self._row_float(row, "target_time_sec", default=self._row_float(row, "center_time_sec")),
                dtype=torch.float32,
            ),
            "center_time_sec": torch.tensor(self._row_float(row, "center_time_sec"), dtype=torch.float32),
            "target_frame": torch.tensor(target, dtype=torch.long),
            "center_frame": torch.tensor(center, dtype=torch.long),
            "recording_id": row["recording_id"],
            "split": row["split"],
            "label_source": self._row_str(row, "label_source"),
            "supervision_type": self._row_str(row, "supervision_type"),
            "measurement_target_source": self._row_str(
                row,
                "measurement_target_source",
                default=self._row_str(row, "distance_target_source"),
            ),
            "validity_target_source": self._row_str(
                row,
                "validity_target_source",
                default=self._row_str(row, "confidence_target_source"),
            ),
            "cache_path": row["cache_path"],
            # Legacy aliases
            "distance_cm": torch.tensor(self._row_float(row, "distance_cm"), dtype=torch.float32),
            "distance_target": torch.tensor(self._row_float(row, "distance_target"), dtype=torch.float32),
            "dist_train_mask": torch.tensor(measurement_distance_train_mask, dtype=torch.float32),
            "confidence_target": torch.tensor(measurement_validity_target, dtype=torch.float32),
            "conf_train_mask": torch.tensor(measurement_validity_train_mask, dtype=torch.float32),
            "distance_target_source": self._row_str(row, "distance_target_source"),
            "confidence_target_source": self._row_str(row, "confidence_target_source"),
            "sign_label": torch.tensor(int(self._row_float(row, "sign_label", default=float(2))), dtype=torch.long),
            "valid_sign_gt_mask": torch.tensor(self._row_float(row, "valid_sign_gt_mask"), dtype=torch.float32),
            "sign_train_mask": torch.tensor(self._row_float(row, "sign_train_mask"), dtype=torch.float32),
            "valid_conf_gt_mask": torch.tensor(self._row_float(row, "valid_conf_gt_mask"), dtype=torch.float32),
            "sign_target_source": self._row_str(row, "sign_target_source"),
        }
        return sample
