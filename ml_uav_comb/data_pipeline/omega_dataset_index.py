"""Compact window index for omega regression datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.lib.format import open_memmap

from ml_uav_comb.features.feature_utils import (
    choose_recording_level_split,
    compute_single_recording_split_intervals,
)


OMEGA_INDEX_SCHEMA_VERSION = 2
OMEGA_INDEX_FORMAT = "omega_compact_v1"


def omega_index_data_dir(index_path: str | Path) -> Path:
    index_path = Path(index_path)
    return index_path.with_name(index_path.stem + "_data")


def load_omega_index_manifest(index_path: str | Path) -> Dict[str, Any]:
    index_path = Path(index_path)
    with open(index_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if int(manifest.get("schema_version", -1)) != OMEGA_INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"expected omega index schema_version {OMEGA_INDEX_SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')}"
        )
    if str(manifest.get("index_format", "")) != OMEGA_INDEX_FORMAT:
        raise ValueError(
            f"expected omega index format {OMEGA_INDEX_FORMAT}, got {manifest.get('index_format')}"
        )
    return manifest


def resolve_omega_index_array_path(index_path: str | Path, relative_path: str) -> Path:
    index_path = Path(index_path)
    return (index_path.parent / relative_path).resolve()


def open_omega_index_split(
    index_path: str | Path,
    split: str,
    *,
    mmap_mode: str | None = "r",
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    manifest = load_omega_index_manifest(index_path)
    if split == "all":
        arrays_by_split = []
        for split_name in manifest["split_names"]:
            split_info = manifest["splits"][split_name]
            arrays_by_split.append(
                {
                    field: np.load(
                        resolve_omega_index_array_path(index_path, str(split_info[f"{field}_path"])),
                        mmap_mode=mmap_mode,
                        allow_pickle=False,
                    )
                    for field in ("recording_code", "start_frame", "sequence_index")
                }
            )
        merged: Dict[str, np.ndarray] = {}
        for field in ("recording_code", "start_frame", "sequence_index"):
            parts = [entry[field] for entry in arrays_by_split if int(entry[field].shape[0]) > 0]
            if parts:
                merged[field] = np.concatenate(parts, axis=0)
            else:
                merged[field] = np.empty((0,), dtype=np.int32)
        split_info = {
            "num_windows": int(sum(int(manifest["splits"][name]["num_windows"]) for name in manifest["split_names"]))
        }
        return manifest, split_info, merged
    if split not in manifest["splits"]:
        raise KeyError(f"unknown split: {split}")
    split_info = manifest["splits"][split]
    arrays = {
        field: np.load(
            resolve_omega_index_array_path(index_path, str(split_info[f"{field}_path"])),
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        for field in ("recording_code", "start_frame", "sequence_index")
    }
    return manifest, split_info, arrays


def _resolve_window_stride(cfg: Dict[str, Any]) -> tuple[int, int]:
    ds = cfg["dataset"]
    if "window_frames" in ds:
        window_frames = int(ds["window_frames"])
    else:
        hop_sec = float(cfg["audio"]["hop_len"]) / float(cfg["audio"]["target_sr"])
        window_frames = int(round(float(ds["window_sec"]) / hop_sec))
    if "stride_frames" in ds:
        stride_frames = int(ds["stride_frames"])
    else:
        hop_sec = float(cfg["audio"]["hop_len"]) / float(cfg["audio"]["target_sr"])
        stride_frames = int(round(float(ds["stride_sec"]) / hop_sec))
    return max(1, window_frames), max(1, stride_frames)


def _recording_split_plan(manifests: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    supervised = [m for m in manifests if m["label_available"] and m["split_hint"] != "exclude"]
    if not supervised:
        raise ValueError("no supervised recordings available for omega dataset index")

    plans: List[Dict[str, Any]] = []
    auto_recordings = [m["recording_id"] for m in supervised if m["split_hint"] == "auto"]
    split_ratio = cfg["dataset"]["split_ratio_single"]
    split_names = cfg["dataset"]["split_names"]
    split_margin_sec = float(cfg["dataset"]["split_margin_sec"])
    split_seed = int(cfg.get("experiment", {}).get("seed", 0))

    if len(supervised) == 1 and supervised[0]["split_hint"] == "auto":
        manifest = supervised[0]
        split_intervals = compute_single_recording_split_intervals(
            manifest["duration_sec"],
            split_ratio,
            split_names,
            split_margin_sec,
        )
        plans.append(
            {
                "manifest": manifest,
                "split_override": None,
                "split_intervals": split_intervals,
            }
        )
        return plans

    for manifest in supervised:
        if manifest["split_hint"] == "auto":
            split_override = choose_recording_level_split(
                manifest["recording_id"],
                auto_recordings,
                split_ratio,
                split_names,
                split_seed,
            )
        else:
            split_override = str(manifest["split_hint"])
        plans.append(
            {
                "manifest": manifest,
                "split_override": split_override,
                "split_intervals": None,
            }
        )
    return plans


def _build_recording_split_arrays(
    cache_path: str | Path,
    cfg: Dict[str, Any],
    *,
    split_override: Optional[str],
    split_intervals: Optional[Dict[str, tuple[float, float]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    cache = np.load(cache_path, allow_pickle=True)
    version = int(np.asarray(cache["schema_version"]).reshape(-1)[0])
    if version not in (4, 5, 6):
        raise ValueError(f"expected omega cache schema_version 4, 5, or 6, got {version} for {cache_path}")

    frame_time_sec = np.asarray(cache["frame_time_sec"], dtype=np.float32)
    split_names = list(cfg["dataset"]["split_names"])
    window_frames, stride_frames = _resolve_window_stride(cfg)
    empty = {
        split_name: {
            "start_frame": np.empty((0,), dtype=np.int32),
            "sequence_index": np.empty((0,), dtype=np.int32),
        }
        for split_name in split_names
    }
    if frame_time_sec.size < window_frames:
        return empty

    start_frames = np.arange(
        0,
        int(frame_time_sec.size) - window_frames + 1,
        stride_frames,
        dtype=np.int32,
    )
    target_frames = start_frames + (window_frames - 1)
    sequence_index = np.arange(start_frames.shape[0], dtype=np.int32)
    start_times = frame_time_sec[start_frames]
    end_times = frame_time_sec[target_frames]

    split_masks: Dict[str, np.ndarray] = {split_name: np.zeros(start_frames.shape[0], dtype=bool) for split_name in split_names}
    if split_override is not None:
        split_masks[str(split_override)] = np.ones(start_frames.shape[0], dtype=bool)
    elif split_intervals is not None:
        for split_name, (split_start, split_end) in split_intervals.items():
            split_masks[str(split_name)] = (start_times >= float(split_start)) & (end_times <= float(split_end))
    else:
        split_masks["train"] = np.ones(start_frames.shape[0], dtype=bool)

    return {
        split_name: {
            "start_frame": start_frames[mask],
            "sequence_index": sequence_index[mask],
        }
        for split_name, mask in split_masks.items()
    }


def _write_empty_array(path: Path, dtype: np.dtype[Any]) -> None:
    np.save(path, np.empty((0,), dtype=dtype))


def write_omega_dataset_index(
    manifests: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    output_path: str | Path,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = omega_index_data_dir(output_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    for old_file in data_dir.glob("*.npy"):
        old_file.unlink()

    plans = _recording_split_plan(manifests, cfg)
    split_names = list(cfg["dataset"]["split_names"])
    split_totals = {split_name: 0 for split_name in split_names}
    recording_entries: List[Dict[str, Any]] = []

    for recording_code, plan in enumerate(plans):
        arrays_by_split = _build_recording_split_arrays(
            plan["manifest"]["cache_path"],
            cfg,
            split_override=plan["split_override"],
            split_intervals=plan["split_intervals"],
        )
        windows_by_split = {
            split_name: int(arrays_by_split[split_name]["start_frame"].shape[0])
            for split_name in split_names
        }
        for split_name, count in windows_by_split.items():
            split_totals[split_name] += int(count)
        recording_entries.append(
            {
                "recording_code": int(recording_code),
                "recording_id": str(plan["manifest"]["recording_id"]),
                "cache_path": str(plan["manifest"]["cache_path"]),
                "split_hint": str(plan["manifest"].get("split_hint", "auto")),
                "label_available": bool(plan["manifest"]["label_available"]),
                "num_frames": int(plan["manifest"]["num_frames"]),
                "duration_sec": float(plan["manifest"]["duration_sec"]),
                "windows_by_split": windows_by_split,
            }
        )

    recording_code_dtype: np.dtype[Any] = np.uint16 if len(recording_entries) < np.iinfo(np.uint16).max else np.uint32
    field_dtypes = {
        "recording_code": recording_code_dtype,
        "start_frame": np.int32,
        "sequence_index": np.int32,
    }

    split_memmaps: Dict[str, Dict[str, np.memmap | None]] = {split_name: {} for split_name in split_names}
    split_paths: Dict[str, Dict[str, Path]] = {split_name: {} for split_name in split_names}
    for split_name in split_names:
        total = int(split_totals[split_name])
        for field_name, dtype in field_dtypes.items():
            path = data_dir / f"{split_name}_{field_name}.npy"
            split_paths[split_name][field_name] = path
            if total <= 0:
                _write_empty_array(path, dtype)
                split_memmaps[split_name][field_name] = None
            else:
                split_memmaps[split_name][field_name] = open_memmap(
                    path,
                    mode="w+",
                    dtype=dtype,
                    shape=(total,),
                )

    write_offsets = {split_name: 0 for split_name in split_names}
    for recording_code, plan in enumerate(plans):
        arrays_by_split = _build_recording_split_arrays(
            plan["manifest"]["cache_path"],
            cfg,
            split_override=plan["split_override"],
            split_intervals=plan["split_intervals"],
        )
        for split_name in split_names:
            start_frame = arrays_by_split[split_name]["start_frame"]
            count = int(start_frame.shape[0])
            if count <= 0:
                continue
            write_start = int(write_offsets[split_name])
            write_end = write_start + count
            split_memmaps[split_name]["recording_code"][write_start:write_end] = recording_code
            split_memmaps[split_name]["start_frame"][write_start:write_end] = start_frame
            split_memmaps[split_name]["sequence_index"][write_start:write_end] = arrays_by_split[split_name]["sequence_index"]
            write_offsets[split_name] = write_end

    for split_name in split_names:
        for field_name in field_dtypes:
            memmap = split_memmaps[split_name][field_name]
            if memmap is not None:
                memmap.flush()
        split_memmaps[split_name].clear()

    window_frames, stride_frames = _resolve_window_stride(cfg)
    manifest = {
        "schema_version": OMEGA_INDEX_SCHEMA_VERSION,
        "index_format": OMEGA_INDEX_FORMAT,
        "window_frames": int(window_frames),
        "stride_frames": int(stride_frames),
        "split_names": split_names,
        "splits": {
            split_name: {
                "num_windows": int(split_totals[split_name]),
                "recording_code_path": str(split_paths[split_name]["recording_code"].relative_to(output_path.parent)),
                "start_frame_path": str(split_paths[split_name]["start_frame"].relative_to(output_path.parent)),
                "sequence_index_path": str(split_paths[split_name]["sequence_index"].relative_to(output_path.parent)),
            }
            for split_name in split_names
        },
        "recordings": recording_entries,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
