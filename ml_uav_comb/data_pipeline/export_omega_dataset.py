"""Build caches and index files for omega-regression training."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ml_uav_comb.data_pipeline.offline_omega_feature_extractor import process_audio_array
from ml_uav_comb.data_pipeline.omega_dataset_index import (
    OMEGA_INDEX_FORMAT,
    OMEGA_INDEX_SCHEMA_VERSION,
    load_omega_index_manifest,
    resolve_omega_index_array_path,
    write_omega_dataset_index,
)
from ml_uav_comb.data_pipeline.omega_normalization import compute_omega_normalization_stats
from ml_uav_comb.features.feature_utils import (
    TerminalProgressBar,
    ensure_dir,
    load_audio_mono,
    load_optional_labels,
    metadata_path_for_index,
)


def _dataset_build_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feature_contract_version": 4,
        "experiment": {"seed": int(cfg.get("experiment", {}).get("seed", 0))},
        "audio": cfg["audio"],
        "front_end": {
            "history_frames": cfg["front_end"]["history_frames"],
            "smooth_sigma_1": cfg["front_end"]["smooth_sigma_1"],
            "smooth_sigma_2": cfg["front_end"]["smooth_sigma_2"],
            "noise_gate_smooth": cfg["front_end"].get("noise_gate_smooth", 0.9),
        },
        "observability": cfg.get("observability", {}),
        "dataset": {
            "recordings": cfg["dataset"]["recordings"],
            "window_frames": cfg["dataset"].get("window_frames"),
            "stride_frames": cfg["dataset"].get("stride_frames"),
            "window_sec": cfg["dataset"].get("window_sec"),
            "stride_sec": cfg["dataset"].get("stride_sec"),
            "split_margin_sec": cfg["dataset"]["split_margin_sec"],
            "split_ratio_single": cfg["dataset"]["split_ratio_single"],
            "split_names": cfg["dataset"]["split_names"],
            "dynamic_epoch_split": bool(cfg["dataset"].get("dynamic_epoch_split", False)),
            "dynamic_epoch_split_ratio": cfg["dataset"].get("dynamic_epoch_split_ratio", [5, 1, 1]),
        },
        "model": {
            "distance_cm_min": cfg["model"]["distance_cm_min"],
            "distance_cm_max": cfg["model"]["distance_cm_max"],
        },
    }


def compute_omega_build_signature(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(_dataset_build_spec(cfg), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def omega_dataset_artifacts_ready(cfg: Dict[str, Any], *, check_cache_files: bool = True) -> tuple[bool, str]:
    index_path = Path(cfg["dataset"]["index_path"])
    normalization_path = Path(cfg["dataset"]["normalization_path"])
    meta_path = Path(cfg["dataset"].get("meta_path") or metadata_path_for_index(index_path))
    if not index_path.exists():
        return False, f"missing index_path: {index_path}"
    if not normalization_path.exists():
        return False, f"missing normalization_path: {normalization_path}"
    if not meta_path.exists():
        return False, f"missing meta_path: {meta_path}"
    try:
        index_manifest = load_omega_index_manifest(index_path)
    except Exception as exc:
        return False, f"invalid index manifest: {exc}"
    if int(index_manifest.get("schema_version", -1)) != OMEGA_INDEX_SCHEMA_VERSION:
        return False, f"unexpected index schema_version: {index_manifest.get('schema_version')}"
    if str(index_manifest.get("index_format", "")) != OMEGA_INDEX_FORMAT:
        return False, f"unexpected index format: {index_manifest.get('index_format')}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("schema_version", -1)) != OMEGA_INDEX_SCHEMA_VERSION:
        return False, f"unexpected schema_version: {meta.get('schema_version')}"
    if str(meta.get("build_signature", "")) != compute_omega_build_signature(cfg):
        return False, "build_signature mismatch"
    split_names = index_manifest.get("split_names", [])
    for split_name in split_names:
        split_info = index_manifest["splits"][split_name]
        for field_name in ("recording_code", "start_frame", "sequence_index"):
            array_path = resolve_omega_index_array_path(index_path, str(split_info[f"{field_name}_path"]))
            if not array_path.exists():
                return False, f"missing index array: {array_path}"
    if check_cache_files:
        for recording_entry in index_manifest.get("recordings", []):
            cache_path = Path(str(recording_entry["cache_path"]))
            if not cache_path.exists():
                return False, f"missing cache file: {cache_path}"
    return True, "ready"


def export_omega_recording_cache(recording_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    audio, sr = load_audio_mono(recording_cfg["audio_path"], int(cfg["audio"]["target_sr"]), cfg["audio"].get("max_duration_sec"))
    label_dict = load_optional_labels(recording_cfg.get("label_path"), recording_cfg["recording_id"])
    features = process_audio_array(audio, sr, cfg, optional_labels=label_dict)
    cache_dir = ensure_dir(cfg["dataset"]["cache_dir"])
    cache_path = cache_dir / f"{recording_cfg['recording_id']}.npz"
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(cache_dir), prefix=f"{recording_cfg['recording_id']}_", suffix=".npz", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        np.savez_compressed(tmp_path, **features)
        tmp_path.replace(cache_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return {
        "recording_id": recording_cfg["recording_id"],
        "split_hint": recording_cfg.get("split_hint", "auto"),
        "cache_path": str(cache_path),
        "label_available": label_dict is not None,
        "num_frames": int(features["frame_time_sec"].shape[0]),
        "duration_sec": float(features["frame_time_sec"][-1]) if int(features["frame_time_sec"].shape[0]) > 0 else 0.0,
    }


def _resolve_build_jobs(cfg: Dict[str, Any], build_jobs: Optional[int]) -> int:
    if build_jobs is None:
        build_jobs = int(cfg["dataset"].get("build_jobs", 1))
    build_jobs = int(build_jobs)
    if build_jobs == 0:
        return 1
    if build_jobs < 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, build_jobs)


def _export_recording_manifests(recordings: List[Dict[str, Any]], cfg: Dict[str, Any], build_jobs: int) -> List[Dict[str, Any]]:
    progress = TerminalProgressBar("Export Caches", len(recordings))
    if build_jobs <= 1 or len(recordings) <= 1:
        manifests = []
        for idx, recording_cfg in enumerate(recordings, start=1):
            manifests.append(export_omega_recording_cache(recording_cfg, cfg))
            progress.update(idx, extra=str(recording_cfg.get("recording_id", idx)))
        progress.finish(extra="done")
        return manifests
    manifests_by_index: Dict[int, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=build_jobs) as executor:
        future_map = {executor.submit(export_omega_recording_cache, rec, cfg): (idx, rec) for idx, rec in enumerate(recordings)}
        completed = 0
        for future in as_completed(future_map):
            idx, rec = future_map[future]
            try:
                manifests_by_index[idx] = future.result()
            except Exception as exc:
                raise RuntimeError(f"failed to export omega cache for recording_id={rec.get('recording_id', idx)}") from exc
            completed += 1
            progress.update(completed, extra=str(rec.get("recording_id", idx)))
    progress.finish(extra=f"jobs={build_jobs}")
    return [manifests_by_index[idx] for idx in range(len(recordings))]


def build_omega_dataset(cfg: Dict[str, Any], build_jobs: Optional[int] = None) -> Dict[str, Any]:
    recordings = list(cfg["dataset"]["recordings"])
    jobs = _resolve_build_jobs(cfg, build_jobs)
    stage_progress = TerminalProgressBar("Build Dataset", 4)
    stage_progress.update(0, extra="export caches")
    manifests = _export_recording_manifests(recordings, cfg, jobs)
    index_path = Path(cfg["dataset"]["index_path"])
    stage_progress.update(1, extra="write index")
    index_manifest = write_omega_dataset_index(manifests, cfg, index_path)
    normalization_path = Path(cfg["dataset"]["normalization_path"])
    normalization_split = "all" if bool(cfg["dataset"].get("dynamic_epoch_split", False)) else "train"
    stage_progress.update(2, extra=f"normalize {normalization_split}")
    normalization_summary = compute_omega_normalization_stats(index_path, normalization_path, split=normalization_split)
    meta_path = Path(cfg["dataset"].get("meta_path") or metadata_path_for_index(index_path))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    stage_progress.update(3, extra="write metadata")
    meta = {
        "schema_version": OMEGA_INDEX_SCHEMA_VERSION,
        "index_path": str(index_path),
        "normalization_path": str(normalization_path),
        "build_signature": compute_omega_build_signature(cfg),
        "supervised_recordings": [m["recording_id"] for m in manifests if m["label_available"] and m["split_hint"] != "exclude"],
        "excluded_recordings": [m["recording_id"] for m in manifests if (not m["label_available"]) or m["split_hint"] == "exclude"],
        "num_recordings": len(recordings),
        "num_windows": int(sum(int(split_info["num_windows"]) for split_info in index_manifest["splits"].values())),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    stage_progress.finish(extra="ready")
    return {
        "index_path": str(index_path),
        "cache_dir": str(cfg["dataset"]["cache_dir"]),
        "normalization_path": str(normalization_path),
        "meta_path": str(meta_path),
        "build_jobs": int(jobs),
        "num_windows": int(meta["num_windows"]),
        **normalization_summary,
    }
