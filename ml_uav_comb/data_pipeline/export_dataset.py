"""Recording-level cache export and dataset index generation."""

from __future__ import annotations

import json
import os
import tempfile
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ml_uav_comb.data_pipeline.dataset_index import build_dataset_index_for_cache, write_dataset_index
from ml_uav_comb.data_pipeline.normalization import compute_normalization_stats
from ml_uav_comb.data_pipeline.offline_feature_extractor import process_audio_array
from ml_uav_comb.features.feature_utils import (
    choose_recording_level_split,
    compute_single_recording_split_intervals,
    ensure_dir,
    load_audio_mono,
    load_optional_labels,
    metadata_path_for_index,
)


def _dataset_build_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    return {
        "experiment": {
            "seed": int(cfg.get("experiment", {}).get("seed", 0)),
        },
        "audio": cfg["audio"],
        "front_end": cfg["front_end"],
        "kf": cfg["kf"],
        "features": cfg["features"],
        "dataset": {
            "recordings": dataset_cfg["recordings"],
            "window_sec": dataset_cfg["window_sec"],
            "stride_sec": dataset_cfg["stride_sec"],
            "split_margin_sec": dataset_cfg["split_margin_sec"],
            "split_ratio_single": dataset_cfg["split_ratio_single"],
            "split_names": dataset_cfg["split_names"],
            "target_space": dataset_cfg["target_space"],
            "gt_motion_eps_cm_per_sec": dataset_cfg["gt_motion_eps_cm_per_sec"],
            "local_fit_half_window_sec": dataset_cfg["local_fit_half_window_sec"],
            "local_fit_min_points": dataset_cfg["local_fit_min_points"],
            "local_fit_max_rmse_cm": dataset_cfg["local_fit_max_rmse_cm"],
            "supervision_mode": dataset_cfg.get("supervision_mode", "strict_gt"),
            "window_anchor": dataset_cfg.get("window_anchor", "trailing"),
            "target_position": dataset_cfg.get("target_position", "window_end"),
            "allow_derived_sign_train": dataset_cfg.get("allow_derived_sign_train", False),
            "allow_pseudo_confidence_train": dataset_cfg.get("allow_pseudo_confidence_train", False),
            "use_physics_gate_as_distance_mask": dataset_cfg.get("use_physics_gate_as_distance_mask", False),
            "use_physics_gate_as_distance_weight": dataset_cfg.get("use_physics_gate_as_distance_weight", True),
            "default_distance_valid_when_missing": dataset_cfg.get("default_distance_valid_when_missing", True),
            "default_sign_annotated_when_missing": dataset_cfg.get("default_sign_annotated_when_missing", False),
            "distance_weight_floor": dataset_cfg.get("distance_weight_floor", 0.25),
            "distance_weight_reliable": dataset_cfg.get("distance_weight_reliable", 1.0),
        },
        "model": {
            "num_candidates": model_cfg.get("num_candidates", 64),
            "distance_grid_cm_min": model_cfg.get("distance_grid_cm_min", 20.0),
            "distance_grid_cm_max": model_cfg.get("distance_grid_cm_max", 300.0),
            "distance_grid_mode": model_cfg.get("distance_grid_mode", "uniform"),
            "distance_target_sigma_cm": model_cfg.get("distance_target_sigma_cm", 12.0),
        },
    }


def compute_dataset_build_signature(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(_dataset_build_spec(cfg), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_artifacts_ready(
    cfg: Dict[str, Any],
    *,
    check_cache_files: bool = True,
) -> tuple[bool, str]:
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
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to read meta_path: {exc}"

    if int(meta.get("schema_version", -1)) != 2:
        return False, f"unexpected schema_version: {meta.get('schema_version')}"

    expected_signature = compute_dataset_build_signature(cfg)
    actual_signature = str(meta.get("build_signature", ""))
    if actual_signature != expected_signature:
        return False, "build_signature mismatch"

    if check_cache_files:
        cache_dir = Path(cfg["dataset"]["cache_dir"])
        for recording_cfg in cfg["dataset"]["recordings"]:
            cache_path = cache_dir / f"{recording_cfg['recording_id']}.npz"
            if not cache_path.exists():
                return False, f"missing cache file: {cache_path}"

    return True, "ready"


def export_recording_cache(
    recording_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    audio, sr = load_audio_mono(
        recording_cfg["audio_path"],
        target_sr=int(cfg["audio"]["target_sr"]),
        max_duration_sec=cfg["audio"].get("max_duration_sec"),
    )
    label_dict = load_optional_labels(recording_cfg.get("label_path"), recording_cfg["recording_id"])
    features = process_audio_array(audio, sr, cfg, optional_labels=label_dict)
    cache_dir = ensure_dir(cfg["dataset"]["cache_dir"])
    cache_path = cache_dir / f"{recording_cfg['recording_id']}.npz"
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(cache_dir),
            prefix=f"{recording_cfg['recording_id']}_",
            suffix=".npz",
            delete=False,
        ) as tmp_file:
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
        "label_dict": label_dict,
        "num_frames": int(features["frame_time_sec"].shape[0]),
        "duration_sec": float(features["frame_time_sec"][-1]) if int(features["frame_time_sec"].shape[0]) > 0 else 0.0,
    }


def _recording_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    label_dict = manifest["label_dict"]
    return {
        "recording_id": manifest["recording_id"],
        "cache_path": manifest["cache_path"],
        "label_available": label_dict is not None,
        "label_points": 0 if label_dict is None else int(label_dict["time_sec"].shape[0]),
        "num_frames": manifest["num_frames"],
        "duration_sec": manifest["duration_sec"],
        "split_hint": manifest["split_hint"],
    }


def _build_supervised_rows(manifests: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    supervised = [m for m in manifests if m["label_dict"] is not None and m["split_hint"] != "exclude"]
    if not supervised:
        raise ValueError("no supervised recordings available for v2 dataset index")

    rows: List[Dict[str, Any]] = []
    auto_recordings = [m["recording_id"] for m in supervised if m["split_hint"] == "auto"]
    split_ratio = cfg["dataset"]["split_ratio_single"]
    split_names = cfg["dataset"]["split_names"]
    split_margin_sec = float(cfg["dataset"]["split_margin_sec"])
    split_seed = int(cfg.get("experiment", {}).get("seed", 0))

    if len(supervised) == 1 and supervised[0]["split_hint"] == "auto":
        manifest = supervised[0]
        split_intervals = compute_single_recording_split_intervals(
            total_duration_sec=manifest["duration_sec"],
            split_ratio=split_ratio,
            split_names=split_names,
            margin_sec=split_margin_sec,
        )
        rows.extend(
            build_dataset_index_for_cache(
                recording_id=manifest["recording_id"],
                cache_path=manifest["cache_path"],
                label_dict=manifest["label_dict"],
                cfg=cfg,
                split_override=None,
                split_intervals=split_intervals,
            )
        )
        return rows

    for manifest in supervised:
        split_hint = manifest["split_hint"]
        if split_hint == "auto":
            split_override: Optional[str] = choose_recording_level_split(
                recording_id=manifest["recording_id"],
                recording_order=auto_recordings,
                split_ratio=split_ratio,
                split_names=split_names,
                seed=split_seed,
            )
        else:
            split_override = split_hint
        rows.extend(
            build_dataset_index_for_cache(
                recording_id=manifest["recording_id"],
                cache_path=manifest["cache_path"],
                label_dict=manifest["label_dict"],
                cfg=cfg,
                split_override=split_override,
                split_intervals=None,
            )
        )
    return rows


def _resolve_build_jobs(cfg: Dict[str, Any], build_jobs: Optional[int]) -> int:
    if build_jobs is None:
        build_jobs = int(cfg["dataset"].get("build_jobs", 1))
    build_jobs = int(build_jobs)
    if build_jobs == 0:
        return 1
    if build_jobs < 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, build_jobs)


def _export_recording_manifests(
    recordings: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    build_jobs: int,
) -> List[Dict[str, Any]]:
    if not recordings:
        return []
    if build_jobs <= 1 or len(recordings) <= 1:
        return [export_recording_cache(recording_cfg, cfg) for recording_cfg in recordings]

    manifests_by_index: Dict[int, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=build_jobs) as executor:
        future_map = {
            executor.submit(export_recording_cache, recording_cfg, cfg): (idx, recording_cfg)
            for idx, recording_cfg in enumerate(recordings)
        }
        for future in as_completed(future_map):
            idx, recording_cfg = future_map[future]
            try:
                manifests_by_index[idx] = future.result()
            except Exception as exc:
                recording_id = recording_cfg.get("recording_id", f"#{idx}")
                raise RuntimeError(f"failed to export cache for recording_id={recording_id}") from exc

    return [manifests_by_index[idx] for idx in range(len(recordings))]


def build_dataset(cfg: Dict[str, Any], build_jobs: Optional[int] = None) -> Dict[str, Any]:
    recordings = list(cfg["dataset"]["recordings"])
    recording_ids = [str(r.get("recording_id", "")) for r in recordings]
    if len(recording_ids) != len(set(recording_ids)):
        counts: Dict[str, int] = {}
        for rid in recording_ids:
            counts[rid] = counts.get(rid, 0) + 1
        duplicated = sorted([rid for rid, count in counts.items() if count > 1])
        preview = duplicated[:10]
        raise ValueError(
            f"dataset.recordings has duplicated recording_id values (showing up to 10): {preview}"
        )
    jobs = _resolve_build_jobs(cfg, build_jobs)
    manifests = _export_recording_manifests(recordings=recordings, cfg=cfg, build_jobs=jobs)
    public_recordings = [_recording_summary(manifest) for manifest in manifests]

    index_rows = _build_supervised_rows(manifests, cfg)
    index_path = Path(cfg["dataset"]["index_path"])
    write_dataset_index(index_rows, index_path)

    normalization_path = Path(cfg["dataset"]["normalization_path"])
    normalization_summary = compute_normalization_stats(index_path=index_path, output_path=normalization_path)

    meta_path = Path(cfg["dataset"].get("meta_path") or metadata_path_for_index(index_path))
    supervised_recordings = [m["recording_id"] for m in manifests if m["label_dict"] is not None and m["split_hint"] != "exclude"]
    excluded_recordings = [m["recording_id"] for m in manifests if (m["label_dict"] is None) or m["split_hint"] == "exclude"]
    meta = {
        "schema_version": 2,
        "index_path": str(index_path),
        "normalization_path": str(normalization_path),
        "target_space": str(cfg["dataset"]["target_space"]),
        "supervision_mode": str(cfg["dataset"].get("supervision_mode", "weak_hybrid")),
        "build_signature": compute_dataset_build_signature(cfg),
        "supervised_recordings": supervised_recordings,
        "excluded_recordings": excluded_recordings,
        "num_recordings": len(recordings),
        "num_windows": len(index_rows),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "index_path": str(index_path),
        "cache_dir": str(cfg["dataset"]["cache_dir"]),
        "normalization_path": str(normalization_path),
        "meta_path": str(meta_path),
        "build_jobs": int(jobs),
        "recordings": public_recordings,
        "supervised_recordings": supervised_recordings,
        "excluded_recordings": excluded_recordings,
        "num_windows": len(index_rows),
        **normalization_summary,
    }
