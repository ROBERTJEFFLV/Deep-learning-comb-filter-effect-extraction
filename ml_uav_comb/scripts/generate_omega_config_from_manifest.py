#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

from ml_uav_comb.features.feature_utils import load_yaml_config


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    required = {"run_id", "split", "output_wav", "labels_csv"}
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _iter_valid_distance_cm(label_path: Path) -> Iterable[float]:
    with label_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            distance_cm = _safe_float(row.get("distance_cm"))
            if distance_cm is None:
                distance_mm = _safe_float(row.get("distance_mm"))
                if distance_mm is not None:
                    distance_cm = distance_mm / 10.0
            if distance_cm is None:
                continue
            distance_valid = _safe_float(row.get("distance_valid"))
            if distance_valid is not None and distance_valid < 0.5:
                continue
            yield float(distance_cm)


def _resolve_distance_bounds_cm(rows: List[Dict[str, str]], margin_cm: float) -> tuple[float, float]:
    train_rows = [row for row in rows if str(row.get("split", "")).strip().lower() == "train"]
    source_rows = train_rows if train_rows else rows
    values: List[float] = []
    for row in source_rows:
        label_path = Path(str(row["labels_csv"])).resolve()
        if not label_path.exists():
            raise FileNotFoundError(f"label file not found: {label_path}")
        values.extend(_iter_valid_distance_cm(label_path))
    if not values:
        raise ValueError("failed to derive distance bounds from labels: no finite distance_cm values found")
    min_cm = max(1.0, float(min(values)) - float(margin_cm))
    max_cm = max(min_cm + 1.0, float(max(values)) + float(margin_cm))
    return min_cm, max_cm


def _normalize_split_name(value: Any) -> str:
    split = str(value or "").strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split value: {value!r}")
    return split


def _normalization_split(rows: List[Dict[str, str]], requested: str) -> str:
    requested = str(requested).strip().lower()
    split_counts: Dict[str, int] = {}
    for row in rows:
        split = _normalize_split_name(row.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1

    if requested in {"", "auto"}:
        if split_counts.get("train", 0) > 0:
            return "train"
        return "all"
    if requested == "all":
        return "all"
    if split_counts.get(requested, 0) <= 0:
        raise ValueError(
            f"requested normalization split {requested!r} has zero manifest rows; counts={split_counts}"
        )
    return requested


def _maybe_override(section: Dict[str, Any], key: str, value: Optional[Any]) -> None:
    if value is not None:
        section[key] = value


def build_config(
    *,
    manifest_path: Path,
    base_config_path: Path,
    out_config_path: Path,
    run_name: str,
    max_runs: Optional[int],
    build_jobs: Optional[int],
    num_workers: Optional[int],
    batch_size: Optional[int],
    eval_batch_size: Optional[int],
    epochs: Optional[int],
    sequence_length: Optional[int],
    eval_sequence_length: Optional[int],
    chunk_step: Optional[int],
    eval_chunk_step: Optional[int],
    normalization_split: str,
    distance_margin_cm: float,
) -> Dict[str, Any]:
    rows = _read_manifest(manifest_path)
    if max_runs is not None:
        if int(max_runs) <= 0:
            raise ValueError("--max-runs must be > 0")
        rows = rows[: int(max_runs)]
        if not rows:
            raise ValueError("manifest produced zero rows after --max-runs")

    cfg = load_yaml_config(base_config_path)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = f"uav_comb_{run_name}"

    recordings: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}
    for row in rows:
        split = _normalize_split_name(row.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
        recordings.append(
            {
                "recording_id": f"{split}_{str(row['run_id']).strip()}",
                "audio_path": str(Path(str(row["output_wav"])).resolve()),
                "label_path": str(Path(str(row["labels_csv"])).resolve()),
                "label_format": "csv",
                "split_hint": split,
            }
        )

    distance_cm_min, distance_cm_max = _resolve_distance_bounds_cm(rows, distance_margin_cm)
    resolved_norm_split = _normalization_split(rows, normalization_split)

    cfg["dataset"]["recordings"] = recordings
    cfg["dataset"]["cache_dir"] = f"ml_uav_comb/cache/{run_name}"
    cfg["dataset"]["index_path"] = f"ml_uav_comb/cache/{run_name}/dataset_index.json"
    cfg["dataset"]["normalization_path"] = f"ml_uav_comb/cache/{run_name}/normalization_stats.npz"
    cfg["dataset"]["meta_path"] = f"ml_uav_comb/cache/{run_name}/dataset_index_meta.json"
    cfg["dataset"]["dynamic_epoch_split"] = False
    cfg["dataset"]["normalization_split"] = resolved_norm_split
    _maybe_override(cfg["dataset"], "build_jobs", build_jobs)

    cfg["model"]["distance_cm_min"] = float(distance_cm_min)
    cfg["model"]["distance_cm_max"] = float(distance_cm_max)

    cfg["training"]["checkpoint_dir"] = f"ml_uav_comb/artifacts/{run_name}"
    cfg["evaluation"]["prediction_csv"] = f"ml_uav_comb/artifacts/{run_name}/eval_predictions.csv"
    cfg["inference"]["prediction_csv"] = f"ml_uav_comb/artifacts/{run_name}/infer_predictions.csv"

    _maybe_override(cfg["training"], "num_workers", num_workers)
    _maybe_override(cfg["training"], "batch_size", batch_size)
    _maybe_override(cfg["training"], "eval_batch_size", eval_batch_size)
    _maybe_override(cfg["training"], "epochs", epochs)
    _maybe_override(cfg["training"], "sequence_length", sequence_length)
    _maybe_override(cfg["training"], "eval_sequence_length", eval_sequence_length)
    _maybe_override(cfg["training"], "chunk_step", chunk_step)
    _maybe_override(cfg["training"], "eval_chunk_step", eval_chunk_step)
    if "persistent_workers" in cfg.get("training", {}) and num_workers is not None:
        cfg["training"]["persistent_workers"] = int(num_workers) > 0

    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    out_config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "config_path": str(out_config_path),
        "run_name": run_name,
        "num_recordings": len(recordings),
        "split_counts": split_counts,
        "normalization_split": resolved_norm_split,
        "distance_cm_min": float(distance_cm_min),
        "distance_cm_max": float(distance_cm_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an omega config from a run_manifest.csv style dataset.")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.csv")
    parser.add_argument(
        "--base-config",
        default="ml_uav_comb/configs/omega_default.yaml",
        help="Base omega config template",
    )
    parser.add_argument("--out-config", default=None, help="Output YAML path; defaults to ml_uav_comb/configs/<run-name>.yaml")
    parser.add_argument("--run-name", default=None, help="Config/cache/artifacts suffix; defaults to output config stem")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on manifest rows")
    parser.add_argument("--build-jobs", type=int, default=None, help="Override dataset.build_jobs")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override training.eval_batch_size")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--sequence-length", type=int, default=None, help="Override training.sequence_length")
    parser.add_argument("--eval-sequence-length", type=int, default=None, help="Override training.eval_sequence_length")
    parser.add_argument("--chunk-step", type=int, default=None, help="Override training.chunk_step")
    parser.add_argument("--eval-chunk-step", type=int, default=None, help="Override training.eval_chunk_step")
    parser.add_argument(
        "--normalization-split",
        default="auto",
        choices=["auto", "train", "val", "test", "all"],
        help="Split used when computing normalization stats",
    )
    parser.add_argument(
        "--distance-margin-cm",
        type=float,
        default=1.0,
        help="Margin applied around observed label distance bounds",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    base_config_path = Path(args.base_config).resolve()
    if args.out_config is not None:
        out_config_path = Path(args.out_config).resolve()
    else:
        fallback_name = args.run_name or manifest_path.stem
        out_config_path = Path("ml_uav_comb/configs") / f"{fallback_name}.yaml"
    run_name = args.run_name or out_config_path.stem

    summary = build_config(
        manifest_path=manifest_path,
        base_config_path=base_config_path,
        out_config_path=out_config_path,
        run_name=run_name,
        max_runs=args.max_runs,
        build_jobs=args.build_jobs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        eval_sequence_length=args.eval_sequence_length,
        chunk_step=args.chunk_step,
        eval_chunk_step=args.eval_chunk_step,
        normalization_split=args.normalization_split,
        distance_margin_cm=args.distance_margin_cm,
    )

    print(f"[config] wrote: {summary['config_path']}")
    print(f"[config] run_name: {summary['run_name']}")
    print(f"[config] num_recordings: {summary['num_recordings']}")
    print(f"[config] split_counts: {summary['split_counts']}")
    print(f"[config] normalization_split: {summary['normalization_split']}")
    print(
        "[config] distance_cm_range: "
        f"[{summary['distance_cm_min']:.3f}, {summary['distance_cm_max']:.3f}]"
    )


if __name__ == "__main__":
    main()
