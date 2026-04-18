#!/usr/bin/env python3
"""
Launch script for omega training with:
  - Synthetic RIR reflection_gain sweep data (train/val)
  - Real flight audio test data
  - Dynamic epoch-level train/val resplit
  - Comprehensive pre-training data distribution checks
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_uav_comb.data_pipeline.export_omega_dataset import build_omega_dataset
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.omega_trainer import train_model


# ─── Config ──────────────────────────────────────────────────────────────────

SYNTHETIC_MANIFEST = (
    "/home/lvmingyang/March24/datasets/simulation/tool_box/RIR/"
    "rir_run_dataset/run_manifest.csv"
)
REAL_TEST_MANIFEST = (
    "/home/lvmingyang/March24/datasets/simulation/original_sound/"
    "copy_test_real/test_dataset/run_manifest.csv"
)
BASE_CONFIG = str(ROOT / "ml_uav_comb" / "configs" / "omega_default.yaml")

CACHE_DIR = str(ROOT / "ml_uav_comb" / "cache" / "omega_rir_sweep_real_test")
CHECKPOINT_DIR = str(ROOT / "ml_uav_comb" / "artifacts" / "omega_rir_sweep_real_test")

EXPECTED_REFLECTION_GAINS = [
    round(0.30 + i * 0.05, 2) for i in range(15)
]  # 0.30, 0.35, ..., 1.00


def _build_recordings_list(
    syn_manifest: pd.DataFrame, real_manifest: pd.DataFrame
) -> list:
    recordings = []
    for _, row in syn_manifest.iterrows():
        split = str(row["split"])
        recordings.append(
            {
                "recording_id": str(row["run_id"]),
                "audio_path": str(row["output_wav"]),
                "label_path": str(row["labels_csv"]),
                "label_format": "csv",
                "split_hint": split,
            }
        )
    for _, row in real_manifest.iterrows():
        recordings.append(
            {
                "recording_id": str(row["run_id"]),
                "audio_path": str(row["output_wav"]),
                "label_path": str(row["labels_csv"]),
                "label_format": "csv",
                "split_hint": "test",
            }
        )
    return recordings


def _build_config(args) -> dict:
    base_cfg = load_yaml_config(BASE_CONFIG)
    syn_df = pd.read_csv(SYNTHETIC_MANIFEST)
    real_df = pd.read_csv(REAL_TEST_MANIFEST)

    recordings = _build_recordings_list(syn_df, real_df)

    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["name"] = "omega_rir_sweep_real_test"
    cfg["experiment"]["seed"] = args.seed

    cfg["dataset"]["recordings"] = recordings
    cfg["dataset"]["cache_dir"] = CACHE_DIR
    cfg["dataset"]["index_path"] = os.path.join(CACHE_DIR, "dataset_index.json")
    cfg["dataset"]["normalization_path"] = os.path.join(
        CACHE_DIR, "normalization_stats.npz"
    )
    cfg["dataset"]["meta_path"] = os.path.join(CACHE_DIR, "dataset_index_meta.json")
    cfg["dataset"]["normalization_split"] = "train"
    cfg["dataset"]["split_names"] = ["train", "val", "test"]
    cfg["dataset"]["split_ratio_single"] = [0.8, 0.2, 0.0]
    cfg["dataset"]["build_jobs"] = args.build_jobs

    # Dynamic epoch split: reshuffle synthetic train/val each epoch
    cfg["dataset"]["dynamic_epoch_split"] = True
    cfg["dataset"]["dynamic_epoch_split_ratio"] = [4, 1, 1]

    cfg["dataset"]["stride_frames"] = args.stride_frames

    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["checkpoint_dir"] = CHECKPOINT_DIR
    cfg["training"]["batch_size"] = args.batch_size
    cfg["training"]["eval_batch_size"] = args.batch_size
    cfg["training"]["sequence_length"] = args.sequence_length
    cfg["training"]["eval_sequence_length"] = args.sequence_length * 2
    cfg["training"]["chunk_step"] = args.sequence_length
    cfg["training"]["eval_chunk_step"] = args.sequence_length * 2
    cfg["training"]["num_workers"] = args.num_workers
    cfg["training"]["log_interval_steps"] = 50
    cfg["training"]["use_compile"] = not args.no_compile

    # Loss rebalancing: smooth_l1 replaces huber (see omega_losses.py),
    # so omega_loss magnitude is now ~0.003 instead of ~1e-6.
    # lambda_omega=7 brings omega contribution (~0.021) on par with pattern BCE (~0.02).
    cfg["training"]["lambda_omega"] = 7.0
    cfg["training"]["lambda_delta"] = 1.4
    cfg["training"]["lambda_acc"] = 0.35
    cfg["training"]["lambda_pattern"] = 1.0
    # Pattern BCE pos_weight: synthetic data has ~96% positive rate.
    # Down-weight positives to balance: (1 - 0.964) / 0.964 ≈ 0.037.
    # This makes the effective contribution of positives and negatives equal.
    cfg["training"]["pattern_pos_weight"] = 0.037

    cfg["evaluation"]["prediction_csv"] = os.path.join(
        CHECKPOINT_DIR, "eval_predictions.csv"
    )

    return cfg


# ─── Verification checks ─────────────────────────────────────────────────────


def _check_synthetic_manifest(syn_df: pd.DataFrame) -> bool:
    ok = True
    print("\n" + "=" * 70)
    print("=== SYNTHETIC DATA CHECKS ===")
    print("=" * 70)

    # Source wavs
    n_sources = syn_df["source_wav"].nunique()
    n_total = len(syn_df)
    print(f"Source wav count: {n_sources}")
    print(f"Total samples: {n_total}")
    print(f"Expected: {n_sources} × 15 = {n_sources * 15}")
    if n_total != n_sources * 15:
        print("  ✗ FAIL: total != n_sources × 15")
        ok = False

    # Reflection gains
    gains = sorted(syn_df["reflection_gain"].unique())
    gains_per_source = syn_df.groupby("source_wav")["reflection_gain"].nunique()
    print(f"\nReflection gains ({len(gains)} values): {gains}")
    if len(gains) != 15:
        print("  ✗ FAIL: expected 15 reflection gain values")
        ok = False
    else:
        expected = [round(0.30 + i * 0.05, 2) for i in range(15)]
        if gains != expected:
            print(f"  ✗ FAIL: expected {expected}")
            ok = False
        else:
            print("  ✓ All 15 reflection gain values correct")

    if (gains_per_source != 15).any():
        bad = gains_per_source[gains_per_source != 15]
        print(f"  ✗ FAIL: {len(bad)} sources have != 15 gains")
        ok = False
    else:
        print(f"  ✓ All {n_sources} sources have exactly 15 gains")

    # Splits
    splits = dict(syn_df["split"].value_counts())
    print(f"\nSplits: {splits}")
    if "test" in splits:
        print("  ✗ FAIL: synthetic test split exists!")
        ok = False
    else:
        print("  ✓ No synthetic test split")

    train_n = splits.get("train", 0)
    val_n = splits.get("val", 0)
    ratio = train_n / max(val_n, 1)
    print(f"  train:val = {train_n}:{val_n} = {ratio:.2f}:1")

    # Distance distribution from manifest
    d_min_cm = syn_df["distance_min_m"] * 100
    d_max_cm = syn_df["distance_max_m"] * 100
    print(f"\nPer-run distance_min_cm: min={d_min_cm.min():.1f}, mean={d_min_cm.mean():.1f}, max={d_min_cm.max():.1f}")
    print(f"Per-run distance_max_cm: min={d_max_cm.min():.1f}, mean={d_max_cm.mean():.1f}, max={d_max_cm.max():.1f}")
    overall_range = (d_min_cm.min(), d_max_cm.max())
    print(f"Overall range: [{overall_range[0]:.1f}, {overall_range[1]:.1f}] cm")

    if overall_range[1] < 30:
        print("  ✗ CRITICAL: distance_max < 30cm! Data still concentrated near range!")
        ok = False
    elif overall_range[1] < 50:
        print("  ⚠ WARNING: distance_max < 50cm, expected ~55cm")
    else:
        print("  ✓ Distance range covers ~5-55cm")

    # Observable ratio
    obs = syn_df["observable_ratio_res"]
    print(f"\nObservable ratio: mean={obs.mean():.3f}, min={obs.min():.3f}, >=0.9: {(obs >= 0.9).mean():.1%}")

    return ok


def _check_real_test_manifest(real_df: pd.DataFrame) -> bool:
    ok = True
    print("\n" + "=" * 70)
    print("=== REAL TEST DATA CHECKS ===")
    print("=" * 70)
    print(f"Test recordings: {len(real_df)}")

    for _, row in real_df.iterrows():
        if not os.path.isfile(row["output_wav"]):
            print(f"  ✗ Missing wav: {row['output_wav']}")
            ok = False
            break
        if not os.path.isfile(row["labels_csv"]):
            print(f"  ✗ Missing labels: {row['labels_csv']}")
            ok = False
            break
    else:
        print("  ✓ All wav and label files exist")

    # Check all splits are test
    if "split" in real_df.columns:
        non_test = real_df[real_df["split"] != "test"]
        if len(non_test) > 0:
            print(f"  ✗ FAIL: {len(non_test)} real recordings not in test split!")
            ok = False
        else:
            print("  ✓ All real recordings in test split")

    return ok


def _check_label_compatibility(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> bool:
    print("\n" + "=" * 70)
    print("=== LABEL FORMAT COMPATIBILITY ===")
    print("=" * 70)

    syn_label = pd.read_csv(syn_df.iloc[0]["labels_csv"], nrows=2)
    real_label = pd.read_csv(real_df.iloc[0]["labels_csv"], nrows=2)
    syn_cols = set(syn_label.columns)
    real_cols = set(real_label.columns)

    print(f"Synthetic labels columns: {sorted(syn_cols)}")
    print(f"Real test labels columns: {sorted(real_cols)}")

    required = {"time_sec", "distance_cm", "distance_valid"}
    if required.issubset(syn_cols) and required.issubset(real_cols):
        print("  ✓ Both formats have required columns")
        return True
    else:
        missing_syn = required - syn_cols
        missing_real = required - real_cols
        if missing_syn:
            print(f"  ✗ Synthetic missing: {missing_syn}")
        if missing_real:
            print(f"  ✗ Real missing: {missing_real}")
        return False


def _check_no_smoke_test_residuals(cfg: dict) -> bool:
    print("\n" + "=" * 70)
    print("=== SMOKE TEST / DEBUG RESIDUAL CHECKS ===")
    print("=" * 70)
    ok = True

    dangerous_keys = [
        "max_runs", "max_files", "max_samples", "debug_subset",
        "smoke_test", "quick_test", "temporary_subset", "near_range_filter",
    ]
    for section_name, section in cfg.items():
        if isinstance(section, dict):
            for key in dangerous_keys:
                if key in section:
                    val = section[key]
                    if val not in (None, False, 0):
                        print(f"  ✗ DANGER: {section_name}.{key} = {val}")
                        ok = False

    # Check training.max_train_steps_per_epoch
    max_steps = cfg.get("training", {}).get("max_train_steps_per_epoch", 0)
    if max_steps > 0:
        print(f"  ⚠ training.max_train_steps_per_epoch = {max_steps} (will limit training)")

    if ok:
        print("  ✓ No smoke test / debug residuals found")
    return ok


def _pre_training_summary(cfg: dict):
    print("\n" + "=" * 70)
    print("=== PRE-TRAINING CONFIGURATION SUMMARY ===")
    print("=" * 70)

    recordings = cfg["dataset"]["recordings"]
    syn_rec = [r for r in recordings if r["split_hint"] != "test"]
    test_rec = [r for r in recordings if r["split_hint"] == "test"]
    train_rec = [r for r in recordings if r["split_hint"] == "train"]
    val_rec = [r for r in recordings if r["split_hint"] == "val"]

    print(f"Config: omega_rir_sweep_real_test (programmatic)")
    print(f"Cache dir: {cfg['dataset']['cache_dir']}")
    print(f"Index path: {cfg['dataset']['index_path']}")
    print(f"Normalization path: {cfg['dataset']['normalization_path']}")
    print(f"Checkpoint dir: {cfg['training']['checkpoint_dir']}")
    print(f"")
    print(f"Total recordings: {len(recordings)}")
    print(f"  Synthetic train: {len(train_rec)}")
    print(f"  Synthetic val: {len(val_rec)}")
    print(f"  Real test: {len(test_rec)}")
    print(f"  Synthetic total: {len(syn_rec)}")
    print(f"")
    print(f"Dynamic epoch split: {cfg['dataset']['dynamic_epoch_split']}")
    print(f"  Ratio: {cfg['dataset']['dynamic_epoch_split_ratio']}")
    print(f"Normalization split: {cfg['dataset']['normalization_split']}")
    print(f"Epochs: {cfg['training']['epochs']}")
    print(f"Batch size: {cfg['training']['batch_size']}")
    print(f"Sequence length: {cfg['training']['sequence_length']}")
    eff = cfg['training']['batch_size'] * cfg['training']['sequence_length']
    print(f"Effective windows/step: {eff}")
    print(f"Window frames: {cfg['dataset']['window_frames']}")
    print(f"Stride frames: {cfg['dataset']['stride_frames']}")


def _post_build_checks(cfg: dict, build_result: dict) -> bool:
    print("\n" + "=" * 70)
    print("=== POST-BUILD DATA DISTRIBUTION CHECKS ===")
    print("=" * 70)

    index_path = Path(cfg["dataset"]["index_path"])
    if not index_path.exists():
        print("  ✗ index_path does not exist!")
        return False

    from ml_uav_comb.data_pipeline.omega_dataset_index import load_omega_index_manifest

    manifest = load_omega_index_manifest(index_path)

    ok = True
    for split_name in ["train", "val", "test"]:
        if split_name in manifest["splits"]:
            n_win = manifest["splits"][split_name]["num_windows"]
            n_rec = manifest["splits"][split_name].get("num_recordings", "?")
            print(f"  {split_name}: {n_win} windows, {n_rec} recordings")
        else:
            print(f"  {split_name}: not present")
            if split_name in ("train", "val"):
                ok = False

    # Check normalization was computed on train only
    norm_split = build_result.get("normalization_split", "unknown")
    print(f"\nNormalization computed on: {norm_split}")
    if norm_split != "train":
        print(f"  ⚠ WARNING: normalization_split={norm_split}, expected 'train'")

    # Load some cache files to check distance distributions
    cache_dir = Path(cfg["dataset"]["cache_dir"])
    npz_files = sorted(cache_dir.glob("*.npz"))
    if npz_files:
        # Sample distance from a few synthetic train caches
        all_dist = []
        sample_count = min(200, len(npz_files))
        rng = np.random.default_rng(0)
        sampled = rng.choice(len(npz_files), size=sample_count, replace=False)
        for idx in sampled:
            f = npz_files[idx]
            try:
                data = np.load(f, allow_pickle=True)
                if "frame_distance_cm" in data:
                    all_dist.append(data["frame_distance_cm"])
            except Exception:
                pass
        if all_dist:
            all_dist = np.concatenate(all_dist)
            print(f"\nSampled frame-level distance_cm ({len(all_dist)} frames from {sample_count} caches):")
            print(f"  min={all_dist.min():.1f}, max={all_dist.max():.1f}, "
                  f"mean={all_dist.mean():.1f}, std={all_dist.std():.1f}")
            for p in [1, 5, 25, 50, 75, 95, 99]:
                print(f"  p{p}={np.percentile(all_dist, p):.1f}", end="")
            print()

            if all_dist.mean() < 15 and all_dist.max() < 20:
                print("  ✗ CRITICAL: distance distribution concentrated in 5-15cm range!")
                print("  ✗ This is the exact bug from the previous training run!")
                ok = False
            else:
                print("  ✓ Distance distribution covers expected range")

    return ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Launch omega RIR sweep training")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--build-jobs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stride-frames", type=int, default=4,
                        help="Window stride in frames (default 4, old default was 1)")
    parser.add_argument("--sequence-length", type=int, default=32,
                        help="Windows per chunk for batch sampler (default 32)")
    parser.add_argument("--force-rebuild", action="store_true", default=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--rebuild-index-only", action="store_true",
                        help="Skip .npz cache export, rebuild only index + normalization")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (saves ~2x VRAM from CUDA graphs)")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run checks, don't train")
    args = parser.parse_args()

    print("=" * 70)
    print("  OMEGA RIR SWEEP TRAINING PIPELINE")
    print("=" * 70)

    # ── Step 1: Load manifests and validate ──────────────────────────────
    syn_df = pd.read_csv(SYNTHETIC_MANIFEST)
    real_df = pd.read_csv(REAL_TEST_MANIFEST)

    syn_ok = _check_synthetic_manifest(syn_df)
    real_ok = _check_real_test_manifest(real_df)
    compat_ok = _check_label_compatibility(syn_df, real_df)

    if not (syn_ok and real_ok and compat_ok):
        print("\n✗ PRE-CHECKS FAILED. Aborting.")
        sys.exit(1)

    # ── Step 2: Build config ─────────────────────────────────────────────
    cfg = _build_config(args)
    _check_no_smoke_test_residuals(cfg)
    _pre_training_summary(cfg)

    # ── Step 3: Build or rebuild dataset ─────────────────────────────────
    if args.rebuild_index_only:
        # Fast path: reuse existing .npz caches, rebuild only index + normalization
        cache_dir = Path(CACHE_DIR)
        print("\n" + "=" * 70)
        print("=== REBUILDING INDEX ONLY (reusing .npz caches) ===")
        print("=" * 70)
        existing_npz = sorted(cache_dir.glob("*.npz"))
        existing_npz = [f for f in existing_npz if f.name not in ("normalization_stats.npz",)]
        print(f"  Found {len(existing_npz)} existing .npz cache files")

        # Reconstruct manifests from existing caches + config recordings
        rec_map = {r["recording_id"]: r for r in cfg["dataset"]["recordings"]}
        manifests = []
        for npz_path in existing_npz:
            rec_id = npz_path.stem
            if rec_id not in rec_map:
                continue
            rec_cfg = rec_map[rec_id]
            data = np.load(npz_path, allow_pickle=True)
            n_frames = int(data["frame_time_sec"].shape[0])
            duration = float(data["frame_time_sec"][-1]) if n_frames > 0 else 0.0
            manifests.append({
                "recording_id": rec_id,
                "split_hint": rec_cfg.get("split_hint", "auto"),
                "cache_path": str(npz_path),
                "label_available": True,
                "num_frames": n_frames,
                "duration_sec": duration,
            })
        print(f"  Matched {len(manifests)} recordings to config")

        from ml_uav_comb.data_pipeline.omega_dataset_index import write_omega_dataset_index
        from ml_uav_comb.data_pipeline.omega_normalization import compute_omega_normalization_stats
        from ml_uav_comb.data_pipeline.export_omega_dataset import compute_omega_build_signature
        from ml_uav_comb.features.feature_utils import metadata_path_for_index

        index_path = Path(cfg["dataset"]["index_path"])
        # Clean old index arrays
        index_data_dir = index_path.parent / (index_path.stem + "_data")
        if index_data_dir.exists():
            for old_file in index_data_dir.glob("*.npy"):
                old_file.unlink()

        print(f"  Rebuilding index with stride_frames={cfg['dataset']['stride_frames']} ...")
        index_manifest = write_omega_dataset_index(manifests, cfg, index_path)
        for s in cfg["dataset"]["split_names"]:
            if s in index_manifest["splits"]:
                print(f"    {s}: {index_manifest['splits'][s]['num_windows']} windows")

        normalization_path = Path(cfg["dataset"]["normalization_path"])
        norm_split = cfg["dataset"].get("normalization_split", "train")
        print(f"  Recomputing normalization on split={norm_split} ...")
        norm_summary = compute_omega_normalization_stats(index_path, normalization_path, split=norm_split)

        meta_path = Path(cfg["dataset"].get("meta_path") or str(metadata_path_for_index(index_path)))
        meta = {
            "schema_version": 2,
            "index_path": str(index_path),
            "normalization_path": str(normalization_path),
            "normalization_split": str(norm_split),
            "build_signature": compute_omega_build_signature(cfg),
            "supervised_recordings": [m["recording_id"] for m in manifests if m["label_available"]],
            "excluded_recordings": [],
            "num_recordings": len(manifests),
            "num_windows": int(sum(
                int(s_info["num_windows"])
                for s_info in index_manifest["splits"].values()
            )),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        build_result = {
            "index_path": str(index_path),
            "cache_dir": str(cfg["dataset"]["cache_dir"]),
            "normalization_path": str(normalization_path),
            "meta_path": str(meta_path),
            "normalization_split": str(norm_split),
            "num_windows": meta["num_windows"],
            **norm_summary,
        }
        print(json.dumps({k: v for k, v in build_result.items()
                          if not isinstance(v, (list, dict)) or k == "normalization_split"}, indent=2))
        post_ok = _post_build_checks(cfg, build_result)
        if not post_ok:
            print("\n✗ POST-BUILD CHECKS FAILED. Aborting training.")
            sys.exit(1)

    elif args.force_rebuild and not args.skip_build:
        cache_dir = Path(CACHE_DIR)
        if cache_dir.exists():
            print(f"\nCleaning old cache: {cache_dir}")
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("=== BUILDING OMEGA DATASET ===")
        print("=" * 70)
        build_result = build_omega_dataset(cfg, build_jobs=args.build_jobs)
        print(json.dumps(build_result, indent=2))

        # Post-build checks
        post_ok = _post_build_checks(cfg, build_result)
        if not post_ok:
            print("\n✗ POST-BUILD CHECKS FAILED. Aborting training.")
            sys.exit(1)

    # ── Step 4: Resolve fixed test recording codes ───────────────────────
    # Load the index to find recording codes for test recordings
    from ml_uav_comb.data_pipeline.omega_dataset_index import load_omega_index_manifest
    index_manifest = load_omega_index_manifest(cfg["dataset"]["index_path"])
    test_recording_ids = set(
        r["recording_id"] for r in cfg["dataset"]["recordings"]
        if r["split_hint"] == "test"
    )
    fixed_test_codes = []
    for rec_entry in index_manifest["recordings"]:
        if rec_entry["recording_id"] in test_recording_ids:
            fixed_test_codes.append(int(rec_entry["recording_code"]))
    cfg["dataset"]["fixed_test_recording_codes"] = fixed_test_codes
    print(f"\nFixed test recording codes: {len(fixed_test_codes)} recordings")

    if args.dry_run:
        print("\n=== DRY RUN COMPLETE ===")
        # Save effective config
        config_path = os.path.join(CHECKPOINT_DIR, "effective_config.json")
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        # Don't save recordings list (too large), save summary
        cfg_save = copy.deepcopy(cfg)
        cfg_save["dataset"]["_recordings_count"] = len(cfg_save["dataset"]["recordings"])
        del cfg_save["dataset"]["recordings"]
        Path(config_path).write_text(json.dumps(cfg_save, indent=2), encoding="utf-8")
        print(f"Effective config saved to: {config_path}")
        return

    # ── Step 5: Train ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== STARTING TRAINING ===")
    print("=" * 70)

    # Save effective config
    config_path = os.path.join(CHECKPOINT_DIR, "effective_config.json")
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    cfg_save = copy.deepcopy(cfg)
    cfg_save["dataset"]["_recordings_count"] = len(cfg_save["dataset"]["recordings"])
    del cfg_save["dataset"]["recordings"]
    cfg_save["dataset"]["fixed_test_recording_codes_count"] = len(fixed_test_codes)
    Path(config_path).write_text(json.dumps(cfg_save, indent=2), encoding="utf-8")
    print(f"Effective config saved to: {config_path}")

    result = train_model(cfg, resume_from=args.resume_from)

    print("\n" + "=" * 70)
    print("=== TRAINING COMPLETE ===")
    print("=" * 70)
    print(json.dumps(result, indent=2))

    # ── Step 6: Final report ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== FINAL REPORT ===")
    print("=" * 70)
    print(f"Config: omega_rir_sweep_real_test (programmatic)")
    print(f"Synthetic manifest: {SYNTHETIC_MANIFEST}")
    print(f"Real test manifest: {REAL_TEST_MANIFEST}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Best checkpoint: {result.get('best_checkpoint', 'N/A')}")
    print(f"History: {result.get('history_path', 'N/A')}")
    print(f"Best epoch: {result.get('best_epoch', 'N/A')}")
    print(f"Best val MAE: {result.get('best_val_distance_mae_cm', 'N/A'):.3f} cm")
    if result.get("final_test_metrics"):
        test_mae = result["final_test_metrics"].get("distance_mae_cm", "N/A")
        print(f"Final test MAE: {test_mae}")


if __name__ == "__main__":
    main()
