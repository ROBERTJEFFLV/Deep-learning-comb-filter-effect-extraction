"""Training entry point for CombCepstralNet (K+1 cepstral-bin classifier).

Usage:
  python -m ml_uav_comb.scripts.train_cepstral \\
      --config ml_uav_comb/configs/cepstral_default.yaml \\
      [--build-cache] [--build-jobs 4] \\
      [--resume path/to/checkpoint.pt]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _load_cfg(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_cache(cfg: dict, build_jobs: int) -> None:
    index_path = Path(cfg["dataset"]["index_path"])
    if index_path.exists():
        print(f"[train_cepstral] Cache already exists at {index_path.parent}, skipping build.")
        return
    print(f"[train_cepstral] Building dataset cache …")
    cmd = [
        sys.executable, "-m", "ml_uav_comb.scripts.build_omega_dataset",
        "--config", str(args_global.config),
        "--build-jobs", str(build_jobs),
    ]
    subprocess.run(cmd, check=True)


args_global: argparse.Namespace


def main() -> None:
    global args_global
    parser = argparse.ArgumentParser(description="Train CombCepstralNet")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--build-cache", action="store_true", help="Build dataset cache if missing")
    parser.add_argument("--build-jobs", type=int, default=4, help="Parallel cache build workers")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    args_global = args

    cfg = _load_cfg(args.config)

    if args.build_cache:
        _ensure_cache(cfg, args.build_jobs)

    from ml_uav_comb.training.cepstral_trainer import train_model
    results = train_model(cfg, resume_from=args.resume)

    print("\n[train_cepstral] Training complete.")
    print(f"  best_epoch         = {results['best_epoch']}")
    print(f"  best_val_mae_cm    = {results['best_val_distance_mae_cm']:.3f}")
    test = results.get("test_metrics", {})
    if test:
        print(f"  test_accuracy      = {test.get('accuracy', 0.0):.4f}")
        print(f"  test_dist_mae_cm   = {test.get('distance_mae_cm', 0.0):.3f}")
        print(f"  test_pat_recall    = {test.get('pattern_recall', 0.0):.4f}")
        print(f"  test_fp_rate       = {test.get('fp_rate', 0.0):.4f}")


if __name__ == "__main__":
    main()
