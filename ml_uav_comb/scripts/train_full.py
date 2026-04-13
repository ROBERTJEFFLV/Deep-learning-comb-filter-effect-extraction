#!/usr/bin/env python3
"""Run full training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_uav_comb.data_pipeline.export_dataset import build_dataset, dataset_artifacts_ready
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml_uav_comb/configs/default.yaml")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    if args.skip_build and args.force_rebuild:
        raise ValueError("--skip-build and --force-rebuild cannot be used together")

    if args.force_rebuild:
        print("[dataset] force rebuild requested")
        build_dataset(cfg, build_jobs=args.build_jobs)
    elif args.skip_build:
        ready, reason = dataset_artifacts_ready(cfg)
        if not ready:
            raise RuntimeError(f"--skip-build requested but dataset artifacts are not ready: {reason}")
        print("[dataset] reusing existing artifacts (--skip-build)")
    else:
        ready, reason = dataset_artifacts_ready(cfg)
        if ready:
            print("[dataset] reusing existing artifacts")
        else:
            print(f"[dataset] building artifacts because: {reason}")
            build_dataset(cfg, build_jobs=args.build_jobs)

    result = train_model(cfg, resume_from=args.resume_from)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
