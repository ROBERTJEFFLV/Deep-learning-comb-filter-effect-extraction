#!/usr/bin/env python3
"""Build recording-level caches and the dataset index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.features.feature_utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-jobs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    summary = build_dataset(cfg, build_jobs=args.build_jobs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
