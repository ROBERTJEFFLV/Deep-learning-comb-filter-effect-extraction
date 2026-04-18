#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml_uav_comb.data_pipeline.export_omega_dataset import build_omega_dataset
from ml_uav_comb.features.feature_utils import load_yaml_config
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--build-jobs', type=int, default=None)
args = parser.parse_args()
cfg = load_yaml_config(args.config)
print(json.dumps(build_omega_dataset(cfg, build_jobs=args.build_jobs), indent=2))
