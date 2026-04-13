#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml_uav_comb.data_pipeline.export_omega_dataset import build_omega_dataset, omega_dataset_artifacts_ready
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.omega_trainer import train_model
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='ml_uav_comb/configs/omega_default.yaml')
parser.add_argument('--build-jobs', type=int, default=None)
parser.add_argument('--skip-build', action='store_true')
parser.add_argument('--force-rebuild', action='store_true')
parser.add_argument('--resume-from', default=None)
args = parser.parse_args()
cfg = load_yaml_config(args.config)
if args.force_rebuild:
    build_omega_dataset(cfg, build_jobs=args.build_jobs)
elif not args.skip_build:
    ready, _ = omega_dataset_artifacts_ready(cfg)
    if not ready:
        build_omega_dataset(cfg, build_jobs=args.build_jobs)
print(json.dumps(train_model(cfg, resume_from=args.resume_from), indent=2))
