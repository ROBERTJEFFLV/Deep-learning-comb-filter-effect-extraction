#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml_uav_comb.data_pipeline.export_omega_dataset import build_omega_dataset
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.omega_trainer import build_model, create_dataloader, evaluate_model
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--build-jobs', type=int, default=None)
parser.add_argument('--split', default='test', choices=['train','val','test'])
args = parser.parse_args()
cfg = load_yaml_config(args.config)
if not Path(cfg['dataset']['index_path']).exists() or not Path(cfg['dataset']['normalization_path']).exists():
    build_omega_dataset(cfg, build_jobs=args.build_jobs)
loader = create_dataloader(cfg, args.split)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(cfg).to(device)
state = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state['model_state'])
if bool(cfg.get("training", {}).get("use_compile", True)) and hasattr(torch, "compile"):
    model = torch.compile(model, mode=str(cfg.get("training", {}).get("compile_mode", "reduce-overhead")))
print(json.dumps(evaluate_model(model, loader, device, cfg, cfg['evaluation']['prediction_csv']), indent=2))
