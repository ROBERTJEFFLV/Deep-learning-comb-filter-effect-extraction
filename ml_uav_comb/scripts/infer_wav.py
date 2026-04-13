#!/usr/bin/env python3
"""Run inference on a single wav file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.infer import infer_single_wav
from ml_uav_comb.training.trainer import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--build-jobs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if not Path(cfg["dataset"]["normalization_path"]).exists():
        build_dataset(cfg, build_jobs=args.build_jobs)
    output_csv = args.output_csv or cfg["inference"]["prediction_csv"]
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state["model_state"])
    summary = infer_single_wav(model, cfg, args.wav, output_csv, device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
