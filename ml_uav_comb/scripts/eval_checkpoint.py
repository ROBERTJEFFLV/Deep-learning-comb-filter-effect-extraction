#!/usr/bin/env python3
"""Evaluate a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.evaluate import evaluate_model
from ml_uav_comb.training.trainer import build_model, create_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--build-jobs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if not Path(cfg["dataset"]["index_path"]).exists() or not Path(cfg["dataset"]["normalization_path"]).exists():
        build_dataset(cfg, build_jobs=args.build_jobs)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    _, _, test_loader = create_dataloaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state["model_state"])
    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        cfg=cfg,
        output_csv=cfg["evaluation"]["prediction_csv"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
