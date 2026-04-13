from __future__ import annotations

import unittest

import torch

from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.omega_losses import combined_omega_loss


class TestOmegaLossGating(unittest.TestCase):
    def test_pattern_target_acts_as_detached_soft_regression_mask(self) -> None:
        cfg = load_yaml_config("ml_uav_comb/configs/omega_tiny_debug.yaml")
        batch = {
            "pattern_target": torch.tensor([0.25, 1.0], dtype=torch.float32),
            "omega_target": torch.tensor([0.2, 0.2], dtype=torch.float32),
            "sequence_index": torch.tensor([0, 1], dtype=torch.long),
            "chunk_id": torch.tensor([0, 0], dtype=torch.long),
            "recording_id": ["rec", "rec"],
        }
        model_out = {
            "omega_pred": torch.tensor([0.6, 0.6], dtype=torch.float32),
            "pattern_logit": torch.tensor([-2.0, 2.0], dtype=torch.float32),
        }

        losses = combined_omega_loss(model_out, batch, cfg)

        self.assertEqual(tuple(losses["sample_weights"].shape), (2,))
        self.assertAlmostEqual(float(losses["sample_weights"][0].item()), 0.25, places=6)
        self.assertGreater(float(losses["sample_weights"][1].item()), 0.0)
        self.assertGreaterEqual(float(losses["loss_pattern"].item()), 0.0)
        self.assertGreater(float(losses["loss_omega"].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
