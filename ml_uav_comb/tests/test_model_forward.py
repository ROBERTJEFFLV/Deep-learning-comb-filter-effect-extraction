from __future__ import annotations

import unittest

import torch

from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.models.uav_comb_observer import UAVCombObserver
from ml_uav_comb.training.losses import combined_loss


class TestModelForward(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        cfg = load_yaml_config("ml_uav_comb/configs/tiny_debug.yaml")
        model = UAVCombObserver(cfg=cfg, scalar_dim=4, use_stpacc=True)
        d = int(cfg["model"]["num_candidates"])
        batch = {
            "phase": torch.randn(2, 3, 240, 43),
            "comb": torch.randn(2, 3, 240, 43),
            "scalar": torch.randn(2, 240, 4),
            "scalar_observed_mask": torch.ones(2, 240, 4),
            "scalar_reliable_mask": torch.ones(2, 240, 4),
            "stpacc": torch.randn(2, 1, 240, 64),
            "frequencies_hz": torch.linspace(1000.0, 5000.0, 43)[None].repeat(2, 1),
            "distance_target_grid": torch.softmax(torch.randn(2, d), dim=-1),
            "measurement_distance_target_cm": torch.tensor([80.0, 120.0], dtype=torch.float32),
            "measurement_distance_train_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "measurement_validity_target": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "measurement_validity_train_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }
        pred = model(batch)
        self.assertIn("distance_logits", pred)
        self.assertIn("measurement_distance_cm", pred)
        self.assertIn("measurement_validity_logit", pred)
        self.assertIn("distance_grid_cm", pred)
        self.assertIn("temporal_repr", pred)
        self.assertIn("last_repr", pred)
        self.assertEqual(pred["distance_logits"].shape, (2, d))
        self.assertEqual(pred["measurement_distance_cm"].shape, (2,))
        self.assertEqual(pred["measurement_validity_logit"].shape, (2,))
        self.assertEqual(pred["distance_grid_cm"].shape, (d,))
        self.assertTrue(torch.allclose(pred["last_repr"], pred["temporal_repr"][:, -1, :], atol=1e-6))

        losses = combined_loss(
            pred=pred,
            batch=batch,
            lambda_likelihood_ce=1.0,
            lambda_measurement_mean=0.3,
            lambda_measurement_validity=0.3,
        )
        losses["total"].backward()
        self.assertFalse(torch.isnan(losses["total"]))


if __name__ == "__main__":
    unittest.main()
