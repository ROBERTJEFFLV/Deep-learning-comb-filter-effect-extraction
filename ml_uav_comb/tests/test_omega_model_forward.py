from __future__ import annotations

import unittest

import torch

from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.models.uav_comb_omega_net import UAVCombOmegaNet


class TestOmegaModelForward(unittest.TestCase):
    def test_forward_returns_minimal_dual_head_outputs(self) -> None:
        cfg = load_yaml_config("ml_uav_comb/configs/omega_tiny_debug.yaml")
        model = UAVCombOmegaNet(cfg)
        batch = {"x": torch.randn(2, 1, 68, 43)}

        pred = model(batch)
        self.assertIn("omega_pred", pred)
        self.assertIn("pattern_logit", pred)
        self.assertIn("pattern_prob", pred)
        self.assertEqual(pred["omega_pred"].shape, (2,))
        self.assertEqual(pred["pattern_logit"].shape, (2,))
        self.assertEqual(pred["pattern_prob"].shape, (2,))
        self.assertTrue(torch.all((pred["pattern_prob"] >= 0.0) & (pred["pattern_prob"] <= 1.0)))

        loss = pred["omega_pred"].mean()
        loss.backward()
        grad_found = any(param.grad is not None for param in model.parameters() if param.requires_grad)
        self.assertTrue(grad_found)

    def test_forward_debug_exposes_frequency_pool_details(self) -> None:
        cfg = load_yaml_config("ml_uav_comb/configs/omega_tiny_debug.yaml")
        model = UAVCombOmegaNet(cfg)
        pred = model({"x": torch.randn(2, 1, 68, 43)}, return_debug=True)
        self.assertIn("frequency_pool_weights", pred)
        self.assertIn("pooled_frequency_hz", pred)
        self.assertEqual(pred["pooled_frequency_hz"].shape[:2], (2, 68))
        self.assertTrue(
            torch.allclose(
                pred["frequency_pool_weights"].sum(dim=-1),
                torch.ones_like(pred["frequency_pool_weights"].sum(dim=-1)),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
