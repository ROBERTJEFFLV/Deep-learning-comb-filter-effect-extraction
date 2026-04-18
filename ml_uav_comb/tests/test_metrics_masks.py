from __future__ import annotations

import unittest

import torch

from ml_uav_comb.training.metrics import compute_batch_metrics


class TestMetricsMasks(unittest.TestCase):
    def test_measurement_metrics_respect_masks(self) -> None:
        pred = {
            "measurement_distance_cm": torch.tensor([10.0, 30.0], dtype=torch.float32),
            "measurement_validity_logit": torch.tensor([10.0, -10.0], dtype=torch.float32),
        }
        batch = {
            "measurement_distance_target_cm": torch.tensor([12.0, 20.0], dtype=torch.float32),
            "valid_dist_gt_mask": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "measurement_distance_train_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "measurement_validity_target": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "measurement_validity_train_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }
        metrics = compute_batch_metrics(pred, batch, target_space="raw")
        self.assertGreater(metrics["measurement_mae_cm_train"], 0.0)
        self.assertEqual(metrics["measurement_mae_cm_gt"], 2.0)
        self.assertEqual(metrics["measurement_validity_acc_train"], 1.0)


if __name__ == "__main__":
    unittest.main()
