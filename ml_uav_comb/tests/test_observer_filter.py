from __future__ import annotations

import unittest

from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.filtering.observer_filter import DistanceGridRangeTracker


class TestObserverFilter(unittest.TestCase):
    def test_invalid_measurement_can_skip_update(self) -> None:
        cfg = load_yaml_config("ml_uav_comb/configs/tiny_debug.yaml")
        tracker = DistanceGridRangeTracker(cfg)
        first = tracker.step(
            measurement_distance_cm=100.0,
            measurement_logvar=2.0,
            measurement_validity_prob=0.9,
            measurement_entropy=0.1,
            measurement_margin=0.8,
            timestamp_sec=0.0,
        )
        second = tracker.step(
            measurement_distance_cm=300.0,
            measurement_logvar=2.0,
            measurement_validity_prob=0.1,
            measurement_entropy=0.1,
            measurement_margin=0.8,
            timestamp_sec=0.1,
        )
        self.assertEqual(first["measurement_used_flag"], 1.0)
        self.assertEqual(second["measurement_used_flag"], 0.0)

    def test_entropy_and_margin_scale_r(self) -> None:
        cfg = load_yaml_config("ml_uav_comb/configs/tiny_debug.yaml")
        tracker = DistanceGridRangeTracker(cfg)
        low_uncertainty = tracker.step(
            measurement_distance_cm=100.0,
            measurement_logvar=2.0,
            measurement_validity_prob=0.9,
            measurement_entropy=0.1,
            measurement_margin=0.9,
            timestamp_sec=0.0,
        )
        high_uncertainty = tracker.step(
            measurement_distance_cm=100.0,
            measurement_logvar=2.0,
            measurement_validity_prob=0.9,
            measurement_entropy=1.5,
            measurement_margin=0.1,
            timestamp_sec=0.1,
        )
        self.assertGreater(high_uncertainty["R_eff"], low_uncertainty["R_eff"])


if __name__ == "__main__":
    unittest.main()
