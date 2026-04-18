from __future__ import annotations

import unittest

import numpy as np

from ml_uav_comb.features.feature_utils import fit_local_motion_sign


class TestSignLabelFit(unittest.TestCase):
    def test_local_fit_motion_sign(self) -> None:
        t = np.linspace(0.0, 1.0, 21, dtype=np.float32)
        approaching = {
            "time_sec": t,
            "distance_cm": 40.0 - 10.0 * t,
            "motion_sign": np.asarray(["unknown"] * len(t), dtype=object),
            "explicit_motion_sign_mask": np.zeros(len(t), dtype=np.float32),
            "confidence_valid": np.zeros(len(t), dtype=np.float32),
            "has_conf_gt": np.asarray([0.0], dtype=np.float32),
        }
        retreating = dict(approaching)
        retreating["distance_cm"] = 20.0 + 10.0 * t

        fit_a = fit_local_motion_sign(0.5, approaching, 0.2, 1.0, 5, 2.0)
        fit_r = fit_local_motion_sign(0.5, retreating, 0.2, 1.0, 5, 2.0)
        self.assertEqual(fit_a["sign_label"], "approach")
        self.assertEqual(fit_r["sign_label"], "retreat")


if __name__ == "__main__":
    unittest.main()
