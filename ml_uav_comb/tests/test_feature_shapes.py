from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.tests.support import make_tiny_dataset_cfg


class TestFeatureShapes(unittest.TestCase):
    def test_feature_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_tiny_dataset_cfg(tmpdir, duration_sec=4.0, include_unlabeled_recording=False)
            build_dataset(cfg)

            cache = np.load(Path(cfg["dataset"]["cache_dir"]) / "rec_1.npz", allow_pickle=True)
            self.assertEqual(int(cache["schema_version"][0]), 2)
            self.assertEqual(cache["phase_stft"].shape[-1], 3)
            self.assertEqual(cache["diff_comb"].shape[-1], 3)
            self.assertEqual(cache["phase_stft"].shape[1], 43)
            self.assertEqual(cache["diff_comb"].shape[1], 43)
            self.assertEqual(cache["stpacc"].shape[-1], 64)
            self.assertEqual(cache["scalar_seq"].shape[-1], 4)
            self.assertEqual(cache["scalar_observed_mask"].shape, cache["scalar_seq"].shape)
            self.assertEqual(cache["scalar_reliable_mask"].shape, cache["scalar_seq"].shape)
            self.assertEqual(cache["teacher_seq"].shape[-1], 9)
            self.assertNotIn("heuristic_distance_kf_cm", cache["scalar_field_names"].tolist())
            self.assertIn("heuristic_distance_kf_cm", cache["teacher_field_names"].tolist())


if __name__ == "__main__":
    unittest.main()
