from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.tests.support import make_tiny_dataset_cfg


class TestSchemaV2(unittest.TestCase):
    def test_schema_v2_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_tiny_dataset_cfg(tmpdir, duration_sec=3.0, include_unlabeled_recording=False)
            build_dataset(cfg)
            cache = np.load(Path(cfg["dataset"]["cache_dir"]) / "rec_1.npz", allow_pickle=True)
            self.assertEqual(int(cache["schema_version"][0]), 2)
            self.assertIn("scalar_observed_mask", cache.files)
            self.assertIn("scalar_reliable_mask", cache.files)
            self.assertIn("teacher_field_names", cache.files)
            self.assertIn("frequencies_hz", cache.files)
            rows = list(np.genfromtxt(cfg["dataset"]["index_path"], delimiter=",", names=True, dtype=None, encoding="utf-8"))
            self.assertTrue(rows)
            self.assertIn("target_frame", rows[0].dtype.names)
            self.assertIn("target_time_sec", rows[0].dtype.names)
            self.assertIn("measurement_distance_train_mask", rows[0].dtype.names)
            self.assertIn("measurement_validity_target", rows[0].dtype.names)
            self.assertIn("distance_target_grid", rows[0].dtype.names)
            self.assertTrue(all(float(row["sign_train_mask"]) == 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
