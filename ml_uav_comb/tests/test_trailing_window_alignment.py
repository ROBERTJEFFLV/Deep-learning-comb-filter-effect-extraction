from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.tests.support import make_tiny_dataset_cfg


class TestTrailingWindowAlignment(unittest.TestCase):
    def test_target_frame_is_window_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_tiny_dataset_cfg(tmpdir, duration_sec=4.0, include_unlabeled_recording=False)
            build_dataset(cfg)

            with open(cfg["dataset"]["index_path"], "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            first = rows[0]
            end_frame = int(first["end_frame"])
            target_frame = int(first["target_frame"])
            self.assertEqual(target_frame, end_frame - 1)

            cache_path = Path(first["cache_path"])
            frame_time = np.load(cache_path, allow_pickle=True)["frame_time_sec"]
            target_time_expected = float(frame_time[target_frame])
            target_time_actual = float(first["target_time_sec"])
            self.assertAlmostEqual(target_time_actual, target_time_expected, places=6)


if __name__ == "__main__":
    unittest.main()
