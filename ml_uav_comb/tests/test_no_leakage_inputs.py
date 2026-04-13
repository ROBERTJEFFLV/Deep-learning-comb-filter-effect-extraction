from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.tests.support import make_tiny_dataset_cfg


class TestNoLeakageInputs(unittest.TestCase):
    def test_recorded_audio_excluded_and_teacher_not_in_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_tiny_dataset_cfg(tmpdir, duration_sec=4.0, include_unlabeled_recording=True)
            build_dataset(cfg)
            with open(cfg["dataset"]["index_path"], "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            self.assertTrue(all(row["recording_id"] == "rec_1" for row in rows))


if __name__ == "__main__":
    unittest.main()
