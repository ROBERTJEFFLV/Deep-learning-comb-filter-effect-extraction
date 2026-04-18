from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_uav_comb.data_pipeline.export_dataset import build_dataset
from ml_uav_comb.tests.support import make_tiny_dataset_cfg


class TestSplitMargin(unittest.TestCase):
    def test_windows_do_not_cross_split_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_tiny_dataset_cfg(tmpdir, duration_sec=12.0, include_unlabeled_recording=False)
            build_dataset(cfg)

            with open(cfg["dataset"]["index_path"], "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            cache = np.load(Path(cfg["dataset"]["cache_dir"]) / "rec_1.npz", allow_pickle=True)
            frame_time = cache["frame_time_sec"]

            spans = {}
            for split in ("train", "val", "test"):
                split_rows = [row for row in rows if row["split"] == split]
                self.assertTrue(split_rows)
                start_times = [float(frame_time[int(row["start_frame"])]) for row in split_rows]
                end_times = [float(frame_time[int(row["end_frame"]) - 1]) for row in split_rows]
                spans[split] = (min(start_times), max(end_times))

            self.assertGreaterEqual(spans["val"][0] - spans["train"][1], float(cfg["dataset"]["window_sec"]))
            self.assertGreaterEqual(spans["test"][0] - spans["val"][1], float(cfg["dataset"]["window_sec"]))


if __name__ == "__main__":
    unittest.main()
