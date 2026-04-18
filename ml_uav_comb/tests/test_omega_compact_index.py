from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_uav_comb.data_pipeline.export_omega_dataset import (
    build_omega_dataset,
    omega_dataset_artifacts_ready,
)
from ml_uav_comb.data_pipeline.omega_dataset import OmegaWindowDataset
from ml_uav_comb.data_pipeline.omega_dataset_index import load_omega_index_manifest, omega_index_data_dir
from ml_uav_comb.tests.support import make_omega_tiny_dataset_cfg
from ml_uav_comb.training.omega_trainer import ContiguousSequenceBatchSampler, create_dataloader


class TestOmegaCompactIndex(unittest.TestCase):
    def test_compact_index_build_and_dataset_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_omega_tiny_dataset_cfg(tmpdir, duration_sec=4.0, include_unlabeled_recording=True)
            summary = build_omega_dataset(cfg)
            ready, reason = omega_dataset_artifacts_ready(cfg)
            self.assertTrue(ready, reason)

            index_path = Path(cfg["dataset"]["index_path"])
            self.assertEqual(index_path.suffix, ".json")
            self.assertTrue(index_path.exists())
            self.assertTrue(omega_index_data_dir(index_path).exists())

            manifest = load_omega_index_manifest(index_path)
            total_windows = sum(int(split_info["num_windows"]) for split_info in manifest["splits"].values())
            self.assertEqual(int(summary["num_windows"]), total_windows)
            self.assertEqual(len(manifest["recordings"]), 1)

            dataset = OmegaWindowDataset(index_path, split="train", max_cache_files=1)
            self.assertGreater(len(dataset), 0)
            sample = dataset[0]
            start_frame = int(dataset.start_frame[0])
            target_frame = start_frame + dataset.window_frames - 1
            cache_path = Path(manifest["recordings"][0]["cache_path"])
            cache = np.load(cache_path, allow_pickle=True)
            self.assertEqual(sample["recording_id"], "omega_rec_1")
            self.assertAlmostEqual(
                float(sample["target_time_sec"].item()),
                float(cache["frame_time_sec"][target_frame]),
                places=6,
            )
            self.assertEqual(tuple(sample["x"].shape), (1, dataset.window_frames, 43))
            self.assertIn("pattern_target", sample)
            self.assertGreaterEqual(float(sample["pattern_target"].item()), 0.0)
            self.assertLessEqual(float(sample["pattern_target"].item()), 1.0)
            self.assertTrue(np.isfinite(float(sample["omega_target"].item())))

    def test_sampler_uses_compact_chunk_starts_and_contiguous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_omega_tiny_dataset_cfg(tmpdir, duration_sec=4.0, include_unlabeled_recording=False)
            build_omega_dataset(cfg)
            dataset = OmegaWindowDataset(cfg["dataset"]["index_path"], split="train", max_cache_files=1)
            sampler = ContiguousSequenceBatchSampler(
                dataset,
                sequence_length=4,
                chunk_step=4,
                chunks_per_batch=2,
                drop_last=False,
                shuffle_chunks=False,
            )
            self.assertIsInstance(sampler._full_chunk_starts, np.ndarray)
            self.assertEqual(sampler._full_chunk_starts.ndim, 1)

            first_batch = next(iter(sampler))
            self.assertEqual(len(first_batch), 8)
            seq_index = np.asarray(dataset.sequence_index[first_batch], dtype=np.int64)
            self.assertTrue(np.all(np.diff(seq_index[:4]) == 1))
            self.assertTrue(np.all(np.diff(seq_index[4:8]) == 1))

            loader = create_dataloader(cfg, "train")
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["x"].shape[-2:]), (dataset.window_frames, 43))
            self.assertEqual(batch["chunk_id"].ndim, 1)
            self.assertIn("pattern_target", batch)
            self.assertIn("omega_target", batch)

    def test_test_only_dataset_uses_all_split_for_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_omega_tiny_dataset_cfg(tmpdir, duration_sec=4.0, include_unlabeled_recording=False)
            cfg["dataset"]["recordings"][0]["split_hint"] = "test"
            summary = build_omega_dataset(cfg)

            self.assertEqual(summary["normalization_split"], "all")

            train_dataset = OmegaWindowDataset(cfg["dataset"]["index_path"], split="train", max_cache_files=1)
            test_dataset = OmegaWindowDataset(cfg["dataset"]["index_path"], split="test", max_cache_files=1)
            self.assertEqual(len(train_dataset), 0)
            self.assertGreater(len(test_dataset), 0)
            sample = test_dataset[0]
            self.assertEqual(sample["recording_id"], "omega_rec_1")
            self.assertTrue(np.isfinite(float(sample["omega_target"].item())))


if __name__ == "__main__":
    unittest.main()
