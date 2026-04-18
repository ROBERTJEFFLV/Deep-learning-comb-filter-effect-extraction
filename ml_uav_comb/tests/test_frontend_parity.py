from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from config import HOP_LEN
from ml_uav_comb.data_pipeline.offline_feature_extractor import process_audio_array
from ml_uav_comb.features.feature_utils import load_audio_mono, load_yaml_config
from ml_uav_comb.tests.support import write_test_wav
from processing.audio_processor import AudioProcessor


def _run_legacy_audio_processor(audio: np.ndarray) -> list[dict]:
    process_queue: Queue = Queue()
    frame_queue: Queue = Queue()
    running_state = {"run": True}

    def running_flag() -> bool:
        return bool(running_state["run"])

    processor = AudioProcessor(process_queue, frame_queue, running_flag)
    processor.start()

    total_hops = int(np.ceil(len(audio) / float(HOP_LEN)))
    for hop_idx in range(total_hops):
        start = hop_idx * HOP_LEN
        block = audio[start : start + HOP_LEN]
        if len(block) < HOP_LEN:
            block = np.pad(block, (0, HOP_LEN - len(block)))
        process_queue.put(block.astype(np.float32))

    frames = []
    deadline = time.time() + 8.0
    while time.time() < deadline and len(frames) < total_hops:
        try:
            frames.append(frame_queue.get(timeout=0.1))
        except Empty:
            if process_queue.empty():
                break
    running_state["run"] = False
    time.sleep(0.2)
    return frames


class TestFrontendParity(unittest.TestCase):
    def test_v2_frontend_matches_legacy_core_outputs(self) -> None:
        cfg = load_yaml_config("ml_uav_comb/configs/tiny_debug.yaml")
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "rec_1.wav"
            write_test_wav(wav_path, duration_sec=4.0)
            audio, sr = load_audio_mono(
                wav_path,
                target_sr=int(cfg["audio"]["target_sr"]),
                max_duration_sec=4.0,
            )
            features = process_audio_array(audio, sr, cfg)
            legacy_frames = _run_legacy_audio_processor(audio)

            warmup_frames = int(cfg["audio"]["n_fft"] // cfg["audio"]["hop_len"])
            legacy_frames = legacy_frames[warmup_frames : warmup_frames + int(features["phase_stft"].shape[0])]
            self.assertGreater(len(legacy_frames), 0)

            smooth_d1_new = features["diff_comb"][: len(legacy_frames), :, 0]
            smooth_d1_old = np.stack([frame["diff_amplitude"] for frame in legacy_frames], axis=0)
            sum_abs_new = features["scalar_seq"][: len(legacy_frames), 0]
            sum_abs_old = np.asarray([frame["sum_abs_d1"] for frame in legacy_frames], dtype=np.float32)

            smooth_d1_mae = float(np.mean(np.abs(smooth_d1_new - smooth_d1_old)))
            sum_abs_mae = float(np.mean(np.abs(sum_abs_new - sum_abs_old)))
            self.assertLess(smooth_d1_mae, 0.15)
            self.assertLess(sum_abs_mae, 1.0)

            teacher_field_names = [str(v) for v in features["teacher_field_names"].tolist()]
            raw_idx = teacher_field_names.index("heuristic_distance_raw_cm")
            raw_avail_idx = teacher_field_names.index("heuristic_distance_raw_available")
            kf_idx = teacher_field_names.index("heuristic_distance_kf_cm")
            kf_avail_idx = teacher_field_names.index("heuristic_distance_kf_available")

            raw_pairs = []
            kf_pairs = []
            for new_row, old_frame in zip(features["teacher_seq"][: len(legacy_frames)], legacy_frames):
                if new_row[raw_avail_idx] > 0.5 and old_frame.get("distance") is not None:
                    raw_pairs.append((float(new_row[raw_idx]), float(old_frame["distance"])))
                if new_row[kf_avail_idx] > 0.5 and old_frame.get("distance_kf") is not None:
                    kf_pairs.append((float(new_row[kf_idx]), float(old_frame["distance_kf"])))

            if not raw_pairs or not kf_pairs:
                self.skipTest("synthetic frontend fixture did not produce heuristic distance pairs")
            raw_mae = float(np.mean([abs(a - b) for a, b in raw_pairs]))
            kf_mae = float(np.mean([abs(a - b) for a, b in kf_pairs]))
            self.assertLess(raw_mae, 8.0)
            self.assertLess(kf_mae, 8.0)


if __name__ == "__main__":
    unittest.main()
