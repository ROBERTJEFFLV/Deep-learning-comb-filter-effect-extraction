from __future__ import annotations

import time
import unittest
from queue import Empty, Queue

import numpy as np

from config import PLAYBACK_BLOCKSIZE, COMB_V2_N_FFT, COMB_V2_HOP
from processing.audio_processor import AudioProcessor


def _run_v2_audio_processor(audio: np.ndarray) -> list[dict]:
    """将音频分块送入新版 AudioProcessor，收集输出帧。"""
    process_queue: Queue = Queue()
    frame_queue: Queue = Queue()
    running_state = {"run": True}

    def running_flag() -> bool:
        return bool(running_state["run"])

    processor = AudioProcessor(process_queue, frame_queue, running_flag)
    processor.start()

    block_size = PLAYBACK_BLOCKSIZE
    total_blocks = int(np.ceil(len(audio) / float(block_size)))
    for i in range(total_blocks):
        start = i * block_size
        block = audio[start : start + block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))
        process_queue.put(block.astype(np.float32))

    expected_frames = max(1, (len(audio) - COMB_V2_N_FFT) // COMB_V2_HOP + 1)
    frames = []
    deadline = time.time() + 8.0
    while time.time() < deadline and len(frames) < expected_frames:
        try:
            frames.append(frame_queue.get(timeout=0.1))
        except Empty:
            if process_queue.empty():
                break
    running_state["run"] = False
    time.sleep(0.2)
    return frames


class TestV2AudioProcessor(unittest.TestCase):
    """验证新版 AudioProcessor 能正常输出 v2 特征帧。"""

    def test_v2_produces_feature_frames(self) -> None:
        rng = np.random.default_rng(42)
        sr = 48000
        duration = 2.0
        audio = rng.normal(0, 0.1, int(sr * duration)).astype(np.float32)

        frames = _run_v2_audio_processor(audio)
        self.assertGreater(len(frames), 0, "AudioProcessor 应产生至少一个输出帧")

        required_keys = {'t', 'smd', 'cpr', 'cpn', 'nda', 'cpq',
                         'obstacle_detected', 'distance_raw', 'distance_kf',
                         'velocity_kf', 'spectrum'}
        for f in frames:
            self.assertTrue(required_keys.issubset(f.keys()),
                            f"输出帧缺少字段: {required_keys - f.keys()}")
            self.assertIsInstance(f['smd'], float)
            self.assertIsInstance(f['obstacle_detected'], (bool, np.bool_))
            self.assertIsNotNone(f['spectrum'])

    def test_v2_detects_comb_in_synthetic(self) -> None:
        """合成梳状滤波器信号，验证 SMD 特征能检测到。"""
        sr = 48000
        duration = 2.0
        rng = np.random.default_rng(123)
        t = np.arange(int(sr * duration)) / sr
        noise = rng.normal(0, 0.3, len(t)).astype(np.float64)

        # 添加梳状滤波器: y(t) = x(t) + 0.8*x(t - tau)
        tau_samples = int(0.001 * sr)  # 1ms delay → d ≈ 17cm
        delayed = np.zeros_like(noise)
        delayed[tau_samples:] = noise[:-tau_samples]
        audio = (noise + 0.8 * delayed).astype(np.float32)

        frames = _run_v2_audio_processor(audio)
        self.assertGreater(len(frames), 5)

        # 后半段帧（EMA + diff 预热后）应有可检测的 SMD
        # diff 管线的 SMD 量级低于原始 log 幅度谱，0.02 是合理下限
        late_frames = frames[len(frames) // 2:]
        smd_vals = [f['smd'] for f in late_frames]
        mean_smd = np.mean(smd_vals)
        self.assertGreater(mean_smd, 0.02,
                           f"合成梳状信号 SMD 均值 {mean_smd:.3f} 应 > 0.02")


if __name__ == "__main__":
    unittest.main()
