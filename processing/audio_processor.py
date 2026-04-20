# processing/audio_processor.py
# -*- coding: utf-8 -*-
"""
实时音频处理器 — 基于倒谱分析的梳状滤波器特征提取 (v2)

从 process_queue 读取音频 chunk，累积到 N_FFT 帧长度后：
1. 使用 CombFeatureExtractor 提取 SMD / CPR / CPN / NDA / CPQ 特征
2. 使用 RangeKF + FixedLagRTS 平滑距离估计
3. 将特征帧推入 frame_queue 供可视化使用
"""

from queue import Queue, Empty
import numpy as np
import time
import threading

from config import (
    SR,
    COMB_V2_N_FFT, COMB_V2_HOP,
    COMB_V2_FREQ_MIN, COMB_V2_FREQ_MAX,
    COMB_V2_TAU_MIN, COMB_V2_TAU_MAX,
    COMB_V2_EMA_ALPHA, COMB_V2_CEP_AVG,
    COMB_V2_SMD_THRESHOLD,
    KF_PARAMS, RTS_PARAMS,
)
from processing.comb_feature_v2 import CombFeatureConfig, CombFeatureExtractor
from processing.range_kf import RangeKF, FixedLagRTS

C_SPEED = 343.0


class AudioProcessor:
    def __init__(
        self,
        process_queue: Queue,
        frame_queue: Queue,
        running_flag: callable,
        kf_params: dict = None,
        rts_params: dict = None,
    ):
        self.pq = process_queue
        self.out_q = frame_queue
        self.running = running_flag

        # 特征提取器
        self.cfg = CombFeatureConfig(
            sr=SR,
            n_fft=COMB_V2_N_FFT,
            hop_length=COMB_V2_HOP,
            freq_min=COMB_V2_FREQ_MIN,
            freq_max=COMB_V2_FREQ_MAX,
            tau_min_s=COMB_V2_TAU_MIN,
            tau_max_s=COMB_V2_TAU_MAX,
            ema_alpha=COMB_V2_EMA_ALPHA,
            cep_avg_frames=COMB_V2_CEP_AVG,
        )
        self.extractor = CombFeatureExtractor(self.cfg)
        self.smd_threshold = COMB_V2_SMD_THRESHOLD

        # 音频累积缓冲区
        self.n_fft = COMB_V2_N_FFT
        self.hop = COMB_V2_HOP
        self._audio_buf = np.zeros(0, dtype=np.float32)

        # Kalman 滤波器（平滑距离估计）
        kf_p = dict(KF_PARAMS) if kf_params is None else dict(kf_params)
        rts_p = dict(RTS_PARAMS) if rts_params is None else dict(rts_params)

        kf_R = kf_p.get('R', 4)
        kf_sigma_a = kf_p.get('sigma_a', 8)
        kf_vmax = kf_p.get('v_max', 8)
        kf_amax = kf_p.get('a_max', 25)

        self.kf = RangeKF(
            R=kf_R, sigma_a=kf_sigma_a,
            v_max=kf_vmax, a_max=kf_amax, d_min=0.0,
        )
        self.dt_hop = self.hop / SR
        rts_lag = rts_p.get('lag', 3)
        self.rts = FixedLagRTS(
            lag=rts_lag, v_max=kf_vmax, a_max=kf_amax,
            dt_window=self.dt_hop, d_min=0.0,
        )

        # 状态跟踪
        self._frame_count = 0
        self._last_distance_kf = None
        self._last_velocity_kf = None

        # 频谱数据（供可视化使用）
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / SR)
        band_mask = (self.freqs >= COMB_V2_FREQ_MIN) & (self.freqs <= COMB_V2_FREQ_MAX)
        self.band_freqs = self.freqs[band_mask]
        self._last_spectrum = np.zeros(len(self.band_freqs))

    def _process_frames(self):
        """从音频缓冲区中提取所有可用帧并处理。"""
        while len(self._audio_buf) >= self.n_fft:
            frame = self._audio_buf[:self.n_fft].astype(np.float64)
            self._audio_buf = self._audio_buf[self.hop:]

            # 提取特征
            feat = self.extractor.process_frame(frame)
            self._frame_count += 1
            t_sec = self._frame_count * self.dt_hop

            # 频谱（供可视化）
            windowed = frame * np.hanning(self.n_fft)
            mag = np.abs(np.fft.rfft(windowed))
            band_mask = (self.freqs >= COMB_V2_FREQ_MIN) & (self.freqs <= COMB_V2_FREQ_MAX)
            self._last_spectrum = mag[band_mask]

            # 检测判定
            obstacle_detected = feat.smd > self.smd_threshold
            dist_raw = feat.dist_est_cm if obstacle_detected else None

            # Kalman 平滑
            dist_kf = None
            vel_kf = None
            if obstacle_detected and dist_raw is not None and dist_raw > 0:
                self.kf.predict(self.dt_hop)
                self.kf.update(dist_raw)
                if self.kf.x is not None:
                    dist_kf = float(self.kf.x[0])
                    vel_kf = float(self.kf.x[1])
                    self._last_distance_kf = dist_kf
                    self._last_velocity_kf = vel_kf

            # 输出帧
            out_frame = {
                't': t_sec,
                'smd': feat.smd,
                'cpr': feat.cpr,
                'cpn': feat.cpn,
                'nda': feat.nda,
                'cpq': feat.cpq,
                'obstacle_detected': obstacle_detected,
                'distance_raw': dist_raw,
                'distance_kf': dist_kf,
                'velocity_kf': vel_kf,
                'spectrum': self._last_spectrum,
            }

            try:
                self.out_q.put_nowait(out_frame)
            except Exception:
                pass

    def _run(self):
        """处理线程主循环。"""
        while self.running():
            try:
                chunk = self.pq.get(timeout=0.05)
            except Empty:
                continue

            self._audio_buf = np.concatenate([self._audio_buf, chunk])
            self._process_frames()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
