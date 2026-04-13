"""Offline extractor for omega-only LS replacement training."""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ml_uav_comb.features.feature_utils import (
    interpolate_distance,
    interpolate_distance_valid,
    interpolate_observability_score_res,
    interpolate_pattern_label_res,
    interpolate_v_perp_mps,
    resolution_pattern_binary_target,
    resolution_pattern_soft_target,
    resolution_observability_score,
)

C_SPEED = 343.0
OMEGA_CACHE_SCHEMA_VERSION = 4


def distance_cm_to_omega(value: Any) -> Any:
    factor = 4.0 * math.pi / (C_SPEED * 100.0)
    if hasattr(value, "detach"):
        return value * factor
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32) * np.float32(factor)
    return float(value) * factor


def omega_to_distance_cm(value: Any) -> Any:
    factor = (C_SPEED * 100.0) / (4.0 * math.pi)
    if hasattr(value, "detach"):
        return value * factor
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32) * np.float32(factor)
    return float(value) * factor


class OfflineOmegaFeatureExtractor:
    """Freeze legacy pre-LS processing up to smooth_d1 only."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        audio_cfg = cfg["audio"]
        front_cfg = cfg["front_end"]
        observability_cfg = cfg.get("observability", {})
        self.sr = int(audio_cfg["target_sr"])
        self.n_fft = int(audio_cfg["n_fft"])
        self.hop_len = int(audio_cfg["hop_len"])
        self.freq_min = float(audio_cfg["freq_min"])
        self.freq_max = float(audio_cfg["freq_max"])
        self.hop_sec = float(self.hop_len) / float(self.sr)
        self.history_frames = int(front_cfg["history_frames"])
        self.smooth_sigma_1 = float(front_cfg["smooth_sigma_1"])
        self.smooth_sigma_2 = float(front_cfg["smooth_sigma_2"])
        self.noise_gate_smooth = float(front_cfg.get("noise_gate_smooth", 0.9))
        self.observability_center_freq_hz = float(observability_cfg.get("center_frequency_hz", 3000.0))
        self.observability_score_threshold = float(observability_cfg.get("score_threshold", 1.0))
        self.observability_soft_target_lower = float(observability_cfg.get("soft_target_lower", 0.8))
        self.observability_soft_target_upper = float(observability_cfg.get("soft_target_upper", 1.2))
        self.observability_delta_t_sec = float(self.history_frames) * self.hop_sec
        self.freq_bin_hz = float(self.sr) / float(self.n_fft)
        self.observability_velocity_coeff_mps_per_m = float(self.freq_bin_hz / (self.observability_center_freq_hz * self.observability_delta_t_sec))

        self.fft_freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        self.freq_idx = np.where((self.fft_freqs >= self.freq_min) & (self.fft_freqs <= self.freq_max))[0]
        self.selected_freqs = self.fft_freqs[self.freq_idx].astype(np.float32)

        self.buf_short = np.zeros(self.n_fft, dtype=np.float32)
        self.s_hist: deque[np.ndarray] = deque(maxlen=150)
        self.h_amp: deque[np.ndarray] = deque(maxlen=self.history_frames)
        self.d1_hist: deque[np.ndarray] = deque(maxlen=150)
        self._smoothed_rms = 0.0

    def _frame_labels(self, frame_time_sec: np.ndarray, label_dict: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        n = int(frame_time_sec.shape[0])
        distance_cm = np.full((n,), np.nan, dtype=np.float32)
        omega_target = np.full((n,), np.nan, dtype=np.float32)
        v_perp_mps = np.full((n,), np.nan, dtype=np.float32)
        observability_score_res = np.full((n,), np.nan, dtype=np.float32)
        pattern_target = np.full((n,), np.nan, dtype=np.float32)
        pattern_binary_target = np.full((n,), np.nan, dtype=np.float32)
        if label_dict is None:
            return {
                "frame_distance_cm": distance_cm,
                "frame_omega_target": omega_target,
                "frame_v_perp_mps": v_perp_mps,
                "frame_observability_score_res": observability_score_res,
                "frame_pattern_target": pattern_target,
                "frame_pattern_binary_target": pattern_binary_target,
            }
        has_distance_valid = bool(label_dict.get("has_distance_valid", np.asarray([0.0], dtype=np.float32))[0] > 0.5)
        for idx, t in enumerate(frame_time_sec.tolist()):
            d = interpolate_distance(float(t), label_dict)
            if d is None or not np.isfinite(d):
                continue
            distance_cm[idx] = float(d)
            distance_is_valid = True
            if has_distance_valid:
                dv = interpolate_distance_valid(float(t), label_dict)
                distance_is_valid = bool(dv is not None and float(dv) >= 0.5)

            vel = interpolate_v_perp_mps(float(t), label_dict)
            if vel is not None and np.isfinite(vel):
                v_perp_mps[idx] = float(vel)

            score = interpolate_observability_score_res(float(t), label_dict)
            if (score is None or not np.isfinite(score)) and np.isfinite(v_perp_mps[idx]):
                score = float(
                    resolution_observability_score(
                        distance_m=np.asarray([float(d) / 100.0], dtype=np.float32),
                        velocity_mps=np.asarray([float(v_perp_mps[idx])], dtype=np.float32),
                        coeff_mps_per_m=self.observability_velocity_coeff_mps_per_m,
                    )[0]
                )
            if score is not None and np.isfinite(score):
                observability_score_res[idx] = float(score)
                pattern_target[idx] = float(
                    resolution_pattern_soft_target(
                        np.asarray([float(score)], dtype=np.float32),
                        lower=self.observability_soft_target_lower,
                        upper=self.observability_soft_target_upper,
                    )[0]
                )
                pattern_binary_target[idx] = float(
                    resolution_pattern_binary_target(
                        np.asarray([float(score)], dtype=np.float32),
                        threshold=self.observability_score_threshold,
                    )[0]
                )
            else:
                pattern = interpolate_pattern_label_res(float(t), label_dict)
                if pattern is not None and np.isfinite(pattern):
                    pattern_binary_target[idx] = float(pattern >= 0.5)
                    pattern_target[idx] = float(np.clip(pattern, 0.0, 1.0))
            if distance_is_valid:
                omega_target[idx] = float(distance_cm_to_omega(distance_cm[idx]))
        valid = np.isfinite(distance_cm) & np.isfinite(omega_target)
        if np.any(valid):
            omega_target[valid] = distance_cm_to_omega(distance_cm[valid]).astype(np.float32)
        return {
            "frame_distance_cm": distance_cm,
            "frame_omega_target": omega_target,
            "frame_v_perp_mps": v_perp_mps,
            "frame_observability_score_res": observability_score_res,
            "frame_pattern_target": pattern_target,
            "frame_pattern_binary_target": pattern_binary_target,
        }

    def process_audio(self, audio: np.ndarray, optional_labels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - audio.size))

        frame_time_sec = []
        smooth_d1_seq = []
        sum_abs_d1_seq = []
        energy_proxy_seq = []
        snr_proxy_seq = []

        frame_idx = 0
        max_start = int(audio.shape[0])
        for start in range(self.n_fft, max_start + self.hop_len, self.hop_len):
            block = audio[start : start + self.hop_len]
            if block.size == 0:
                break
            if block.size < self.hop_len:
                block = np.pad(block, (0, self.hop_len - block.size))

            self.buf_short[:-self.hop_len] = self.buf_short[self.hop_len :]
            self.buf_short[-self.hop_len :] = block

            rms = float(np.sqrt(np.mean(block**2) + 1e-12))
            self._smoothed_rms = self.noise_gate_smooth * self._smoothed_rms + (1.0 - self.noise_gate_smooth) * rms
            energy_proxy_seq.append(float(self._smoothed_rms))
            frame_time_sec.append(float(frame_idx) * self.hop_sec)

            win = self.buf_short * np.hanning(self.n_fft)
            fft_vals = np.fft.rfft(win, n=self.n_fft)
            mag = np.abs(fft_vals[self.freq_idx]).astype(np.float32)
            snr_proxy_seq.append(float(20.0 * np.log10((np.max(mag) + 1e-8) / (np.mean(mag) + 1e-8))))

            norm = mag / (float(np.max(mag)) + 1e-12)
            smooth = gaussian_filter1d(norm, sigma=self.smooth_sigma_1)
            centered = smooth - float(np.mean(smooth))
            denom = float(np.max(centered) - np.min(centered))
            if denom > 1e-6:
                current = ((centered - float(np.min(centered))) / denom).astype(np.float32)
            else:
                current = centered.astype(np.float32)

            self.s_hist.append(current)
            if len(self.s_hist) > 1:
                current = np.mean(np.stack(self.s_hist, axis=0), axis=0).astype(np.float32)

            old = self.h_amp[0] if len(self.h_amp) == self.h_amp.maxlen else None
            self.h_amp.append(current.copy())
            d1 = current - old if old is not None else np.zeros_like(current)
            d1_freq_sm = gaussian_filter1d(d1, sigma=self.smooth_sigma_1)
            self.d1_hist.append(d1_freq_sm.astype(np.float32))
            if len(self.d1_hist) > 1:
                smooth_d1 = gaussian_filter1d(np.stack(self.d1_hist, axis=0), sigma=self.smooth_sigma_2, axis=0)[-1].astype(np.float32)
            else:
                smooth_d1 = d1.astype(np.float32)

            smooth_d1_seq.append(smooth_d1)
            sum_abs_d1_seq.append(float(np.sum(np.abs(smooth_d1))))
            frame_idx += 1

        frame_time_sec_np = np.asarray(frame_time_sec, dtype=np.float32)
        labels = self._frame_labels(frame_time_sec_np, optional_labels)
        return {
            "schema_version": np.asarray([OMEGA_CACHE_SCHEMA_VERSION], dtype=np.int64),
            "frame_time_sec": frame_time_sec_np,
            "smooth_d1": np.asarray(smooth_d1_seq, dtype=np.float32),
            "sum_abs_d1": np.asarray(sum_abs_d1_seq, dtype=np.float32),
            "energy_proxy": np.asarray(energy_proxy_seq, dtype=np.float32),
            "snr_proxy": np.asarray(snr_proxy_seq, dtype=np.float32),
            "frequencies_hz": self.selected_freqs.astype(np.float32),
            **labels,
        }


def process_audio_array(audio: np.ndarray, sr: int, cfg: Dict[str, Any], optional_labels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    if int(sr) != int(cfg["audio"]["target_sr"]):
        raise ValueError(f"expected sr={cfg['audio']['target_sr']}, got {sr}")
    extractor = OfflineOmegaFeatureExtractor(cfg)
    return extractor.process_audio(audio, optional_labels=optional_labels)
