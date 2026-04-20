"""Offline feature extractor for omega regression — v2 cepstral pipeline.

Replaces the legacy smooth_d1 extraction with log-magnitude-band features
aligned with the v2 cepstral analysis pipeline (processing/comb_feature_v2.py).

The primary ML channel is ``log_mag_band`` — the log-magnitude spectrum in the
800–8000 Hz analysis band — which directly encodes the comb filter's periodic
spectral modulation.  Temporal derivatives (dt1, dt_long, abs_dt1) are derived
from this base channel, giving the model 4 input channels [C, T, F].

Additionally, per-frame cepstral scalar features (SMD, CPR, CPN) are stored as
auxiliary data for potential multi-task learning or analysis.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

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
OMEGA_CACHE_SCHEMA_VERSION = 6
OMEGA_CACHE_SCHEMA_VERSION_COMPAT = (6,)


def _temporal_difference(arr: np.ndarray, lag: int = 1) -> np.ndarray:
    """Compute frame-wise temporal difference: out[t] = arr[t] - arr[t-lag].

    For t < lag, out[t] = 0 (no history available).
    """
    out = np.zeros_like(arr)
    if lag < arr.shape[0]:
        out[lag:] = arr[lag:] - arr[:-lag]
    return out.astype(np.float32)


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
    """Offline feature extractor using v2 cepstral log-magnitude pipeline.

    Produces:
    - log_mag_band [T, F]  — log magnitude spectrum in the analysis band
    - log_mag_band_dt1, log_mag_band_dt_long, log_mag_band_abs_dt1 (temporal diffs)
    - smd, cpr, cpn  — per-frame cepstral scalar features
    - energy_proxy, snr_proxy — auxiliary per-frame scalars
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        audio_cfg = cfg["audio"]
        front_cfg = cfg.get("front_end", {})
        observability_cfg = cfg.get("observability", {})

        self.sr = int(audio_cfg["target_sr"])
        self.n_fft = int(audio_cfg["n_fft"])
        self.hop_len = int(audio_cfg["hop_len"])
        self.freq_min = float(audio_cfg["freq_min"])
        self.freq_max = float(audio_cfg["freq_max"])
        self.hop_sec = float(self.hop_len) / float(self.sr)

        # Cepstral analysis parameters (from v2 pipeline)
        cep_cfg = cfg.get("cepstral", {})
        self.tau_min_s = float(cep_cfg.get("tau_min_s", 0.00025))
        self.tau_max_s = float(cep_cfg.get("tau_max_s", 0.004))
        self.cep_avg_frames = int(cep_cfg.get("cep_avg_frames", 4))

        # Temporal difference lags
        self.dt_short_lag = int(front_cfg.get("dt_short_lag", 1))
        self.dt_long_lag = int(front_cfg.get("dt_long_lag", 9))

        # Observability parameters
        self.observability_score_threshold = float(observability_cfg.get("score_threshold", 1.0))
        self.observability_soft_target_lower = float(observability_cfg.get("soft_target_lower", 0.8))
        self.observability_soft_target_upper = float(observability_cfg.get("soft_target_upper", 1.2))
        history_frames = int(front_cfg.get("history_frames", 18))
        obs_center_freq = float(observability_cfg.get("center_frequency_hz", 3000.0))
        obs_delta_t = float(history_frames) * self.hop_sec
        freq_bin_hz = float(self.sr) / float(self.n_fft)
        self.observability_velocity_coeff_mps_per_m = float(
            freq_bin_hz / (obs_center_freq * max(obs_delta_t, 1e-6))
        )

        # Frequency axis and band selection
        self.fft_freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        self.band_mask = (self.fft_freqs >= self.freq_min) & (self.fft_freqs <= self.freq_max)
        self.freq_idx = np.where(self.band_mask)[0]
        self.selected_freqs = self.fft_freqs[self.freq_idx].astype(np.float32)
        self.n_band = int(self.band_mask.sum())

        # Cepstral bin range for comb search
        df = float(self.sr) / float(self.n_fft)
        self.quef_tau_factor = 1.0 / (self.n_band * df)
        self.cep_min_bin = max(2, round(self.tau_min_s / self.quef_tau_factor))
        self.cep_max_bin = min(self.n_band // 2,
                               round(self.tau_max_s / self.quef_tau_factor) + 1)
        self.ref_min_bin = self.cep_max_bin + 5
        self.ref_max_bin = min(self.n_band // 2, self.ref_min_bin + 30)

        # Hann window
        self.window = np.hanning(self.n_fft).astype(np.float64)

    def _cepstral_scalars(self, cep_abs: np.ndarray) -> tuple:
        """Extract CPR and CPN from magnitude cepstrum."""
        qmin, qmax = self.cep_min_bin, self.cep_max_bin
        if qmin >= qmax or qmax > len(cep_abs) // 2:
            return 0.0, 0.0
        search = cep_abs[qmin:qmax]
        if len(search) == 0:
            return 0.0, 0.0
        peak_val = float(np.max(search))
        median_val = float(np.median(search))
        cpr = peak_val / (median_val + 1e-12)
        rmin, rmax = self.ref_min_bin, self.ref_max_bin
        if rmin < rmax and rmax <= len(cep_abs) // 2:
            ref_baseline = float(np.mean(cep_abs[rmin:rmax]))
        else:
            ref_baseline = median_val
        cpn = peak_val / (ref_baseline + 1e-12)
        return cpr, cpn

    def _frame_labels(self, frame_time_sec: np.ndarray,
                      label_dict: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
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
        has_distance_valid = bool(
            label_dict.get("has_distance_valid", np.asarray([0.0], dtype=np.float32))[0] > 0.5
        )
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

        # ── Empirical observability gate ──────────────────────────────
        _HARD_CUTOFF_CM = 25.0
        _OBS_SLOPE = 3.30
        _OBS_INTERCEPT = -0.50
        d_m = distance_cm / 100.0
        has_d = np.isfinite(distance_cm)
        has_v = np.isfinite(v_perp_mps)
        not_observable = has_d & (
            (distance_cm >= _HARD_CUTOFF_CM)
            | (has_v & (v_perp_mps < _OBS_SLOPE * d_m + _OBS_INTERCEPT))
        )
        pattern_target[not_observable] = 0.0
        pattern_binary_target[not_observable] = 0.0
        omega_target[not_observable] = np.nan
        # ──────────────────────────────────────────────────────────────

        return {
            "frame_distance_cm": distance_cm,
            "frame_omega_target": omega_target,
            "frame_v_perp_mps": v_perp_mps,
            "frame_observability_score_res": observability_score_res,
            "frame_pattern_target": pattern_target,
            "frame_pattern_binary_target": pattern_binary_target,
        }

    def process_audio(self, audio: np.ndarray,
                      optional_labels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Extract v2 cepstral features from an audio array.

        Returns a dict suitable for saving as .npz cache:
        - log_mag_band [T, F] — log magnitude spectrum in the analysis band
        - log_mag_band_dt1 [T, F] — short-term temporal difference
        - log_mag_band_dt_long [T, F] — long-term temporal difference
        - log_mag_band_abs_dt1 [T, F] — absolute short-term diff
        - smd [T] — spectral modulation depth (per-frame scalar)
        - cpr [T] — cepstral peak ratio
        - cpn [T] — cepstral peak normalized
        - energy_proxy [T], snr_proxy [T] — auxiliary scalars
        - frame labels (distance, omega, pattern, etc.)
        """
        audio = np.asarray(audio, dtype=np.float64).reshape(-1)
        if audio.size < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - audio.size))

        n_samples = int(audio.shape[0])
        n_frames = max(0, (n_samples - self.n_fft) // self.hop_len + 1)

        log_mag_band_seq = np.zeros((n_frames, self.n_band), dtype=np.float32)
        smd_seq = np.zeros(n_frames, dtype=np.float32)
        cpr_seq = np.zeros(n_frames, dtype=np.float32)
        cpn_seq = np.zeros(n_frames, dtype=np.float32)
        energy_proxy_seq = np.zeros(n_frames, dtype=np.float32)
        snr_proxy_seq = np.zeros(n_frames, dtype=np.float32)
        frame_time_sec = np.zeros(n_frames, dtype=np.float32)

        # Cepstrum averaging buffer
        cep_buffer = []

        for i in range(n_frames):
            start = i * self.hop_len
            frame = audio[start:start + self.n_fft]
            windowed = frame * self.window
            fft_vals = np.fft.rfft(windowed, n=self.n_fft)
            mag_full = np.abs(fft_vals)
            mag_band = mag_full[self.freq_idx].astype(np.float32)

            # Log magnitude (primary ML channel)
            log_band = np.log(mag_band + 1e-12).astype(np.float32)
            log_mag_band_seq[i] = log_band

            # SMD: std of mean-removed log spectrum
            centered = log_band - log_band.mean()
            smd_seq[i] = float(np.std(centered))

            # Cepstrum for CPR/CPN (with averaging)
            cep_frame = np.abs(np.fft.fft(centered.astype(np.float64)))
            cep_buffer.append(cep_frame)
            if len(cep_buffer) > self.cep_avg_frames:
                cep_buffer.pop(0)
            avg_cep = np.mean(cep_buffer, axis=0)
            cpr_val, cpn_val = self._cepstral_scalars(avg_cep)
            cpr_seq[i] = cpr_val
            cpn_seq[i] = cpn_val

            # Energy and SNR proxies
            rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
            energy_proxy_seq[i] = rms
            snr_proxy_seq[i] = float(
                20.0 * np.log10((np.max(mag_band) + 1e-8) / (np.mean(mag_band) + 1e-8))
            )
            frame_time_sec[i] = float(i) * self.hop_sec

        # Derive multi-channel temporal differences from log_mag_band
        log_mag_band_dt1 = _temporal_difference(log_mag_band_seq, lag=self.dt_short_lag)
        log_mag_band_dt_long = _temporal_difference(log_mag_band_seq, lag=self.dt_long_lag)
        log_mag_band_abs_dt1 = np.abs(log_mag_band_dt1)

        labels = self._frame_labels(frame_time_sec, optional_labels)
        return {
            "schema_version": np.asarray([OMEGA_CACHE_SCHEMA_VERSION], dtype=np.int64),
            "frame_time_sec": frame_time_sec,
            # Primary ML channels [T, F]
            "log_mag_band": log_mag_band_seq,
            "log_mag_band_dt1": log_mag_band_dt1,
            "log_mag_band_dt_long": log_mag_band_dt_long,
            "log_mag_band_abs_dt1": log_mag_band_abs_dt1,
            # Per-frame cepstral scalars [T]
            "smd": smd_seq,
            "cpr": cpr_seq,
            "cpn": cpn_seq,
            # Auxiliary scalars
            "energy_proxy": energy_proxy_seq,
            "snr_proxy": snr_proxy_seq,
            "frequencies_hz": self.selected_freqs.astype(np.float32),
            **labels,
        }


def process_audio_array(audio: np.ndarray, sr: int, cfg: Dict[str, Any],
                        optional_labels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    if int(sr) != int(cfg["audio"]["target_sr"]):
        raise ValueError(f"expected sr={cfg['audio']['target_sr']}, got {sr}")
    extractor = OfflineOmegaFeatureExtractor(cfg)
    return extractor.process_audio(audio, optional_labels=optional_labels)
