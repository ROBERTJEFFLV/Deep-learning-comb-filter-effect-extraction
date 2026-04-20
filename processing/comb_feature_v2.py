# -*- coding: utf-8 -*-
"""
Comb Filter Feature Extraction Pipeline v2

Extracts per-frame features that discriminate comb-filter-present vs absent.

Core idea:  Y(f) = X(f) · H(f),  where H(f) = 1 + A_rel·exp(-j2πfτ).
In log domain the additive comb modulation creates periodic ripple.
We detect this periodicity via cepstral analysis on the log magnitude spectrum.

Theory:
  log|Y(f)| = log|X(f)| + log|H(f)|
  The comb creates cos(2πfτ) modulation in |H(f)|², so log|H(f)| has periodic
  structure with spectral period Δf = 1/τ.  The FFT of log|Y(f)| (= cepstrum)
  shows a peak at quefrency bin  n = N_band · df · τ  where df = SR/N_FFT.

  For d = 5–55 cm:   quefrency bins 2–23
  For rotor harmonics f0=200 Hz: quefrency bin 36  (well separated!)

Features extracted per frame:
  1. SMD  – Spectral Modulation Depth (std of mean-removed log spectrum)
  2. CPR  – Cepstral Peak Ratio (peak / median in comb quefrency range)
  3. CPN  – Cepstral Peak Normalized (peak / baseline from non-comb region)
  4. NDA  – Normalized Differential Activity (improved frame-diff S(t))
  5. CPQ  – Cepstral Peak Quefrency (peak location → τ → distance)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

C_SPEED = 343.0  # m/s


@dataclass
class CombFeatureConfig:
    """All tunable parameters in one place."""
    sr: int = 48000
    n_fft: int = 2048
    hop_length: int = 512
    # Analysis band (Hz)
    freq_min: float = 800.0
    freq_max: float = 8000.0
    # Quefrency search range (seconds) — maps to distance 4cm–69cm
    tau_min_s: float = 0.00025
    tau_max_s: float = 0.004
    # Preprocessing: EMA → diff → spectral smooth
    ema_alpha: float = 0.1        # 时域 EMA 平滑系数 (压快变噪声)
    diff_dt: int = 15             # 差分帧间隔 (~0.32s, 提取慢变梳状滤波)
    spectral_sigma: float = 5.0   # 差分后频谱高斯平滑 σ
    # NDA: frame lag for differential (number of hops)
    diff_lag: int = 1
    # Number of frames to average cepstrum over (sliding window)
    cep_avg_frames: int = 4


@dataclass
class FrameFeatures:
    """Per-frame feature vector."""
    smd: float = 0.0     # Spectral Modulation Depth
    cpr: float = 0.0     # Cepstral Peak Ratio (peak / median in search range)
    cpn: float = 0.0     # Cepstral Peak Normalized (peak / non-comb baseline)
    nda: float = 0.0     # Normalized Differential Activity
    cpq: float = 0.0     # Cepstral Peak Quefrency (seconds)
    dist_est_cm: float = 0.0  # Distance estimate from CPQ


class CombFeatureExtractor:
    """Stateful per-frame comb filter feature extractor.

    Preprocessing pipeline (applied in process_file):
      log|Y(f)| → EMA(α) → diff(dt) → spectral_smooth(σ) → features
    """

    def __init__(self, cfg: CombFeatureConfig | None = None):
        self.cfg = cfg or CombFeatureConfig()
        c = self.cfg

        # Frequency axis
        self.freqs = np.fft.rfftfreq(c.n_fft, 1.0 / c.sr)
        self.band_mask = (self.freqs >= c.freq_min) & (self.freqs <= c.freq_max)
        self.band_freqs = self.freqs[self.band_mask]
        self.n_band = int(self.band_mask.sum())

        # Quefrency mapping
        self.df = c.sr / c.n_fft
        self.quef_tau_factor = 1.0 / (self.n_band * self.df)

        # Comb quefrency search range (in cepstral bins)
        self.cep_min_bin = max(2, round(c.tau_min_s / self.quef_tau_factor))
        self.cep_max_bin = min(self.n_band // 2,
                               round(c.tau_max_s / self.quef_tau_factor) + 1)

        # Non-comb reference range
        self.ref_min_bin = self.cep_max_bin + 5
        self.ref_max_bin = min(self.n_band // 2, self.ref_min_bin + 30)

        # Hann window for STFT
        self.window = np.hanning(c.n_fft).astype(np.float64)

        # Stateful buffers for streaming process_frame()
        self._ema_state = None           # EMA accumulator [F]
        self._log_band_history = []      # ring buffer for diff
        self._cep_buffer = []            # cepstrum averaging
        self._frame_idx = 0

    def reset(self):
        """Reset internal state (call between independent audio files)."""
        self._ema_state = None
        self._log_band_history = []
        self._cep_buffer = []
        self._frame_idx = 0

    def _stft_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrum of a single frame."""
        windowed = frame * self.window
        return np.abs(np.fft.rfft(windowed))

    def process_frame(self, frame: np.ndarray) -> FrameFeatures:
        """Streaming per-frame feature extraction (for real-time pipeline).

        Implements the same EMA → diff → spectral_smooth pipeline as process_file
        but in a stateful frame-by-frame manner.
        """
        mag = self._stft_frame(frame)
        mag_band = mag[self.band_mask]
        log_band = np.log(mag_band + 1e-12)

        # Step 1: EMA temporal smoothing
        alpha = self.cfg.ema_alpha
        if alpha > 0:
            if self._ema_state is None:
                self._ema_state = log_band.copy()
            else:
                self._ema_state = (1 - alpha) * self._ema_state + alpha * log_band
            ema_out = self._ema_state.copy()
        else:
            ema_out = log_band

        # Step 2: Temporal differencing
        self._log_band_history.append(ema_out)
        dt = self.cfg.diff_dt
        if dt > 0 and len(self._log_band_history) > dt:
            diff_out = ema_out - self._log_band_history[-dt - 1]
            # Keep history bounded
            if len(self._log_band_history) > dt + 5:
                self._log_band_history = self._log_band_history[-(dt + 2):]
        else:
            diff_out = np.zeros_like(ema_out)

        # Step 3: Spectral smoothing
        sigma = self.cfg.spectral_sigma
        if sigma > 0:
            working = gaussian_filter1d(diff_out, sigma=sigma)
        else:
            working = diff_out

        # Features from working signal
        centered = working - working.mean()
        smd = float(np.std(centered))

        cep_frame = np.abs(np.fft.fft(centered))
        self._cep_buffer.append(cep_frame)
        if len(self._cep_buffer) > self.cfg.cep_avg_frames:
            self._cep_buffer.pop(0)
        avg_cep = np.mean(self._cep_buffer, axis=0)
        cpr, cpn, cpq_sec, dist_cm = self._compute_cep_features(avg_cep)

        nda = 0.0  # NDA is less meaningful in preprocessed domain
        self._frame_idx += 1

        return FrameFeatures(
            smd=smd, cpr=cpr, cpn=cpn,
            nda=nda, cpq=cpq_sec, dist_est_cm=dist_cm,
        )

    def _compute_smd(self, log_band: np.ndarray) -> float:
        """Spectral Modulation Depth = std of mean-removed log spectrum."""
        centered = log_band - log_band.mean()
        return float(np.std(centered))

    def _compute_cepstrum(self, log_band: np.ndarray) -> np.ndarray:
        """Compute magnitude cepstrum via FFT of mean-removed log spectrum."""
        centered = log_band - log_band.mean()
        return np.abs(np.fft.fft(centered))

    def _compute_cep_features(self, cep_abs: np.ndarray
                               ) -> Tuple[float, float, float, float]:
        """Extract CPR, CPN, quefrency, distance from a (possibly averaged) cepstrum."""
        qmin, qmax = self.cep_min_bin, self.cep_max_bin
        if qmin >= qmax or qmax > len(cep_abs) // 2:
            return 0.0, 0.0, 0.0, 0.0

        search = cep_abs[qmin:qmax]
        if len(search) == 0:
            return 0.0, 0.0, 0.0, 0.0

        peak_idx_local = int(np.argmax(search))
        peak_val = float(search[peak_idx_local])

        median_val = float(np.median(search))
        cpr = peak_val / (median_val + 1e-12)

        rmin, rmax = self.ref_min_bin, self.ref_max_bin
        if rmin < rmax and rmax <= len(cep_abs) // 2:
            ref_baseline = float(np.mean(cep_abs[rmin:rmax]))
        else:
            ref_baseline = median_val
        cpn = peak_val / (ref_baseline + 1e-12)

        peak_bin = qmin + peak_idx_local
        cpq_sec = peak_bin * self.quef_tau_factor
        dist_cm = C_SPEED * cpq_sec / 2.0 * 100.0

        return cpr, cpn, cpq_sec, dist_cm

    def process_file(self, audio: np.ndarray, sr: int | None = None
                     ) -> Tuple[np.ndarray, list]:
        """Process an entire audio array with EMA → diff → spectral_smooth pipeline.

        Returns
        -------
        feature_matrix : (N_frames, 6) float64
            Columns: [smd, cpr, cpn, nda, cpq, dist_est_cm]
        frame_times : list of float (seconds)
        """
        if sr is not None and sr != self.cfg.sr:
            raise ValueError(f"Sample rate mismatch: got {sr}, expected {self.cfg.sr}")

        n_fft = self.cfg.n_fft
        hop = self.cfg.hop_length
        n_samples = len(audio)
        n_frames = max(0, (n_samples - n_fft) // hop + 1)

        if n_frames == 0:
            return np.zeros((0, 6), dtype=np.float64), []

        # Step 1: Vectorized STFT → log magnitude in analysis band
        idx = (np.arange(n_frames) * hop)[:, None] + np.arange(n_fft)[None, :]
        frames = audio[idx] * self.window[None, :]
        mags = np.abs(np.fft.rfft(frames, axis=1))
        log_band = np.log(mags[:, self.band_mask] + 1e-12)  # [T, F]

        # Step 2: EMA temporal smoothing (suppress fast-changing noise)
        alpha = self.cfg.ema_alpha
        if alpha > 0:
            log_band = lfilter([alpha], [1, -(1 - alpha)], log_band, axis=0)

        # Step 3: Temporal differencing (extract slow-varying comb filter changes)
        dt = self.cfg.diff_dt
        if dt > 0:
            diff_band = np.zeros_like(log_band)
            diff_band[dt:] = log_band[dt:] - log_band[:-dt]
            working = diff_band
            valid_start = dt
        else:
            working = log_band
            valid_start = 0

        # Step 4: Spectral Gaussian smoothing (smooth out spike noise after diff)
        sigma = self.cfg.spectral_sigma
        if sigma > 0:
            working = gaussian_filter1d(working, sigma=sigma, axis=1)

        # Step 5: Extract features from the preprocessed signal
        features = np.zeros((n_frames, 6), dtype=np.float64)
        times = []
        cep_buffer = []

        for i in range(n_frames):
            t = (i * hop) / self.cfg.sr
            times.append(t)

            w = working[i]
            centered = w - w.mean()

            # SMD: spectral modulation depth
            smd = float(np.std(centered))

            # Cepstrum for CPR/CPN
            cep_frame = np.abs(np.fft.fft(centered))
            cep_buffer.append(cep_frame)
            if len(cep_buffer) > self.cfg.cep_avg_frames:
                cep_buffer.pop(0)
            avg_cep = np.mean(cep_buffer, axis=0)
            cpr, cpn, cpq_sec, dist_cm = self._compute_cep_features(avg_cep)

            # NDA: normalized differential activity (from working signal)
            nda = 0.0
            if i >= 1:
                prev = working[i - 1]
                diff = w - prev
                nda = float(np.mean(np.abs(diff)))

            features[i] = [smd, cpr, cpn, nda, cpq_sec, dist_cm]

        # Zero out features for invalid diff frames
        if valid_start > 0:
            features[:valid_start] = 0

        return features, times


def extract_features_from_wav(wav_path: str,
                              cfg: CombFeatureConfig | None = None
                              ) -> Tuple[np.ndarray, list, float]:
    """Convenience: load WAV and extract features.

    Returns (feature_matrix, times, duration_sec).
    """
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype='float64')
    if audio.ndim > 1:
        audio = audio[:, 0]

    if cfg is None:
        cfg = CombFeatureConfig(sr=sr)
    elif cfg.sr != sr:
        cfg.sr = sr

    extractor = CombFeatureExtractor(cfg)
    feats, times = extractor.process_file(audio, sr)
    duration = len(audio) / sr
    return feats, times, duration
