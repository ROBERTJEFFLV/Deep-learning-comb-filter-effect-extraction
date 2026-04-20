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
    # Running-mean normalization EMA coefficient
    ema_alpha: float = 0.02
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
    """Stateful per-frame comb filter feature extractor."""

    def __init__(self, cfg: CombFeatureConfig | None = None):
        self.cfg = cfg or CombFeatureConfig()
        c = self.cfg

        # Frequency axis
        self.freqs = np.fft.rfftfreq(c.n_fft, 1.0 / c.sr)
        self.band_mask = (self.freqs >= c.freq_min) & (self.freqs <= c.freq_max)
        self.band_freqs = self.freqs[self.band_mask]
        self.n_band = int(self.band_mask.sum())

        # Quefrency mapping: cepstral bin n corresponds to spectral period
        # P = N_band / n bins,  and time delay τ = P·df⁻¹ = n / (N_band·df)
        # So:  n = N_band · df · τ,  and  τ = n / (N_band · df)
        self.df = c.sr / c.n_fft  # Hz per frequency bin
        self.quef_tau_factor = 1.0 / (self.n_band * self.df)  # τ = n * factor

        # Comb quefrency search range (in cepstral bins)
        self.cep_min_bin = max(2, round(c.tau_min_s / self.quef_tau_factor))
        self.cep_max_bin = min(self.n_band // 2,
                               round(c.tau_max_s / self.quef_tau_factor) + 1)

        # Non-comb reference range: bins beyond comb range but before Nyquist
        # Used as baseline for normalization
        self.ref_min_bin = self.cep_max_bin + 5
        self.ref_max_bin = min(self.n_band // 2, self.ref_min_bin + 30)

        # Hann window for STFT
        self.window = np.hanning(c.n_fft).astype(np.float64)

        # Running-mean state
        self._ema_mag = None
        self._prev_norm = None
        # Cepstrum averaging buffer
        self._cep_buffer = []

    def reset(self):
        """Reset internal state (call between independent audio files)."""
        self._ema_mag = None
        self._prev_norm = None
        self._cep_buffer = []

    def _stft_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrum of a single frame."""
        windowed = frame * self.window
        return np.abs(np.fft.rfft(windowed))

    def _normalize(self, mag_band: np.ndarray) -> np.ndarray:
        """Running-mean normalized spectrum (zero-mean modulation)."""
        alpha = self.cfg.ema_alpha
        if self._ema_mag is None:
            self._ema_mag = mag_band.copy()
        else:
            self._ema_mag = (1 - alpha) * self._ema_mag + alpha * mag_band
        ref = np.maximum(self._ema_mag, 1e-12)
        return mag_band / ref - 1.0

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
        """Extract CPR, CPN, quefrency, distance from a (possibly averaged) cepstrum.

        Returns (cpr, cpn, cpq_sec, dist_cm).
        """
        qmin, qmax = self.cep_min_bin, self.cep_max_bin
        if qmin >= qmax or qmax > len(cep_abs) // 2:
            return 0.0, 0.0, 0.0, 0.0

        search = cep_abs[qmin:qmax]
        if len(search) == 0:
            return 0.0, 0.0, 0.0, 0.0

        peak_idx_local = int(np.argmax(search))
        peak_val = float(search[peak_idx_local])

        # CPR: peak / median in search range
        median_val = float(np.median(search))
        cpr = peak_val / (median_val + 1e-12)

        # CPN: peak / mean in non-comb reference range
        rmin, rmax = self.ref_min_bin, self.ref_max_bin
        if rmin < rmax and rmax <= len(cep_abs) // 2:
            ref_baseline = float(np.mean(cep_abs[rmin:rmax]))
        else:
            ref_baseline = median_val
        cpn = peak_val / (ref_baseline + 1e-12)

        # Quefrency → τ → distance
        peak_bin = qmin + peak_idx_local
        cpq_sec = peak_bin * self.quef_tau_factor
        dist_cm = C_SPEED * cpq_sec / 2.0 * 100.0

        return cpr, cpn, cpq_sec, dist_cm

    def _compute_nda(self, norm_band: np.ndarray) -> float:
        """Normalized Differential Activity — improved S(t)."""
        if self._prev_norm is None:
            self._prev_norm = norm_band.copy()
            return 0.0
        diff = norm_band - self._prev_norm
        nda = float(np.mean(np.abs(diff)))
        self._prev_norm = norm_band.copy()
        return nda

    def process_frame(self, frame: np.ndarray) -> FrameFeatures:
        """Extract features from one audio frame of length n_fft."""
        mag = self._stft_frame(frame)
        mag_band = mag[self.band_mask]
        log_band = np.log(mag_band + 1e-12)
        norm_band = self._normalize(mag_band)

        smd = self._compute_smd(log_band)

        # Compute per-frame cepstrum and add to averaging buffer
        cep_frame = self._compute_cepstrum(log_band)
        self._cep_buffer.append(cep_frame)
        if len(self._cep_buffer) > self.cfg.cep_avg_frames:
            self._cep_buffer.pop(0)

        # Average cepstrum over buffer for robustness
        avg_cep = np.mean(self._cep_buffer, axis=0)
        cpr, cpn, cpq_sec, dist_cm = self._compute_cep_features(avg_cep)

        nda = self._compute_nda(norm_band)

        return FrameFeatures(
            smd=smd, cpr=cpr, cpn=cpn,
            nda=nda, cpq=cpq_sec, dist_est_cm=dist_cm,
        )

    def process_file(self, audio: np.ndarray, sr: int | None = None
                     ) -> Tuple[np.ndarray, list]:
        """Process an entire audio array.

        Returns
        -------
        feature_matrix : (N_frames, 6) float64
            Columns: [smd, cpr, cpn, nda, cpq, dist_est_cm]
        frame_times : list of float (seconds)
        """
        if sr is not None and sr != self.cfg.sr:
            raise ValueError(f"Sample rate mismatch: got {sr}, expected {self.cfg.sr}")

        self.reset()
        n_fft = self.cfg.n_fft
        hop = self.cfg.hop_length
        n_samples = len(audio)
        n_frames = max(0, (n_samples - n_fft) // hop + 1)

        features = np.zeros((n_frames, 6), dtype=np.float64)
        times = []

        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + n_fft].astype(np.float64)
            ff = self.process_frame(frame)
            features[i] = [ff.smd, ff.cpr, ff.cpn, ff.nda, ff.cpq, ff.dist_est_cm]
            times.append(start / self.cfg.sr)

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
