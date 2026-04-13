"""Offline physics-guided feature extraction aligned with the current project."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from processing.comb_shift import estimate_comb_filter_shift, lag_to_shift_direction
from processing.range_kf import FixedLagRTS, RangeKF
from processing.sine_fit_22 import _make_omega_grid, _weights_from_amp

from ml_uav_comb.features.direction import SHIFT_DOWN_FREQ, SHIFT_NONE, SHIFT_UP_FREQ, SHIFT_TO_NUM
from ml_uav_comb.features.feature_utils import shift_num
from ml_uav_comb.features.stpacc import compute_stpacc_frame


C_SPEED = 343.0


@dataclass
class FitResult:
    omega_win: float
    A_win: float
    rmse: float
    r2: float
    distance_cm: float
    success: bool


class OfflineCombFeatureExtractor:
    """Pure offline extractor that mirrors the existing physics front end."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        audio_cfg = cfg["audio"]
        front_cfg = cfg["front_end"]
        feat_cfg = cfg["features"]
        kf_cfg = cfg["kf"]

        self.sr = int(audio_cfg["target_sr"])
        self.n_fft = int(audio_cfg["n_fft"])
        self.hop_len = int(audio_cfg["hop_len"])
        self.freq_min = float(audio_cfg["freq_min"])
        self.freq_max = float(audio_cfg["freq_max"])
        self.hop_sec = float(self.hop_len) / float(self.sr)

        self.fft_freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        self.freq_idx = np.where(
            (self.fft_freqs >= self.freq_min) & (self.fft_freqs <= self.freq_max)
        )[0]
        self.selected_freqs = self.fft_freqs[self.freq_idx].astype(np.float32)

        self.history_frames = int(front_cfg["history_frames"])
        self.smooth_sigma_1 = float(front_cfg["smooth_sigma_1"])
        self.smooth_sigma_2 = float(front_cfg["smooth_sigma_2"])
        self.d1_amp_scale = float(front_cfg["d1_amp_scale"])
        self.sound_threshold = float(front_cfg["sound_threshold"])
        self.noise_gate_enabled = bool(front_cfg["noise_gate_enabled"])
        self.noise_gate_threshold = float(front_cfg["noise_gate_threshold"])
        self.noise_gate_smooth = float(front_cfg["noise_gate_smooth"])
        self.amp_threshold = float(front_cfg["amp_threshold"])
        self.sum_abs_smooth_len = int(front_cfg["sum_abs_smooth_len"])
        self.k_compare = int(front_cfg["k_compare"])
        self.trend_n = int(front_cfg["trend_n"])
        self.trend_min_votes = int(front_cfg["trend_min_votes"])
        self.dir_window_size = int(front_cfg["dir_window_size"])
        self.dir_vote_threshold = int(front_cfg["dir_vote_threshold"])
        self.comb_max_lag = int(front_cfg["comb_max_lag"])
        self.comb_rho_thresh = float(front_cfg["comb_rho_thresh"])
        self.window_size = int(front_cfg["window_size"])
        self.window_hop = int(front_cfg["window_hop"])
        self.window_pick = int(front_cfg["window_pick"])
        self.window_gaussian_sigma = float(front_cfg["gaussian_sigma_window"])
        self.fit_d_min_m = float(front_cfg["fit_d_min_m"])
        self.fit_oversample = float(front_cfg["fit_oversample"])
        self.fit_gaussian_sigma = float(front_cfg["fit_gaussian_sigma"])
        self.local_radius_first = int(front_cfg["local_radius_first"])
        self.local_radius_segment = int(front_cfg["local_radius_segment"])
        self.missing_threshold_sec = float(front_cfg["missing_threshold_sec"])
        self.min_amplitude = float(front_cfg["min_amplitude"])

        self.use_stpacc = bool(feat_cfg["use_stpacc"])
        self.stp_n_fft = int(feat_cfg["stp_n_fft"])
        self.stp_hop_length = int(feat_cfg["stp_hop_length"])
        self.stp_downsample_bins = int(feat_cfg["stp_downsample_bins"])
        self.acoustic_scalar_fields = list(feat_cfg["acoustic_scalar_fields"])
        self.teacher_only_fields = list(feat_cfg["teacher_only_fields"])

        self.kf = RangeKF(
            R=float(kf_cfg["R"]),
            sigma_a=float(kf_cfg["sigma_a"]),
            v_max=float(kf_cfg["v_max"]),
            a_max=float(kf_cfg["a_max"]),
            d_min=float(self.fit_d_min_m * 100.0),
        )
        self.dt_window = self.window_hop * self.hop_sec
        self.rts = FixedLagRTS(
            lag=int(kf_cfg["lag"]),
            v_max=float(kf_cfg["v_max"]),
            a_max=float(kf_cfg["a_max"]),
            dt_window=self.dt_window,
            d_min=float(self.fit_d_min_m * 100.0),
        )

        self.omega_grid_global = _make_omega_grid(
            fmin=self.freq_min,
            fmax=self.freq_max,
            n_points=len(self.selected_freqs),
            d_min_m=self.fit_d_min_m,
            c_speed=C_SPEED,
            oversample=self.fit_oversample,
        )
        phi = np.outer(self.omega_grid_global, self.selected_freqs).astype(np.float32)
        self.cos_global = np.cos(phi)

        self.buf_short = np.zeros(self.n_fft, dtype=np.float32)
        self.buf_stp = np.zeros(self.stp_n_fft, dtype=np.float32)
        self.s_hist: deque[np.ndarray] = deque(maxlen=150)
        self.h_amp: deque[np.ndarray] = deque(maxlen=self.history_frames)
        self.d1_hist: deque[np.ndarray] = deque(maxlen=150)
        self.sum_abs_d1_hist: deque[float] = deque(maxlen=self.sum_abs_smooth_len)
        self.d1_window: deque[np.ndarray] = deque(maxlen=self.k_compare)
        self.shift_window: deque[str] = deque(maxlen=self.dir_window_size)
        self.window_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.prev_smooth_d1: Optional[np.ndarray] = None
        self.prev_window_omega: Optional[float] = None
        self._smoothed_rms = 0.0
        self._frames_since_window = 0
        self.last_fit_result: Optional[FitResult] = None
        self.last_valid_shift = SHIFT_NONE
        self.last_distance_kf: Optional[float] = None
        self.last_velocity_kf: Optional[float] = None
        self.last_acceleration_kf: float = 0.0
        self.prev_velocity_kf: Optional[float] = None
        self.prev_measure_time: Optional[float] = None
        self.last_measure_time: Optional[float] = None
        self.force_reinit = False

    def _nearest_omega_idx(self, omega: float) -> int:
        idx = int(np.searchsorted(self.omega_grid_global, omega))
        if idx == len(self.omega_grid_global):
            idx -= 1
        if idx > 0 and abs(self.omega_grid_global[idx] - omega) > abs(self.omega_grid_global[idx - 1] - omega):
            idx -= 1
        return idx

    def _fit_window_phi0_inherit(self, block_11xF: np.ndarray) -> FitResult:
        block_s = gaussian_filter1d(
            block_11xF,
            sigma=self.fit_gaussian_sigma,
            axis=0,
            mode="reflect",
        )
        omegas = []
        amplitudes = []
        omega_prev_seg = None

        for idx, y in enumerate(block_s):
            weights = _weights_from_amp(y)
            sw = np.sqrt(weights).astype(np.float32)

            if idx == 0 and self.prev_window_omega is not None:
                center = self._nearest_omega_idx(self.prev_window_omega)
                radius = self.local_radius_first
            elif omega_prev_seg is not None:
                center = self._nearest_omega_idx(omega_prev_seg)
                radius = self.local_radius_segment
            else:
                center = None
                radius = None

            if center is None:
                lo, hi = 0, len(self.omega_grid_global)
            else:
                lo = max(0, center - radius)
                hi = min(len(self.omega_grid_global), center + radius + 1)

            cos_loc = self.cos_global[lo:hi, :]
            omega_loc = self.omega_grid_global[lo:hi]
            cosw = cos_loc * sw[None, :]
            yw = y * sw
            denom = np.einsum("gf,gf->g", cosw, cosw, optimize=True) + 1e-12
            numer = np.einsum("gf,f->g", cosw, yw, optimize=True)
            amplitude = np.maximum(numer / denom, 0.0)
            yhat = amplitude[:, None] * cos_loc
            residual = (yhat - y) * sw[None, :]
            sse = np.einsum("gf,gf->g", residual, residual, optimize=True)
            best = int(np.argmin(sse))
            omegas.append(float(omega_loc[best]))
            amplitudes.append(float(amplitude[best]))
            omega_prev_seg = omegas[-1]

        if not omegas:
            return FitResult(0.0, 0.0, float("inf"), -1.0, 0.0, False)

        y_rep = np.median(block_s, axis=0).astype(np.float32)
        candidate_idx = np.asarray([self._nearest_omega_idx(v) for v in omegas], dtype=int)
        cos_cands = self.cos_global[candidate_idx, :]
        amplitude_vec = np.asarray(amplitudes, dtype=np.float32)
        yhat_all = amplitude_vec[:, None] * cos_cands
        weight_stack = np.stack([_weights_from_amp(block_s[k]) for k in range(len(omegas))], axis=0)
        sqrt_weight_stack = np.sqrt(weight_stack).astype(np.float32)
        residual = (yhat_all[:, None, :] - block_s[None, :, :]) * sqrt_weight_stack[None, :, :]
        sse = np.einsum("jkf,jkf->jk", residual, residual, optimize=True)
        np.fill_diagonal(sse, 0.0)
        total_sse = sse.sum(axis=1)
        best_idx = int(np.argmin(total_sse))

        omega_win = float(omegas[best_idx])
        amp_win = float(amplitudes[best_idx])
        yhat = amp_win * cos_cands[best_idx]
        rmse = float(np.sqrt(np.mean((yhat - y_rep) ** 2)))
        ss_res = float(np.sum((y_rep - yhat) ** 2))
        ss_tot = float(np.sum((y_rep - np.mean(y_rep)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        distance_cm = C_SPEED * 100.0 * omega_win / (4.0 * math.pi)
        return FitResult(omega_win, amp_win, rmse, r2, distance_cm, True)

    def _update_direction(self, smooth_d1: np.ndarray, smooth_amp: float, noise_blocked: bool) -> Dict[str, Any]:
        shift_direction_raw = SHIFT_NONE
        comb_lag = 0.0
        comb_rho = 0.0

        self.d1_window.append(smooth_d1)
        if smooth_amp > self.amp_threshold and len(self.d1_window) >= self.k_compare:
            new = self.d1_window[-1]
            candidates = list(range(0, len(self.d1_window) - 1))
            n = min(self.trend_n, len(candidates))
            step = max(1, (len(candidates) // max(1, n))) if n > 0 else 1
            ref_idx = candidates[::step][:n]
            oldest = self.d1_window[ref_idx[0]] if ref_idx else self.d1_window[0]
            lag, rho = estimate_comb_filter_shift(
                oldest,
                new,
                max_lag=self.comb_max_lag,
                rho_thresh=self.comb_rho_thresh,
            )
            comb_lag = float(lag)
            comb_rho = float(rho)

            votes_up = 0
            votes_down = 0
            valid_pairs = 0
            for idx in ref_idx:
                old = self.d1_window[idx]
                lag_pair, rho_pair = estimate_comb_filter_shift(
                    old,
                    new,
                    max_lag=self.comb_max_lag,
                    rho_thresh=self.comb_rho_thresh,
                )
                shift_vote = lag_to_shift_direction(lag_pair, rho_pair, rho_thresh=self.comb_rho_thresh)
                if shift_vote == SHIFT_UP_FREQ:
                    votes_up += 1
                    valid_pairs += 1
                elif shift_vote == SHIFT_DOWN_FREQ:
                    votes_down += 1
                    valid_pairs += 1

            if valid_pairs >= self.trend_min_votes:
                if votes_up > votes_down:
                    shift_direction_raw = SHIFT_UP_FREQ
                elif votes_down > votes_up:
                    shift_direction_raw = SHIFT_DOWN_FREQ

        window_direction = SHIFT_NONE
        if shift_direction_raw in (SHIFT_UP_FREQ, SHIFT_DOWN_FREQ):
            self.shift_window.append(shift_direction_raw)
            cnt_up = sum(1 for item in self.shift_window if item == SHIFT_UP_FREQ)
            cnt_down = sum(1 for item in self.shift_window if item == SHIFT_DOWN_FREQ)
            if cnt_up >= self.dir_vote_threshold and cnt_up > cnt_down:
                window_direction = SHIFT_UP_FREQ
            elif cnt_down >= self.dir_vote_threshold and cnt_down > cnt_up:
                window_direction = SHIFT_DOWN_FREQ

        shift_direction_hysteresis = window_direction
        if window_direction == SHIFT_NONE:
            if smooth_amp > self.amp_threshold:
                shift_direction_hysteresis = self.last_valid_shift
            else:
                self.last_valid_shift = SHIFT_NONE
        else:
            self.last_valid_shift = window_direction

        if noise_blocked:
            shift_direction_hysteresis = SHIFT_NONE

        return {
            "shift_direction_raw": shift_direction_raw,
            "shift_direction_hysteresis": shift_direction_hysteresis,
            "comb_lag": comb_lag,
            "comb_rho": comb_rho,
        }

    def _step_kf(
        self,
        current_time: float,
        fit_result: Optional[FitResult],
        smooth_amp: float,
        shift_direction_hysteresis: str,
        noise_blocked: bool,
    ) -> Dict[str, Any]:
        measurement_available = 0.0
        if fit_result is None:
            if self.last_measure_time is not None and (current_time - self.last_measure_time > self.missing_threshold_sec):
                self.kf.reset()
                self.rts.reset()
                self.last_distance_kf = None
                self.last_velocity_kf = None
                self.last_acceleration_kf = 0.0
                self.prev_velocity_kf = None
                self.prev_measure_time = None
            return {
                "distance_raw_cm": self.last_fit_result.distance_cm if self.last_fit_result else None,
                "distance_kf_cm": self.last_distance_kf,
                "velocity_kf_cm_s": self.last_velocity_kf,
                "acceleration_kf_cm_s2": self.last_acceleration_kf,
                "heuristic_measure_available": measurement_available,
            }

        apply_gate = (not noise_blocked) and (smooth_amp >= self.min_amplitude)
        effective_measure = apply_gate and shift_direction_hysteresis != SHIFT_NONE and fit_result.success
        if not effective_measure:
            if self.last_measure_time is not None and (current_time - self.last_measure_time > self.missing_threshold_sec):
                self.kf.reset()
                self.rts.reset()
                self.last_distance_kf = None
                self.last_velocity_kf = None
                self.last_acceleration_kf = 0.0
                self.prev_velocity_kf = None
                self.prev_measure_time = None
            return {
                "distance_raw_cm": fit_result.distance_cm,
                "distance_kf_cm": self.last_distance_kf,
                "velocity_kf_cm_s": self.last_velocity_kf,
                "acceleration_kf_cm_s2": self.last_acceleration_kf,
                "heuristic_measure_available": measurement_available,
            }

        if self.force_reinit:
            self.kf.reset()
            self.rts.reset()
            self.last_distance_kf = None
            self.last_velocity_kf = None
            self.last_acceleration_kf = 0.0
            self.prev_velocity_kf = None
            self.prev_measure_time = None
            self.force_reinit = False

        measurement_available = 1.0
        self.kf.predict(self.dt_window)
        x_pred_next = None if self.kf.x is None else self.kf.x.copy()
        p_pred_next = None if self.kf.P is None else self.kf.P.copy()
        self.kf.update(fit_result.distance_cm)

        if x_pred_next is not None and self.kf.x is not None:
            F = np.array([[1.0, self.dt_window], [0.0, 1.0]], dtype=np.float32)
            self.rts.push(F, x_pred_next, p_pred_next, self.kf.x, self.kf.P, timestamp=current_time)

        x_s = self.rts.get_smoothed()
        if x_s is not None:
            raw_distance = float(x_s[0][0])
            raw_velocity = float(x_s[0][1])
        else:
            raw_distance = float(self.kf.distance) if self.kf.distance is not None else None
            raw_velocity = float(self.kf.velocity) if self.kf.velocity is not None else None

        self.last_distance_kf = raw_distance
        self.last_velocity_kf = raw_velocity
        if (
            self.prev_velocity_kf is not None
            and self.prev_measure_time is not None
            and raw_velocity is not None
            and current_time > self.prev_measure_time
        ):
            dt = current_time - self.prev_measure_time
            self.last_acceleration_kf = float(raw_velocity - self.prev_velocity_kf) / float(dt)
        else:
            self.last_acceleration_kf = 0.0

        self.prev_velocity_kf = raw_velocity
        self.prev_measure_time = current_time
        self.last_measure_time = current_time
        self.last_fit_result = fit_result
        return {
            "distance_raw_cm": fit_result.distance_cm,
            "distance_kf_cm": self.last_distance_kf,
            "velocity_kf_cm_s": self.last_velocity_kf,
            "acceleration_kf_cm_s2": self.last_acceleration_kf,
            "heuristic_measure_available": measurement_available,
        }

    def process_audio_array(self, audio: np.ndarray) -> Dict[str, Any]:
        if audio.ndim != 1:
            raise ValueError("audio must be mono")

        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        self.buf_short[:] = audio[: self.n_fft]
        if self.use_stpacc:
            if len(audio) >= self.stp_n_fft:
                self.buf_stp[:] = audio[: self.stp_n_fft]
            else:
                self.buf_stp[: len(audio)] = audio[:]

        phase_stft = []
        diff_comb = []
        scalar_seq = []
        scalar_observed_mask = []
        scalar_reliable_mask = []
        teacher_seq = []
        stpacc_seq = []
        frame_time_sec = []

        max_start = len(audio)
        frame_idx = 0
        for start in range(self.n_fft, max_start + self.hop_len, self.hop_len):
            block = audio[start : start + self.hop_len]
            if len(block) == 0:
                break
            if len(block) < self.hop_len:
                block = np.pad(block, (0, self.hop_len - len(block)))

            self.buf_short[:-self.hop_len] = self.buf_short[self.hop_len :]
            self.buf_short[-self.hop_len :] = block
            if self.use_stpacc:
                self.buf_stp[:-self.hop_len] = self.buf_stp[self.hop_len :]
                self.buf_stp[-self.hop_len :] = block

            rms = float(np.sqrt(np.mean(block**2)))
            self._smoothed_rms = self.noise_gate_smooth * self._smoothed_rms + (1.0 - self.noise_gate_smooth) * rms
            noise_blocked = bool(self.noise_gate_enabled and self._smoothed_rms < self.noise_gate_threshold)

            current_time = float(frame_idx) * self.hop_sec
            frame_time_sec.append(current_time)

            win = self.buf_short * np.hanning(self.n_fft)
            fft_vals = np.fft.rfft(win, n=self.n_fft)
            selected_complex = fft_vals[self.freq_idx]
            mag = np.abs(selected_complex).astype(np.float32)
            log_mag = np.log(mag**2 + 1e-7).astype(np.float32)
            phase = np.angle(selected_complex)
            phase_stft.append(
                np.stack(
                    (
                        log_mag,
                        np.sin(phase).astype(np.float32),
                        np.cos(phase).astype(np.float32),
                    ),
                    axis=-1,
                )
            )

            norm = mag / (float(np.max(mag)) + 1e-12)
            smooth = gaussian_filter1d(norm, sigma=self.smooth_sigma_1)
            centered = smooth - float(np.mean(smooth))
            denom = float(np.max(centered) - np.min(centered))
            if denom > 1e-6:
                current = ((centered - float(np.min(centered))) / denom).astype(np.float32)
            else:
                current = centered.astype(np.float32)
            is_sound_present = 0.0 if noise_blocked else float(np.any(current > self.sound_threshold))

            self.s_hist.append(current)
            if len(self.s_hist) > 1:
                current = np.mean(np.stack(self.s_hist, axis=0), axis=0).astype(np.float32)

            old = self.h_amp[0] if len(self.h_amp) == self.h_amp.maxlen else None
            self.h_amp.append(current.copy())
            d1 = current - old if old is not None else np.zeros_like(current)
            d1_freq_sm = gaussian_filter1d(d1, sigma=self.smooth_sigma_1)
            self.d1_hist.append(d1_freq_sm.astype(np.float32))
            if len(self.d1_hist) > 1:
                smooth_d1 = gaussian_filter1d(
                    np.stack(self.d1_hist, axis=0),
                    sigma=self.smooth_sigma_2,
                    axis=0,
                )[-1].astype(np.float32)
            else:
                smooth_d1 = d1.astype(np.float32)

            abs_d1 = np.abs(smooth_d1).astype(np.float32)
            if self.prev_smooth_d1 is None:
                delta_d1 = np.zeros_like(smooth_d1)
            else:
                delta_d1 = (smooth_d1 - self.prev_smooth_d1).astype(np.float32)
            self.prev_smooth_d1 = smooth_d1.copy()
            diff_comb.append(np.stack((smooth_d1, abs_d1, delta_d1), axis=-1))

            sum_abs_d1 = float(np.sum(abs_d1))
            self.sum_abs_d1_hist.append(sum_abs_d1)
            smooth_amp = float(np.mean(np.asarray(self.sum_abs_d1_hist, dtype=np.float32)))

            direction_info = self._update_direction(smooth_d1, smooth_amp, noise_blocked)
            shift_direction_raw = str(direction_info["shift_direction_raw"])
            shift_direction_hysteresis = str(direction_info["shift_direction_hysteresis"])
            comb_lag = float(direction_info["comb_lag"])
            comb_rho = float(direction_info["comb_rho"])

            fit_result = self.last_fit_result
            if not noise_blocked:
                self.window_buffer.append(self.d1_amp_scale * smooth_d1)
                if len(self.window_buffer) == self.window_size:
                    self._frames_since_window += 1
                    if self._frames_since_window >= self.window_hop:
                        window_block_np = np.stack(list(self.window_buffer), axis=0)
                        pick_idx = np.linspace(0, self.window_size - 1, self.window_pick, dtype=int)
                        picked_block = window_block_np[pick_idx]
                        picked_smoothed = gaussian_filter1d(
                            picked_block,
                            sigma=self.window_gaussian_sigma,
                            axis=0,
                        )
                        fit_result = self._fit_window_phi0_inherit(picked_smoothed)
                        self.last_fit_result = fit_result
                        self._frames_since_window = 0

            kf_state = self._step_kf(
                current_time=current_time,
                fit_result=fit_result,
                smooth_amp=smooth_amp,
                shift_direction_hysteresis=shift_direction_hysteresis,
                noise_blocked=noise_blocked,
            )

            reliable_comb = float(
                np.isfinite(comb_lag)
                and np.isfinite(comb_rho)
                and (abs(comb_lag) >= 1.0)
                and (abs(comb_rho) >= self.comb_rho_thresh)
                and (smooth_amp >= self.amp_threshold)
                and (is_sound_present > 0.5)
            )

            acoustic_values = {
                "sum_abs_d1_smooth": smooth_amp,
                "comb_shift_lag": comb_lag,
                "comb_shift_rho": comb_rho,
                "is_sound_present": is_sound_present,
            }
            observed_mask = {
                "sum_abs_d1_smooth": 1.0,
                "comb_shift_lag": 1.0 if np.isfinite(comb_lag) else 0.0,
                "comb_shift_rho": 1.0 if np.isfinite(comb_rho) else 0.0,
                "is_sound_present": 1.0,
            }
            reliable_mask = {
                "sum_abs_d1_smooth": float((smooth_amp >= self.amp_threshold) and (is_sound_present > 0.5)),
                "comb_shift_lag": reliable_comb,
                "comb_shift_rho": reliable_comb,
                "is_sound_present": 1.0,
            }
            teacher_values = {
                "heuristic_distance_raw_cm": 0.0 if kf_state["distance_raw_cm"] is None else float(kf_state["distance_raw_cm"]),
                "heuristic_distance_kf_cm": 0.0 if kf_state["distance_kf_cm"] is None else float(kf_state["distance_kf_cm"]),
                "heuristic_distance_raw_available": 0.0 if kf_state["distance_raw_cm"] is None else 1.0,
                "heuristic_distance_kf_available": 0.0 if kf_state["distance_kf_cm"] is None else 1.0,
                "velocity_kf_cm_s": 0.0 if kf_state["velocity_kf_cm_s"] is None else float(kf_state["velocity_kf_cm_s"]),
                "acceleration_kf_cm_s2": 0.0 if kf_state["acceleration_kf_cm_s2"] is None else float(kf_state["acceleration_kf_cm_s2"]),
                "heuristic_measure_available": float(kf_state["heuristic_measure_available"]),
                "shift_direction_raw": shift_num(shift_direction_raw),
                "shift_direction_hysteresis": shift_num(shift_direction_hysteresis),
            }

            scalar_seq.append(
                np.asarray([acoustic_values[field] for field in self.acoustic_scalar_fields], dtype=np.float32)
            )
            scalar_observed_mask.append(
                np.asarray([observed_mask[field] for field in self.acoustic_scalar_fields], dtype=np.float32)
            )
            scalar_reliable_mask.append(
                np.asarray([reliable_mask[field] for field in self.acoustic_scalar_fields], dtype=np.float32)
            )
            teacher_seq.append(
                np.asarray([teacher_values[field] for field in self.teacher_only_fields], dtype=np.float32)
            )

            if self.use_stpacc:
                if start >= self.stp_n_fft:
                    stp_frame = compute_stpacc_frame(
                        self.buf_stp.copy(),
                        downsample_bins=self.stp_downsample_bins,
                    )
                else:
                    stp_frame = np.zeros(self.stp_downsample_bins, dtype=np.float32)
                stpacc_seq.append(stp_frame.astype(np.float32))

            frame_idx += 1

        result = {
            "schema_version": np.asarray([2], dtype=np.int64),
            "frame_time_sec": np.asarray(frame_time_sec, dtype=np.float32),
            "phase_stft": np.asarray(phase_stft, dtype=np.float32),
            "diff_comb": np.asarray(diff_comb, dtype=np.float32),
            "scalar_seq": np.asarray(scalar_seq, dtype=np.float32),
            "scalar_field_names": np.asarray(self.acoustic_scalar_fields, dtype=object),
            "scalar_observed_mask": np.asarray(scalar_observed_mask, dtype=np.float32),
            "scalar_reliable_mask": np.asarray(scalar_reliable_mask, dtype=np.float32),
            "teacher_seq": np.asarray(teacher_seq, dtype=np.float32),
            "teacher_field_names": np.asarray(self.teacher_only_fields, dtype=object),
            "frequencies_hz": self.selected_freqs.astype(np.float32),
        }
        if self.use_stpacc:
            result["stpacc"] = np.asarray(stpacc_seq, dtype=np.float32)
        return result


def process_audio_array(
    audio: np.ndarray,
    sr: int,
    cfg: Dict[str, Any],
    optional_labels: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    if sr != int(cfg["audio"]["target_sr"]):
        raise ValueError(f"expected sr={cfg['audio']['target_sr']}, got {sr}")
    extractor = OfflineCombFeatureExtractor(cfg)
    return extractor.process_audio_array(audio)
