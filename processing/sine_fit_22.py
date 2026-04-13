# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
基于幅值-频率点的余弦拟合（φ=0；采用 A = K·exp(-beta·ω) 的指数先验并带 ±A_margin 硬约束；
带 ω 继承与局部搜索；窗口输出 ω 均值）
--------------------------------------------------------------------------------
改动点：
1) 继承策略：
    - 窗口内 11 段：第 1 段用“上一窗口 ω”为初值（local_bins=5）；
      后续段用“上一段 ω”为初值（local_bins=2）；局部网格为空则回退全局。
    - 窗口完成后，将本窗口 ω 均值作为下一窗口第 1 段的初值。
2) 目标函数：
    - 仅搜索 ω；给定 ω → 通过闭式最小二乘求 A_ls，再将 A 限定到 [max(0, A0-Δ), A0+Δ]，其中 A0(ω)=K·exp(-beta·ω)、Δ=A_margin；
    - 以加权 SSE 选 ω。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

# 优先使用 soundfile 读取音频（支持更多格式），若不可用则后备 scipy.io.wavfile
try:
    import soundfile as sf  # 首选读取库
    _HAVE_SF = True
except Exception:  # 如果 soundfile 不可用，使用 scipy.io.wavfile
    from scipy.io import wavfile
    _HAVE_SF = False

from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq


# ============================ 数据类 ============================
@dataclass
class FitResult:
    """存储一次拟合的结果指标（面向 CSV 输出和后续分析）"""
    A: float                # 幅值（窗口/段的幅值估计，可能受指数先验约束）
    omega: float            # 角频率 (rad/Hz)，模型中为 ω * f（f 单位为 Hz）
    phi: float              # 相位 (rad)，恒为 0
    ripple_freq: float      # 空间频率 (cycles/Hz) = omega/(2π)
    rmse: float             # 均方根误差（模型与观测幅值之间，未加权）
    r2: float               # 拟合优度 R^2（未加权）
    success: bool           # 是否成功（找到至少一个候选 ω）
    nit: int                # 遍历的候选 ω 数量（本窗口内总计）
    cost: float             # 加权 SSE（在代表谱上的代价）
    elapsed: float          # 本窗口耗时（秒）


# ============================ 辅助函数 ============================
def _demean_and_smooth(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    对时域帧去均值并做高斯平滑：
    - 去均值消除 DC，避免低频抬升；
    - 高斯平滑抑制时域噪声，使谱更平滑。
    """
    x = x - float(np.mean(x))
    if sigma and sigma > 0:
        x = gaussian_filter1d(x, sigma=sigma, mode="reflect")
    return x


def _interp_amp_in_band(
    x: np.ndarray, fs: float, fmin: float, fmax: float, n_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 rFFT 幅值谱，并在目标频带上均匀插值 n_points 个频率点。
    返回 (f_grid, amp)。
    """
    if fmax >= fs / 2.0:
        raise ValueError(f"fmax={fmax} 必须小于 Nyquist={fs/2:.1f} Hz")
    X = np.abs(rfft(x))
    freqs = rfftfreq(len(x), d=1.0 / fs)
    f_grid = np.linspace(fmin, fmax, n_points)
    amp = np.interp(f_grid, freqs, X)
    return f_grid, amp


def _weights_from_amp(amp: np.ndarray) -> np.ndarray:
    """
    基于幅值构造权重：
    - 噪声基线取 25% 分位数；
    - w = amp / noise，裁剪到 [0.5, 10]；
    - 加权最小二乘时使用 sqrt(w)。
    """
    noise = np.percentile(amp, 25)
    eps = 1e-12
    w = amp / (noise + eps)
    return np.clip(w, 0.5, 10.0)



def _make_omega_grid(
    fmin: float,
    fmax: float,
    n_points: int,
    d_min_m: float = 0.05,
    c_speed: float = 343.0,
    oversample: float = 6.0,
) -> np.ndarray:
    """
    生成候选 omega（rad/Hz）网格。
    - 带宽 B = fmax - fmin，Rayleigh 分辨率 Δω ≈ 2π/B；
    - 频率插值间隔 df ≈ B/(n_points-1) → ω_max ≈ π/df；
    - ω_min ≈ 2π*(2*d_min/c)；
    - 步长 domega = (2π/B)/oversample。
    """
    B = float(fmax - fmin)
    if n_points < 2 or B <= 0:
        raise ValueError("n_points 必须 >=2 且 fmax > fmin")

    df = B / (n_points - 1)
    omega_max = math.pi / df

    tau_min = 2.0 * d_min_m / c_speed
    omega_min = 2.0 * math.pi * tau_min

    domega = (2.0 * math.pi) / B / max(1.0, float(oversample))

    k_min = int(math.ceil(omega_min / domega))
    k_max = int(math.floor(omega_max / domega))
    if k_max < k_min:
        grid = np.array([max(omega_min, 0.0)])
    else:
        grid = (np.arange(k_min, k_max + 1, dtype=float) * domega)
    # 确保 ω 网格为正（去除非正值，至少保留一个极小正值）
    grid = grid[grid > 0.0]
    if grid.size == 0:
        grid = np.array([max(domega, 1e-9)], dtype=float)
    return grid


def _select_local_grid(omega_grid: np.ndarray, omega_prev: float, local_bins: int) -> np.ndarray:
    """
    在全局 omega 网格中选择围绕上一次估计 omega_prev 的局部子网格（±local_bins）。
    - 若 omega_prev 超出范围，自动就近截取；
    - 若 local_bins < 1 或 omega_grid 为空，返回空数组。
    """
    if omega_grid.size == 0 or local_bins < 1:
        return np.array([], dtype=float)
    k = int(np.argmin(np.abs(omega_grid - omega_prev)))
    i0 = max(0, k - local_bins)
    i1 = min(len(omega_grid) - 1, k + local_bins)
    return omega_grid[i0:i1 + 1]


# ============================ 指数先验模型：A = K·exp(-beta·ω) ============================


# NOTE: legacy physA-specific coarse search removed. Use _coarse_search_fit_phi0
# which implements the same exponential-prior + hard-constraint behavior when
# provided with (K, beta, A_margin).


# ============================ 兼容实时管线：无参简化版封装 ============================
def _fit_A_phi0_weighted(
    f: np.ndarray,
    y: np.ndarray,
    omega: float,
    w: Optional[np.ndarray],
    K: Optional[float] = None,
    beta: Optional[float] = None,
    A_margin: Optional[float] = None,
) -> Tuple[float, float]:
    """
    指数先验 + 硬约束版本（仅保留此逻辑）：
      y ≈ A·cos(ω f)
      A0(ω) = K·exp(-beta·ω)，A ∈ [max(0, A0-Δ), A0+Δ]，Δ=A_margin
      先计算闭式解 A_ls = ⟨w y, cos⟩ / ⟨w cos, cos⟩；再将 A = clip(A_ls, 区间)
      返回 (A*, sse)
    注意：必须提供 K、beta、A_margin，否则将抛出异常。
    """
    cvec = np.cos(omega * f)
    if w is not None:
        num = float(np.sum(w * y * cvec))
        den = float(np.sum(w * cvec * cvec)) + 1e-12
    else:
        num = float(np.dot(y, cvec))
        den = float(np.dot(cvec, cvec)) + 1e-12

    use_prior = (K is not None) and (beta is not None) and (A_margin is not None)

    if not use_prior:
        raise ValueError("_fit_A_phi0_weighted 需要提供 K、beta、A_margin（仅支持 A=K·exp(-beta·ω) 约束）")

    # 先验 A0 与硬约束
    try:
        A0 = float(K) * math.exp(-float(beta) * float(omega))
    except OverflowError:
        A0 = 0.0
    A0 = max(0.0, A0)
    A_min = max(0.0, A0 - float(A_margin))
    A_max = A0 + float(A_margin)
    if A_min > A_max:
        A_min, A_max = A_max, A_min

    A_ls = num / den
    A_hat = float(np.clip(A_ls, A_min, A_max))

    yhat = A_hat * cvec
    if w is not None:
        r = np.sqrt(w) * (yhat - y)
    else:
        r = (yhat - y)
    sse = float(np.sum(r * r))
    return A_hat, sse


def _coarse_search_fit_phi0(
    f: np.ndarray,
    y: np.ndarray,
    omega_grid: np.ndarray,
    use_weights: bool = True,
    K: Optional[float] = None,
    beta: Optional[float] = None,
    A_margin: Optional[float] = None,
) -> Tuple[float, float, float, int]:
    """
    仅支持“指数先验 + 硬约束”的 ω 粗搜索：
      对每个 ω，A0(ω)=K·exp(-beta·ω)，A 被夹到 [max(0, A0-Δ), A0+Δ]，Δ=A_margin，
      计算加权 SSE 并取最小者。
    返回 (omega_best, A_best, sse_best, nit)。
    注意：必须提供 K、beta、A_margin。
    """
    if (K is None) or (beta is None) or (A_margin is None):
        raise ValueError("_coarse_search_fit_phi0 需要提供 K、beta、A_margin（仅支持 A=K·exp(-beta·ω) 约束）")
    w = _weights_from_amp(y) if use_weights else None
    best = None  # (sse, omega, A)
    for omega in omega_grid:
        A_hat, sse = _fit_A_phi0_weighted(
            f, y, omega, w,
            K=K, beta=beta, A_margin=A_margin,
        )
        if (best is None) or (sse < best[0]):
            best = (sse, float(omega), float(A_hat))

    if best is None:
        return 0.0, 0.0, float("inf"), 0

    sse_best, omega_best, A_best = best
    return omega_best, A_best, sse_best, len(omega_grid)




# ============================ 主流程 ============================
def process_wav(
    wav_path: str = "white_noise_echo.wav",
    fmin: float = 1000.0,
    fmax: float = 5000.0,
    n_points: int = 47,
    sigma: float = 1.5,
    frame_ms: float = 200.0,
    hop_ms: float = 100.0,
    channel: Optional[int] = None,
    use_weights: bool = True,
    out_csv: Optional[str] = None,
    d_min_m: float = 0.05,
    c_speed: float = 343.0,
    oversample: float = 6.0,
    # 与原物理一致性相关的参数已移除（仅保留指数先验模型）
) -> Dict:
    """
    逐帧处理并将每个“窗口”的拟合结果写入 CSV（每窗口一行）。
    - 窗口内 11 段：链式继承 ω（local_bins=2），首段继承上一窗口 ω（local_bins=5）。
    - 窗口 ω 输出为 11 段 ω 的简单平均；在该 ω 下对窗口代表谱估计 A 与指标（A 由指数先验 + 硬约束限制）。
    """
    # ---------- 读取音频 ----------
    if _HAVE_SF:
        x, fs = sf.read(wav_path, always_2d=True)
        x = x[:, int(channel)] if channel is not None else np.mean(x, axis=1)
    else:
        fs, x = wavfile.read(wav_path)
        x = np.asarray(x, dtype=float)
        if x.ndim > 1:
            x = x[:, int(channel)] if channel is not None else np.mean(x, axis=1)
        if x.dtype.kind in ("i", "u"):
            x = x / np.iinfo(x.dtype).max  # 将整数 PCM 归一化到 [-1,1]

    # 采样率检查
    if fs < 2 * fmax:
        raise ValueError(f"采样率 {fs} Hz 太低，无法覆盖 {fmax} Hz 频带")

    frame_len = int(round(fs * frame_ms / 1000.0))
    hop_len = int(round(fs * hop_ms / 1000.0))
    if frame_len <= 32 or hop_len <= 0:
        raise ValueError("frame_ms/hop_ms 设置过小")

    results: List[FitResult] = []
    t_stamps: List[float] = []
    amp_list: List[np.ndarray] = []
    frame_times: List[float] = []

    # 预先构造全局 ω 网格
    omega_grid_global = _make_omega_grid(
        fmin=fmin, fmax=fmax, n_points=n_points,
        d_min_m=d_min_m, c_speed=c_speed, oversample=oversample
    )

    # 逐帧计算频带幅值并缓存
    f_grid: Optional[np.ndarray] = None
    for start in range(0, max(0, len(x) - frame_len + 1), hop_len):
        frame = x[start:start + frame_len]
        t0s = start / fs
        frame_filt = _demean_and_smooth(frame, sigma=sigma)
        f_grid_cur, amp = _interp_amp_in_band(frame_filt, fs, fmin, fmax, n_points)
        if f_grid is None:
            f_grid = f_grid_cur
        amp_list.append(amp.astype(np.float64, copy=False))
        frame_times.append(t0s)

    if f_grid is None or len(amp_list) == 0:
        # 无可用帧
        out_csv = out_csv or "sine_fit_results.csv"
        with open(out_csv, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "t_start_s", "A", "omega_rad_per_Hz", "phi_rad", "ripple_freq_cycles_per_Hz",
                "rmse", "r2", "success", "nit", "cost", "d_cm", "elapsed_s"
            ])
        return {
            "wav_path": wav_path, "fs": fs, "frames": 0, "frame_ms": frame_ms, "hop_ms": hop_ms,
            "band": [fmin, fmax], "n_points": n_points, "csv": out_csv, "last_fit": {}
        }

    # --- 窗口参数：0.2s 窗口，0.1s 步长 ---
    frames_per_sec = fs / float(hop_len)
    window_frames = max(1, int(round(0.2 * frames_per_sec)))   # 约 68 帧 @ 44.1k/128
    step_frames   = max(1, int(round(0.1 * frames_per_sec)))   # 约 34 帧
    n_pick = 11

    # 继承用局部搜索半径
    local_bins_within = 2   # 窗口内逐段
    local_bins_cross  = 5   # 窗口间首段

    import time as _time
    prev_window_omega: Optional[float] = None  # 上一窗口输出 ω

    for i0 in range(0, max(0, len(amp_list) - window_frames + 1), step_frames):
        t_win0 = _time.perf_counter()
        i1 = i0 + window_frames
        window_block = np.stack(amp_list[i0:i1], axis=0)  # (W, F)

        # 分 11 段等分平均，得到 (11, F) 及每段时间戳
        seg_means: List[np.ndarray] = []
        seg_times: List[float] = []
        for k in range(n_pick):
            s_rel = int(np.floor(k * window_frames / n_pick))
            e_rel = int(np.ceil((k + 1) * window_frames / n_pick))
            s_rel = max(0, min(s_rel, window_frames - 1))
            e_rel = max(s_rel + 1, min(e_rel, window_frames))
            seg = window_block[s_rel:e_rel, :]
            # 对应的时间戳（原始帧时间的平均）
            if seg.shape[0] == 0:
                seg = window_block[max(0, min(s_rel, window_frames - 1)):max(1, min(s_rel + 1, window_frames)), :]
                seg_time_vals = [frame_times[min(i0 + s_rel, len(frame_times) - 1)]]
            else:
                t_slice = frame_times[i0 + s_rel:i0 + e_rel]
                seg_time_vals = t_slice if len(t_slice) > 0 else [frame_times[min(i0 + s_rel, len(frame_times) - 1)]]

            seg_means.append(np.mean(seg, axis=0))
            seg_times.append(float(np.mean(seg_time_vals)))

        block = np.stack(seg_means, axis=0)  # (11, F)
        # 时间轴高斯平滑（可按需调 sigma；过大将过度平滑时变细节）
        block_s = gaussian_filter1d(block, sigma=2.0, axis=0, mode="reflect")

        # —— 继承 + 局部搜索（物理一致性）—— #
        omegas: List[float] = []
        total_evals = 0
        omega_prev: Optional[float] = None  # 窗口内逐段继承

        for k, (row, t_row) in enumerate(zip(block_s, seg_times)):
            # 选择局部/全局网格
            if k == 0 and (prev_window_omega is not None):
                # 窗口首段：继承上一窗口 ω，local_bins=5
                local_grid = _select_local_grid(omega_grid_global, prev_window_omega, local_bins_cross)
                omega_grid_use = local_grid if local_grid.size > 0 else omega_grid_global
            elif omega_prev is not None:
                # 窗口内后续段：继承上一段 ω，local_bins=2
                local_grid = _select_local_grid(omega_grid_global, omega_prev, local_bins_within)
                omega_grid_use = local_grid if local_grid.size > 0 else omega_grid_global
            else:
                # 无初值：全局
                omega_grid_use = omega_grid_global

            # 使用指数先验的粗搜索（传入先验常数）
            omega_best, A_best_seg, sse_best_seg, nit = _coarse_search_fit_phi0(
                f_grid, row, omega_grid_use, use_weights=use_weights,
                c_speed=c_speed, K=57.77, beta=298.17, A_margin=7.0,
            )
            omegas.append(omega_best)
            total_evals += nit
            omega_prev = omega_best  # 作为下一段初值

        # 窗口 ω 输出：简单均值
        omega_win = float(np.mean(omegas)) if len(omegas) > 0 else (prev_window_omega or 0.0)

    # 用窗口代表谱（中位数）在 ω_win 下估计 A（使用指数先验 + 硬约束）与指标
        y_rep = np.median(block_s, axis=0)
        w_rep = _weights_from_amp(y_rep) if use_weights else None
        # 代表谱上再求一次 κ 与 A（对应 ω_win）
        
        # 窗口代表谱上使用指数先验 + 硬约束来估计 A
        K_prior = 57.77
        beta_prior = 298.17
        A_margin_prior = 7.0
        A_win, sse_win = _fit_A_phi0_weighted(
            f_grid, y_rep, omega_win, w_rep,
            K=K_prior, beta=beta_prior, A_margin=A_margin_prior,
        )

        yhat = A_win * np.cos(omega_win * f_grid)
        rmse = float(np.sqrt(np.mean((yhat - y_rep) ** 2)))
        ss_res = float(np.sum((y_rep - yhat) ** 2))
        ss_tot = float(np.sum((y_rep - np.mean(y_rep)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        ripple_freq = omega_win / (2.0 * math.pi)

        elapsed = _time.perf_counter() - t_win0

        # 窗口中心时间
        center_idx = i0 + window_frames // 2
        t_center = frame_times[min(center_idx, len(frame_times) - 1)]

        results.append(
            FitResult(A=A_win, omega=omega_win, phi=0.0, ripple_freq=ripple_freq,
                      rmse=rmse, r2=r2, success=True, nit=total_evals, cost=sse_win, elapsed=elapsed)
        )
        t_stamps.append(t_center)

        # 该窗口 ω 作为下一窗口首段的初值
        prev_window_omega = omega_win

    # ---------- 保存结果到 CSV（每窗口一行） ----------
    out_csv = out_csv or "sine_fit_results.csv"
    try:
        fpath = out_csv
        with open(fpath, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "t_start_s", "A", "omega_rad_per_Hz", "phi_rad", "ripple_freq_cycles_per_Hz",
                "rmse", "r2", "success", "nit", "cost", "d_cm", "elapsed_s"
            ])
            for t0, fit in zip(t_stamps, results):
                # 路径差 (cm): d = c * 100 * omega / (4*pi)
                d_cm = c_speed * 100.0 * fit.omega / (4.0 * math.pi)
                writer.writerow([
                    f"{t0:.6f}", f"{fit.A:.9g}", f"{fit.omega:.9g}", f"{fit.phi:.9g}",
                    f"{fit.ripple_freq:.9g}", f"{fit.rmse:.9g}", f"{fit.r2:.6f}",
                    int(fit.success), fit.nit, f"{fit.cost:.9g}", f"{d_cm:.6f}", f"{fit.elapsed:.6f}"
                ])
    except PermissionError:
        ts = int(time.time())
        base = out_csv.rsplit('.', 1)[0]
        ext = ('.' + out_csv.rsplit('.', 1)[1]) if ('.' in out_csv) else ''
        fpath = f"{base}_{ts}{ext}"
        with open(fpath, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "t_start_s", "A", "omega_rad_per_Hz", "phi_rad", "ripple_freq_cycles_per_Hz",
                "rmse", "r2", "success", "nit", "cost", "d_cm", "elapsed_s"
            ])
            for t0, fit in zip(t_stamps, results):
                d_cm = c_speed * 100.0 * fit.omega / (4.0 * math.pi)
                writer.writerow([
                    f"{t0:.6f}", f"{fit.A:.9g}", f"{fit.omega:.9g}", f"{fit.phi:.9g}",
                    f"{fit.ripple_freq:.9g}", f"{fit.rmse:.9g}", f"{fit.r2:.6f}",
                    int(fit.success), fit.nit, f"{fit.cost:.9g}", f"{d_cm:.6f}", f"{fit.elapsed:.6f}"
                ])

    # 返回摘要（最后一个窗口）
    summary: Dict[str, float] = {}
    if results:
        last = results[-1]
        d_cm = c_speed * 100.0 * last.omega / (4.0 * math.pi)
        summary = {
            "A": last.A, "omega": last.omega, "phi": last.phi,
            "ripple_freq": last.ripple_freq, "rmse": last.rmse,
            "r2": last.r2, "d_cm": d_cm, "elapsed": last.elapsed,
        }

    return {
        "wav_path": wav_path,
        "fs": fs,
        "frames": len(results),            # 这里的“帧”指“窗口条目数”
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "band": [fmin, fmax],
        "n_points": n_points,
        "csv": out_csv,
        "last_fit": summary,
    }


# ============================ 主入口 ============================
def main() -> None:
    p = argparse.ArgumentParser(description="φ=0、A=K·exp(-beta·ω) 指数先验的余弦拟合 + 继承局部搜索 + 窗口均值 ω")
    p.add_argument("--wav", type=str, default=None, help="WAV 文件路径 (默认取当前目录第一个)")
    p.add_argument("--fmin", type=float, default=1000.0)
    p.add_argument("--fmax", type=float, default=5000.0)
    p.add_argument("--n_points", type=int, default=47)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--frame_ms", type=float, default=200.0)
    p.add_argument("--hop_ms", type=float, default=100.0)
    p.add_argument("--channel", type=int, default=None)
    p.add_argument("--no_weights", action="store_true")
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--print_last", action="store_true")
    # 网格/物理参数
    p.add_argument("--d_min_m", type=float, default=0.05, help="最小路径 (米)，用于 ω_min")
    p.add_argument("--c_speed", type=float, default=343.0, help="传播速度 (m/s)")
    p.add_argument("--oversample", type=float, default=6.0, help="Rayleigh 步长过采样因子 (>=1)")
    # 物理一致性新增参数
    # 已移除旧的物理一致性 CLI 参数

    args = p.parse_args()

    # 选择 WAV（保持与原始行为一致）
    wav_path = r"C:\Users\A\Desktop\workshop_10_7\第一版\src\white_noise_echo.wav" if args.wav is None else args.wav
    if wav_path is None:
        wavs = sorted(glob.glob("*.wav"))
        if not wavs:
            raise SystemExit("未指定 WAV 文件且目录下未找到 .wav")
        wav_path = wavs[0]

    meta = process_wav(
        wav_path=wav_path,
        fmin=args.fmin,
        fmax=args.fmax,
        n_points=args.n_points,
        sigma=args.sigma,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        channel=args.channel,
        use_weights=(not args.no_weights),
        out_csv=args.out_csv,
        d_min_m=args.d_min_m,
        c_speed=args.c_speed,
        oversample=args.oversample,
    # 旧的物理参数已移除
    )

    print(f"[done] {meta['frames']} 个窗口结果写出 @ fs={meta['fs']} Hz. CSV -> {meta['csv']}")
    if args.print_last and meta.get("last_fit"):
        print(json.dumps(meta["last_fit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
