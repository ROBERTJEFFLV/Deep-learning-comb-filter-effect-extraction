# src/processing/comb_shift.py

import numpy as np
from typing import Tuple, List


def estimate_comb_filter_shift(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_lag: int = 12,
    rho_thresh: float = 0.6
) -> Tuple[int, float]:
    """
    估计 frame2 相对 frame1 的频移（按 bin 计数）。
    - 先用互相关在 [-max_lag, +max_lag] 搜索最佳 lag；
    - 再在“与该 lag 对齐的重叠区”计算 Pearson 相关系数 rho；
    - 若 |rho| < rho_thresh，则返回 (0, rho) 视为无效移动。
    """
    assert frame1.ndim == 1 and frame2.ndim == 1 and frame1.shape == frame2.shape
    N = frame1.shape[0]

    # 互相关（全相关）并在窗口内找峰
    corr = np.correlate(frame1, frame2, mode='full')
    center = N - 1
    start = center - max_lag
    end   = center + max_lag + 1
    subcorr = corr[start:end]
    rel_idx = int(np.argmax(subcorr))
    lag = rel_idx - max_lag

    # 只在重叠区计算 Pearson（避免 np.roll 的环绕污染）
    if lag > 0:
        x = frame1[0:N-lag]
        y = frame2[lag:N]
    elif lag < 0:
        k = -lag
        x = frame1[k:N]
        y = frame2[0:N-k]
    else:
        x = frame1
        y = frame2

    xm = x.mean()
    ym = y.mean()
    num = float(np.sum((x - xm) * (y - ym)))
    den = float(np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2)))
    rho = num / den if den > 0 else 0.0

    if abs(rho) < rho_thresh:
        return 0, rho
    return lag, rho


def detect_zero_crossings(x: np.ndarray) -> List[int]:
    """
    返回相邻元素符号改变的位置索引（i 表示 i 与 i+1 之间过零）。
    若 x[i] 恰为 0 也记为过零点。
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("detect_zero_crossings: 输入必须是一维数组")

    zero_pts = list(np.where(x == 0)[0])
    s = np.sign(x)
    sign_changes = np.where(s[:-1] * s[1:] < 0)[0]
    crossings = sorted(set(zero_pts + sign_changes.tolist()))
    return crossings


def detect_comb_direction(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_lag: int = 12,
    rho_thresh: float = 0.6
):
    """
    判断 frame2 相对 frame1 的整体平移方向，并在“对齐后的 new 片段”上取零交叉。
    返回: (direction, zeros)
      direction ∈ {"Left","Right","None"}
      zeros: 对齐后的 new 片段的零交叉索引列表
    """
    lag, rho = estimate_comb_filter_shift(frame1, frame2, max_lag, rho_thresh)

    # 对齐到重叠片段，用于更稳健地计算 zeros
    N = len(frame1)
    if lag > 0:
        y_aligned = frame2[lag:N]
    elif lag < 0:
        y_aligned = frame2[0:N+lag]  # lag 为负
    else:
        y_aligned = frame2

    zeros = detect_zero_crossings(y_aligned)

    if abs(rho) < rho_thresh or lag == 0:
        return "None", zeros

    # 约定：lag>0 → frame2 向右（高频）→ "Right"；lag<0 → "Left"
    direction = "Left" if lag > 0 else "Right"
    return direction, zeros


def average_zero_crossing_freq_spacing(
    zeros: List[int],
    freqs: np.ndarray
) -> float:
    """
    计算相邻零点频率间隔的平均值，并换算为距离（cm）。
    若零点少于2个，返回 0.0。
    """
    if len(zeros) < 2:
        return 0.0

    zero_freqs = freqs[np.array(zeros)]
    diffs = np.diff(zero_freqs)
    diffs_avg = float(np.mean(diffs))
    if diffs_avg <= 0:
        return 0.0

    time_delay = 1.0 / diffs_avg
    distance_cm = 340.0 * 100.0 * time_delay / 2.0
    return distance_cm


def average_zero_crossing_freq_spacing_mixed(
    zeros1: List[int],
    zeros2: List[int],
    freqs: np.ndarray,
    quarter_step: bool = True
) -> float:
    """
    d1 与 d2 的零点融合后估距：
    - 将 zeros1 与 zeros2 合并、去重、按频率从小到大排序；
    - 计算相邻零点频率差的平均值 diffs_avg；
    - 若 quarter_step=True，认为相邻两点对应 1/4 波长步距（Δf = 1/(4T)）；
      => T = 1 / (4 * diffs_avg)
    - 距离 = c * T / 2（来回程折半），单位 cm。
    - 零点少于 2 个返回 0.0。
    """
    merged = sorted(set(list(zeros1) + list(zeros2)))
    if len(merged) < 2:
        return 0.0

    zero_freqs = freqs[np.array(merged)]
    diffs = np.diff(zero_freqs)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 0.0

    diffs_avg = float(np.mean(diffs))
    if diffs_avg <= 0:
        return 0.0

    # comb 理论：谱间距 Δf = 1/T
    # 若采用 1/4 波长步距（d1 与 d2 间隔），Δf_mixed = 1/(4T)
    if quarter_step:
        time_delay = 1.0 / (4.0 * diffs_avg)
    else:
        time_delay = 1.0 / diffs_avg

    distance_cm = 340.0 * 100.0 * time_delay / 2.0
    return distance_cm
