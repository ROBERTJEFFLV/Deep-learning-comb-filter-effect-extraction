#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两个 CSV 测量结果的脚本

固定设定：
- 基线: range_1.csv
    - 时间列: time_sec
    - 距离列: distance_cm  (单位: cm)
- 测量: sine_fit_log.csv
    - 时间列: t
    - 距离列: distance_kf  (单位: cm)

功能：
1. 对测量数据时间轴加一个固定延时 DELAY (秒)：
      t_est_aligned = t_est_raw + DELAY
2. 在 t_est_aligned 上用线性插值取得基线真值 y_true
3. 只保留 ref / est 在同一时间点都有数据的样本
4. 计算：
   - 全局误差指标 (MAE, RMSE, bias, median |err|, P95, max|err|)
   - 按真值分段的局部指标 (基于分位数)
   - 绝对误差尾部 1% 指标
   - 误差最大点附近一个时间窗口 (默认 ±0.5s) 的局部指标
   - 每个时间窗口的单独指标分析和汇总可视化
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号


# ==================== 固定配置 ====================

# 文件与列名
REF_CSV = "range_1.csv"
EST_CSV = "sine_fit_log.csv"

REF_TIME_COL = "time_sec"
REF_VALUE_COL = "distance_cm"

EST_TIME_COL = "t"
EST_VALUE_COL = "distance_kf"

# 测量数据加的时间延时 (秒)，正值为加时间，负值为减时间
DELAY = -0.5

# 单位换算：原始数据单位 * UNIT_SCALE = 输出单位
# 当前两边都是 cm，所以 UNIT_SCALE = 1
UNIT_SCALE = 1.0
UNIT_NAME = "cm"

# 最大误差点附近局部窗口半宽 (秒)，默认 ±0.5s
LOCAL_WINDOW = 0.5

# 时间窗口分割参数
GAP_THRESHOLD_FACTOR = 5.0  # 多少倍标准差作为断点阈值
MIN_WINDOW_SIZE = 50        # 最小窗口样本数

# 基线数据过滤条件
REF_DISTANCE_THRESHOLD = 12.0  # 只分析基线距离小于此值(cm)的数据


# ==================== 时间窗口分割函数 ====================

def find_time_windows(times: np.ndarray, gap_threshold_factor: float = 5.0) -> list:
    """
    根据时间间隔找出连续的时间窗口
    
    Args:
        times: 排序后的时间数组
        gap_threshold_factor: 多少倍标准差作为断点阈值
        
    Returns:
        list of tuples: [(start_idx, end_idx), ...] 每个窗口的起始和结束索引
    """
    if len(times) <= 1:
        return [(0, len(times))]
    
    time_diffs = np.diff(times)
    mean_diff = np.mean(time_diffs)
    std_diff = np.std(time_diffs)
    threshold = mean_diff + gap_threshold_factor * std_diff
    
    # 找出大间隔的位置
    large_gaps = time_diffs > threshold
    gap_indices = np.where(large_gaps)[0]
    
    # 构建窗口边界
    windows = []
    start_idx = 0
    
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1  # gap_idx 是间隔前的最后一个点
        if end_idx > start_idx:
            windows.append((start_idx, end_idx))
        start_idx = end_idx
    
    # 添加最后一个窗口
    if start_idx < len(times):
        windows.append((start_idx, len(times)))
    
    return windows


def analyze_time_window(t_window, y_true_window, y_est_window, window_idx: int) -> dict:
    """
    分析单个时间窗口的指标
    
    Args:
        t_window: 窗口内的时间数组
        y_true_window: 窗口内的真值数组  
        y_est_window: 窗口内的估计值数组
        window_idx: 窗口索引
        
    Returns:
        dict: 包含窗口分析结果的字典
    """
    err_window = y_est_window - y_true_window
    stats = compute_basic_stats(err_window)
    
    if stats is None:
        return None
    
    window_info = {
        'window_idx': window_idx,
        'time_start': float(np.min(t_window)),
        'time_end': float(np.max(t_window)),
        'duration': float(np.max(t_window) - np.min(t_window)),
        'n_samples': len(t_window),
        'true_range': (float(np.min(y_true_window)), float(np.max(y_true_window))),
        'est_range': (float(np.min(y_est_window)), float(np.max(y_est_window))),
        'stats': stats
    }
    
    return window_info


# ==================== 基本统计函数 ====================

def compute_basic_stats(err: np.ndarray) -> dict | None:
    """给定误差数组 err，返回一组基础统计量。"""
    err = np.asarray(err, dtype=float)
    if err.size == 0:
        return None

    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    med_err = float(np.median(err))  # 新增：中位数误差(带正负号)
    med_ae = float(np.median(abs_err))
    p95 = float(np.percentile(abs_err, 95))
    max_abs = float(np.max(abs_err))
    idx_max = int(np.argmax(abs_err))

    return {
        "n": int(err.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "med_err": med_err,
        "med_abs_err": med_ae,
        "p95": p95,
        "max_abs_err": max_abs,
        "idx_max_abs": idx_max,
    }


def print_stats_block(title: str, stats: dict | None, unit_name: str) -> None:
    """统一格式打印一块统计结果。"""
    if not stats or stats["n"] == 0:
        print(f"{title}: 样本数为 0，无法统计。")
        return

    print(title)
    print(f"  样本数: {stats['n']}")
    print(
        f"  MAE: {stats['mae']:.3f} {unit_name}  |  "
        f"RMSE: {stats['rmse']:.3f} {unit_name}  |  "
        f"偏置(bias): {stats['bias']:.3f} {unit_name}"
    )
    print(
        f"  中位数误差: {stats['med_err']:.3f} {unit_name}  |  "
        f"中位绝对误差: {stats['med_abs_err']:.3f} {unit_name}  |  "
        f"P95: {stats['p95']:.3f} {unit_name}"
    )
    print(
        f"  最大绝对误差: {stats['max_abs_err']:.3f} {unit_name}"
    )


def create_window_comparison_plots(window_results: list, unit_name: str) -> None:
    """
    创建时间窗口对比分析的可视化图表
    
    Args:
        window_results: 各个窗口的分析结果列表
        unit_name: 单位名称
    """
    if not window_results:
        print("没有有效的窗口数据，无法创建图表。")
        return
    
    # 提取数据
    window_indices = [w['window_idx'] for w in window_results]
    durations = [w['duration'] for w in window_results]
    n_samples = [w['n_samples'] for w in window_results]
    mae_values = [w['stats']['mae'] for w in window_results]
    rmse_values = [w['stats']['rmse'] for w in window_results]
    bias_values = [w['stats']['bias'] for w in window_results]
    med_err_values = [w['stats']['med_err'] for w in window_results]  # 新增：中位数误差
    med_abs_err = [w['stats']['med_abs_err'] for w in window_results]
    p95_values = [w['stats']['p95'] for w in window_results]
    max_abs_err = [w['stats']['max_abs_err'] for w in window_results]
    
    # 创建子图 - 改为3x2布局以容纳中位数误差图
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'时间窗口误差指标对比分析 (基线距离<{REF_DISTANCE_THRESHOLD}cm)', fontsize=16, fontweight='bold')
    
    # 1. MAE 对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(window_indices, mae_values, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title('平均绝对误差 (MAE)')
    ax1.set_xlabel('时间窗口编号')
    ax1.set_ylabel(f'MAE ({unit_name})')
    ax1.grid(True, alpha=0.3)
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, mae_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. RMSE 对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(window_indices, rmse_values, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_title('均方根误差 (RMSE)')
    ax2.set_xlabel('时间窗口编号')
    ax2.set_ylabel(f'RMSE ({unit_name})')
    ax2.grid(True, alpha=0.3)
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 偏置 (Bias) 对比
    ax3 = axes[1, 0]
    colors = ['red' if b < 0 else 'green' for b in bias_values]
    bars3 = ax3.bar(window_indices, bias_values, alpha=0.7, color=colors, edgecolor='black')
    ax3.set_title('系统偏置 (Bias)')
    ax3.set_xlabel('时间窗口编号')
    ax3.set_ylabel(f'Bias ({unit_name})')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(True, alpha=0.3)
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars3, bias_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (max(bias_values) - min(bias_values))*0.02 if val >= 0 
                else bar.get_height() - (max(bias_values) - min(bias_values))*0.02, 
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    
    # 4. 中位数误差 对比 - 新增
    ax4 = axes[1, 1]
    colors4 = ['red' if m < 0 else 'green' for m in med_err_values]
    bars4 = ax4.bar(window_indices, med_err_values, alpha=0.7, color=colors4, edgecolor='black')
    ax4.set_title('中位数误差 (Median Error)')
    ax4.set_xlabel('时间窗口编号')
    ax4.set_ylabel(f'中位数误差 ({unit_name})')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(True, alpha=0.3)
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars4, med_err_values)):
        ax4.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (max(med_err_values) - min(med_err_values))*0.02 if val >= 0 
                else bar.get_height() - (max(med_err_values) - min(med_err_values))*0.02, 
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    
    # 5. P95 对比
    ax5 = axes[2, 0]
    bars5 = ax5.bar(window_indices, p95_values, alpha=0.7, color='gold', edgecolor='orange')
    ax5.set_title('95% 分位数误差 (P95)')
    ax5.set_xlabel('时间窗口编号')
    ax5.set_ylabel(f'P95 ({unit_name})')
    ax5.grid(True, alpha=0.3)
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars5, p95_values)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p95_values)*0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 6. 综合误差分布趋势
    ax6 = axes[2, 1]
    
    # 绘制多个误差指标的趋势线
    ax6.plot(window_indices, mae_values, 'o-', label='MAE', linewidth=2, markersize=6)
    ax6.plot(window_indices, rmse_values, 's-', label='RMSE', linewidth=2, markersize=6)
    ax6.plot(window_indices, med_abs_err, '^-', label='中位绝对误差', linewidth=2, markersize=6)
    ax6.plot(window_indices, med_err_values, 'd-', label='中位数误差', linewidth=2, markersize=6)
    
    ax6.set_title('误差指标趋势对比')
    ax6.set_xlabel('时间窗口编号')
    ax6.set_ylabel(f'误差 ({unit_name})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('window_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建误差分布统计表
    create_window_summary_table(window_results, unit_name)


def create_window_summary_table(window_results: list, unit_name: str) -> None:
    """
    创建时间窗口汇总统计表
    
    Args:
        window_results: 各个窗口的分析结果列表
        unit_name: 单位名称
    """
    # 创建汇总统计表
    summary_data = []
    for w in window_results:
        summary_data.append({
            '窗口': w['window_idx'],
            '开始时间(s)': f"{w['time_start']:.1f}",
            '结束时间(s)': f"{w['time_end']:.1f}",
            '持续时间(s)': f"{w['duration']:.1f}",
            '样本数': w['n_samples'],
            f'MAE({unit_name})': f"{w['stats']['mae']:.3f}",
            f'RMSE({unit_name})': f"{w['stats']['rmse']:.3f}",
            f'Bias({unit_name})': f"{w['stats']['bias']:.3f}",
            f'中位数误差({unit_name})': f"{w['stats']['med_err']:.3f}",
            f'中位绝对误差({unit_name})': f"{w['stats']['med_abs_err']:.3f}",
            f'P95({unit_name})': f"{w['stats']['p95']:.3f}",
            f'最大误差({unit_name})': f"{w['stats']['max_abs_err']:.3f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # 保存到 CSV
    df_summary.to_csv('window_analysis_summary.csv', index=False, encoding='utf-8-sig')
    
    # 打印汇总表
    print("\n" + "="*120)
    print("时间窗口分析汇总表")
    print("="*120)
    print(df_summary.to_string(index=False))
    print("="*120)
    
    # 计算整体统计
    all_mae = [w['stats']['mae'] for w in window_results]
    all_rmse = [w['stats']['rmse'] for w in window_results]
    all_bias = [w['stats']['bias'] for w in window_results]
    all_med_err = [w['stats']['med_err'] for w in window_results]  # 新增：中位数误差统计
    
    print(f"\n各窗口指标统计:")
    print(f"  MAE     - 最小: {min(all_mae):.3f}, 最大: {max(all_mae):.3f}, 平均: {np.mean(all_mae):.3f}, 标准差: {np.std(all_mae):.3f}")
    print(f"  RMSE    - 最小: {min(all_rmse):.3f}, 最大: {max(all_rmse):.3f}, 平均: {np.mean(all_rmse):.3f}, 标准差: {np.std(all_rmse):.3f}")
    print(f"  Bias    - 最小: {min(all_bias):.3f}, 最大: {max(all_bias):.3f}, 平均: {np.mean(all_bias):.3f}, 标准差: {np.std(all_bias):.3f}")
    print(f"  中位数误差 - 最小: {min(all_med_err):.3f}, 最大: {max(all_med_err):.3f}, 平均: {np.mean(all_med_err):.3f}, 标准差: {np.std(all_med_err):.3f}")
    print(f"\n分析结果已保存到: window_analysis_summary.csv")
    print(f"可视化图表已保存到: window_comparison_analysis.png")


# ==================== 主流程 ====================

def main():
    # 1. 读取 CSV
    df_ref = pd.read_csv(REF_CSV)
    df_est = pd.read_csv(EST_CSV)

    # 2. 抽取需要的列，并转换为 float
    t_ref = df_ref[REF_TIME_COL].astype(float).to_numpy()
    y_ref = df_ref[REF_VALUE_COL].astype(float).to_numpy()

    # 2.1 新增：过滤基线数据，只保留距离小于阈值的数据点
    ref_filter_mask = y_ref < REF_DISTANCE_THRESHOLD
    t_ref = t_ref[ref_filter_mask]
    y_ref = y_ref[ref_filter_mask]
    
    print(f"[filter] 基线数据过滤: 原始{len(df_ref)}个点 -> 距离<{REF_DISTANCE_THRESHOLD}cm的{len(t_ref)}个点")
    
    if len(t_ref) == 0:
        print(f"过滤后没有基线数据满足条件（距离<{REF_DISTANCE_THRESHOLD}cm），请检查阈值设置。")
        return

    t_est_raw = df_est[EST_TIME_COL].astype(float).to_numpy()
    y_est = df_est[EST_VALUE_COL].astype(float).to_numpy()

    # 3. 过滤掉无效的测量数据（距离为0或NaN的数据）
    valid_mask = np.isfinite(y_est) & (y_est != 0)
    t_est_raw = t_est_raw[valid_mask]
    y_est = y_est[valid_mask]

    if len(t_est_raw) == 0:
        print("没有有效的测量数据，请检查输入文件。")
        return

    # 4. 排序 (确保时间单调递增)
    sort_idx_ref = np.argsort(t_ref)
    t_ref = t_ref[sort_idx_ref]
    y_ref = y_ref[sort_idx_ref]

    sort_idx_est = np.argsort(t_est_raw)
    t_est_raw = t_est_raw[sort_idx_est]
    y_est = y_est[sort_idx_est]

    # 5. 对测量时间加延时
    t_est_aligned = t_est_raw + DELAY

    # 6. 仅保留那些落在 ref 时间范围内的测量点
    t_min_ref = float(np.min(t_ref))
    t_max_ref = float(np.max(t_ref))
    mask_in = (t_est_aligned >= t_min_ref) & (t_est_aligned <= t_max_ref)

    t_est_aligned = t_est_aligned[mask_in]
    t_est_raw = t_est_raw[mask_in]
    y_est = y_est[mask_in]

    if t_est_aligned.size == 0:
        print("对齐后没有任何测量点落在基线时间范围内，请检查 DELAY 和时间轴。")
        return

    # 7. 用线性插值在 t_est_aligned 上取得基线真值
    # 重要：只在基线数据真正存在的连续区域进行插值
    f_ref = interp1d(
        t_ref,
        y_ref,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    y_true_interp = f_ref(t_est_aligned)

    # 8. 删除插值无效的点 (NaN)
    valid_mask = np.isfinite(y_true_interp) & np.isfinite(y_est)
    t_est_aligned = t_est_aligned[valid_mask]
    t_est_raw = t_est_raw[valid_mask]
    y_est = y_est[valid_mask]
    y_true_interp = y_true_interp[valid_mask]

    # 8.1 新增：严格过滤，确保插值后的真值满足距离条件
    # 并且确保每个测量点附近确实有基线数据支撑
    final_filter_mask = y_true_interp < REF_DISTANCE_THRESHOLD
    
    # 额外检查：确保每个测量点附近(±1秒内)有真实的基线数据点
    baseline_support_mask = np.zeros(len(t_est_aligned), dtype=bool)
    for i, t_meas in enumerate(t_est_aligned):
        # 检查测量时间点附近是否有基线数据
        nearby_baseline = np.any((np.abs(t_ref - t_meas) <= 1.0))
        baseline_support_mask[i] = nearby_baseline
    
    # 结合两个条件
    final_filter_mask = final_filter_mask & baseline_support_mask
    
    t_est_aligned = t_est_aligned[final_filter_mask]
    t_est_raw = t_est_raw[final_filter_mask]
    y_est = y_est[final_filter_mask]
    y_true_interp = y_true_interp[final_filter_mask]
    
    filtered_out_count = len(valid_mask) - np.sum(final_filter_mask)
    print(f"[filter] 严格过滤: 过滤掉{filtered_out_count}个缺乏基线支撑的点")
    print(f"[filter] 最终保留真值<{REF_DISTANCE_THRESHOLD}cm且有基线支撑的{len(t_est_aligned)}个测量点")

    if t_est_aligned.size == 0:
        print("过滤后没有有效样本，请检查输入数据和过滤条件。")
        return

    # 9. 换算单位 (例如 m -> cm；当前为 1:1)
    y_true_u = y_true_interp * UNIT_SCALE
    y_est_u = y_est * UNIT_SCALE
    err_u = y_est_u - y_true_u  # 误差: est - true

    # ============ 新增：时间窗口分析 ============
    print(f"[compare] 开始时间窗口分析...")
    
    # 10. 找出时间窗口
    windows = find_time_windows(t_est_aligned, GAP_THRESHOLD_FACTOR)
    print(f"  发现 {len(windows)} 个时间窗口")
    
    # 11. 分析每个时间窗口
    window_results = []
    for i, (start_idx, end_idx) in enumerate(windows):
        if end_idx - start_idx < MIN_WINDOW_SIZE:
            print(f"  跳过窗口 {i+1}: 样本数太少 ({end_idx - start_idx} < {MIN_WINDOW_SIZE})")
            continue
        
        t_window = t_est_aligned[start_idx:end_idx]
        y_true_window = y_true_u[start_idx:end_idx]
        y_est_window = y_est_u[start_idx:end_idx]
        
        window_info = analyze_time_window(t_window, y_true_window, y_est_window, i+1)
        if window_info:
            window_results.append(window_info)
            print(f"  窗口 {i+1}: t={window_info['time_start']:.1f}~{window_info['time_end']:.1f}s, "
                  f"样本数={window_info['n_samples']}, MAE={window_info['stats']['mae']:.3f}{UNIT_NAME}")
    
    # 12. 创建可视化分析
    if window_results:
        create_window_comparison_plots(window_results, UNIT_NAME)
    else:
        print("  没有足够的有效窗口进行分析")

    # ============ 原有的全局分析 ============
    stats_global = compute_basic_stats(err_u)

    print(
        f"\n[compare] 全局对比 {REF_CSV} (过滤条件: 距离<{REF_DISTANCE_THRESHOLD}cm)"
        f"\n  (测量文件: {EST_CSV}, 估计延迟 {DELAY:.3f} s)"
    )
    print(
        f"  对齐样本数: {stats_global['n']}  |  "
        f"时间范围: {t_est_aligned[0]:.3f} s ~ {t_est_aligned[-1]:.3f} s"
    )
    print(
        f"  真值范围: {y_true_u.min():.3f} ~ {y_true_u.max():.3f} {UNIT_NAME} "
        f"(所有数据都<{REF_DISTANCE_THRESHOLD}{UNIT_NAME})"
    )
    print_stats_block("  全局指标:", stats_global, UNIT_NAME)
    print()

    # 13. 按真值分段 (基于分位数 0, 33, 66, 100)
    true_vals = y_true_u
    qs = np.percentile(true_vals, [0, 33, 66, 100])
    segments = []
    for i in range(len(qs) - 1):
        lo = qs[i]
        hi = qs[i + 1]
        if i < len(qs) - 2:
            mask_seg = (true_vals >= lo) & (true_vals < hi)
        else:
            mask_seg = (true_vals >= lo) & (true_vals <= hi)
        segments.append((lo, hi, mask_seg))

    print("  按真值分段的局部指标 (基于分位数):")
    for (lo, hi, m_seg) in segments:
        if np.sum(m_seg) < 20:
            # 样本太少就略过，避免无意义统计
            continue
        stats_seg = compute_basic_stats(err_u[m_seg])
        if not stats_seg:
            continue
        print_stats_block(
            f"    真值区间 [{lo:.3f}, {hi:.3f}] {UNIT_NAME}:",
            stats_seg,
            UNIT_NAME,
        )
        print()

    # 14. 绝对误差尾部 1% (最差 1% 样本)
    abs_err = np.abs(err_u)
    thr_tail = np.percentile(abs_err, 99.0)
    mask_tail = abs_err >= thr_tail
    if np.sum(mask_tail) > 0:
        stats_tail = compute_basic_stats(err_u[mask_tail])
        print_stats_block("  绝对误差尾部 1% (最差部分):", stats_tail, UNIT_NAME)
        print()
    else:
        print("  绝对误差尾部 1%: 样本数为 0 (数据太少或误差过小)。")
        print()

    # 15. 最大误差点附近局部窗口 (±LOCAL_WINDOW 秒)
    idx_max = stats_global["idx_max_abs"]
    t_true_star = t_est_aligned[idx_max]
    t_est_raw_star = t_est_raw[idx_max]
    est_star = y_est_u[idx_max]
    true_star = y_true_u[idx_max]
    err_star = err_u[idx_max]

    mask_local = (
        (t_est_aligned >= t_true_star - LOCAL_WINDOW)
        & (t_est_aligned <= t_true_star + LOCAL_WINDOW)
    )
    stats_local = compute_basic_stats(err_u[mask_local])

    print_stats_block(
        f"  最大误差点附近局部窗口 (t_true≈{t_true_star:.3f}s, "
        f"窗口半宽 {LOCAL_WINDOW:.3f}s):",
        stats_local,
        UNIT_NAME,
    )
    print("  最大误差样本详细信息:")
    print(
        f"    t_est_raw = {t_est_raw_star:.3f} s, "
        f"t_true = {t_true_star:.3f} s, "
        f"est = {est_star:.3f} {UNIT_NAME}, "
        f"true = {true_star:.3f} {UNIT_NAME}, "
        f"err = {err_star:.3f} {UNIT_NAME}"
    )


if __name__ == "__main__":
    main()
