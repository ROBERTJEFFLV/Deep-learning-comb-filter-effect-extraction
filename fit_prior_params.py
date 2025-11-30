#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_prior_params.py
---------------------------------
基于真实距离轨迹和声学测量数据，拟合先验参数 K 和 beta。

先验模型：
    A0(ω) = K * exp(-β * ω)
    
在实际使用中，幅度会被约束在：
    A ∈ [max(0, A0(ω) - Δ), A0(ω) + Δ]
    
本脚本的工作流程：
    1. 直接处理 WAV 文件：
       - STFT 变换
       - 提取 47 个选定频率 bin 的幅度
       - 计算相邻帧差分 d1
       - 高斯平滑
       - 计算 sum_abs_d1 作为幅度 A
    2. 读取真实距离轨迹 (range_1.csv)，包含 t_ref, d_ref_cm
    3. 将真实距离插值到每一帧时间戳，并转换为 omega_true
    4. 对 (omega_true, A) 进行对数线性拟合，得到 K_fit 和 beta_fit
    5. 根据残差分布计算合适的 prior_A_margin
    6. 生成新的参数建议，可直接更新到 config.py 的 LS_PARAMS 区段

参数关系：
    距离 d (cm) 与角频率 ω 的关系：
        ω = 4π * d / (c * 100)
        d = c * 100 * ω / (4π)
    
    其中 c = 343 m/s (声速)
    
    A 定义：
        送进 LS 前的 47 个 bin 的幅度总和 (sum_abs_d1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
import json
import librosa

from config import SR, N_FFT, HOP_LEN, FREQ_MIN, FREQ_MAX

# 声速 (m/s)
SPEED_OF_SOUND = 343.0

# ============ 配置区域 ============
CONFIG = {
    # 输入文件路径
    "wav_file": r"C:\Users\A\Desktop\workshop_10_7\第四版（临时）\src\rec_1.wav",
    "range_reference": r"C:\Users\A\Desktop\workshop_10_7\第四版（临时）\src\range_1.csv",
    
    # 输出路径
    "output_json": r"C:\Users\A\Desktop\workshop_10_7\第四版（临时）\src\fitted_prior_params.json",
    "plot_output": r"C:\Users\A\Desktop\workshop_10_7\第四版（临时）\src\prior_fit_analysis.png",
    
    # 音频处理参数
    "gaussian_sigma": 6.0,    # 高斯平滑参数
    
    # 数据筛选阈值
    "min_amplitude": 0.1,     # 最小 sum_abs_d1 阈值（平滑后的 47 bin 总幅度）
    
    # margin 计算
    "margin_percentile": 90,  # 使用残差的第 90 百分位数作为 margin
}


def distance_cm_to_omega(d_cm: float) -> float:
    """
    将距离 (cm) 转换为角频率 ω
    
    ω = 4π * d / (c * 100)
    """
    return 4 * np.pi * d_cm / (SPEED_OF_SOUND * 100)


def omega_to_distance_cm(omega: float) -> float:
    """
    将角频率 ω 转换为距离 (cm)
    
    d = c * 100 * ω / (4π)
    """
    return SPEED_OF_SOUND * 100 * omega / (4 * np.pi)


def process_wav_file(wav_path: str, gaussian_sigma: float = 6.0) -> pd.DataFrame:
    """
    处理 WAV 文件，提取每一帧的幅度信息
    
    处理流程（参考 AudioProcessor）：
    1. 加载音频
    2. STFT 变换
    3. 提取 47 个选定频率 bin 的幅度
    4. 计算相邻帧差分 d1
    5. 高斯平滑
    6. 计算 sum_abs_d1 作为幅度 A
    
    返回：DataFrame with columns [t, A]
    """
    print(f"[fit_prior] 处理 WAV 文件: {wav_path}")
    
    # 1. 加载音频
    samples, sr = librosa.load(wav_path, sr=SR)
    print(f"[fit_prior] 音频长度: {len(samples)/sr:.2f} 秒, 采样率: {sr} Hz")
    
    # 2. STFT
    S = librosa.stft(samples, n_fft=N_FFT, hop_length=HOP_LEN)
    mag = np.abs(S)  # 幅度谱
    
    # 3. 计算频率 bin 并提取选定范围的频率
    fft_freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    freq_idx = np.where((fft_freqs >= FREQ_MIN) & (fft_freqs <= FREQ_MAX))[0]
    selected_freqs = fft_freqs[freq_idx]
    print(f"[fit_prior] 提取 {len(freq_idx)} 个频率 bin (范围: {FREQ_MIN}-{FREQ_MAX} Hz)")
    
    # 4. 提取这些 bin 的幅度
    selected_mag = mag[freq_idx, :]  # shape: (n_bins, n_frames)
    
    # 5. 计算相邻帧差分 d1
    d1 = np.diff(selected_mag, axis=1)  # shape: (n_bins, n_frames-1)
    
    # 6. 对每个 bin 进行高斯平滑
    smoothed_d1 = np.zeros_like(d1)
    for i in range(d1.shape[0]):
        smoothed_d1[i, :] = gaussian_filter1d(d1[i, :], sigma=gaussian_sigma)
    
    # 7. 计算每一帧的 sum_abs_d1（所有 bin 的绝对值之和）
    sum_abs_d1 = np.sum(np.abs(smoothed_d1), axis=0)  # shape: (n_frames-1,)
    
    # 8. 构建 DataFrame
    n_frames = sum_abs_d1.shape[0]
    hop_sec = HOP_LEN / sr
    
    # 时间从第二帧开始（因为 d1 是差分）
    times = np.arange(1, n_frames + 1) * hop_sec
    
    df = pd.DataFrame({
        't': times,
        'A': sum_abs_d1
    })
    
    print(f"[fit_prior] 处理完成: {len(df)} 帧")
    print(f"[fit_prior] A 范围: [{df['A'].min():.4f}, {df['A'].max():.4f}]")
    
    return df


def load_offline_results(csv_path: str) -> pd.DataFrame:
    """
    [已弃用] 此函数不再使用，改为直接处理 WAV 文件
    """
    raise NotImplementedError("请使用 process_wav_file() 直接处理 WAV 文件")


def load_range_reference(csv_path: str) -> pd.DataFrame:
    """
    加载真实距离参考轨迹
    
    期望列：t (或 time), d_ref (或 distance_cm, range 等)
    单位假设：时间为秒，距离为厘米
    """
    print(f"[fit_prior] 加载参考距离: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 尝试识别时间列
    time_col = None
    for col in ['t', 'time', 'time_sec']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError("参考距离文件中找不到时间列 (t, time, timestamp)")
    
    # 尝试识别距离列
    dist_col = None
    for col in ['d_ref', 'distance_cm', 'range', 'dist', 'distance']:
        if col in df.columns:
            dist_col = col
            break
    
    if dist_col is None:
        raise ValueError("参考距离文件中找不到距离列 (d_ref, distance_cm, range, dist, distance)")
    
    # 统一列名
    df = df.rename(columns={time_col: 't_ref', dist_col: 'd_ref_cm'})
    
    print(f"[fit_prior] 参考轨迹有 {len(df)} 个点，时间范围: [{df['t_ref'].min():.2f}, {df['t_ref'].max():.2f}] 秒")
    print(f"[fit_prior] 距离范围: [{df['d_ref_cm'].min():.3f}, {df['d_ref_cm'].max():.3f}] 厘米")
    
    return df[['t_ref', 'd_ref_cm']]


def interpolate_reference_distance(offline_df: pd.DataFrame, 
                                   reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    将参考距离插值到离线结果的每一帧时间戳
    
    返回：添加了 d_true_cm 和 omega_true 列的 DataFrame
    """
    print(f"[fit_prior] 插值参考距离到每一帧...")
    
    # 创建插值函数
    interp_func = interp1d(
        reference_df['t_ref'], 
        reference_df['d_ref_cm'],  # 已经是厘米单位
        kind='linear',
        fill_value='extrapolate'
    )
    
    # 对每一帧插值
    df = offline_df.copy()
    df['d_true_cm'] = interp_func(df['t'])  # 直接使用厘米，不需要转换
    df['omega_true'] = df['d_true_cm'].apply(distance_cm_to_omega)
    
    print(f"[fit_prior] d_true_cm 范围: [{df['d_true_cm'].min():.2f}, {df['d_true_cm'].max():.2f}] 厘米")
    print(f"[fit_prior] omega_true 范围: [{df['omega_true'].min():.6f}, {df['omega_true'].max():.6f}]")
    
    return df


def filter_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    根据质量指标筛选数据
    """
    print(f"[fit_prior] 筛选数据...")
    print(f"[fit_prior] 原始数据: {len(df)} 帧")
    
    # 过滤条件
    mask = (
        (df['A'] >= config['min_amplitude']) &
        (df['omega_true'] > 0) &  # omega 必须为正
        (np.isfinite(df['A']))     # A 必须是有限值
    )
    
    filtered = df[mask].copy()
    
    print(f"[fit_prior] 筛选后: {len(filtered)} 帧 ({100*len(filtered)/len(df):.1f}%)")
    print(f"[fit_prior] A 范围: [{filtered['A'].min():.4f}, {filtered['A'].max():.4f}]")
    print(f"[fit_prior] omega_true 范围: [{filtered['omega_true'].min():.6f}, {filtered['omega_true'].max():.6f}]")
    
    return filtered


def fit_exponential_model(df: pd.DataFrame) -> tuple[float, float, dict]:
    """
    拟合指数模型 A = K * exp(-beta * omega)
    
    通过对数变换转为线性回归：
        ln(A) = ln(K) - beta * omega
        y = a + b * x
        
    其中：
        x = omega_true
        y = ln(A)
        a = ln(K) => K = exp(a)
        b = -beta => beta = -b
    
    返回：(K_fit, beta_fit, fit_stats)
    """
    print(f"[fit_prior] 拟合指数模型...")
    
    x = df['omega_true'].values
    y = np.log(df['A'].values)
    
    # 线性回归
    result = linregress(x, y)
    
    # 提取参数
    a = result.intercept
    b = result.slope
    
    K_fit = np.exp(a)
    beta_fit = -b
    
    # 统计信息
    fit_stats = {
        'K': float(K_fit),
        'beta': float(beta_fit),
        'r_squared': float(result.rvalue ** 2),
        'p_value': float(result.pvalue),
        'stderr': float(result.stderr),
        'n_samples': len(df)
    }
    
    print(f"[fit_prior] 拟合结果:")
    print(f"[fit_prior]   K = {K_fit:.4f}")
    print(f"[fit_prior]   beta = {beta_fit:.4f}")
    print(f"[fit_prior]   R² = {fit_stats['r_squared']:.4f}")
    print(f"[fit_prior]   样本数 = {fit_stats['n_samples']}")
    
    return K_fit, beta_fit, fit_stats


def calculate_margin(df: pd.DataFrame, K: float, beta: float, percentile: int) -> float:
    """
    根据残差分布计算合适的 prior_A_margin
    
    A0(ω) = K * exp(-beta * ω)
    残差 r = |A - A0(omega_true)|
    margin = percentile(r)
    """
    print(f"[fit_prior] 计算 margin (使用第 {percentile} 百分位数)...")
    
    # 计算先验预测值
    A0_pred = K * np.exp(-beta * df['omega_true'])
    
    # 计算残差
    residuals = np.abs(df['A'] - A0_pred)
    
    # 计算百分位数
    margin = np.percentile(residuals, percentile)
    
    print(f"[fit_prior] 残差统计:")
    print(f"[fit_prior]   最小值: {residuals.min():.4f}")
    print(f"[fit_prior]   中位数: {np.median(residuals):.4f}")
    print(f"[fit_prior]   平均值: {residuals.mean():.4f}")
    print(f"[fit_prior]   第{percentile}百分位: {margin:.4f}")
    print(f"[fit_prior]   最大值: {residuals.max():.4f}")
    
    return float(margin)


def plot_analysis(df: pd.DataFrame, K: float, beta: float, margin: float, 
                 output_path: str):
    """
    生成分析图表
    """
    print(f"[fit_prior] 生成分析图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 原始数据散点图 (omega_true vs A)
    ax1 = axes[0, 0]
    ax1.scatter(df['omega_true'], df['A'], alpha=0.3, s=10, label='实测数据')
    
    # 拟合曲线
    omega_range = np.linspace(df['omega_true'].min(), df['omega_true'].max(), 200)
    A_fit = K * np.exp(-beta * omega_range)
    A_upper = A_fit + margin
    A_lower = np.maximum(0, A_fit - margin)
    
    ax1.plot(omega_range, A_fit, 'r-', linewidth=2, label=f'拟合: A = {K:.2f} * exp(-{beta:.2f} * ω)')
    ax1.fill_between(omega_range, A_lower, A_upper, alpha=0.2, color='red', 
                     label=f'±{margin:.2f} margin')
    
    ax1.set_xlabel('ω (角频率)')
    ax1.set_ylabel('A (幅度)')
    ax1.set_title('幅度-角频率关系')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 对数坐标 (ln(A) vs omega_true)
    ax2 = axes[0, 1]
    ax2.scatter(df['omega_true'], np.log(df['A']), alpha=0.3, s=10, label='ln(A) 数据')
    ax2.plot(omega_range, np.log(K) - beta * omega_range, 'r-', linewidth=2, 
             label=f'线性拟合: ln(A) = {np.log(K):.2f} - {beta:.2f} * ω')
    ax2.set_xlabel('ω (角频率)')
    ax2.set_ylabel('ln(A)')
    ax2.set_title('对数线性拟合')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 残差分布
    ax3 = axes[1, 0]
    A0_pred = K * np.exp(-beta * df['omega_true'])
    residuals = df['A'] - A0_pred
    
    ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='r', linestyle='--', linewidth=2, label='零残差')
    ax3.axvline(margin, color='g', linestyle='--', linewidth=2, label=f'+margin = {margin:.2f}')
    ax3.axvline(-margin, color='g', linestyle='--', linewidth=2, label=f'-margin = {margin:.2f}')
    ax3.set_xlabel('残差 (A - A0)')
    ax3.set_ylabel('频数')
    ax3.set_title('残差分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 残差 vs omega_true
    ax4 = axes[1, 1]
    ax4.scatter(df['omega_true'], residuals, alpha=0.3, s=10)
    ax4.axhline(0, color='r', linestyle='--', linewidth=2)
    ax4.axhline(margin, color='g', linestyle='--', linewidth=2, label=f'±margin = {margin:.2f}')
    ax4.axhline(-margin, color='g', linestyle='--', linewidth=2)
    ax4.set_xlabel('ω (角频率)')
    ax4.set_ylabel('残差 (A - A0)')
    ax4.set_title('残差 vs 角频率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[fit_prior] 图表已保存: {output_path}")
    plt.close()


def save_fitted_params(K: float, beta: float, margin: float, 
                       fit_stats: dict, output_path: str):
    """
    保存拟合结果到 JSON 文件
    """
    print(f"[fit_prior] 保存拟合参数到: {output_path}")
    
    output = {
        "description": "基于实验数据拟合的先验参数",
        "fitted_prior_params": {
            "prior_K": K,
            "prior_beta": beta,
            "prior_A_margin": margin
        },
        "fit_statistics": fit_stats,
        "usage": "将这些参数复制到 config.py 的 LS_PARAMS 中"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[fit_prior] 参数已保存")
    print(f"[fit_prior] ")
    print(f"[fit_prior] ========== 建议更新 config.py -> LS_PARAMS ==========")
    print(f"[fit_prior] \"prior_K\": {K:.4f},")
    print(f"[fit_prior] \"prior_beta\": {beta:.4f},")
    print(f"[fit_prior] \"prior_A_margin\": {margin:.4f}")
    print(f"[fit_prior] ===================================================")


def main():
    """
    主流程
    """
    print("=" * 60)
    print("先验参数拟合工具 (直接处理 WAV 文件)")
    print("=" * 60)
    
    try:
        # 1. 处理 WAV 文件，提取每一帧的幅度
        print("\n[步骤 1/7] 处理 WAV 文件...")
        wav_df = process_wav_file(CONFIG['wav_file'], CONFIG['gaussian_sigma'])
        
        # 2. 加载参考距离
        print("\n[步骤 2/7] 加载参考距离...")
        reference_df = load_range_reference(CONFIG['range_reference'])
        
        # 3. 插值真实距离
        print("\n[步骤 3/7] 插值参考距离到每一帧...")
        merged_df = interpolate_reference_distance(wav_df, reference_df)
        
        # 4. 筛选数据
        print("\n[步骤 4/7] 筛选数据...")
        filtered_df = filter_data(merged_df, CONFIG)
        
        if len(filtered_df) < 10:
            print(f"[fit_prior] 错误: 筛选后样本过少 ({len(filtered_df)} < 10)，无法拟合")
            print(f"[fit_prior] 建议放宽筛选条件（min_amplitude）")
            return
        
        # 5. 拟合指数模型
        print("\n[步骤 5/7] 拟合指数模型...")
        K_fit, beta_fit, fit_stats = fit_exponential_model(filtered_df)
        
        # 6. 计算 margin
        print("\n[步骤 6/7] 计算 margin...")
        margin_fit = calculate_margin(filtered_df, K_fit, beta_fit, 
                                      CONFIG['margin_percentile'])
        
        # 7. 保存结果
        print("\n[步骤 7/7] 保存结果...")
        save_fitted_params(K_fit, beta_fit, margin_fit, fit_stats, 
                          CONFIG['output_json'])
        
        # 8. 生成图表
        plot_analysis(filtered_df, K_fit, beta_fit, margin_fit, 
                     CONFIG['plot_output'])
        
        print("")
        print("=" * 60)
        print("拟合完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"[fit_prior] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
