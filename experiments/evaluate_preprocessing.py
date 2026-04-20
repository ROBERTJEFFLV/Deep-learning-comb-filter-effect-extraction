#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理消融实验 — 倒谱管线 v2

测试从老管线中提取的预处理块对 SMD/CPR/CPN 检测性能的影响:
  1. 噪声门 (Noise Gate)
  2. 频谱高斯平滑 (Spectral Gaussian Smoothing)
  3. 预白化 / 频谱包络去除 (Pre-whitening)
  4. 时序 EMA 平滑 (Temporal Smoothing)
  5. 低频倒谱截除 (Liftering)

三阶段实验:
  Phase 1 — 单块实验 (逐个测试)
  Phase 2 — 组合实验 (最优单块组合)
  Phase 3 — 消融实验 (从最优组合逐个移除)
"""

import sys, os, glob, time
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, median_filter
from scipy.signal import lfilter

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from processing.comb_feature_v2 import CombFeatureConfig

DATA_DIR = "/home/lvmingyang/March24/datasets/simulation/original_sound/copy_test_real/test_dataset"
REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "preprocessing_ablation_report.md")
C_SPEED = 343.0

# ─────────────── dataset loading (reused from evaluate_real_68.py) ───────────────

def load_dataset():
    wav_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    csv_map = {}
    for csv in csv_files:
        try:
            df_test = pd.read_csv(csv, nrows=1)
            df_test.columns = [c.strip("'\"").strip() for c in df_test.columns]
            if "time_sec" in df_test.columns:
                base = os.path.basename(csv).replace(".csv", "")
                csv_map[base] = csv
        except Exception:
            pass
    recordings = []
    for wav in wav_files:
        wav_name = os.path.basename(wav).replace(".wav", "")
        parts = wav_name.split("__t")
        csv_base = parts[0] if len(parts) >= 2 else wav_name
        if csv_base not in csv_map:
            continue
        bn = os.path.basename(wav)
        if bn.startswith("3_prop_90_deg"):
            cat = "90deg"
        elif bn.startswith("70_deg"):
            cat = "70deg"
        elif bn.startswith("80_deg"):
            cat = "80deg"
        elif bn.startswith("moving"):
            cat = "moving"
        elif bn.startswith("static"):
            cat = "static"
        else:
            cat = "other"
        recordings.append({
            "wav_path": wav, "csv_path": csv_map[csv_base],
            "name": wav_name, "category": cat,
        })
    print(f"Loaded {len(recordings)} recordings")
    return recordings


def read_labels(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip("'\"").strip() for c in df.columns]
    return df["time_sec"].values, df["distance_cm"].values, df["pattern_label_res"].values


# ─────────────── STFT precomputation ───────────────

def precompute_all(recordings, comb_cfg):
    """Precompute band-limited magnitude spectra and labels for all recordings."""
    import soundfile as sf

    n_fft = comb_cfg.n_fft
    hop = comb_cfg.hop_length
    sr = comb_cfg.sr
    window = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band_mask = (freqs >= comb_cfg.freq_min) & (freqs <= comb_cfg.freq_max)
    n_band = int(band_mask.sum())

    # Cepstral bin range (same as CombFeatureExtractor.__init__)
    df = sr / n_fft
    quef_factor = 1.0 / (n_band * df)
    cep_min = max(2, round(comb_cfg.tau_min_s / quef_factor))
    cep_max = min(n_band // 2, round(comb_cfg.tau_max_s / quef_factor) + 1)
    ref_min = cep_max + 5
    ref_max = min(n_band // 2, ref_min + 30)

    cache = []
    total_frames = 0
    chunk_sz = 8000

    for ri, rec in enumerate(recordings):
        audio, file_sr = sf.read(rec["wav_path"], dtype="float64")
        if audio.ndim > 1:
            audio = audio[:, 0]
        n_frames = max(0, (len(audio) - n_fft) // hop + 1)
        if n_frames == 0:
            continue

        mag_bands = np.empty((n_frames, n_band), dtype=np.float32)
        rms_arr = np.empty(n_frames, dtype=np.float32)

        for s in range(0, n_frames, chunk_sz):
            e = min(s + chunk_sz, n_frames)
            idx = (np.arange(s, e) * hop)[:, None] + np.arange(n_fft)[None, :]
            frames = audio[idx]
            rms_arr[s:e] = np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)
            windowed = frames * window[None, :]
            mags = np.abs(np.fft.rfft(windowed, axis=1))
            mag_bands[s:e] = mags[:, band_mask].astype(np.float32)

        # Labels
        lt, ld, lp = read_labels(rec["csv_path"])
        frame_center = (np.arange(n_frames) * hop + n_fft // 2) / sr
        label_idx = np.clip(np.searchsorted(lt, frame_center), 0, len(lt) - 1)
        pattern = lp[label_idx].astype(np.float32)
        dist_true = ld[label_idx].astype(np.float32)

        cache.append({
            "name": rec["name"], "cat": rec["category"],
            "mag_bands": mag_bands, "rms": rms_arr,
            "pattern": pattern, "dist_true": dist_true,
            "n_frames": n_frames,
        })
        total_frames += n_frames

        if (ri + 1) % 20 == 0:
            print(f"  precomputed {ri+1}/{len(recordings)}...")

    meta = {
        "n_band": n_band, "cep_min": cep_min, "cep_max": cep_max,
        "ref_min": ref_min, "ref_max": ref_max, "quef_factor": quef_factor,
    }
    print(f"  precomputed {len(cache)} recordings, {total_frames:,} total frames")
    return cache, meta


# ─────────────── feature extraction with preprocessing ───────────────

def extract_features(mag_bands, rms, pp, meta, cep_avg):
    """
    Vectorized feature extraction with configurable preprocessing.

    Parameters
    ----------
    mag_bands : (N, B) float32 — band-limited magnitude spectrum
    rms : (N,) float32 — per-frame RMS of raw audio
    pp : dict — preprocessing config
    meta : dict — cepstral bin ranges etc.
    cep_avg : int — cepstrum averaging window size

    Returns
    -------
    features : (N, 5) — [smd, cpr, cpn, nda, dist_est_cm]
    """
    N, B = mag_bands.shape
    cep_min = meta["cep_min"]
    cep_max = meta["cep_max"]
    ref_min = meta["ref_min"]
    ref_max = meta["ref_max"]
    quef_factor = meta["quef_factor"]

    # Noise gate mask
    if pp.get("noise_gate"):
        gate = rms >= pp["noise_gate_rms"]
    else:
        gate = np.ones(N, dtype=bool)

    mb = mag_bands.astype(np.float64)

    # Pre-whitening
    if pp.get("prewhiten"):
        w = pp["prewhiten_window"]
        env = median_filter(mb, size=(1, w))
        mb = mb / (env + 1e-12)

    lb = np.log(mb + 1e-12)

    # Spectral Gaussian smoothing (along frequency axis)
    if pp.get("spectral_smooth"):
        lb = gaussian_filter1d(lb, sigma=pp["spectral_smooth_sigma"], axis=1)

    # Temporal EMA smoothing (along time axis)
    if pp.get("temporal_smooth"):
        alpha = pp["temporal_alpha"]
        lb = lfilter([alpha], [1, -(1 - alpha)], lb, axis=0)

    # SMD (per-frame std of mean-removed log-spectrum)
    means = lb.mean(axis=1, keepdims=True)
    centered = lb - means
    smd = centered.std(axis=1)

    # Cepstrum (magnitude cepstrum = |FFT(centered)|)
    ceps = np.abs(np.fft.fft(centered, axis=1))

    # Liftering
    if pp.get("lifter"):
        n_rm = pp["lifter_n"]
        ceps[:, :n_rm] = 0
        # Mirror: cepstrum is symmetric for real input
        if n_rm > 0 and B > n_rm:
            ceps[:, -n_rm:] = 0

    # Cepstrum temporal averaging
    if cep_avg > 1:
        avg_ceps = uniform_filter1d(ceps, size=cep_avg, axis=0, mode="nearest")
    else:
        avg_ceps = ceps

    # Cepstral peak search
    if cep_min >= cep_max or cep_max > B // 2:
        cpr = np.zeros(N)
        cpn = np.zeros(N)
        dist_cm = np.zeros(N)
    else:
        search = avg_ceps[:, cep_min:cep_max]
        peak_local = np.argmax(search, axis=1)
        peak_bin = peak_local + cep_min
        peak_vals = avg_ceps[np.arange(N), peak_bin]

        # CPR: peak / median in search range
        med = np.median(search, axis=1)
        cpr = peak_vals / (med + 1e-12)

        # CPN: peak / mean in non-comb reference range
        if ref_min < ref_max and ref_max <= B // 2:
            baseline = np.mean(avg_ceps[:, ref_min:ref_max], axis=1)
        else:
            baseline = med
        cpn = peak_vals / (baseline + 1e-12)

        cpq_sec = peak_bin * quef_factor
        dist_cm = cpq_sec * C_SPEED / 2.0 * 100.0

    # NDA (mean absolute difference of consecutive log-spectra)
    nda = np.zeros(N)
    if N > 1:
        nda[1:] = np.mean(np.abs(lb[1:] - lb[:-1]), axis=1)

    features = np.column_stack([smd, cpr, cpn, nda, dist_cm])
    # Apply noise gate: gated frames get score 0
    features[~gate] = 0
    return features


# ─────────────── evaluation metrics ───────────────

def compute_metrics(features, pattern, dist_true):
    """Compute AUC-ROC, Youden's J, distance MAE."""
    smd, cpr, cpn = features[:, 0], features[:, 1], features[:, 2]
    dist_est = features[:, 4]
    res = {}

    for name, vals in [("smd", smd), ("cpr", cpr), ("cpn", cpn)]:
        pos = vals[pattern == 1]
        neg = vals[pattern == 0]
        if len(pos) < 10 or len(neg) < 10:
            res[f"{name}_auc"] = 0.5
            res[f"{name}_j"] = 0.0
            res[f"{name}_tpr"] = 0.0
            res[f"{name}_fpr"] = 0.0
            continue
        # Fast AUC via percentile thresholds
        all_v = np.concatenate([pos, neg])
        ths = np.sort(np.unique(np.percentile(all_v, np.linspace(0, 100, 500))))
        best_j, best_tpr, best_fpr = -1, 0, 1
        tprs_l, fprs_l = [], []
        for th in ths:
            tp = (pos >= th).sum()
            fn = len(pos) - tp
            fp = (neg >= th).sum()
            tn = len(neg) - fp
            tpr = tp / (tp + fn) if (tp + fn) else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0
            tprs_l.append(tpr)
            fprs_l.append(fpr)
            j = tpr - fpr
            if j > best_j:
                best_j, best_tpr, best_fpr = j, tpr, fpr
        fprs_a = np.array(fprs_l)
        tprs_a = np.array(tprs_l)
        idx = np.argsort(fprs_a)
        auc = float(np.trapz(tprs_a[idx], fprs_a[idx]))
        res[f"{name}_auc"] = auc
        res[f"{name}_j"] = best_j
        res[f"{name}_tpr"] = best_tpr
        res[f"{name}_fpr"] = best_fpr

    p1 = pattern == 1
    if p1.sum() > 0:
        err = np.abs(dist_est[p1] - dist_true[p1])
        res["dist_mae"] = float(np.mean(err))
        res["dist_within5"] = float((err <= 5).mean())
        res["dist_within10"] = float((err <= 10).mean())
    else:
        res["dist_mae"] = float("nan")
        res["dist_within5"] = float("nan")
        res["dist_within10"] = float("nan")

    return res


# ─────────────── run one configuration across all recordings ───────────────

def run_config(cache, meta, pp, cep_avg=4):
    """Run one preprocessing config on all cached recordings."""
    all_feats = []
    all_pattern = []
    all_dist = []

    for rec in cache:
        feats = extract_features(rec["mag_bands"], rec["rms"], pp, meta, cep_avg)
        all_feats.append(feats)
        all_pattern.append(rec["pattern"])
        all_dist.append(rec["dist_true"])

    all_feats = np.vstack(all_feats)
    all_pattern = np.concatenate(all_pattern)
    all_dist = np.concatenate(all_dist)

    return compute_metrics(all_feats, all_pattern, all_dist)


# ─────────────── experiment configuration builder ───────────────

def pp_base(**kw):
    """Create a preprocessing config dict with defaults."""
    d = {
        "noise_gate": False, "noise_gate_rms": 0.01,
        "spectral_smooth": False, "spectral_smooth_sigma": 2.0,
        "prewhiten": False, "prewhiten_window": 21,
        "temporal_smooth": False, "temporal_alpha": 0.3,
        "lifter": False, "lifter_n": 2,
    }
    d.update(kw)
    return d


def build_phase1_configs():
    """Phase 1: individual preprocessing blocks."""
    cfgs = {"baseline": pp_base()}

    # Noise gate sweep
    for rms in [0.003, 0.005, 0.01, 0.02, 0.05]:
        cfgs[f"gate_{rms}"] = pp_base(noise_gate=True, noise_gate_rms=rms)

    # Spectral smoothing sweep
    for sigma in [1.0, 2.0, 3.0, 5.0, 8.0]:
        cfgs[f"smooth_{sigma}"] = pp_base(spectral_smooth=True, spectral_smooth_sigma=sigma)

    # Pre-whitening sweep
    for w in [5, 9, 15, 21, 31, 41]:
        cfgs[f"whiten_{w}"] = pp_base(prewhiten=True, prewhiten_window=w)

    # Temporal EMA sweep
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
        cfgs[f"temporal_{alpha}"] = pp_base(temporal_smooth=True, temporal_alpha=alpha)

    # Liftering sweep
    for n in [1, 2, 3, 5]:
        cfgs[f"lifter_{n}"] = pp_base(lifter=True, lifter_n=n)

    return cfgs


def find_best(results, prefix, metric="smd_auc"):
    """Find best config among those with given prefix."""
    cands = {k: v for k, v in results.items() if k.startswith(prefix) and k != "baseline"}
    if not cands:
        return None, None
    best_name = max(cands, key=lambda k: cands[k].get(metric, 0))
    return best_name, results[best_name]


def build_phase2_configs(results, phase1_cfgs):
    """Phase 2: combinations of best individual blocks."""
    combos = {}

    # Find best of each block
    best = {}
    for prefix in ["gate_", "smooth_", "whiten_", "temporal_", "lifter_"]:
        name, _ = find_best(results, prefix)
        if name:
            best[prefix.rstrip("_")] = phase1_cfgs[name]

    if not best:
        return combos

    # Pairwise: smooth + gate
    if "smooth" in best and "gate" in best:
        p = pp_base()
        p.update({k: v for k, v in best["smooth"].items() if "spectral" in k})
        p.update({k: v for k, v in best["gate"].items() if "noise" in k})
        combos["combo_smooth+gate"] = p

    # Triple: smooth + gate + temporal
    if "smooth" in best and "gate" in best and "temporal" in best:
        p = pp_base()
        p.update({k: v for k, v in best["smooth"].items() if "spectral" in k})
        p.update({k: v for k, v in best["gate"].items() if "noise" in k})
        p.update({k: v for k, v in best["temporal"].items() if "temporal" in k})
        combos["combo_smooth+gate+temporal"] = p

    # Smooth + whiten
    if "smooth" in best and "whiten" in best:
        p = pp_base()
        p.update({k: v for k, v in best["smooth"].items() if "spectral" in k})
        p.update({k: v for k, v in best["whiten"].items() if "prewhiten" in k})
        combos["combo_smooth+whiten"] = p

    # Smooth + gate + whiten
    if "smooth" in best and "gate" in best and "whiten" in best:
        p = pp_base()
        p.update({k: v for k, v in best["smooth"].items() if "spectral" in k})
        p.update({k: v for k, v in best["gate"].items() if "noise" in k})
        p.update({k: v for k, v in best["whiten"].items() if "prewhiten" in k})
        combos["combo_smooth+gate+whiten"] = p

    # All blocks combined
    p_all = pp_base()
    for b_pp in best.values():
        for k, v in b_pp.items():
            if v is not True and v is not False:
                p_all[k] = v
            elif v is True:
                p_all[k] = True
    combos["combo_ALL"] = p_all

    # Smooth + gate + lifter
    if "smooth" in best and "gate" in best and "lifter" in best:
        p = pp_base()
        p.update({k: v for k, v in best["smooth"].items() if "spectral" in k})
        p.update({k: v for k, v in best["gate"].items() if "noise" in k})
        p.update({k: v for k, v in best["lifter"].items() if "lifter" in k})
        combos["combo_smooth+gate+lifter"] = p

    return combos


def build_phase3_configs(best_combo_pp):
    """Phase 3: ablation — remove one block at a time from best combo."""
    ablations = {}
    blocks = [
        ("noise_gate", {"noise_gate": False}),
        ("spectral_smooth", {"spectral_smooth": False}),
        ("prewhiten", {"prewhiten": False}),
        ("temporal_smooth", {"temporal_smooth": False}),
        ("lifter", {"lifter": False}),
    ]
    for block_name, overrides in blocks:
        if best_combo_pp.get(block_name):
            p = dict(best_combo_pp)
            p.update(overrides)
            ablations[f"ablate_no_{block_name}"] = p
    return ablations


# ─────────────── report generation ───────────────

def gen_report(results, phase1_names, phase2_names, phase3_names,
               best_combo_name, phase1_cfgs, phase2_cfgs, phase3_cfgs):
    L = []
    w = L.append

    w("# 预处理消融实验报告 — 倒谱管线 v2\n")
    w(f"**实验时间**: 2026-04-20\n")

    # ───── Theory section ─────
    w("---\n")
    w("## 1. 理论分析\n")
    w("### 1.1 信号模型\n")
    w("接收信号频谱:\n")
    w("```")
    w("Y(f) = X(f) · H(f) + N(f)")
    w("```\n")
    w("- **X(f)**: 声源（无人机螺旋桨噪声）——强结构化频谱，具有 f₀≈200Hz 的谐波系列")
    w("- **H(f)**: 梳状滤波器 H(f) = 1 + A·exp(-j2πfτ)，在真实数据中 A ≪ 1")
    w("- **N(f)**: 加性噪声（电噪声、量化噪声、环境噪声）\n")
    w("对数谱:\n")
    w("```")
    w("log|Y(f)| ≈ log|X(f)| + A·cos(2πfτ) + noise")
    w("```\n")
    w("倒谱 C[n] = |FFT{log|Y(f)| - mean}| 在 quefrency n=τ/(quef_factor) 处产生梳状峰。\n")
    w("**核心挑战**: 真实数据中 A ≪ 1，梳状调制被 log|X(f)| 的自身结构和 N(f) 淹没。\n")

    w("### 1.2 各预处理块的数学分析\n")

    w("#### (a) 噪声门 — Noise Gate\n")
    w("- **原理**: 帧 RMS < 阈值 → 标记为「无梳状效应」(SMD=0)")
    w("- **数学效应**: 排除 SNR 极低帧（倒谱被量化/环境噪声主导）")
    w("- **对 SMD 的影响**: 消除虚假高 SMD（纯噪声帧的 log 谱方差可能偏高）")
    w("- **预测**: ↓ FPR，TPR 几乎不变（comb pattern 只出现在近距离、信号强时）")
    w("- **理论判定**: ✅ 安全且有依据\n")

    w("#### (b) 频谱高斯平滑 — Spectral Gaussian Smoothing\n")
    w("- **原理**: 沿频率轴对 log|Y(f)| 做 Gaussian 卷积 (σ bins)")
    w("- **数学效应**: 等价于倒谱域乘以 Gaussian 窗 → 衰减高 quefrency 成分")
    w("- **对梳状信号的衰减**: exp(-2π²σ_f²τ²)")
    w("  - σ=2 bins (≈47Hz), d=10cm (τ=0.58ms): 衰减 = 0.997 → **可忽略**")
    w("  - σ=2 bins, d=40cm (τ=2.3ms): 衰减 = 0.95 → **轻微**")
    w("  - σ=5 bins, d=40cm: 衰减 = 0.72 → **需注意**")
    w("- **对窄带噪声**: 尖峰 δ(f) → Gaussian(σ)，峰值降低 ~σ 倍")
    w("- **预测**: σ=1~3 时改善 SNR 而不损失信号；σ>5 可能损失远距离信号")
    w("- **理论判定**: ✅ 强依据，σ ≤ 3 安全\n")

    w("#### (c) 预白化 — Pre-whitening (Spectral Envelope Removal)\n")
    w("- **原理**: mag_band / median_filter(mag_band) → 去除频谱包络")
    w("- **数学效应**: log 域等价于减去平滑的 log|X(f)| → 剩余 ≈ A·cos(2πfτ) + 残差")
    w("- **对 SMD**: 声源频谱形状贡献被去除，SMD 更纯粹反映梳状调制")
    w("- **关键参数**: median 窗口大小 W")
    w("  - 梳状调制频率 1/τ：d=10cm → Δf≈1724Hz ≈ 74 bins; d=40cm → Δf≈435Hz ≈ 19 bins")
    w("  - 螺旋桨谐波间距: f₀≈200Hz ≈ 9 bins")
    w("  - **W < 19** 可能平滑掉远距离梳状信号")
    w("  - **W ≈ 9-15**: 平滑谐波但保留梳状调制（推荐范围）")
    w("- **理论判定**: ✅ 理论可行，W=9~15 为安全范围\n")

    w("#### (d) 时序 EMA 平滑 — Temporal Smoothing\n")
    w("- **原理**: log|Y_t(f)| → α·log|Y_t| + (1-α)·EMA_{t-1}")
    w("- **数学效应**: 压制帧间快变噪声（电噪声），保留慢变梳状信号")
    w("- **等效时间常数**: τ_ema ≈ 1/α 帧 = (hop/sr)/α 秒")
    w("  - α=0.1: τ≈107ms (强平滑); α=0.3: τ≈36ms; α=0.5: τ≈21ms")
    w("- **对梳状信号**: 距离变化速度 ~cm/s → 梳状 τ 变化 ~60μs/s → **远慢于** EMA 时间常数")
    w("- **预测**: α=0.1~0.3 应改善 SNR")
    w("- **理论判定**: ✅ 物理逻辑正确\n")

    w("#### (e) 低频倒谱截除 — Liftering\n")
    w("- **原理**: 将倒谱前 k 个 bin 置零（bin 0 = 均值，bin 1 = 线性趋势 ...）")
    w("- **当前已有**: bin 0 已通过 centered = log_band - mean 去除")
    w("- **梳状峰位**: bin 2~23 (d=5~55cm)，截除 bin 0~1 **安全**")
    w("- **预测**: 对 CPR/CPN 有轻微改善（去除残余趋势干扰），SMD 不受影响")
    w("- **理论判定**: ✅ 安全但效果可能有限\n")

    w("---\n")

    # ───── Phase 1 results ─────
    w("## 2. Phase 1 — 单块实验结果\n")
    baseline_r = results["baseline"]
    w(f"**Baseline (无预处理)**: SMD AUC = {baseline_r['smd_auc']:.4f}, "
      f"J = {baseline_r['smd_j']:.3f} (TPR={baseline_r['smd_tpr']:.1%}, FPR={baseline_r['smd_fpr']:.1%}), "
      f"距离 MAE = {baseline_r['dist_mae']:.1f} cm\n")

    groups = [
        ("2.1 噪声门", "gate_"),
        ("2.2 频谱高斯平滑", "smooth_"),
        ("2.3 预白化", "whiten_"),
        ("2.4 时序 EMA 平滑", "temporal_"),
        ("2.5 低频倒谱截除", "lifter_"),
    ]

    for title, prefix in groups:
        w(f"### {title}\n")
        w("| 配置 | SMD AUC | Δ AUC | J | TPR | FPR | CPR AUC | CPN AUC | dist MAE |")
        w("|------|---------|-------|---|-----|-----|---------|---------|----------|")
        w(f"| baseline | {baseline_r['smd_auc']:.4f} | — | {baseline_r['smd_j']:.3f} | "
          f"{baseline_r['smd_tpr']:.1%} | {baseline_r['smd_fpr']:.1%} | "
          f"{baseline_r.get('cpr_auc',0):.4f} | {baseline_r.get('cpn_auc',0):.4f} | "
          f"{baseline_r['dist_mae']:.1f} |")
        names_in_group = [n for n in phase1_names if n.startswith(prefix)]
        for n in names_in_group:
            r = results[n]
            d_auc = r["smd_auc"] - baseline_r["smd_auc"]
            sign = "+" if d_auc >= 0 else ""
            w(f"| {n} | {r['smd_auc']:.4f} | {sign}{d_auc:.4f} | {r['smd_j']:.3f} | "
              f"{r['smd_tpr']:.1%} | {r['smd_fpr']:.1%} | "
              f"{r.get('cpr_auc',0):.4f} | {r.get('cpn_auc',0):.4f} | "
              f"{r['dist_mae']:.1f} |")
        best_n, _ = find_best(results, prefix)
        if best_n:
            w(f"\n**最优**: `{best_n}` (SMD AUC = {results[best_n]['smd_auc']:.4f})\n")

    # ───── Phase 2 results ─────
    w("---\n")
    w("## 3. Phase 2 — 组合实验结果\n")
    w("| 配置 | SMD AUC | Δ AUC | J | TPR | FPR | CPR AUC | CPN AUC | dist MAE |")
    w("|------|---------|-------|---|-----|-----|---------|---------|----------|")
    w(f"| baseline | {baseline_r['smd_auc']:.4f} | — | {baseline_r['smd_j']:.3f} | "
      f"{baseline_r['smd_tpr']:.1%} | {baseline_r['smd_fpr']:.1%} | "
      f"{baseline_r.get('cpr_auc',0):.4f} | {baseline_r.get('cpn_auc',0):.4f} | "
      f"{baseline_r['dist_mae']:.1f} |")
    for n in phase2_names:
        r = results[n]
        d_auc = r["smd_auc"] - baseline_r["smd_auc"]
        sign = "+" if d_auc >= 0 else ""
        w(f"| {n} | {r['smd_auc']:.4f} | {sign}{d_auc:.4f} | {r['smd_j']:.3f} | "
          f"{r['smd_tpr']:.1%} | {r['smd_fpr']:.1%} | "
          f"{r.get('cpr_auc',0):.4f} | {r.get('cpn_auc',0):.4f} | "
          f"{r['dist_mae']:.1f} |")

    if best_combo_name:
        br = results[best_combo_name]
        w(f"\n**最优组合**: `{best_combo_name}`")
        w(f"- SMD AUC = {br['smd_auc']:.4f} (Δ = +{br['smd_auc']-baseline_r['smd_auc']:.4f})")
        w(f"- J = {br['smd_j']:.3f} (TPR={br['smd_tpr']:.1%}, FPR={br['smd_fpr']:.1%})")
        w(f"- 距离 MAE = {br['dist_mae']:.1f} cm\n")

    # ───── Phase 3 results ─────
    if phase3_names:
        w("---\n")
        w("## 4. Phase 3 — 消融实验\n")
        w(f"基准: `{best_combo_name}` (SMD AUC = {results[best_combo_name]['smd_auc']:.4f})\n")
        w("| 移除的块 | SMD AUC | Δ AUC | J | TPR | FPR | 结论 |")
        w("|---------|---------|-------|---|-----|-----|------|")
        combo_auc = results[best_combo_name]["smd_auc"]
        for n in phase3_names:
            r = results[n]
            d_auc = r["smd_auc"] - combo_auc
            sign = "+" if d_auc >= 0 else ""
            removed = n.replace("ablate_no_", "")
            verdict = "有贡献 ✅" if d_auc < -0.002 else ("无影响 ⚪" if abs(d_auc) < 0.002 else "有害 ❌")
            w(f"| {removed} | {r['smd_auc']:.4f} | {sign}{d_auc:.4f} | {r['smd_j']:.3f} | "
              f"{r['smd_tpr']:.1%} | {r['smd_fpr']:.1%} | {verdict} |")
        w("")

    # ───── Conclusion ─────
    w("---\n")
    w("## 5. 结论与推荐配置\n")

    # Find overall best
    all_names = list(results.keys())
    overall_best = max(all_names, key=lambda k: results[k].get("smd_auc", 0))
    ob = results[overall_best]
    w(f"### 全局最优配置: `{overall_best}`\n")
    w(f"- **SMD AUC** = {ob['smd_auc']:.4f} (baseline {baseline_r['smd_auc']:.4f}, Δ = +{ob['smd_auc']-baseline_r['smd_auc']:.4f})")
    w(f"- **Youden's J** = {ob['smd_j']:.3f} (TPR = {ob['smd_tpr']:.1%}, FPR = {ob['smd_fpr']:.1%})")
    w(f"- **CPR AUC** = {ob.get('cpr_auc',0):.4f}, **CPN AUC** = {ob.get('cpn_auc',0):.4f}")
    w(f"- **距离 MAE** = {ob['dist_mae']:.1f} cm")
    w(f"- **相对 baseline 提升**: SMD AUC +{ob['smd_auc']-baseline_r['smd_auc']:.4f}, "
      f"J +{ob['smd_j']-baseline_r['smd_j']:.3f}\n")

    # Top 10 overall
    w("### Top-10 配置排名\n")
    w("| # | 配置 | SMD AUC | J | TPR | FPR | dist MAE |")
    w("|---|------|---------|---|-----|-----|----------|")
    sorted_names = sorted(all_names, key=lambda k: results[k].get("smd_auc", 0), reverse=True)
    for i, n in enumerate(sorted_names[:10]):
        r = results[n]
        w(f"| {i+1} | {n} | {r['smd_auc']:.4f} | {r['smd_j']:.3f} | "
          f"{r['smd_tpr']:.1%} | {r['smd_fpr']:.1%} | {r['dist_mae']:.1f} |")

    # Write report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n报告已保存: {REPORT_PATH}")


# ─────────────── main ───────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("预处理消融实验 — 倒谱管线 v2")
    print("=" * 60)

    # Load dataset
    recordings = load_dataset()

    # Default comb config
    comb_cfg = CombFeatureConfig()

    # Precompute STFT
    print("\n[Precompute] 预计算 STFT...")
    cache, meta = precompute_all(recordings, comb_cfg)

    results = {}

    # ───── Phase 1 ─────
    print("\n[Phase 1] 单块实验...")
    phase1_cfgs = build_phase1_configs()
    phase1_names = list(phase1_cfgs.keys())
    for i, (name, pp) in enumerate(phase1_cfgs.items()):
        t0 = time.time()
        results[name] = run_config(cache, meta, pp, cep_avg=comb_cfg.cep_avg_frames)
        dt = time.time() - t0
        r = results[name]
        print(f"  [{i+1}/{len(phase1_cfgs)}] {name:30s} "
              f"SMD AUC={r['smd_auc']:.4f}  J={r['smd_j']:.3f}  "
              f"TPR={r['smd_tpr']:.1%}  FPR={r['smd_fpr']:.1%}  "
              f"dist={r['dist_mae']:.1f}cm  ({dt:.1f}s)")

    # ───── Phase 2 ─────
    print("\n[Phase 2] 组合实验...")
    phase2_cfgs = build_phase2_configs(results, phase1_cfgs)
    phase2_names = list(phase2_cfgs.keys())
    for i, (name, pp) in enumerate(phase2_cfgs.items()):
        t0 = time.time()
        results[name] = run_config(cache, meta, pp, cep_avg=comb_cfg.cep_avg_frames)
        dt = time.time() - t0
        r = results[name]
        print(f"  [{i+1}/{len(phase2_cfgs)}] {name:35s} "
              f"SMD AUC={r['smd_auc']:.4f}  J={r['smd_j']:.3f}  "
              f"TPR={r['smd_tpr']:.1%}  FPR={r['smd_fpr']:.1%}  "
              f"dist={r['dist_mae']:.1f}cm  ({dt:.1f}s)")

    # Find best combo
    all_combo = {**{k: results[k] for k in phase2_names}}
    best_combo_name = max(all_combo, key=lambda k: all_combo[k].get("smd_auc", 0)) if all_combo else None

    # ───── Phase 3 ─────
    phase3_cfgs = {}
    phase3_names = []
    if best_combo_name:
        print(f"\n[Phase 3] 消融实验 (基准: {best_combo_name})...")
        phase3_cfgs = build_phase3_configs(phase2_cfgs[best_combo_name])
        phase3_names = list(phase3_cfgs.keys())
        for i, (name, pp) in enumerate(phase3_cfgs.items()):
            t0 = time.time()
            results[name] = run_config(cache, meta, pp, cep_avg=comb_cfg.cep_avg_frames)
            dt = time.time() - t0
            r = results[name]
            print(f"  [{i+1}/{len(phase3_cfgs)}] {name:35s} "
                  f"SMD AUC={r['smd_auc']:.4f}  J={r['smd_j']:.3f}  "
                  f"TPR={r['smd_tpr']:.1%}  FPR={r['smd_fpr']:.1%}  "
                  f"dist={r['dist_mae']:.1f}cm  ({dt:.1f}s)")

    # ───── Report ─────
    print("\n[Report] 生成报告...")
    gen_report(results, phase1_names, phase2_names, phase3_names,
               best_combo_name, phase1_cfgs, phase2_cfgs, phase3_cfgs)

    dt_total = time.time() - t_start
    print(f"\n总耗时: {dt_total:.0f}s ({dt_total/60:.1f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
