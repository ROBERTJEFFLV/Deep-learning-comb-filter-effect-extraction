#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序差分实验 — 倒谱管线 v2

测试时序差分 (前后帧 Δt 间隔取差) 对 SMD/CPR/CPN 的影响。
核心逻辑 (用户原始设计哲学):
  1. EMA 平滑 → 压住快变噪声
  2. 差分 → 提取慢变信号，去除静态声源频谱
  3. 倒谱/SMD → 提取梳状信息

额外测试:
  - 纯差分 (无 EMA)
  - EMA → 差分
  - 差分 → 取绝对值 (保留调制幅度，不区分方向)
  - 分类别分析 (moving vs static vs angle)
"""

import sys, os, glob, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from processing.comb_feature_v2 import CombFeatureConfig

DATA_DIR = "/home/lvmingyang/March24/datasets/simulation/original_sound/copy_test_real/test_dataset"
REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "temporal_diff_experiment_report.md")
C_SPEED = 343.0


# ─────────────── dataset (reuse from evaluate_preprocessing.py) ───────────────

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


def precompute_all(recordings, comb_cfg):
    import soundfile as sf
    n_fft = comb_cfg.n_fft
    hop = comb_cfg.hop_length
    sr = comb_cfg.sr
    window = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band_mask = (freqs >= comb_cfg.freq_min) & (freqs <= comb_cfg.freq_max)
    n_band = int(band_mask.sum())

    df = sr / n_fft
    quef_factor = 1.0 / (n_band * df)
    cep_min = max(2, round(comb_cfg.tau_min_s / quef_factor))
    cep_max = min(n_band // 2, round(comb_cfg.tau_max_s / quef_factor) + 1)
    ref_min = cep_max + 5
    ref_max = min(n_band // 2, ref_min + 30)

    cache = []
    total_frames = 0
    for ri, rec in enumerate(recordings):
        audio, file_sr = sf.read(rec["wav_path"], dtype="float64")
        if audio.ndim > 1:
            audio = audio[:, 0]
        n_frames = max(0, (len(audio) - n_fft) // hop + 1)
        if n_frames == 0:
            continue
        mag_bands = np.empty((n_frames, n_band), dtype=np.float32)
        rms_arr = np.empty(n_frames, dtype=np.float32)
        chunk = 8000
        for s in range(0, n_frames, chunk):
            e = min(s + chunk, n_frames)
            idx = (np.arange(s, e) * hop)[:, None] + np.arange(n_fft)[None, :]
            frames = audio[idx]
            rms_arr[s:e] = np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)
            windowed = frames * window[None, :]
            mags = np.abs(np.fft.rfft(windowed, axis=1))
            mag_bands[s:e] = mags[:, band_mask].astype(np.float32)

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


# ─────────────── feature extraction with diff ───────────────

def extract_features_diff(mag_bands, rms, pp, meta, cep_avg):
    """
    Feature extraction with optional EMA + temporal differencing.

    pp keys:
      ema_alpha: float or None (None = no EMA)
      diff_dt: int or None (None = no diff)
      diff_abs: bool (if True, take |diff| before computing features)
    """
    N, B = mag_bands.shape
    cep_min = meta["cep_min"]
    cep_max = meta["cep_max"]
    ref_min = meta["ref_min"]
    ref_max = meta["ref_max"]
    quef_factor = meta["quef_factor"]

    mb = mag_bands.astype(np.float64)
    lb = np.log(mb + 1e-12)

    # Step 1: Optional EMA smoothing on log-spectrum
    if pp.get("ema_alpha") is not None:
        alpha = pp["ema_alpha"]
        lb = lfilter([alpha], [1, -(1 - alpha)], lb, axis=0)

    # Step 2: Optional temporal differencing
    dt = pp.get("diff_dt")
    if dt is not None and dt > 0:
        diff_lb = np.zeros_like(lb)
        diff_lb[dt:] = lb[dt:] - lb[:-dt]
        # First dt frames have no valid diff → zero
        valid_mask = np.zeros(N, dtype=bool)
        valid_mask[dt:] = True

        if pp.get("diff_abs", False):
            diff_lb = np.abs(diff_lb)

        lb_for_features = diff_lb
    else:
        lb_for_features = lb
        valid_mask = np.ones(N, dtype=bool)

    # SMD
    means = lb_for_features.mean(axis=1, keepdims=True)
    centered = lb_for_features - means
    smd = centered.std(axis=1)

    # Cepstrum
    ceps = np.abs(np.fft.fft(centered, axis=1))

    # Cepstrum averaging
    if cep_avg > 1:
        avg_ceps = uniform_filter1d(ceps, size=cep_avg, axis=0, mode="nearest")
    else:
        avg_ceps = ceps

    # Cepstral features
    if cep_min >= cep_max or cep_max > B // 2:
        cpr = np.zeros(N)
        cpn = np.zeros(N)
        dist_cm = np.zeros(N)
    else:
        search = avg_ceps[:, cep_min:cep_max]
        peak_local = np.argmax(search, axis=1)
        peak_bin = peak_local + cep_min
        peak_vals = avg_ceps[np.arange(N), peak_bin]
        med = np.median(search, axis=1)
        cpr = peak_vals / (med + 1e-12)
        if ref_min < ref_max and ref_max <= B // 2:
            baseline = np.mean(avg_ceps[:, ref_min:ref_max], axis=1)
        else:
            baseline = med
        cpn = peak_vals / (baseline + 1e-12)
        cpq_sec = peak_bin * quef_factor
        dist_cm = cpq_sec * C_SPEED / 2.0 * 100.0

    features = np.column_stack([smd, cpr, cpn, dist_cm])
    features[~valid_mask] = 0
    return features


# ─────────────── metrics ───────────────

def compute_auc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return 0.5, 0.0, 0.0, 1.0
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
    fprs_a, tprs_a = np.array(fprs_l), np.array(tprs_l)
    idx = np.argsort(fprs_a)
    auc = float(np.trapz(tprs_a[idx], fprs_a[idx]))
    return auc, best_j, best_tpr, best_fpr


def eval_config(cache, meta, pp, cep_avg=4):
    """Evaluate a config globally and per-category."""
    # Global
    all_smd, all_cpr, all_cpn = [], [], []
    all_pattern, all_dist, all_dist_est = [], [], []
    # Per-category
    cat_data = {}

    for rec in cache:
        feats = extract_features_diff(rec["mag_bands"], rec["rms"], pp, meta, cep_avg)
        smd, cpr, cpn, d_est = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3]
        pat, d_true = rec["pattern"], rec["dist_true"]

        all_smd.append(smd)
        all_cpr.append(cpr)
        all_cpn.append(cpn)
        all_pattern.append(pat)
        all_dist.append(d_true)
        all_dist_est.append(d_est)

        cat = rec["cat"]
        if cat not in cat_data:
            cat_data[cat] = {"smd": [], "cpr": [], "cpn": [], "pat": [], "dist": [], "dist_est": []}
        cat_data[cat]["smd"].append(smd)
        cat_data[cat]["cpr"].append(cpr)
        cat_data[cat]["cpn"].append(cpn)
        cat_data[cat]["pat"].append(pat)
        cat_data[cat]["dist"].append(d_true)
        cat_data[cat]["dist_est"].append(d_est)

    all_smd = np.concatenate(all_smd)
    all_cpr = np.concatenate(all_cpr)
    all_cpn = np.concatenate(all_cpn)
    all_pattern = np.concatenate(all_pattern)
    all_dist = np.concatenate(all_dist)
    all_dist_est = np.concatenate(all_dist_est)

    res = {"global": {}, "per_cat": {}}

    # Global metrics
    for name, vals in [("smd", all_smd), ("cpr", all_cpr), ("cpn", all_cpn)]:
        auc, j, tpr, fpr = compute_auc(vals[all_pattern == 1], vals[all_pattern == 0])
        res["global"][f"{name}_auc"] = auc
        res["global"][f"{name}_j"] = j
        res["global"][f"{name}_tpr"] = tpr
        res["global"][f"{name}_fpr"] = fpr

    p1 = all_pattern == 1
    if p1.sum() > 0:
        err = np.abs(all_dist_est[p1] - all_dist[p1])
        res["global"]["dist_mae"] = float(np.mean(err))
    else:
        res["global"]["dist_mae"] = float("nan")

    # Per-category metrics
    for cat, cd in cat_data.items():
        smd_c = np.concatenate(cd["smd"])
        pat_c = np.concatenate(cd["pat"])
        auc, j, tpr, fpr = compute_auc(smd_c[pat_c == 1], smd_c[pat_c == 0])
        p1c = pat_c == 1
        if p1c.sum() > 0:
            d_err = np.abs(np.concatenate(cd["dist_est"])[p1c] - np.concatenate(cd["dist"])[p1c])
            d_mae = float(np.mean(d_err))
        else:
            d_mae = float("nan")
        res["per_cat"][cat] = {
            "smd_auc": auc, "smd_j": j, "smd_tpr": tpr, "smd_fpr": fpr,
            "n_frames": len(pat_c), "p1_rate": float(p1c.mean()),
            "dist_mae": d_mae,
        }

    return res


# ─────────────── experiment configs ───────────────

def build_configs():
    cfgs = {}

    # Baseline: no preprocessing
    cfgs["baseline"] = {}

    # Best from previous experiment: EMA only
    cfgs["ema_0.2"] = {"ema_alpha": 0.2}

    # ─── Pure diff (no EMA) ───
    for dt in [1, 2, 3, 5, 10, 20, 40]:
        cfgs[f"diff_{dt}"] = {"diff_dt": dt}
        cfgs[f"diff_{dt}_abs"] = {"diff_dt": dt, "diff_abs": True}

    # ─── EMA → diff (user's design philosophy) ───
    for alpha in [0.1, 0.2, 0.3]:
        for dt in [1, 2, 3, 5, 10, 20, 40]:
            cfgs[f"ema{alpha}_diff{dt}"] = {"ema_alpha": alpha, "diff_dt": dt}
            cfgs[f"ema{alpha}_diff{dt}_abs"] = {"ema_alpha": alpha, "diff_dt": dt, "diff_abs": True}

    return cfgs


# ─────────────── report ───────────────

def gen_report(all_results, configs):
    L = []
    w = L.append

    w("# 时序差分实验报告 — 倒谱管线 v2\n")

    w("## 1. 理论分析\n")
    w("### 差分的数学效应\n")
    w("```")
    w("ΔL_t(f) = log|Y_t(f)| - log|Y_{t-Δt}(f)|")
    w("       ≈ Δlog|X(f)| + A·[cos(2πfτ_t) - cos(2πfτ_{t-Δt})] + ΔN(f)")
    w("```\n")
    w("**三个分量的命运**:\n")
    w("| 分量 | 差分后 | 原因 |")
    w("|------|--------|------|")
    w("| log|X(f)| 声源频谱 | ≈ 0 (静态/慢变) | 螺旋桨 RPM 近似恒定 |")
    w("| A·cos(2πfτ) 梳状调制 | 存活 (if τ 变化) / ≈ 0 (if τ 恒定) | 取决于无人机运动状态 |")
    w("| N(f) 噪声 | ×√2 (随机噪声) / ↓ (EMA 预平滑后) | 先 EMA 再差分可抑制 |")
    w("")
    w("**关键洞察**: 差分是**双刃剑**")
    w("- ✅ **Moving 场景**: τ 持续变化 → 梳状信号存活，声源频谱被消除 → SNR ↑")
    w("- ❌ **Static 场景**: τ 恒定 → 梳状信号也被消除 → 检测失效")
    w("- 参数 Δt 控制差分的时间尺度：")
    w("  - Δt=1 (~21ms): 检测快速变化")
    w("  - Δt=10 (~213ms): 检测中等速度运动")
    w("  - Δt=40 (~853ms): 检测慢速运动\n")

    # ─── Global results table ───
    w("---\n")
    w("## 2. 全局结果\n")

    # Sort by SMD AUC
    sorted_names = sorted(all_results.keys(),
                          key=lambda k: all_results[k]["global"].get("smd_auc", 0), reverse=True)

    w("### 2.1 Top-20 配置\n")
    w("| # | 配置 | SMD AUC | Δ AUC | J | TPR | FPR | CPR AUC | CPN AUC | dist MAE |")
    w("|---|------|---------|-------|---|-----|-----|---------|---------|----------|")
    bl = all_results["baseline"]["global"]
    for i, n in enumerate(sorted_names[:20]):
        g = all_results[n]["global"]
        da = g["smd_auc"] - bl["smd_auc"]
        s = "+" if da >= 0 else ""
        w(f"| {i+1} | {n} | {g['smd_auc']:.4f} | {s}{da:.4f} | "
          f"{g['smd_j']:.3f} | {g['smd_tpr']:.1%} | {g['smd_fpr']:.1%} | "
          f"{g.get('cpr_auc',0):.4f} | {g.get('cpn_auc',0):.4f} | "
          f"{g['dist_mae']:.1f} |")

    # ─── Group analysis: pure diff ───
    w("\n### 2.2 纯差分 (无 EMA)\n")
    w("| Δt | 帧间隔(ms) | SMD AUC | Δ AUC | J | |Δ|版 AUC | |Δ|版 J |")
    w("|-----|-----------|---------|-------|---|---------|----|")
    hop_ms = 1024 / 48000 * 1000
    for dt in [1, 2, 3, 5, 10, 20, 40]:
        n1, n2 = f"diff_{dt}", f"diff_{dt}_abs"
        g1, g2 = all_results[n1]["global"], all_results[n2]["global"]
        da1 = g1["smd_auc"] - bl["smd_auc"]
        da2 = g2["smd_auc"] - bl["smd_auc"]
        s1 = "+" if da1 >= 0 else ""
        s2 = "+" if da2 >= 0 else ""
        w(f"| {dt} | {dt*hop_ms:.0f} | {g1['smd_auc']:.4f} | {s1}{da1:.4f} | "
          f"{g1['smd_j']:.3f} | {g2['smd_auc']:.4f} | {g2['smd_j']:.3f} |")

    # ─── Group analysis: EMA + diff ───
    w("\n### 2.3 EMA → 差分\n")
    for alpha in [0.1, 0.2, 0.3]:
        w(f"\n**EMA α={alpha}**\n")
        w("| Δt | ms | SMD AUC | Δ AUC | J | TPR | FPR | |Δ|版 AUC | |Δ|版 J |")
        w("|-----|---|---------|-------|---|-----|-----|---------|----|")
        # EMA-only baseline
        ema_n = f"ema_{alpha}" if f"ema_{alpha}" in all_results else None
        if ema_n:
            eg = all_results[ema_n]["global"]
            w(f"| 0(EMA only) | — | {eg['smd_auc']:.4f} | +{eg['smd_auc']-bl['smd_auc']:.4f} | "
              f"{eg['smd_j']:.3f} | {eg['smd_tpr']:.1%} | {eg['smd_fpr']:.1%} | — | — |")
        for dt in [1, 2, 3, 5, 10, 20, 40]:
            n1 = f"ema{alpha}_diff{dt}"
            n2 = f"ema{alpha}_diff{dt}_abs"
            g1 = all_results[n1]["global"]
            g2 = all_results[n2]["global"]
            da1 = g1["smd_auc"] - bl["smd_auc"]
            s1 = "+" if da1 >= 0 else ""
            w(f"| {dt} | {dt*hop_ms:.0f} | {g1['smd_auc']:.4f} | {s1}{da1:.4f} | "
              f"{g1['smd_j']:.3f} | {g1['smd_tpr']:.1%} | {g1['smd_fpr']:.1%} | "
              f"{g2['smd_auc']:.4f} | {g2['smd_j']:.3f} |")

    # ─── Per-category analysis for top configs ───
    w("\n---\n")
    w("## 3. 分类别分析\n")
    w("理论预测: 差分对 moving 场景有利，对 static 有害。\n")

    # Select key configs for comparison
    key_cfgs = ["baseline", "ema_0.2"]
    # Add best pure diff and best EMA+diff
    best_diff = max([n for n in sorted_names if n.startswith("diff_") and not n.endswith("_abs")],
                    key=lambda k: all_results[k]["global"]["smd_auc"], default=None)
    best_ema_diff = max([n for n in sorted_names if n.startswith("ema") and "diff" in n and not n.endswith("_abs")],
                        key=lambda k: all_results[k]["global"]["smd_auc"], default=None)
    best_abs = max([n for n in sorted_names if n.endswith("_abs")],
                   key=lambda k: all_results[k]["global"]["smd_auc"], default=None)
    for c in [best_diff, best_ema_diff, best_abs]:
        if c and c not in key_cfgs:
            key_cfgs.append(c)

    cats_order = ["static", "moving", "70deg", "80deg", "90deg"]
    w("| 配置 | " + " | ".join(f"{c} AUC / J" for c in cats_order) + " |")
    w("|------| " + " | ".join("---" for _ in cats_order) + " |")
    for cfg_name in key_cfgs:
        pc = all_results[cfg_name]["per_cat"]
        cells = []
        for cat in cats_order:
            if cat in pc:
                cells.append(f"{pc[cat]['smd_auc']:.4f} / {pc[cat]['smd_j']:.3f}")
            else:
                cells.append("—")
        w(f"| {cfg_name} | " + " | ".join(cells) + " |")

    # Detailed per-cat for each key config
    for cfg_name in key_cfgs:
        w(f"\n**{cfg_name}**:\n")
        pc = all_results[cfg_name]["per_cat"]
        for cat in cats_order:
            if cat not in pc:
                continue
            c = pc[cat]
            w(f"- {cat}: AUC={c['smd_auc']:.4f}, J={c['smd_j']:.3f}, "
              f"TPR={c['smd_tpr']:.1%}, FPR={c['smd_fpr']:.1%}, "
              f"p1={c['p1_rate']:.1%}, MAE={c['dist_mae']:.1f}cm")

    # ─── Conclusion ───
    w("\n---\n")
    w("## 4. 结论\n")

    overall_best = sorted_names[0]
    ob = all_results[overall_best]["global"]
    w(f"### 全局最优: `{overall_best}`")
    w(f"- SMD AUC = {ob['smd_auc']:.4f} (baseline {bl['smd_auc']:.4f}, Δ = +{ob['smd_auc']-bl['smd_auc']:.4f})")
    w(f"- J = {ob['smd_j']:.3f}, TPR = {ob['smd_tpr']:.1%}, FPR = {ob['smd_fpr']:.1%}\n")

    # Moving-only best
    moving_best = None
    moving_best_auc = 0
    for n in sorted_names:
        pc = all_results[n]["per_cat"]
        if "moving" in pc and pc["moving"]["smd_auc"] > moving_best_auc:
            moving_best = n
            moving_best_auc = pc["moving"]["smd_auc"]
    if moving_best:
        mc = all_results[moving_best]["per_cat"]["moving"]
        w(f"### Moving 场景最优: `{moving_best}`")
        w(f"- Moving SMD AUC = {mc['smd_auc']:.4f}, J = {mc['smd_j']:.3f}\n")

    static_best = None
    static_best_auc = 0
    for n in sorted_names:
        pc = all_results[n]["per_cat"]
        if "static" in pc and pc["static"]["smd_auc"] > static_best_auc:
            static_best = n
            static_best_auc = pc["static"]["smd_auc"]
    if static_best:
        sc = all_results[static_best]["per_cat"]["static"]
        w(f"### Static 场景最优: `{static_best}`")
        w(f"- Static SMD AUC = {sc['smd_auc']:.4f}, J = {sc['smd_j']:.3f}\n")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n报告已保存: {REPORT_PATH}")


# ─────────────── main ───────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("时序差分实验 — 倒谱管线 v2")
    print("=" * 60)

    recordings = load_dataset()
    comb_cfg = CombFeatureConfig()

    print("\n[Precompute] 预计算 STFT...")
    cache, meta = precompute_all(recordings, comb_cfg)

    configs = build_configs()
    print(f"\n[Experiment] 共 {len(configs)} 个配置")

    all_results = {}
    for i, (name, pp) in enumerate(configs.items()):
        t0 = time.time()
        all_results[name] = eval_config(cache, meta, pp, cep_avg=comb_cfg.cep_avg_frames)
        dt = time.time() - t0
        g = all_results[name]["global"]
        print(f"  [{i+1:3d}/{len(configs)}] {name:30s} "
              f"SMD AUC={g['smd_auc']:.4f}  J={g['smd_j']:.3f}  "
              f"TPR={g['smd_tpr']:.1%}  FPR={g['smd_fpr']:.1%}  "
              f"dist={g['dist_mae']:.1f}cm  ({dt:.1f}s)")

    print("\n[Report] 生成报告...")
    gen_report(all_results, configs)

    dt_total = time.time() - t_start
    print(f"\n总耗时: {dt_total:.0f}s ({dt_total/60:.1f}min)")


if __name__ == "__main__":
    main()
