#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-analysis: Macro-Average AUC across categories.

发现了 Simpson's Paradox —— diff 管线在每个类别上的 AUC 都高于 ema_only，
但 pooled global AUC 反而低。原因是 ema_only 在 static/moving 上 FPR≈100%，
pooled AUC 被虚高。

本脚本对精选配置计算 macro-avg AUC，给出正确排名。
"""

import sys, os, glob, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import lfilter

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from processing.comb_feature_v2 import CombFeatureConfig

DATA_DIR = "/home/lvmingyang/March24/datasets/simulation/original_sound/copy_test_real/test_dataset"
REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "full_pipeline_experiment_report.md")
C_SPEED = 343.0


# ─────── reuse from evaluate_full_pipeline.py ───────

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


def extract_features(mag_bands, rms, pp, meta, cep_avg):
    N, B = mag_bands.shape
    cep_min, cep_max = meta["cep_min"], meta["cep_max"]
    ref_min, ref_max = meta["ref_min"], meta["ref_max"]
    quef_factor = meta["quef_factor"]

    mb = mag_bands.astype(np.float64)
    if pp.get("pre_ema_alpha") is not None:
        alpha = pp["pre_ema_alpha"]
        mb = lfilter([alpha], [1, -(1 - alpha)], mb, axis=0)
    lb = np.log(mb + 1e-12)
    if pp.get("ema_alpha") is not None:
        alpha = pp["ema_alpha"]
        lb = lfilter([alpha], [1, -(1 - alpha)], lb, axis=0)

    dt = pp.get("diff_dt")
    if dt is not None and dt > 0:
        diff_lb = np.zeros_like(lb)
        diff_lb[dt:] = lb[dt:] - lb[:-dt]
        valid_mask = np.zeros(N, dtype=bool)
        valid_mask[dt:] = True
        if pp.get("diff_abs", False):
            diff_lb = np.abs(diff_lb)
        working = diff_lb
    else:
        working = lb
        valid_mask = np.ones(N, dtype=bool)

    if pp.get("spectral_sigma") is not None:
        working = gaussian_filter1d(working, sigma=pp["spectral_sigma"], axis=1)
    if pp.get("post_ema_alpha") is not None:
        alpha = pp["post_ema_alpha"]
        working = lfilter([alpha], [1, -(1 - alpha)], working, axis=0)

    means = working.mean(axis=1, keepdims=True)
    centered = working - means
    smd = centered.std(axis=1)

    ceps = np.abs(np.fft.fft(centered, axis=1))
    if cep_avg > 1:
        avg_ceps = uniform_filter1d(ceps, size=cep_avg, axis=0, mode="nearest")
    else:
        avg_ceps = ceps

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


def eval_config_full(cache, meta, pp, cep_avg=4):
    """Returns global + per-category metrics."""
    all_smd, all_cpr, all_cpn = [], [], []
    all_pattern, all_dist, all_dist_est = [], [], []
    cat_data = {}

    for rec in cache:
        feats = extract_features(rec["mag_bands"], rec["rms"], pp, meta, cep_avg)
        smd, cpr, cpn, d_est = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3]
        pat, d_true = rec["pattern"], rec["dist_true"]

        all_smd.append(smd); all_cpr.append(cpr); all_cpn.append(cpn)
        all_pattern.append(pat); all_dist.append(d_true); all_dist_est.append(d_est)

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

    cats_order = ["static", "moving", "70deg", "80deg", "90deg"]
    cat_aucs = {}
    for cat in cats_order:
        if cat not in cat_data:
            cat_aucs[cat] = 0.5
            res["per_cat"][cat] = {"smd_auc": 0.5, "smd_j": 0, "smd_tpr": 0, "smd_fpr": 1,
                                    "cpr_auc": 0.5, "cpn_auc": 0.5, "p1_rate": 0, "dist_mae": float("nan")}
            continue
        cd = cat_data[cat]
        smd_c = np.concatenate(cd["smd"])
        cpr_c = np.concatenate(cd["cpr"])
        cpn_c = np.concatenate(cd["cpn"])
        pat_c = np.concatenate(cd["pat"])
        d_est_c = np.concatenate(cd["dist_est"])
        d_true_c = np.concatenate(cd["dist"])

        smd_auc, smd_j, smd_tpr, smd_fpr = compute_auc(smd_c[pat_c == 1], smd_c[pat_c == 0])
        cpr_auc, _, _, _ = compute_auc(cpr_c[pat_c == 1], cpr_c[pat_c == 0])
        cpn_auc, _, _, _ = compute_auc(cpn_c[pat_c == 1], cpn_c[pat_c == 0])

        p1c = pat_c == 1
        d_mae = float(np.mean(np.abs(d_est_c[p1c] - d_true_c[p1c]))) if p1c.sum() > 0 else float("nan")
        cat_aucs[cat] = smd_auc
        res["per_cat"][cat] = {
            "smd_auc": smd_auc, "smd_j": smd_j, "smd_tpr": smd_tpr, "smd_fpr": smd_fpr,
            "cpr_auc": cpr_auc, "cpn_auc": cpn_auc,
            "p1_rate": float(p1c.mean()), "dist_mae": d_mae,
        }

    # Macro-average AUC
    valid_cats = [c for c in cats_order if c in cat_data]
    macro_smd = np.mean([cat_aucs[c] for c in valid_cats])
    res["macro_smd_auc"] = float(macro_smd)

    # Macro-average J-statistic
    macro_j = np.mean([res["per_cat"][c]["smd_j"] for c in valid_cats])
    res["macro_smd_j"] = float(macro_j)

    return res


# ─────── configs to test ───────

def build_configs():
    cfgs = {}

    # References
    cfgs["baseline"] = {}
    cfgs["ema_only_0.1"] = {"ema_alpha": 0.1}
    cfgs["ema_only_0.15"] = {"ema_alpha": 0.15}
    cfgs["ema_only_0.2"] = {"ema_alpha": 0.2}
    cfgs["ema_only_0.3"] = {"ema_alpha": 0.3}

    # Best diff combos (α=0.1, σ=5.0, non-abs)
    for dt in [3, 4, 5, 6, 7, 8, 10]:
        cfgs[f"ema0.1_dt{dt}_s5.0"] = {"ema_alpha": 0.1, "diff_dt": dt, "spectral_sigma": 5.0}

    # α=0.1, σ=3.0
    for dt in [5, 6, 7, 8]:
        cfgs[f"ema0.1_dt{dt}_s3.0"] = {"ema_alpha": 0.1, "diff_dt": dt, "spectral_sigma": 3.0}

    # α=0.1, σ=7.0 and 10.0 (push σ higher)
    for sigma in [7.0, 10.0]:
        for dt in [5, 6, 7, 8]:
            cfgs[f"ema0.1_dt{dt}_s{sigma}"] = {"ema_alpha": 0.1, "diff_dt": dt, "spectral_sigma": sigma}

    # α=0.05 (even less smoothing)
    for dt in [5, 6, 7, 8]:
        cfgs[f"ema0.05_dt{dt}_s5.0"] = {"ema_alpha": 0.05, "diff_dt": dt, "spectral_sigma": 5.0}
        cfgs[f"ema0.05_dt{dt}_s7.0"] = {"ema_alpha": 0.05, "diff_dt": dt, "spectral_sigma": 7.0}

    # Other alphas at the sweet spot (dt=7, σ=5.0)
    for alpha in [0.15, 0.2, 0.3]:
        cfgs[f"ema{alpha}_dt7_s5.0"] = {"ema_alpha": alpha, "diff_dt": 7, "spectral_sigma": 5.0}
        cfgs[f"ema{alpha}_dt7_s7.0"] = {"ema_alpha": alpha, "diff_dt": 7, "spectral_sigma": 7.0}

    # Larger dt (push further into user's 0.12s range → dt=8,10,12)
    for dt in [10, 12, 15]:
        cfgs[f"ema0.1_dt{dt}_s5.0"] = {"ema_alpha": 0.1, "diff_dt": dt, "spectral_sigma": 5.0}
        cfgs[f"ema0.1_dt{dt}_s7.0"] = {"ema_alpha": 0.1, "diff_dt": dt, "spectral_sigma": 7.0}

    # Full pipeline with post-EMA
    for dt in [5, 7]:
        for sigma in [5.0, 7.0]:
            cfgs[f"full_dt{dt}_s{sigma}_p0.3"] = {
                "ema_alpha": 0.1, "diff_dt": dt, "spectral_sigma": sigma, "post_ema_alpha": 0.3}

    return cfgs


def main():
    t_start = time.time()
    print("=" * 70)
    print("Macro-Avg AUC 重排名 — Simpson's Paradox 修正")
    print("=" * 70)

    recordings = load_dataset()
    print(f"Loaded {len(recordings)} recordings")
    comb_cfg = CombFeatureConfig()

    print("\n[Precompute] 预计算 STFT...")
    cache, meta = precompute_all(recordings, comb_cfg)

    configs = build_configs()
    print(f"\n[Experiment] 共 {len(configs)} 个配置")

    all_results = {}
    for i, (name, pp) in enumerate(configs.items()):
        t0 = time.time()
        all_results[name] = eval_config_full(cache, meta, pp, cep_avg=comb_cfg.cep_avg_frames)
        dt = time.time() - t0
        g = all_results[name]["global"]
        m_auc = all_results[name]["macro_smd_auc"]
        m_j = all_results[name]["macro_smd_j"]
        print(f"  [{i+1:3d}/{len(configs)}] {name:35s} "
              f"global={g['smd_auc']:.4f}  macro={m_auc:.4f}  macroJ={m_j:.3f}  "
              f"dist={g['dist_mae']:.1f}cm  ({dt:.1f}s)")

    # ─── Generate report ───
    print("\n[Report] 生成报告...")

    # Sort by macro AUC
    sorted_macro = sorted(all_results.keys(), key=lambda k: all_results[k]["macro_smd_auc"], reverse=True)
    sorted_global = sorted(all_results.keys(), key=lambda k: all_results[k]["global"]["smd_auc"], reverse=True)

    cats_order = ["static", "moving", "70deg", "80deg", "90deg"]

    L = []
    w = L.append
    w("# 完整管线实验报告 — Macro-Avg AUC 修正版\n")
    w("## 0. Simpson's Paradox 发现\n")
    w("在之前的实验中，`ema_only_0.2` 的 global pooled AUC 最高 (0.6662)，")
    w("但 diff 管线在 **每个类别** 上的 AUC 都高于 ema_only。")
    w("这是经典的 Simpson's Paradox：\n")
    w("- `ema_only` 在 static/moving 上 FPR≈100%（完全不判别），但因为这些类别的")
    w("  pattern=1 帧占总 p1 帧的 ~43%，pooled 时 AUC 被虚高。")
    w("- diff 管线真正在做 **判别**，FPR 大幅降低 (20-32% vs 44-100%)。\n")
    w("**修正方案**: 使用 macro-average AUC = mean(per-category AUC) 作为主指标。\n")

    w("---\n")
    w("## 1. Macro-Avg AUC 排名 (Top-30)\n")
    bl = all_results["baseline"]
    ema_ref = all_results["ema_only_0.2"]
    w(f"参考: baseline macro={bl['macro_smd_auc']:.4f} global={bl['global']['smd_auc']:.4f} | "
      f"ema_only_0.2 macro={ema_ref['macro_smd_auc']:.4f} global={ema_ref['global']['smd_auc']:.4f}\n")

    w("| # | 配置 | **Macro AUC** | Macro J | Global AUC | Global J | dist MAE |")
    w("|---|------|-------------|---------|------------|----------|----------|")
    for i, n in enumerate(sorted_macro[:30]):
        r = all_results[n]
        g = r["global"]
        w(f"| {i+1} | {n} | **{r['macro_smd_auc']:.4f}** | {r['macro_smd_j']:.3f} | "
          f"{g['smd_auc']:.4f} | {g['smd_j']:.3f} | {g['dist_mae']:.1f} |")

    w("\n---\n")
    w("## 2. 全类别详情 (Top 配置)\n")
    key_cfgs = ["baseline", "ema_only_0.2"] + sorted_macro[:8]
    seen = set()
    key_cfgs_dedup = []
    for c in key_cfgs:
        if c not in seen:
            seen.add(c)
            key_cfgs_dedup.append(c)
    key_cfgs = key_cfgs_dedup

    w("| 配置 | macro AUC | global AUC | " + " | ".join(cats_order) + " |")
    w("|------|-----------|------------|" + "|".join("---|" for _ in cats_order))
    for cfg_name in key_cfgs:
        r = all_results[cfg_name]
        g, pc = r["global"], r["per_cat"]
        cells = [f"{r['macro_smd_auc']:.4f}", f"{g['smd_auc']:.4f}"]
        for cat in cats_order:
            if cat in pc:
                cells.append(f"{pc[cat]['smd_auc']:.4f}")
            else:
                cells.append("—")
        w(f"| {cfg_name} | " + " | ".join(cells) + " |")

    w("")
    for cfg_name in key_cfgs:
        w(f"\n**{cfg_name}** (macro={all_results[cfg_name]['macro_smd_auc']:.4f}):")
        pc = all_results[cfg_name]["per_cat"]
        for cat in cats_order:
            if cat in pc:
                c = pc[cat]
                w(f"  {cat}: AUC={c['smd_auc']:.4f} J={c['smd_j']:.3f} "
                  f"TPR={c['smd_tpr']:.1%} FPR={c['smd_fpr']:.1%} "
                  f"p1={c['p1_rate']:.1%} MAE={c['dist_mae']:.1f}cm")

    # Parameter trends
    w("\n---\n")
    w("## 3. 参数趋势 (Macro AUC)\n")

    # σ sweep at α=0.1
    w("### EMA α=0.1, diff dt sweep, σ sweep\n")
    w("| dt \\ σ | 3.0 | 5.0 | 7.0 | 10.0 |")
    w("|--------|-----|-----|-----|------|")
    for dt in [5, 6, 7, 8]:
        cells = []
        for sigma in [3.0, 5.0, 7.0, 10.0]:
            name = f"ema0.1_dt{dt}_s{sigma}"
            if name in all_results:
                cells.append(f"{all_results[name]['macro_smd_auc']:.4f}")
            else:
                cells.append("—")
        w(f"| dt={dt} | " + " | ".join(cells) + " |")

    # Conclusion
    w("\n---\n")
    w("## 4. 结论\n")

    best = sorted_macro[0]
    br = all_results[best]
    w(f"### 🏆 Macro-Avg 最优: `{best}`\n")
    w(f"- **Macro AUC = {br['macro_smd_auc']:.4f}**")
    w(f"- Macro J = {br['macro_smd_j']:.3f}")
    w(f"- Global AUC = {br['global']['smd_auc']:.4f}")
    w(f"- dist MAE = {br['global']['dist_mae']:.1f} cm\n")
    w("Per-category:")
    for cat in cats_order:
        if cat in br["per_cat"]:
            c = br["per_cat"][cat]
            w(f"  - {cat}: AUC={c['smd_auc']:.4f} J={c['smd_j']:.3f} "
              f"TPR={c['smd_tpr']:.1%} FPR={c['smd_fpr']:.1%}")

    # Compare with old winner
    w(f"\n### 对比: `ema_only_0.2` (之前的 global AUC 冠军)\n")
    er = all_results["ema_only_0.2"]
    w(f"- Macro AUC = {er['macro_smd_auc']:.4f} (排名 {sorted_macro.index('ema_only_0.2')+1})")
    w(f"- Global AUC = {er['global']['smd_auc']:.4f}")
    delta = br["macro_smd_auc"] - er["macro_smd_auc"]
    w(f"- **Macro AUC 差距: {delta:+.4f}**\n")

    w("### Simpson's Paradox 总结\n")
    w("| 指标 | ema_only_0.2 | 最优差分管线 | 赢家 |")
    w("|------|-------------|------------|------|")
    w(f"| Global AUC | {er['global']['smd_auc']:.4f} | {br['global']['smd_auc']:.4f} | "
      f"{'ema_only' if er['global']['smd_auc'] > br['global']['smd_auc'] else 'diff'} |")
    w(f"| Macro AUC | {er['macro_smd_auc']:.4f} | {br['macro_smd_auc']:.4f} | "
      f"{'ema_only' if er['macro_smd_auc'] > br['macro_smd_auc'] else '**diff**'} |")
    for cat in cats_order:
        e_auc = er["per_cat"].get(cat, {}).get("smd_auc", 0.5)
        b_auc = br["per_cat"].get(cat, {}).get("smd_auc", 0.5)
        winner = "ema_only" if e_auc > b_auc else "**diff**"
        w(f"| {cat} AUC | {e_auc:.4f} | {b_auc:.4f} | {winner} |")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n报告已保存: {REPORT_PATH}")

    dt_total = time.time() - t_start
    print(f"\n总耗时: {dt_total:.0f}s ({dt_total/60:.1f}min)")


if __name__ == "__main__":
    main()
