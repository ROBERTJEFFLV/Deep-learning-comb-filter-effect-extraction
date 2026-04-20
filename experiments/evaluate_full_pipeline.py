#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整预处理管线实验 — EMA → 差分 → 频谱平滑 → 特征

用户设计哲学:
  1. EMA 平滑 → 压住快变噪声 (电噪声等)
  2. 差分 Δt ≈ 0.07~0.12s → 提取慢变，去除静态声源频谱
  3. 频谱高斯平滑 → 凸显 comb filter effect
  4. SMD / 倒谱 → 提取距离信息

与上一轮实验的区别: 差分后必须做频谱平滑，否则差分残差淹没 comb 信号。
"""

import sys, os, glob, time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import lfilter

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from processing.comb_feature_v2 import CombFeatureConfig

DATA_DIR = "/home/lvmingyang/March24/datasets/simulation/original_sound/copy_test_real/test_dataset"
REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "full_pipeline_experiment_report.md")
C_SPEED = 343.0


# ─────────────── dataset loading ───────────────

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


# ─────────────── feature extraction: full pipeline ───────────────

def extract_features(mag_bands, rms, pp, meta, cep_avg):
    """
    Full pipeline: [pre_ema] → log → [ema] → [diff] → [abs] → [spectral_smooth] → features

    pp keys:
      pre_ema_alpha: float or None — EMA on raw magnitude before log (None=skip)
      ema_alpha: float or None — EMA on log-spectrum (None=skip)
      diff_dt: int or None — temporal differencing (None=skip)
      diff_abs: bool — take |diff| (default False)
      spectral_sigma: float or None — Gaussian smoothing along frequency after diff (None=skip)
      post_ema_alpha: float or None — EMA after diff+smooth (None=skip)
    """
    N, B = mag_bands.shape
    cep_min = meta["cep_min"]
    cep_max = meta["cep_max"]
    ref_min = meta["ref_min"]
    ref_max = meta["ref_max"]
    quef_factor = meta["quef_factor"]

    mb = mag_bands.astype(np.float64)

    # Optional pre-log EMA on magnitudes
    if pp.get("pre_ema_alpha") is not None:
        alpha = pp["pre_ema_alpha"]
        mb = lfilter([alpha], [1, -(1 - alpha)], mb, axis=0)

    lb = np.log(mb + 1e-12)

    # EMA on log-spectrum (压快变噪声)
    if pp.get("ema_alpha") is not None:
        alpha = pp["ema_alpha"]
        lb = lfilter([alpha], [1, -(1 - alpha)], lb, axis=0)

    # Temporal differencing (提取慢变)
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

    # Spectral Gaussian smoothing (凸显 comb 结构)
    if pp.get("spectral_sigma") is not None:
        working = gaussian_filter1d(working, sigma=pp["spectral_sigma"], axis=1)

    # Optional post-diff temporal smoothing
    if pp.get("post_ema_alpha") is not None:
        alpha = pp["post_ema_alpha"]
        working = lfilter([alpha], [1, -(1 - alpha)], working, axis=0)

    # SMD
    means = working.mean(axis=1, keepdims=True)
    centered = working - means
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
            cat_data[cat] = {"smd": [], "pat": [], "dist": [], "dist_est": []}
        cat_data[cat]["smd"].append(smd)
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

    for cat, cd in cat_data.items():
        smd_c = np.concatenate(cd["smd"])
        pat_c = np.concatenate(cd["pat"])
        auc, j, tpr, fpr = compute_auc(smd_c[pat_c == 1], smd_c[pat_c == 0])
        p1c = pat_c == 1
        d_mae = float(np.mean(np.abs(
            np.concatenate(cd["dist_est"])[p1c] - np.concatenate(cd["dist"])[p1c]
        ))) if p1c.sum() > 0 else float("nan")
        res["per_cat"][cat] = {
            "smd_auc": auc, "smd_j": j, "smd_tpr": tpr, "smd_fpr": fpr,
            "p1_rate": float(p1c.mean()), "dist_mae": d_mae,
        }

    return res


# ─────────────── experiment configs ───────────────

def build_configs():
    cfgs = {}

    # References
    cfgs["baseline"] = {}
    cfgs["ema_only_0.2"] = {"ema_alpha": 0.2}

    # ═══════════════════════════════════════════════════
    # Phase 1: EMA → diff → spectral_smooth (user's pipeline)
    # Δt ≈ 0.07~0.12s → 3~6 frames (at hop=1024, sr=48000)
    # ═══════════════════════════════════════════════════

    # Grid search: ema_alpha × diff_dt × spectral_sigma
    ema_vals = [0.1, 0.15, 0.2, 0.3]
    dt_vals = [3, 4, 5, 6, 7]          # 0.064s ~ 0.149s
    sigma_vals = [1.0, 2.0, 3.0, 5.0]
    abs_vals = [False, True]

    for alpha, dt, sigma, use_abs in product(ema_vals, dt_vals, sigma_vals, abs_vals):
        abs_tag = "_abs" if use_abs else ""
        name = f"ema{alpha}_dt{dt}_s{sigma}{abs_tag}"
        cfgs[name] = {
            "ema_alpha": alpha,
            "diff_dt": dt,
            "diff_abs": use_abs,
            "spectral_sigma": sigma,
        }

    # ═══════════════════════════════════════════════════
    # Phase 2: diff → spectral_smooth (no pre-EMA)
    # ═══════════════════════════════════════════════════
    for dt in [3, 4, 5, 6]:
        for sigma in [2.0, 3.0, 5.0]:
            for use_abs in [False, True]:
                abs_tag = "_abs" if use_abs else ""
                cfgs[f"noema_dt{dt}_s{sigma}{abs_tag}"] = {
                    "diff_dt": dt,
                    "diff_abs": use_abs,
                    "spectral_sigma": sigma,
                }

    # ═══════════════════════════════════════════════════
    # Phase 3: EMA → diff → spectral_smooth → post_ema
    # ═══════════════════════════════════════════════════
    for dt in [4, 5, 6]:
        for sigma in [2.0, 3.0]:
            for post_a in [0.3, 0.5]:
                cfgs[f"full_dt{dt}_s{sigma}_p{post_a}"] = {
                    "ema_alpha": 0.2,
                    "diff_dt": dt,
                    "spectral_sigma": sigma,
                    "post_ema_alpha": post_a,
                }
                cfgs[f"full_dt{dt}_s{sigma}_p{post_a}_abs"] = {
                    "ema_alpha": 0.2,
                    "diff_dt": dt,
                    "diff_abs": True,
                    "spectral_sigma": sigma,
                    "post_ema_alpha": post_a,
                }

    return cfgs


# ─────────────── report ───────────────

def gen_report(all_results, configs):
    L = []
    w = L.append

    w("# 完整管线实验报告 — EMA → 差分 → 频谱平滑\n")

    w("## 1. 管线设计\n")
    w("```")
    w("log|Y(f)| → EMA(α) → Δt差分 → [|·|] → 频谱高斯平滑(σ) → [post-EMA] → SMD/倒谱")
    w("```\n")
    w("- **EMA**: 压住电噪声等快变干扰")
    w("- **差分 Δt=3~6帧 (64~128ms)**: 去除静态声源频谱，提取变化分量")
    w("- **频谱平滑**: 抹平差分残差，凸显 comb filter 的周期性条纹")
    w("- **|·|**: 可选取绝对值，保留调制幅度\n")

    bl = all_results["baseline"]["global"]
    ema_r = all_results["ema_only_0.2"]["global"]

    # Sort all by SMD AUC
    sorted_names = sorted(all_results.keys(),
                          key=lambda k: all_results[k]["global"].get("smd_auc", 0), reverse=True)

    w("---\n")
    w("## 2. Top-30 配置\n")
    w(f"参考: baseline AUC={bl['smd_auc']:.4f}, ema_only AUC={ema_r['smd_auc']:.4f}\n")
    w("| # | 配置 | SMD AUC | Δ(vs bl) | Δ(vs ema) | J | TPR | FPR | CPR AUC | dist MAE |")
    w("|---|------|---------|----------|-----------|---|-----|-----|---------|----------|")
    for i, n in enumerate(sorted_names[:30]):
        g = all_results[n]["global"]
        da_bl = g["smd_auc"] - bl["smd_auc"]
        da_ema = g["smd_auc"] - ema_r["smd_auc"]
        s1 = "+" if da_bl >= 0 else ""
        s2 = "+" if da_ema >= 0 else ""
        w(f"| {i+1} | {n} | {g['smd_auc']:.4f} | {s1}{da_bl:.4f} | {s2}{da_ema:.4f} | "
          f"{g['smd_j']:.3f} | {g['smd_tpr']:.1%} | {g['smd_fpr']:.1%} | "
          f"{g.get('cpr_auc',0):.4f} | {g['dist_mae']:.1f} |")

    # ─── Heatmap-style: ema × dt for fixed sigma ───
    w("\n---\n")
    w("## 3. 参数热力图 (SMD AUC)\n")

    for use_abs in [False, True]:
        abs_tag = "_abs" if use_abs else ""
        abs_label = " |Δ|" if use_abs else " Δ"
        for sigma in [1.0, 2.0, 3.0, 5.0]:
            w(f"\n### σ={sigma}{abs_label}\n")
            header = "| ema\\dt | " + " | ".join(f"dt={dt}" for dt in [3, 4, 5, 6, 7]) + " |"
            sep = "|--------|" + "|".join("------" for _ in [3,4,5,6,7]) + "|"
            w(header)
            w(sep)
            for alpha in [0.1, 0.15, 0.2, 0.3]:
                cells = []
                for dt in [3, 4, 5, 6, 7]:
                    name = f"ema{alpha}_dt{dt}_s{sigma}{abs_tag}"
                    if name in all_results:
                        auc = all_results[name]["global"]["smd_auc"]
                        cells.append(f"{auc:.4f}")
                    else:
                        cells.append("—")
                w(f"| α={alpha} | " + " | ".join(cells) + " |")

    # ─── Per-category for top configs ───
    w("\n---\n")
    w("## 4. 分类别分析 (Top 配置)\n")

    cats_order = ["static", "moving", "70deg", "80deg", "90deg"]
    key_cfgs = ["baseline", "ema_only_0.2"] + sorted_names[:5]
    # Deduplicate
    seen = set()
    key_cfgs_dedup = []
    for c in key_cfgs:
        if c not in seen:
            seen.add(c)
            key_cfgs_dedup.append(c)
    key_cfgs = key_cfgs_dedup

    w("| 配置 | global | " + " | ".join(cats_order) + " |")
    w("|------|--------| " + " | ".join("---" for _ in cats_order) + " |")
    for cfg_name in key_cfgs:
        g = all_results[cfg_name]["global"]
        pc = all_results[cfg_name]["per_cat"]
        cells = [f"{g['smd_auc']:.4f}"]
        for cat in cats_order:
            if cat in pc:
                cells.append(f"{pc[cat]['smd_auc']:.4f}/{pc[cat]['smd_j']:.3f}")
            else:
                cells.append("—")
        w(f"| {cfg_name} | " + " | ".join(cells) + " |")

    for cfg_name in key_cfgs:
        w(f"\n**{cfg_name}**:")
        pc = all_results[cfg_name]["per_cat"]
        for cat in cats_order:
            if cat in pc:
                c = pc[cat]
                w(f"  {cat}: AUC={c['smd_auc']:.4f} J={c['smd_j']:.3f} "
                  f"TPR={c['smd_tpr']:.1%} FPR={c['smd_fpr']:.1%} "
                  f"p1={c['p1_rate']:.1%} MAE={c['dist_mae']:.1f}cm")

    # ─── Conclusion ───
    w("\n---\n")
    w("## 5. 结论\n")
    best = sorted_names[0]
    bg = all_results[best]["global"]
    w(f"### 全局最优: `{best}`\n")
    w(f"- SMD AUC = {bg['smd_auc']:.4f}")
    w(f"- vs baseline: +{bg['smd_auc']-bl['smd_auc']:.4f}")
    w(f"- vs ema_only: +{bg['smd_auc']-ema_r['smd_auc']:.4f}")
    w(f"- J = {bg['smd_j']:.3f} (TPR={bg['smd_tpr']:.1%}, FPR={bg['smd_fpr']:.1%})")
    w(f"- CPR AUC = {bg.get('cpr_auc',0):.4f}, CPN AUC = {bg.get('cpn_auc',0):.4f}")
    w(f"- dist MAE = {bg['dist_mae']:.1f} cm\n")

    # Best diff pipeline specifically
    diff_names = [n for n in sorted_names if "dt" in n]
    if diff_names:
        best_diff = diff_names[0]
        dg = all_results[best_diff]["global"]
        w(f"### 最优差分管线: `{best_diff}`\n")
        w(f"- SMD AUC = {dg['smd_auc']:.4f}")
        w(f"- vs baseline: +{dg['smd_auc']-bl['smd_auc']:.4f}")
        w(f"- vs ema_only: +{dg['smd_auc']-ema_r['smd_auc']:.4f}")
        w(f"- J = {dg['smd_j']:.3f} (TPR={dg['smd_tpr']:.1%}, FPR={dg['smd_fpr']:.1%})")
        w(f"- dist MAE = {dg['dist_mae']:.1f} cm\n")

        # Per cat for best diff
        bpc = all_results[best_diff]["per_cat"]
        w("Per-category:")
        for cat in cats_order:
            if cat in bpc:
                c = bpc[cat]
                w(f"  - {cat}: AUC={c['smd_auc']:.4f} J={c['smd_j']:.3f} p1={c['p1_rate']:.1%}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n报告已保存: {REPORT_PATH}")


# ─────────────── main ───────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("完整管线实验 — EMA → 差分 → 频谱平滑")
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
        print(f"  [{i+1:3d}/{len(configs)}] {name:40s} "
              f"SMD={g['smd_auc']:.4f}  J={g['smd_j']:.3f}  "
              f"TPR={g['smd_tpr']:.1%}  FPR={g['smd_fpr']:.1%}  "
              f"dist={g['dist_mae']:.1f}cm  ({dt:.1f}s)")

    print("\n[Report] 生成报告...")
    gen_report(all_results, configs)

    dt_total = time.time() - t_start
    print(f"\n总耗时: {dt_total:.0f}s ({dt_total/60:.1f}min)")


if __name__ == "__main__":
    main()
