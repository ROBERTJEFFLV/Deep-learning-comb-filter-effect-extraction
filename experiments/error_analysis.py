#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误分析: 最优预处理管线的误判频谱分析

对 68 个真实 UAV 录音使用最优管线 EMA(0.1)→diff(15)→smooth(5.0) 进行评估,
详细分析 FP/FN/距离误差的频谱特征, 找出误判原因和改善方向.
"""

import sys, os, glob, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter
from collections import defaultdict

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = "/home/lvmingyang/March24/datasets/simulation/original_sound/copy_test_real/test_dataset"
REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "error_analysis_report.md")
C_SPEED = 343.0

# ─── Optimal pipeline parameters ───
OPTIMAL_PP = {"ema_alpha": 0.1, "diff_dt": 15, "spectral_sigma": 5.0}


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
    cols = {}
    cols["time_sec"] = df["time_sec"].values
    cols["distance_cm"] = df["distance_cm"].values
    cols["pattern"] = df["pattern_label_res"].values
    if "v_perp_mps" in df.columns:
        cols["v_perp"] = df["v_perp_mps"].values
    else:
        cols["v_perp"] = np.zeros_like(cols["time_sec"])
    return cols


def precompute_recording(wav_path, csv_path, sr=48000, n_fft=2048, hop=512):
    import soundfile as sf
    audio, file_sr = sf.read(wav_path, dtype="float64")
    if audio.ndim > 1:
        audio = audio[:, 0]

    window = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band_mask = (freqs >= 800.0) & (freqs <= 8000.0)
    n_band = int(band_mask.sum())

    n_frames = max(0, (len(audio) - n_fft) // hop + 1)
    if n_frames == 0:
        return None

    # Vectorized STFT
    idx = (np.arange(n_frames) * hop)[:, None] + np.arange(n_fft)[None, :]
    frames = audio[idx] * window[None, :]
    mags = np.abs(np.fft.rfft(frames, axis=1))
    mag_bands = mags[:, band_mask].astype(np.float32)
    rms_arr = np.sqrt(np.mean(audio[idx] ** 2, axis=1)).astype(np.float32)

    # Labels
    labels = read_labels(csv_path)
    frame_center = (np.arange(n_frames) * hop + n_fft // 2) / sr
    label_idx = np.clip(np.searchsorted(labels["time_sec"], frame_center), 0, len(labels["time_sec"]) - 1)
    pattern = labels["pattern"][label_idx].astype(np.float32)
    dist_true = labels["distance_cm"][label_idx].astype(np.float32)
    v_perp = labels["v_perp"][label_idx].astype(np.float32)

    # Cepstral analysis setup
    df_hz = sr / n_fft
    quef_factor = 1.0 / (n_band * df_hz)
    cep_min = max(2, round(0.00025 / quef_factor))
    cep_max = min(n_band // 2, round(0.004 / quef_factor) + 1)
    ref_min = cep_max + 5
    ref_max = min(n_band // 2, ref_min + 30)

    return {
        "mag_bands": mag_bands,
        "rms": rms_arr,
        "pattern": pattern,
        "dist_true": dist_true,
        "v_perp": v_perp,
        "frame_time": frame_center,
        "n_frames": n_frames,
        "n_band": n_band,
        "cep_min": cep_min,
        "cep_max": cep_max,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "quef_factor": quef_factor,
    }


def extract_features(mag_bands, pp, meta, cep_avg=4):
    """Extract features with the optimal preprocessing pipeline."""
    N, B = mag_bands.shape
    cep_min, cep_max = meta["cep_min"], meta["cep_max"]
    ref_min, ref_max = meta["ref_min"], meta["ref_max"]
    quef_factor = meta["quef_factor"]

    mb = mag_bands.astype(np.float64)
    lb = np.log(mb + 1e-12)

    # EMA
    if pp.get("ema_alpha") is not None:
        alpha = pp["ema_alpha"]
        lb = lfilter([alpha], [1, -(1 - alpha)], lb, axis=0)

    # Diff
    dt = pp.get("diff_dt", 0)
    if dt > 0:
        diff_lb = np.zeros_like(lb)
        diff_lb[dt:] = lb[dt:] - lb[:-dt]
        valid_mask = np.zeros(N, dtype=bool)
        valid_mask[dt:] = True
        working = diff_lb
    else:
        working = lb
        valid_mask = np.ones(N, dtype=bool)

    # Spectral smooth
    if pp.get("spectral_sigma") is not None:
        working = gaussian_filter1d(working, sigma=pp["spectral_sigma"], axis=1)

    # Features
    means = working.mean(axis=1, keepdims=True)
    centered = working - means
    smd = centered.std(axis=1)

    ceps = np.abs(np.fft.fft(centered, axis=1))
    from scipy.ndimage import uniform_filter1d
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
    return features, working, valid_mask


def compute_auc_with_threshold(pos, neg):
    """Compute AUC and optimal threshold."""
    if len(pos) < 5 or len(neg) < 5:
        return 0.5, 0.0, 0.0, 1.0, 0.0
    all_v = np.concatenate([pos, neg])
    ths = np.sort(np.unique(np.percentile(all_v, np.linspace(0, 100, 500))))
    best_j, best_tpr, best_fpr, best_th = -1, 0, 1, 0
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
            best_j, best_tpr, best_fpr, best_th = j, tpr, fpr, th
    fprs_a, tprs_a = np.array(fprs_l), np.array(tprs_l)
    idx = np.argsort(fprs_a)
    auc = float(np.trapz(tprs_a[idx], fprs_a[idx]))
    return auc, best_j, best_tpr, best_fpr, best_th


def main():
    t_start = time.time()
    print("=" * 70)
    print("错误分析: 最优管线 EMA(0.1)→diff(15)→smooth(5.0) 误判频谱分析")
    print("=" * 70)

    recordings = load_dataset()
    print(f"Loaded {len(recordings)} recordings")

    cats_order = ["static", "moving", "70deg", "80deg", "90deg"]

    # ── Phase 1: 收集所有录音的特征和标签 ──
    print("\n[Phase 1] 预计算特征...")
    all_data = []
    for ri, rec in enumerate(recordings):
        data = precompute_recording(rec["wav_path"], rec["csv_path"])
        if data is None:
            continue
        meta = {k: data[k] for k in ["n_band", "cep_min", "cep_max", "ref_min", "ref_max", "quef_factor"]}
        feats, working, valid = extract_features(data["mag_bands"], OPTIMAL_PP, meta)
        all_data.append({
            "name": rec["name"],
            "cat": rec["category"],
            "feats": feats,
            "working": working,
            "valid": valid,
            "pattern": data["pattern"],
            "dist_true": data["dist_true"],
            "v_perp": data["v_perp"],
            "frame_time": data["frame_time"],
            "mag_bands": data["mag_bands"],
            "rms": data["rms"],
            "meta": meta,
        })
        if (ri + 1) % 20 == 0:
            print(f"  {ri+1}/{len(recordings)}...")
    print(f"  完成 {len(all_data)} 录音")

    # ── Phase 2: 按类别计算 AUC 和最优阈值 ──
    print("\n[Phase 2] 计算 per-category AUC...")
    cat_results = {}
    for cat in cats_order:
        cat_recs = [d for d in all_data if d["cat"] == cat]
        if not cat_recs:
            continue
        smd_p1 = np.concatenate([d["feats"][:, 0][d["pattern"] == 1] for d in cat_recs])
        smd_p0 = np.concatenate([d["feats"][:, 0][d["pattern"] == 0] for d in cat_recs])
        auc, j, tpr, fpr, th = compute_auc_with_threshold(smd_p1, smd_p0)
        n_p1 = len(smd_p1)
        n_p0 = len(smd_p0)
        cat_results[cat] = {"auc": auc, "j": j, "tpr": tpr, "fpr": fpr,
                            "threshold": th, "n_p1": n_p1, "n_p0": n_p0}
        print(f"  {cat:8s}: AUC={auc:.4f}  J={j:.3f}  TPR={tpr:.2f}  FPR={fpr:.2f}  "
              f"threshold={th:.4f}  p1={n_p1}  p0={n_p0}")

    # ── Phase 3: 逐录音逐帧分类,收集 FP/FN ──
    print("\n[Phase 3] 逐帧分类,收集 FP/FN...")
    error_details = defaultdict(list)  # cat → list of error records

    for d in all_data:
        cat = d["cat"]
        if cat not in cat_results:
            continue
        th = cat_results[cat]["threshold"]
        smd = d["feats"][:, 0]
        dist_est = d["feats"][:, 3]
        pattern = d["pattern"]
        dist_true = d["dist_true"]
        valid = d["valid"]

        predicted = (smd >= th).astype(float)
        for i in range(len(smd)):
            if not valid[i]:
                continue
            actual = int(pattern[i])
            pred = int(predicted[i])
            if actual != pred:
                error_type = "FP" if pred == 1 else "FN"
                error_details[cat].append({
                    "rec": d["name"],
                    "frame_idx": i,
                    "time_sec": float(d["frame_time"][i]),
                    "error_type": error_type,
                    "smd": float(smd[i]),
                    "dist_true": float(dist_true[i]),
                    "dist_est": float(dist_est[i]),
                    "v_perp": float(d["v_perp"][i]),
                    "rms": float(d["rms"][i]),
                    "working_std": float(np.std(d["working"][i])),
                    "working_max": float(np.max(np.abs(d["working"][i]))),
                    "working_mean": float(np.mean(d["working"][i])),
                })

    # ── Phase 4: 错误模式分析 ──
    print("\n[Phase 4] 错误模式分析...")

    analysis = {}
    for cat in cats_order:
        if cat not in error_details:
            continue
        errors = error_details[cat]
        fp_errors = [e for e in errors if e["error_type"] == "FP"]
        fn_errors = [e for e in errors if e["error_type"] == "FN"]

        cat_recs = [d for d in all_data if d["cat"] == cat]
        total_p1 = sum((d["pattern"][d["valid"]] == 1).sum() for d in cat_recs)
        total_p0 = sum((d["pattern"][d["valid"]] == 0).sum() for d in cat_recs)

        result = {
            "total_p1": int(total_p1),
            "total_p0": int(total_p0),
            "n_fp": len(fp_errors),
            "n_fn": len(fn_errors),
            "fp_rate": len(fp_errors) / max(total_p0, 1),
            "fn_rate": len(fn_errors) / max(total_p1, 1),
        }

        # FP analysis: why are pattern=0 frames misclassified?
        if fp_errors:
            fp_dist = np.array([e["dist_true"] for e in fp_errors])
            fp_smd = np.array([e["smd"] for e in fp_errors])
            fp_rms = np.array([e["rms"] for e in fp_errors])
            fp_v = np.array([e["v_perp"] for e in fp_errors])
            result["fp_dist_mean"] = float(np.mean(fp_dist))
            result["fp_dist_std"] = float(np.std(fp_dist))
            result["fp_dist_range"] = [float(np.min(fp_dist)), float(np.max(fp_dist))]
            result["fp_smd_mean"] = float(np.mean(fp_smd))
            result["fp_rms_mean"] = float(np.mean(fp_rms))
            result["fp_v_perp_mean"] = float(np.mean(fp_v))

            # FP recordings distribution
            fp_recs = defaultdict(int)
            for e in fp_errors:
                fp_recs[e["rec"]] += 1
            result["fp_top_recordings"] = sorted(fp_recs.items(), key=lambda x: -x[1])[:5]

        # FN analysis: why are pattern=1 frames missed?
        if fn_errors:
            fn_dist = np.array([e["dist_true"] for e in fn_errors])
            fn_smd = np.array([e["smd"] for e in fn_errors])
            fn_rms = np.array([e["rms"] for e in fn_errors])
            fn_v = np.array([e["v_perp"] for e in fn_errors])
            result["fn_dist_mean"] = float(np.mean(fn_dist))
            result["fn_dist_std"] = float(np.std(fn_dist))
            result["fn_dist_range"] = [float(np.min(fn_dist)), float(np.max(fn_dist))]
            result["fn_smd_mean"] = float(np.mean(fn_smd))
            result["fn_rms_mean"] = float(np.mean(fn_rms))
            result["fn_v_perp_mean"] = float(np.mean(fn_v))

            # FN by distance bucket
            fn_dist_buckets = {}
            for bucket_start in range(0, 30, 5):
                bucket_end = bucket_start + 5
                in_bucket = [(e["dist_true"] >= bucket_start) and (e["dist_true"] < bucket_end) for e in fn_errors]
                fn_dist_buckets[f"{bucket_start}-{bucket_end}cm"] = sum(in_bucket)
            result["fn_by_distance"] = fn_dist_buckets

            # FN recordings distribution
            fn_recs = defaultdict(int)
            for e in fn_errors:
                fn_recs[e["rec"]] += 1
            result["fn_top_recordings"] = sorted(fn_recs.items(), key=lambda x: -x[1])[:5]

        analysis[cat] = result
        print(f"  {cat}: FP={result['n_fp']} ({result['fp_rate']:.1%})  "
              f"FN={result['n_fn']} ({result['fn_rate']:.1%})")

    # ── Phase 5: 距离估计精度分析 ──
    print("\n[Phase 5] 距离估计精度分析...")
    dist_analysis = {}
    for cat in cats_order:
        cat_recs = [d for d in all_data if d["cat"] == cat]
        if not cat_recs:
            continue
        p1_frames = []
        for d in cat_recs:
            mask = (d["pattern"] == 1) & d["valid"]
            if mask.sum() > 0:
                p1_frames.append({
                    "dist_true": d["dist_true"][mask],
                    "dist_est": d["feats"][mask, 3],
                    "smd": d["feats"][mask, 0],
                })
        if not p1_frames:
            continue
        d_true = np.concatenate([p["dist_true"] for p in p1_frames])
        d_est = np.concatenate([p["dist_est"] for p in p1_frames])
        err = d_est - d_true
        abs_err = np.abs(err)
        dist_analysis[cat] = {
            "n_frames": len(d_true),
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
            "median_err": float(np.median(abs_err)),
            "dist_true_range": [float(np.min(d_true)), float(np.max(d_true))],
            "dist_est_range": [float(np.min(d_est)), float(np.max(d_est))],
        }
        # By distance bucket
        bucket_stats = {}
        for bs in range(0, 30, 5):
            be = bs + 5
            mask_b = (d_true >= bs) & (d_true < be)
            if mask_b.sum() > 0:
                bucket_stats[f"{bs}-{be}cm"] = {
                    "n": int(mask_b.sum()),
                    "mae": float(np.mean(abs_err[mask_b])),
                    "bias": float(np.mean(err[mask_b])),
                }
        dist_analysis[cat]["by_distance"] = bucket_stats
        print(f"  {cat}: MAE={dist_analysis[cat]['mae']:.1f}cm  "
              f"RMSE={dist_analysis[cat]['rmse']:.1f}cm  "
              f"bias={dist_analysis[cat]['bias']:.1f}cm")

    # ── Phase 6: 跨类别特征分布对比 ──
    print("\n[Phase 6] 跨类别特征分布...")
    feature_dist = {}
    for cat in cats_order:
        cat_recs = [d for d in all_data if d["cat"] == cat]
        if not cat_recs:
            continue
        all_smd_p0 = np.concatenate([d["feats"][:, 0][d["pattern"] == 0] for d in cat_recs])
        all_smd_p1 = np.concatenate([d["feats"][:, 0][d["pattern"] == 1] for d in cat_recs]) if any(
            (d["pattern"] == 1).sum() > 0 for d in cat_recs) else np.array([])
        feature_dist[cat] = {
            "smd_p0_mean": float(np.mean(all_smd_p0)) if len(all_smd_p0) > 0 else 0,
            "smd_p0_std": float(np.std(all_smd_p0)) if len(all_smd_p0) > 0 else 0,
            "smd_p0_q25": float(np.percentile(all_smd_p0, 25)) if len(all_smd_p0) > 0 else 0,
            "smd_p0_q75": float(np.percentile(all_smd_p0, 75)) if len(all_smd_p0) > 0 else 0,
            "smd_p1_mean": float(np.mean(all_smd_p1)) if len(all_smd_p1) > 0 else 0,
            "smd_p1_std": float(np.std(all_smd_p1)) if len(all_smd_p1) > 0 else 0,
            "smd_p1_q25": float(np.percentile(all_smd_p1, 25)) if len(all_smd_p1) > 0 else 0,
            "smd_p1_q75": float(np.percentile(all_smd_p1, 75)) if len(all_smd_p1) > 0 else 0,
            "separation": (float(np.mean(all_smd_p1)) - float(np.mean(all_smd_p0))) /
                          (float(np.std(all_smd_p0)) + 1e-12) if len(all_smd_p1) > 0 else 0,
        }
        print(f"  {cat}: p0_mean={feature_dist[cat]['smd_p0_mean']:.4f}  "
              f"p1_mean={feature_dist[cat]['smd_p1_mean']:.4f}  "
              f"separation={feature_dist[cat]['separation']:.2f}σ")

    # ── Phase 7: 频谱形态分析 (p1 vs p0) ──
    print("\n[Phase 7] 频谱形态分析...")
    spectral_analysis = {}
    for cat in cats_order:
        cat_recs = [d for d in all_data if d["cat"] == cat]
        if not cat_recs:
            continue
        # Average working spectra for p0 and p1
        p0_spectra = []
        p1_spectra = []
        for d in cat_recs:
            valid = d["valid"]
            for i in range(len(d["pattern"])):
                if not valid[i]:
                    continue
                if d["pattern"][i] == 1:
                    p1_spectra.append(d["working"][i])
                else:
                    p0_spectra.append(d["working"][i])

        result = {}
        if p0_spectra:
            p0_arr = np.array(p0_spectra)
            result["p0_mean_spectrum"] = np.mean(p0_arr, axis=0)
            result["p0_std_spectrum"] = np.std(p0_arr, axis=0)
            # Cepstral analysis of mean p0 spectrum
            c0 = result["p0_mean_spectrum"] - result["p0_mean_spectrum"].mean()
            cep_p0 = np.abs(np.fft.fft(c0))
            result["p0_cep_peak"] = float(np.max(cep_p0[2:len(cep_p0)//2]))
            result["p0_cep_mean"] = float(np.mean(cep_p0[2:len(cep_p0)//2]))

        if p1_spectra:
            p1_arr = np.array(p1_spectra)
            result["p1_mean_spectrum"] = np.mean(p1_arr, axis=0)
            result["p1_std_spectrum"] = np.std(p1_arr, axis=0)
            c1 = result["p1_mean_spectrum"] - result["p1_mean_spectrum"].mean()
            cep_p1 = np.abs(np.fft.fft(c1))
            result["p1_cep_peak"] = float(np.max(cep_p1[2:len(cep_p1)//2]))
            result["p1_cep_mean"] = float(np.mean(cep_p1[2:len(cep_p1)//2]))

            # Spectrum difference: what spectral regions differ most?
            if p0_spectra:
                diff_spectrum = result["p1_mean_spectrum"] - result["p0_mean_spectrum"]
                result["diff_max_bin"] = int(np.argmax(np.abs(diff_spectrum)))
                result["diff_abs_mean"] = float(np.mean(np.abs(diff_spectrum)))

        spectral_analysis[cat] = result
        if "p1_cep_peak" in result and "p0_cep_peak" in result:
            print(f"  {cat}: p0_cep_peak={result['p0_cep_peak']:.4f}  "
                  f"p1_cep_peak={result['p1_cep_peak']:.4f}  "
                  f"ratio={result['p1_cep_peak']/(result['p0_cep_peak']+1e-12):.2f}")

    # ── Phase 8: 时间连续性分析 ──
    print("\n[Phase 8] 时间连续性分析...")
    temporal_analysis = {}
    for cat in cats_order:
        cat_recs = [d for d in all_data if d["cat"] == cat]
        if not cat_recs:
            continue

        # Track FP/FN burst lengths
        if cat not in error_details:
            continue
        th = cat_results[cat]["threshold"]

        burst_lengths_fp = []
        burst_lengths_fn = []

        for d in cat_recs:
            smd = d["feats"][:, 0]
            pattern = d["pattern"]
            valid = d["valid"]
            pred = (smd >= th).astype(int)

            current_burst = 0
            current_type = None
            for i in range(len(smd)):
                if not valid[i]:
                    if current_burst > 0:
                        if current_type == "FP":
                            burst_lengths_fp.append(current_burst)
                        elif current_type == "FN":
                            burst_lengths_fn.append(current_burst)
                    current_burst = 0
                    current_type = None
                    continue

                err_type = None
                if pred[i] != int(pattern[i]):
                    err_type = "FP" if pred[i] == 1 else "FN"

                if err_type == current_type and err_type is not None:
                    current_burst += 1
                else:
                    if current_burst > 0:
                        if current_type == "FP":
                            burst_lengths_fp.append(current_burst)
                        elif current_type == "FN":
                            burst_lengths_fn.append(current_burst)
                    current_burst = 1 if err_type else 0
                    current_type = err_type

            if current_burst > 0 and current_type:
                if current_type == "FP":
                    burst_lengths_fp.append(current_burst)
                elif current_type == "FN":
                    burst_lengths_fn.append(current_burst)

        temporal_analysis[cat] = {
            "fp_bursts": len(burst_lengths_fp),
            "fp_mean_length": float(np.mean(burst_lengths_fp)) if burst_lengths_fp else 0,
            "fp_max_length": int(np.max(burst_lengths_fp)) if burst_lengths_fp else 0,
            "fn_bursts": len(burst_lengths_fn),
            "fn_mean_length": float(np.mean(burst_lengths_fn)) if burst_lengths_fn else 0,
            "fn_max_length": int(np.max(burst_lengths_fn)) if burst_lengths_fn else 0,
        }
        ta = temporal_analysis[cat]
        print(f"  {cat}: FP_bursts={ta['fp_bursts']}(mean_len={ta['fp_mean_length']:.1f})  "
              f"FN_bursts={ta['fn_bursts']}(mean_len={ta['fn_mean_length']:.1f})")

    # ── Phase 9: 能量和信噪比与误判关系 ──
    print("\n[Phase 9] 能量-误判关系...")
    energy_analysis = {}
    for cat in cats_order:
        if cat not in error_details:
            continue
        cat_recs = [d for d in all_data if d["cat"] == cat]
        all_rms_correct = []
        all_rms_error = []
        for d in cat_recs:
            th = cat_results[cat]["threshold"]
            smd = d["feats"][:, 0]
            pred = (smd >= th).astype(int)
            valid = d["valid"]
            for i in range(len(smd)):
                if not valid[i]:
                    continue
                is_error = (pred[i] != int(d["pattern"][i]))
                if is_error:
                    all_rms_error.append(d["rms"][i])
                else:
                    all_rms_correct.append(d["rms"][i])

        if all_rms_error and all_rms_correct:
            energy_analysis[cat] = {
                "correct_rms_mean": float(np.mean(all_rms_correct)),
                "error_rms_mean": float(np.mean(all_rms_error)),
                "correct_rms_std": float(np.std(all_rms_correct)),
                "error_rms_std": float(np.std(all_rms_error)),
            }
            ea = energy_analysis[cat]
            print(f"  {cat}: correct_rms={ea['correct_rms_mean']:.4f}±{ea['correct_rms_std']:.4f}  "
                  f"error_rms={ea['error_rms_mean']:.4f}±{ea['error_rms_std']:.4f}")

    # ═══ Generate Report ═══
    print("\n[Report] 生成报告...")
    elapsed = time.time() - t_start

    L = []
    w = L.append

    w("# 错误分析报告: 最优预处理管线\n")
    w(f"管线: `EMA(α={OPTIMAL_PP['ema_alpha']}) → diff(dt={OPTIMAL_PP['diff_dt']}) → smooth(σ={OPTIMAL_PP['spectral_sigma']})`\n")
    w(f"数据: {len(recordings)} 个真实 UAV 录音  |  运行时间: {elapsed:.0f}s\n")

    w("---\n")
    w("## 1. Per-Category AUC 概览\n")
    w("| Category | AUC | J-stat | TPR | FPR | Threshold | p1帧 | p0帧 |")
    w("|----------|-----|--------|-----|-----|-----------|------|------|")
    macro_auc_vals = []
    for cat in cats_order:
        if cat in cat_results:
            r = cat_results[cat]
            w(f"| {cat} | {r['auc']:.4f} | {r['j']:.3f} | {r['tpr']:.2f} | {r['fpr']:.2f} | "
              f"{r['threshold']:.4f} | {r['n_p1']} | {r['n_p0']} |")
            macro_auc_vals.append(r['auc'])
    macro_auc = np.mean(macro_auc_vals) if macro_auc_vals else 0
    w(f"\n**Macro-avg AUC = {macro_auc:.4f}**\n")

    w("---\n")
    w("## 2. FP/FN 错误统计\n")
    w("| Category | Total p1 | Total p0 | FP | FN | FP Rate | FN Rate |")
    w("|----------|----------|----------|----|----|---------|---------|")
    for cat in cats_order:
        if cat in analysis:
            a = analysis[cat]
            w(f"| {cat} | {a['total_p1']} | {a['total_p0']} | {a['n_fp']} | {a['n_fn']} | "
              f"{a['fp_rate']:.1%} | {a['fn_rate']:.1%} |")

    w("\n### 2.1 FP 错误分析 (pattern=0 被误判为 pattern=1)\n")
    for cat in cats_order:
        if cat not in analysis or analysis[cat]["n_fp"] == 0:
            continue
        a = analysis[cat]
        w(f"**{cat}**: {a['n_fp']} FP 帧")
        if "fp_dist_mean" in a:
            w(f"- 真实距离: 均值={a['fp_dist_mean']:.1f}cm, 范围={a['fp_dist_range']}")
            w(f"- SMD 均值={a['fp_smd_mean']:.4f}, RMS={a['fp_rms_mean']:.4f}")
            w(f"- v_perp 均值={a['fp_v_perp_mean']:.4f} m/s")
        if "fp_top_recordings" in a:
            w(f"- 集中录音: {a['fp_top_recordings'][:3]}")
        w("")

    w("### 2.2 FN 错误分析 (pattern=1 被漏判)\n")
    for cat in cats_order:
        if cat not in analysis or analysis[cat]["n_fn"] == 0:
            continue
        a = analysis[cat]
        w(f"**{cat}**: {a['n_fn']} FN 帧")
        if "fn_dist_mean" in a:
            w(f"- 真实距离: 均值={a['fn_dist_mean']:.1f}cm, 范围={a['fn_dist_range']}")
            w(f"- SMD 均值={a['fn_smd_mean']:.4f}, RMS={a['fn_rms_mean']:.4f}")
        if "fn_by_distance" in a:
            w(f"- 按距离分布: {a['fn_by_distance']}")
        if "fn_top_recordings" in a:
            w(f"- 集中录音: {a['fn_top_recordings'][:3]}")
        w("")

    w("---\n")
    w("## 3. 特征分布分析\n")
    w("| Category | p0_smd_mean | p0_smd_std | p1_smd_mean | p1_smd_std | Separation(σ) |")
    w("|----------|-------------|------------|-------------|------------|---------------|")
    for cat in cats_order:
        if cat in feature_dist:
            fd = feature_dist[cat]
            w(f"| {cat} | {fd['smd_p0_mean']:.4f} | {fd['smd_p0_std']:.4f} | "
              f"{fd['smd_p1_mean']:.4f} | {fd['smd_p1_std']:.4f} | {fd['separation']:.2f} |")

    w("\n**解读**: separation 值越高,分类越容易。< 0.5σ 表示严重重叠。\n")

    w("---\n")
    w("## 4. 距离估计精度\n")
    w("| Category | N帧 | MAE(cm) | RMSE(cm) | Bias(cm) | Median(cm) |")
    w("|----------|-----|---------|----------|----------|------------|")
    for cat in cats_order:
        if cat in dist_analysis:
            da = dist_analysis[cat]
            w(f"| {cat} | {da['n_frames']} | {da['mae']:.1f} | {da['rmse']:.1f} | "
              f"{da['bias']:.1f} | {da['median_err']:.1f} |")

    w("\n### 4.1 距离估计按距离桶分析\n")
    for cat in cats_order:
        if cat in dist_analysis and "by_distance" in dist_analysis[cat]:
            w(f"\n**{cat}**:")
            for bucket, stats in dist_analysis[cat]["by_distance"].items():
                w(f"  - {bucket}: N={stats['n']}, MAE={stats['mae']:.1f}cm, bias={stats['bias']:.1f}cm")

    w("\n---\n")
    w("## 5. 频谱形态分析\n")
    for cat in cats_order:
        if cat not in spectral_analysis:
            continue
        sa = spectral_analysis[cat]
        w(f"\n### {cat}")
        if "p0_cep_peak" in sa:
            w(f"- p0 倒谱峰值: {sa['p0_cep_peak']:.4f}")
        if "p1_cep_peak" in sa:
            w(f"- p1 倒谱峰值: {sa['p1_cep_peak']:.4f}")
            if "p0_cep_peak" in sa:
                ratio = sa["p1_cep_peak"] / (sa["p0_cep_peak"] + 1e-12)
                w(f"- 倒谱峰值比: {ratio:.2f}x")
        if "diff_abs_mean" in sa:
            w(f"- 频谱差异绝对值均值: {sa['diff_abs_mean']:.6f}")

    w("\n---\n")
    w("## 6. 时间连续性分析\n")
    w("| Category | FP Bursts | FP Mean Len | FP Max Len | FN Bursts | FN Mean Len | FN Max Len |")
    w("|----------|-----------|-------------|------------|-----------|-------------|------------|")
    for cat in cats_order:
        if cat in temporal_analysis:
            ta = temporal_analysis[cat]
            w(f"| {cat} | {ta['fp_bursts']} | {ta['fp_mean_length']:.1f} | {ta['fp_max_length']} | "
              f"{ta['fn_bursts']} | {ta['fn_mean_length']:.1f} | {ta['fn_max_length']} |")

    w("\n**解读**: 长 burst 说明误判是系统性的 (持续多帧), 短 burst 说明是随机/噪声导致。\n")

    w("---\n")
    w("## 7. 能量与误判关系\n")
    w("| Category | Correct RMS | Error RMS | 差异 |")
    w("|----------|-------------|-----------|------|")
    for cat in cats_order:
        if cat in energy_analysis:
            ea = energy_analysis[cat]
            diff_pct = (ea["error_rms_mean"] - ea["correct_rms_mean"]) / (ea["correct_rms_mean"] + 1e-12) * 100
            w(f"| {cat} | {ea['correct_rms_mean']:.4f}±{ea['correct_rms_std']:.4f} | "
              f"{ea['error_rms_mean']:.4f}±{ea['error_rms_std']:.4f} | {diff_pct:+.1f}% |")

    w("\n---\n")
    w("## 8. 关键发现与改善建议\n")
    w("\n### 误判根因分析\n")
    w("(由脚本自动识别的模式)\n\n")

    # Auto-detect patterns
    issues = []

    # Check if any category has very low separation
    for cat in cats_order:
        if cat in feature_dist and feature_dist[cat]["separation"] < 0.5:
            issues.append(f"- **{cat}**: SMD p0/p1 分离度仅 {feature_dist[cat]['separation']:.2f}σ, "
                         f"分类本质困难。可能原因: 该角度/运动模式下梳状滤波效应极弱。")

    # Check if FN correlates with distance
    for cat in cats_order:
        if cat in analysis and analysis[cat].get("fn_dist_mean", 0) > 15:
            issues.append(f"- **{cat}**: FN 帧平均距离 {analysis[cat]['fn_dist_mean']:.1f}cm, "
                         f"远距离处梳状滤波效应减弱是主要漏检原因。")

    # Check for bursty errors
    for cat in cats_order:
        if cat in temporal_analysis and temporal_analysis[cat]["fp_mean_length"] > 10:
            issues.append(f"- **{cat}**: FP burst 平均长度 {temporal_analysis[cat]['fp_mean_length']:.1f} 帧, "
                         f"说明存在持续性虚警 — 某些非梳状信号特征被误识别。")

    # Check for energy correlation
    for cat in cats_order:
        if cat in energy_analysis:
            ea = energy_analysis[cat]
            diff_pct = abs(ea["error_rms_mean"] - ea["correct_rms_mean"]) / (ea["correct_rms_mean"] + 1e-12) * 100
            if diff_pct > 30:
                issues.append(f"- **{cat}**: 误判帧能量与正确帧差异 {diff_pct:.0f}%, "
                             f"能量/SNR 可能是有用的辅助判据。")

    for issue in issues:
        w(issue)

    if not issues:
        w("- 未检测到显著的系统性误判模式")

    w("\n### 网络设计改善建议\n")
    w("1. **多通道输入优势**: 新管线的 4 通道 (raw, preprocessed, dt1, abs) 比旧管线更有信息量\n")
    w("2. **能量特征补充**: energy_proxy / snr_proxy 作为辅助输入可帮助区分低信噪比场景\n")
    w("3. **时间上下文**: 长 FP/FN burst 表明帧级分类不足, 需要序列级别的时间建模\n")
    w("4. **距离感知**: FN 集中在远距离处, 模型应学习距离与信号强度的关系\n")

    report_text = "\n".join(L)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存: {REPORT_PATH}")
    print(f"总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
