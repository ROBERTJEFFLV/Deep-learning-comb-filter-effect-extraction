#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_process_kalman.py
---------------------------------
将本仓库内的在线处理流程（processing.audio_processor.AudioProcessor）改造成离线批处理：
- 读取单个 .wav 文件（在 INLINE_RUN['input'] 中指定）
- 以与在线一致的窗口/步长与算法（频域特征 + 方向判定 + 窗口余弦拟合 + Kalman 平滑）运行
- 将逐帧结果写入 CSV（包含时间、方向、ω、A、r2、rmse、原始 distance、KF 平滑距离等）
- LS/KF/RTS/输出门控参数直接从 config.py 读取

配置后直接运行本脚本，无需命令行参数。
"""

from __future__ import annotations

import os
import sys
import csv
import time
import json
from pathlib import Path
from typing import Optional, Iterable, Tuple, List, Dict
from queue import Queue, Empty
from copy import deepcopy

import numpy as np

from config import (
    SR, N_FFT, HOP_LEN,
    LS_PARAMS, KF_PARAMS, RTS_PARAMS, OUTPUT_FILTER
)

# 直接复用在线处理的核心实现，确保一致性
from processing.audio_processor import AudioProcessor

# ============ 参考轨迹加载与对齐 ============
def _safe_float(x: str) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def load_reference_range(csv_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    加载参考距离文件 range_1.csv，返回 (t_ref_secs, d_ref_cm)；若找不到则返回 None。
    预期列名（容错识别）：
      时间列: t, time, time_sec, timestamp
      距离列: distance_cm, d_ref, range, distance, dist, distance_mm(自动 /10 转 cm)
      过滤列: valid (若存在，仅保留 True/1)
    """
    if not os.path.exists(csv_path):
        print(f"[offline][ref] reference file not found: {csv_path} — skip comparison")
        return None

    times: List[float] = []
    dists_cm: List[float] = []
    time_col = None
    dist_col_cm = None
    dist_col_mm = None
    valid_col = None  # 保留但不强制过滤

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[offline][ref] empty reference file: {csv_path}")
            return None

        # 识别列
        lowered = [h.strip().lower() for h in header]
        def _find(cols: List[str]) -> Optional[int]:
            for name in cols:
                if name in lowered:
                    return lowered.index(name)
            return None

        t_idx = _find(['t', 'time', 'time_sec', 'timestamp'])
        cm_idx = _find(['d_ref', 'distance_cm', 'range', 'distance', 'dist'])
        mm_idx = _find(['distance_mm'])
        v_idx  = _find(['valid'])

        if t_idx is None:
            print(f"[offline][ref] no time column found in {csv_path}")
            return None
        if (cm_idx is None) and (mm_idx is None):
            print(f"[offline][ref] no distance column found in {csv_path}")
            return None

        for row in reader:
            if not row:
                continue
            t_val = _safe_float(row[t_idx]) if t_idx < len(row) else None
            if t_val is None:
                continue
            # 不使用 valid 过滤，参考文件部分数据 valid=0 但距离仍可用
            d_val_cm = None
            if cm_idx is not None and cm_idx < len(row):
                d_val_cm = _safe_float(row[cm_idx])
            if d_val_cm is None and mm_idx is not None and mm_idx < len(row):
                d_val_mm = _safe_float(row[mm_idx])
                if d_val_mm is not None:
                    d_val_cm = d_val_mm / 10.0
            if d_val_cm is None:
                continue
            times.append(t_val)
            dists_cm.append(d_val_cm)

    if not times:
        print(f"[offline][ref] no valid rows parsed in {csv_path}")
        return None

    t_ref = np.asarray(times, dtype=np.float64)
    d_ref = np.asarray(dists_cm, dtype=np.float64)
    # 确保按时间递增
    order = np.argsort(t_ref)
    t_ref = t_ref[order]
    d_ref = d_ref[order]
    print(f"[offline][ref] loaded reference: {len(t_ref)} pts, t=[{t_ref[0]:.2f},{t_ref[-1]:.2f}]s, d=[{d_ref.min():.2f},{d_ref.max():.2f}]cm")
    return t_ref, d_ref


def evaluate_against_reference(
    est_samples: List[Tuple[float, float]],
    ref: Optional[Tuple[np.ndarray, np.ndarray]],
    delay_sec: float = 0.5,
) -> None:
    """
    将 (t_est, d_est_cm) 与参考 (t_ref, d_ref_cm) 对齐比较，考虑估计延迟 delay_sec。
    规则：比较时使用真值 t_ref = t_est - delay；若超出参考时间范围则跳过。
    打印 MAE/RMSE/偏置等统计，以及若干误差最大的样本。
    """
    if ref is None:
        print("[offline][ref] no reference provided — skip comparison")
        return

    t_ref, d_ref = ref
    if len(t_ref) < 2:
        print("[offline][ref] reference too short — skip comparison")
        return

    t0, t1 = float(t_ref[0]), float(t_ref[-1])
    comp: List[Dict[str, float]] = []

    for t_est, d_est in est_samples:
        t_truth = t_est - delay_sec
        if (t_truth < t0) or (t_truth > t1):
            continue
        # 线性插值真值
        d_true = float(np.interp(t_truth, t_ref, d_ref))
        err = d_est - d_true
        comp.append({
            't_est': t_est,
            't_true': t_truth,
            'd_est': d_est,
            'd_true': d_true,
            'err': err,
            'abs_err': abs(err),
        })

    if not comp:
        print("[offline][ref] no overlapping time after delay — skip comparison")
        return

    errs = np.array([c['err'] for c in comp], dtype=np.float64)
    abs_errs = np.abs(errs)
    mae = float(abs_errs.mean())
    rmse = float(np.sqrt((errs**2).mean()))
    bias = float(errs.mean())
    med = float(np.median(abs_errs))
    p95 = float(np.percentile(abs_errs, 95))
    worst = max(comp, key=lambda c: c['abs_err'])

    print("\n[offline][ref] 对比 range_1.csv（考虑估计延迟 0.5s）")
    print(f"  样本数: {len(comp)}")
    print(f"  MAE: {mae:.3f} cm  |  RMSE: {rmse:.3f} cm  |  偏置: {bias:.3f} cm")
    print(f"  中位绝对误差: {med:.3f} cm  |  P95: {p95:.3f} cm  |  最大绝对误差: {worst['abs_err']:.3f} cm")
    print("  误差最大样本:")
    print(f"    t_est={worst['t_est']:.3f}s, t_true={worst['t_true']:.3f}s, est={worst['d_est']:.3f}cm, true={worst['d_true']:.3f}cm, err={worst['err']:.3f}cm")

# ============ 内联配置（请在此设置路径） ============
BASE_DIR = Path(__file__).resolve().parent
INLINE_RUN = {
    # 输入：单个 .wav 文件路径
    "input": str(BASE_DIR / "rec_1.wav"),

    # 输出 CSV 路径；为空则默认写到当前目录 offline_results.csv
    "output": str(BASE_DIR / "sine_fit_log.csv"),
}


def load_kalman_params():
    """
    返回 config.py 中的 LS/KF/RTS/输出门控配置副本（不再依赖 JSON 文件）。
    """
    return (
        deepcopy(LS_PARAMS),
        deepcopy(KF_PARAMS),
        deepcopy(RTS_PARAMS),
        deepcopy(OUTPUT_FILTER),
    )


def _try_import_soundfile():
    try:
        import soundfile as sf  # type: ignore
        return sf
    except Exception:
        return None


def _try_import_librosa():
    try:
        import librosa  # type: ignore
        return librosa
    except Exception:
        return None


def _try_import_scipy_signal():
    try:
        from scipy import signal  # type: ignore
        return signal
    except Exception:
        return None


def load_audio_mono_float(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    加载音频为 mono/float32，必要时重采样到 target_sr。
    优先 soundfile -> librosa -> scipy.io.wavfile -> wave。
    """
    sf = _try_import_soundfile()
    if sf is not None:
        data, sr = sf.read(path, dtype='float32', always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        if sr != target_sr:
            librosa = _try_import_librosa()
            sig = _try_import_scipy_signal()
            if librosa is not None:
                data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr).astype(np.float32)
                sr = target_sr
            elif sig is not None:
                # polyphase resample
                g = np.gcd(int(sr), int(target_sr))
                up = int(target_sr // g)
                down = int(sr // g)
                data = sig.resample_poly(data, up=up, down=down).astype(np.float32)
                sr = target_sr
            else:
                raise RuntimeError(f"需要重采样 {sr}-> {target_sr}Hz，但未找到 librosa 或 scipy.signal")
        return data.astype(np.float32), int(sr)

    # fallback: librosa
    librosa = _try_import_librosa()
    if librosa is not None:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32), int(sr)

    # fallback: scipy.io.wavfile
    try:
        from scipy.io import wavfile  # type: ignore
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        # int -> float
        if np.issubdtype(data.dtype, np.integer):
            maxv = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max(1, float(maxv))
        else:
            data = data.astype(np.float32)
        if sr != target_sr:
            sig = _try_import_scipy_signal()
            if sig is None:
                raise RuntimeError(f"需要重采样 {sr}->{target_sr}Hz，但未找到 scipy.signal")
            g = np.gcd(int(sr), int(target_sr))
            up = int(target_sr // g)
            down = int(sr // g)
            data = sig.resample_poly(data, up=up, down=down).astype(np.float32)
            sr = target_sr
        return data.astype(np.float32), int(sr)
    except Exception:
        pass

    # ultimate fallback: wave (PCM 16)
    import wave
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1).astype(np.float32)
        if sr != target_sr:
            sig = _try_import_scipy_signal()
            if sig is None:
                raise RuntimeError(f"需要重采样 {sr}->{target_sr}Hz，但未找到 scipy.signal")
            g = np.gcd(int(sr), int(target_sr))
            up = int(target_sr // g)
            down = int(sr // g)
            data = sig.resample_poly(data, up=up, down=down).astype(np.float32)
            sr = target_sr
    return data.astype(np.float32), int(sr)


def _make_running_flag():
    flag = {'run': True}
    def running(stop: bool = False) -> bool:
        if stop:
            flag['run'] = False
        return flag['run']
    return running

def _dump_params(ap: "AudioProcessor",
                 ls_params: dict | None,
                 kf_params: dict | None,
                 rts_params: dict | None,
                 output_filter: dict | None):
    """打印本次运行的‘实际生效参数’（含默认回落后的ap内部值），便于复现实验。"""
    try:
        print("\n[offline][params] 使用的配置（实际生效）")
        print("  来源: config.py 中的 LS_PARAMS / KF_PARAMS / RTS_PARAMS / OUTPUT_FILTER")
        # 基本采样/步长
        hop_sec = HOP_LEN / SR
        print(f"  采样: SR={SR}, N_FFT={N_FFT}, HOP_LEN={HOP_LEN} (hop_sec={hop_sec:.6f}s)")

        # LS / 窗口相关（取 ap 内部的最终值）
        print("  [LS/Window]")
        print(f"    window_size={ap.window_size}, window_hop={ap.window_hop}, window_pick={ap.window_pick}")
        print(f"    d_min_m={ap.d_min_m}, oversample={ap.oversample}, gaussian_sigma={ap.gaussian_sigma}")
        print(f"    local_radius_first={ap.local_radius_first}, local_radius_segment={ap.local_radius_segment}")

        # 输出门控/缺测分段
        print("  [Gating/Seg]")
        print(f"    min_amplitude={getattr(ap, 'min_amplitude', None)}, _missing_threshold={getattr(ap, '_missing_threshold', None)}")
        print(f"    direction_required=True")

        # KF/RTS（从 ap.kf/ap.rts 取）
        print("  [KF/RTS]")
        v_max = getattr(ap, 'v_max', None)
        a_max = getattr(ap, 'a_max', None)
        dt_window = getattr(ap, 'dt_window', None)
        R = getattr(ap.kf, 'R', None)
        sigma_a = getattr(ap.kf, 'sigma_a', None)
        print(f"    KF: R={R}, sigma_a={sigma_a}, v_max={v_max}, a_max={a_max}, d_min_cm={getattr(ap, 'd_min_cm', None)}")
        print(f"    RTS: lag={getattr(ap.rts, 'lag', None)}, dt_window={dt_window}")

        # 原始配置（供快速比对）
        def _safe_d(d):
            try:
                return json.dumps(d or {}, ensure_ascii=False)
            except Exception:
                return str(d)

        print("  [配置原始片段]")
        print(f"    ls_params={_safe_d(ls_params)}")
        print(f"    kf_params={_safe_d(kf_params)}")
        print(f"    rts_params={_safe_d(rts_params)}")
        print(f"    output_filter={_safe_d(output_filter)}")
    except Exception as e:
        print(f"[offline][params] dump failed: {e}")


def process_one_file(wav_path: str, writer, include_header_once: dict):
    """
    将单个 WAV 以在线同款流程离线处理，并写入 CSV。
    writer: csv.DictWriter
    include_header_once: 用于仅写一次表头的可变字典标志
    """
    print(f"[offline] processing: {wav_path}")
    samples, sr = load_audio_mono_float(wav_path, target_sr=SR)
    assert sr == SR, f"resample failed: {sr} != {SR}"

    # 加载参数配置（直接使用 config.py 中的常量）
    ls_params, kf_params, rts_params, output_filter = load_kalman_params()

    # 队列/线程化处理器（重用在线 AudioProcessor）
    proc_q: Queue = Queue(maxsize=1000)  # 输入队列不需要太大
    frame_q: Queue = Queue(maxsize=1000)  # 输出队列也不需要太大，边处理边消费
    running = _make_running_flag()
    ap = AudioProcessor(proc_q, frame_q, running, ls_params, kf_params, rts_params, output_filter)

    # 尝试加载参考轨迹（同目录下 range_1.csv）
    ref_path = os.path.join(os.path.dirname(wav_path), "range_1.csv")
    ref = load_reference_range(ref_path)
    # 收集用于对比的估计样本 (t_est, d_est_cm)
    est_samples: List[Tuple[float, float]] = []
    ap.start()

    # 写入表头（如果是第一个文件）
    if not include_header_once.get('done'):
        writer.writeheader()
        include_header_once['done'] = True

    # 将音频分块推入在线处理器（HOP_LEN 一帧）+ 同时消费输出队列
    total = int(np.ceil(len(samples) / HOP_LEN))
    idx = 0
    frame_idx = 0
    
    for i in range(0, len(samples), HOP_LEN):
        block = samples[i:i+HOP_LEN]
        if len(block) < HOP_LEN:
            pad = np.zeros(HOP_LEN - len(block), dtype=np.float32)
            block = np.concatenate([block, pad], axis=0)
        
        # 推入处理队列
        try:
            proc_q.put(block.astype(np.float32), timeout=1.0)
        except Exception:
            # 若队列满，稍等再试
            time.sleep(0.01)
            proc_q.put(block.astype(np.float32), timeout=1.0)
        idx += 1
        
        # 同时消费输出队列，避免队列堆积
        while True:
            try:
                fr = frame_q.get_nowait()
                row = {
                    'file': os.path.basename(wav_path),
                    'frame_idx': frame_idx,
                    't': fr.get('t'),
                    'direction_d1': fr.get('direction_d1'),
                    'omega': fr.get('omega'),
                    'A': fr.get('A'),
                    'r2': fr.get('r2'),
                    'rmse': fr.get('rmse'),
                    'distance': fr.get('distance'),
                    'distance_kf': fr.get('distance_kf'),
                    'velocity_kf': fr.get('velocity_kf'),
                    'acceleration_kf': fr.get('acceleration_kf'),
                    'has_measure': fr.get('has_measure'),
                    'n_valid_bins': fr.get('n_valid_bins'),
                    'valid_band_hz': fr.get('valid_band_hz'),
                    'sum_abs_d1': fr.get('sum_abs_d1'),
                    'is_sound_present': fr.get('is_sound_present'),
                }
                writer.writerow(row)
                # 收集估计值用于与参考对比（None 不参与）
                t_est = fr.get('t')
                d_est = fr.get('distance_kf') if fr.get('distance_kf') is not None else fr.get('distance')
                if (t_est is not None) and (d_est is not None):
                    try:
                        est_samples.append((float(t_est), float(d_est)))
                    except Exception:
                        pass
                frame_idx += 1
            except Empty:
                break  # 输出队列已空，继续处理下一个音频块

    # 等待处理器处理完所有输入
    while not proc_q.empty():
        time.sleep(0.05)
        # 同时继续消费输出
        while True:
            try:
                fr = frame_q.get_nowait()
                row = {
                    'file': os.path.basename(wav_path),
                    'frame_idx': frame_idx,
                    't': fr.get('t'),
                    'direction_d1': fr.get('direction_d1'),
                    'omega': fr.get('omega'),
                    'A': fr.get('A'),
                    'r2': fr.get('r2'),
                    'rmse': fr.get('rmse'),
                    'distance': fr.get('distance'),
                    'distance_kf': fr.get('distance_kf'),
                    'velocity_kf': fr.get('velocity_kf'),
                    'acceleration_kf': fr.get('acceleration_kf'),
                    'has_measure': fr.get('has_measure'),
                    'n_valid_bins': fr.get('n_valid_bins'),
                    'valid_band_hz': fr.get('valid_band_hz'),
                    'sum_abs_d1': fr.get('sum_abs_d1'),
                    'is_sound_present': fr.get('is_sound_present'),
                }
                writer.writerow(row)
                # 收集估计值（None 不参与）
                t_est = fr.get('t')
                d_est = fr.get('distance_kf') if fr.get('distance_kf') is not None else fr.get('distance')
                if (t_est is not None) and (d_est is not None):
                    try:
                        est_samples.append((float(t_est), float(d_est)))
                    except Exception:
                        pass
                frame_idx += 1
            except Empty:
                break

    # 通知处理线程退出
    running(stop=True)
    time.sleep(0.1)

    # 最后再消费一次，确保所有输出都被写入
    idle_rounds = 0
    while idle_rounds < 20:  # 最多等待1秒（20 * 0.05）
        time.sleep(0.05)
        had_frames = False
        while True:
            try:
                fr = frame_q.get_nowait()
                row = {
                    'file': os.path.basename(wav_path),
                    'frame_idx': frame_idx,
                    't': fr.get('t'),
                    'direction_d1': fr.get('direction_d1'),
                    'omega': fr.get('omega'),
                    'A': fr.get('A'),
                    'r2': fr.get('r2'),
                    'rmse': fr.get('rmse'),
                    'distance': fr.get('distance'),
                    'distance_kf': fr.get('distance_kf'),
                    'velocity_kf': fr.get('velocity_kf'),
                    'acceleration_kf': fr.get('acceleration_kf'),
                    'has_measure': fr.get('has_measure'),
                    'n_valid_bins': fr.get('n_valid_bins'),
                    'valid_band_hz': fr.get('valid_band_hz'),
                    'sum_abs_d1': fr.get('sum_abs_d1'),
                    'is_sound_present': fr.get('is_sound_present'),
                }
                writer.writerow(row)
                # 收集估计值（None 不参与）
                t_est = fr.get('t')
                d_est = fr.get('distance_kf') if fr.get('distance_kf') is not None else fr.get('distance')
                if (t_est is not None) and (d_est is not None):
                    try:
                        est_samples.append((float(t_est), float(d_est)))
                    except Exception:
                        pass
                frame_idx += 1
                had_frames = True
            except Empty:
                break
        
        if had_frames:
            idle_rounds = 0
        else:
            idle_rounds += 1
    
    print(f"[offline] wrote {frame_idx} frames for {os.path.basename(wav_path)}")
    # 估计与参考对比（考虑 0.5s 延迟）
    try:
        evaluate_against_reference(est_samples, ref, delay_sec=0.5)
    except Exception as e:
        print(f"[offline][ref] evaluation failed: {e}")

    # 紧跟着把“本次实际生效的参数”也打印出来，形成终端尾部“对齐评估 + 参数”并列输出
    try:
        _dump_params(ap, ls_params, kf_params, rts_params, output_filter)
    except Exception as e:
        print(f"[offline][params] print failed: {e}")


def main(argv: Optional[Iterable[str]] = None):
    # 使用内联配置
    input_path = str(INLINE_RUN.get("input") or "").strip()
    output_path = str(INLINE_RUN.get("output") or "offline_results.csv").strip()

    if not input_path:
        print("请在 INLINE_RUN['input'] 中设置 .wav 文件路径。")
        return

    # 检查输入文件是否存在
    wav_file = Path(input_path)
    if not wav_file.exists():
        print(f"[offline] 文件不存在: {input_path}")
        return
    
    if not wav_file.is_file():
        print(f"[offline] 输入路径不是文件: {input_path}")
        print("[offline] 请在 INLINE_RUN['input'] 中指定单个 .wav 文件路径")
        return
    
    if wav_file.suffix.lower() not in ['.wav']:
        print(f"[offline] 不是 .wav 文件: {input_path}")
        return

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'frame_idx', 't', 'direction_d1',
            'omega', 'A', 'r2', 'rmse', 'distance', 'distance_kf', 'velocity_kf', 'acceleration_kf',
            'has_measure', 'n_valid_bins', 'valid_band_hz',
            'sum_abs_d1', 'is_sound_present'
        ])
        header_once = {}
        try:
            process_one_file(str(wav_file), writer, header_once)
        except Exception as e:
            print(f"[offline] failed on {wav_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
