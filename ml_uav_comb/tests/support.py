from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ml_uav_comb.features.feature_utils import load_yaml_config, resolution_observability_score


def write_test_wav(path: Path, duration_sec: float, sr: int = 48_000) -> None:
    num_samples = int(round(float(duration_sec) * float(sr)))
    t = np.arange(num_samples, dtype=np.float32) / float(sr)
    audio = (
        0.22 * np.sin(2.0 * np.pi * 1_400.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 2_200.0 * t)
        + 0.04 * np.sin(2.0 * np.pi * 40.0 * t)
    )
    pcm = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def write_omega_labels(path: Path, duration_sec: float) -> None:
    times = np.arange(0.0, float(duration_sec) + 1e-6, 0.1, dtype=np.float32)
    # Use distances within observable range (< 25cm cutoff) and
    # velocity that passes the empirical observability gate
    distance_cm = 15.0 - 1.5 * times
    distance_cm = np.clip(distance_cm, 5.0, 24.0)
    v_perp_mps = np.full_like(times, 0.5, dtype=np.float32)
    score = resolution_observability_score(distance_cm / 100.0, v_perp_mps).astype(np.float32)
    pattern_label = (score >= 1.0).astype(np.float32)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_sec",
                "distance_cm",
                "distance_valid",
                "v_perp_mps",
                "observability_score_res",
                "pattern_label_res",
            ],
        )
        writer.writeheader()
        for t, d, v, s, p in zip(times, distance_cm, v_perp_mps, score, pattern_label):
            writer.writerow(
                {
                    "time_sec": f"{float(t):.3f}",
                    "distance_cm": f"{float(d):.3f}",
                    "distance_valid": "1",
                    "v_perp_mps": f"{float(v):.6f}",
                    "observability_score_res": f"{float(s):.6f}",
                    "pattern_label_res": str(int(float(p) >= 0.5)),
                }
            )


def make_omega_tiny_dataset_cfg(
    tmpdir: str | Path,
    duration_sec: float,
    include_unlabeled_recording: bool = False,
) -> Dict[str, Any]:
    tmpdir = Path(tmpdir)
    cfg = load_yaml_config("ml_uav_comb/configs/omega_tiny_debug.yaml")
    cfg["audio"]["max_duration_sec"] = float(duration_sec)
    cfg["dataset"]["cache_dir"] = str(tmpdir / "omega_cache")
    cfg["dataset"]["index_path"] = str(tmpdir / "omega_cache" / "dataset_index.json")
    cfg["dataset"]["normalization_path"] = str(tmpdir / "omega_cache" / "normalization_stats.npz")
    cfg["dataset"]["meta_path"] = str(tmpdir / "omega_cache" / "dataset_index_meta.json")

    labeled_wav = tmpdir / "omega_rec_1.wav"
    label_csv = tmpdir / "omega_range_1.csv"
    write_test_wav(labeled_wav, duration_sec=duration_sec)
    write_omega_labels(label_csv, duration_sec=duration_sec)

    recordings = [
        {
            "recording_id": "omega_rec_1",
            "audio_path": str(labeled_wav),
            "label_path": str(label_csv),
            "label_format": "csv",
            "split_hint": "auto",
        }
    ]
    if include_unlabeled_recording:
        unlabeled_wav = tmpdir / "omega_recorded_audio.wav"
        write_test_wav(unlabeled_wav, duration_sec=duration_sec)
        recordings.append(
            {
                "recording_id": "omega_recorded_audio",
                "audio_path": str(unlabeled_wav),
                "label_path": None,
                "label_format": None,
                "split_hint": "exclude",
            }
        )

    cfg["dataset"]["recordings"] = recordings
    cfg["training"]["num_workers"] = 0
    cfg["training"]["pin_memory"] = False
    cfg["training"]["persistent_workers"] = False
    cfg["training"]["use_compile"] = False
    cfg["training"]["epochs"] = 1
    cfg["training"]["checkpoint_dir"] = str(tmpdir / "omega_artifacts")
    cfg["evaluation"]["prediction_csv"] = str(tmpdir / "omega_artifacts" / "eval_predictions.csv")
    cfg["inference"]["prediction_csv"] = str(tmpdir / "omega_artifacts" / "infer_predictions.csv")
    return cfg
