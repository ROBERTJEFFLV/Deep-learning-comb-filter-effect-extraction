#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml_uav_comb.data_pipeline.offline_omega_feature_extractor import omega_to_distance_cm, process_audio_array
from ml_uav_comb.data_pipeline.omega_normalization import load_omega_normalization_stats
from ml_uav_comb.features.feature_utils import load_audio_mono, load_optional_labels, load_yaml_config
from ml_uav_comb.models.uav_comb_omega_net import UAVCombOmegaNet
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--audio', required=True)
parser.add_argument('--labels', default=None)
parser.add_argument('--output', default=None)
args = parser.parse_args()
cfg = load_yaml_config(args.config)
audio, sr = load_audio_mono(args.audio, int(cfg['audio']['target_sr']), cfg['audio'].get('max_duration_sec'))
labels = load_optional_labels(args.labels, 'infer') if args.labels else None
cache = process_audio_array(audio, sr, cfg, optional_labels=labels)
norm = load_omega_normalization_stats(cfg['dataset']['normalization_path'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UAVCombOmegaNet(cfg).to(device)
state = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state['model_state'])
if bool(cfg.get("training", {}).get("use_compile", True)) and hasattr(torch, "compile"):
    model = torch.compile(model, mode=str(cfg.get("training", {}).get("compile_mode", "reduce-overhead")))
use_amp = bool(cfg.get("training", {}).get("use_amp", True)) and device.type == "cuda"
amp_dtype_name = str(cfg.get("training", {}).get("amp_dtype", "auto")).lower()
if amp_dtype_name == "bfloat16":
    amp_dtype = torch.bfloat16
elif amp_dtype_name == "float16":
    amp_dtype = torch.float16
elif hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16
else:
    amp_dtype = torch.float16
model.eval(); torch.set_num_threads(1)
window_frames = int(cfg['dataset'].get('window_frames', 68)); stride_frames = int(cfg['dataset'].get('stride_frames', 1))
mean = norm['smooth_d1_mean'].reshape(1, -1); std = norm['smooth_d1_std'].reshape(1, -1)
pattern_threshold = float(cfg.get('inference', {}).get('pattern_threshold', 0.5))
rows=[]
with torch.inference_mode():
    for start in range(0, int(cache['frame_time_sec'].shape[0]) - window_frames + 1, stride_frames):
        end = start + window_frames; target = end - 1
        x = ((cache['smooth_d1'][start:end] - mean) / std).astype(np.float32)
        with torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext():
            out = model({"x": torch.from_numpy(x[None, None, :, :]).to(device)})
        omega_pred = float(out['omega_pred'].item())
        pattern_prob = float(out['pattern_prob'].item())
        pattern_pred = float(pattern_prob >= pattern_threshold)
        row = {
            'start_time_sec': float(cache['frame_time_sec'][start]),
            'center_time_sec': float(cache['frame_time_sec'][start + (window_frames // 2)]),
            'target_time_sec': float(cache['frame_time_sec'][target]),
            'omega_pred': omega_pred,
            'pattern_prob': pattern_prob,
            'pattern_pred': pattern_pred,
            'distance_pred_cm': float(omega_to_distance_cm(omega_pred)),
            'distance_pred_cm_gated': float(omega_to_distance_cm(omega_pred)) if pattern_pred > 0.5 else float('nan'),
        }
        if 'frame_omega_target' in cache:
            row['omega_target'] = float(cache['frame_omega_target'][target]) if np.isfinite(cache['frame_omega_target'][target]) else float('nan')
            row['distance_target_cm'] = float(cache['frame_distance_cm'][target]) if np.isfinite(cache['frame_distance_cm'][target]) else float('nan')
            row['distance_valid'] = float(np.isfinite(cache['frame_omega_target'][target]))
        if 'frame_pattern_target' in cache:
            row['pattern_target'] = float(cache['frame_pattern_target'][target]) if np.isfinite(cache['frame_pattern_target'][target]) else float('nan')
        if 'frame_pattern_binary_target' in cache:
            row['pattern_binary_target'] = float(cache['frame_pattern_binary_target'][target]) if np.isfinite(cache['frame_pattern_binary_target'][target]) else float('nan')
        if 'frame_observability_score_res' in cache:
            row['observability_score_res'] = float(cache['frame_observability_score_res'][target]) if np.isfinite(cache['frame_observability_score_res'][target]) else float('nan')
        rows.append(row)
output = Path(args.output or cfg['inference']['prediction_csv']); output.parent.mkdir(parents=True, exist_ok=True)
with open(output,'w',encoding='utf-8',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['target_time_sec']); writer.writeheader(); writer.writerows(rows)
print(json.dumps({'output_csv': str(output), 'num_predictions': len(rows)}, indent=2))
