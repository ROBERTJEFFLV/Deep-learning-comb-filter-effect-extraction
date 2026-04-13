#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
One-click omega-only training for the synthetic RIR run dataset.

Usage:
  bash ml_uav_comb/scripts/train_omega_rir_global.sh [options]

Options:
  --manifest PATH          run_manifest.csv path
                          (default: /home/lvmingyang/March24/datasets/simulation/tool_box/RIR/rir_run_dataset/run_manifest.csv)
  --run-name NAME          experiment suffix used for config/cache/artifacts
                          (default: omega_rir_global)
  --base-config PATH       base yaml config
                          (default: ml_uav_comb/configs/omega_default.yaml)
  --max-runs N             optional cap on number of runs (for debug)
  --build-jobs N           parallel jobs for cache building stage
                          (default: auto, all CPU cores)
  --num-workers N          training.num_workers
                          (default: 4)
  --batch-size N           number of chunks per optimizer step
                          (default: 16)
  --eval-batch-size N      number of chunks per eval step
                          (default: batch-size)
  --epochs N               training.epochs
                          (default: 30)
  --sequence-length N      training.sequence_length
                          (default: 16)
  --eval-sequence-length N training.eval_sequence_length
                          (default: 32)
  --chunk-step N           training.chunk_step
                          (default: sequence_length, no overlap)
  --eval-chunk-step N      training.eval_chunk_step
                          (default: eval_sequence_length)
  --skip-build             train directly from existing cache/index/normalization files
  --force-rebuild          rebuild cache/index even if reusable artifacts already exist
  --resume-from PATH       resume checkpoint for train_omega.py
  --post-eval              run standalone eval_omega_checkpoint after training
                          (default: off; training already runs final test on best checkpoint)
  --skip-eval              deprecated alias for disabling standalone post-eval
  -h, --help               show this message
EOF
}

MANIFEST_PATH="/home/lvmingyang/March24/datasets/simulation/tool_box/RIR/rir_run_dataset/run_manifest.csv"
RUN_NAME="omega_rir_global"
BASE_CONFIG="ml_uav_comb/configs/omega_default.yaml"
MAX_RUNS=""
if command -v nproc >/dev/null 2>&1; then
  CPU_CORES="$(nproc)"
else
  CPU_CORES="8"
fi
if [[ "${CPU_CORES}" -lt 2 ]]; then
  CPU_CORES="2"
fi
BUILD_JOBS="${CPU_CORES}"
NUM_WORKERS="4"
BATCH_SIZE="16"
EVAL_BATCH_SIZE=""
EPOCHS="30"
SEQUENCE_LENGTH="16"
EVAL_SEQUENCE_LENGTH="32"
CHUNK_STEP=""
EVAL_CHUNK_STEP=""
SKIP_BUILD="0"
FORCE_REBUILD="0"
POST_EVAL="0"
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"; shift 2 ;;
    --run-name)
      RUN_NAME="$2"; shift 2 ;;
    --base-config)
      BASE_CONFIG="$2"; shift 2 ;;
    --max-runs)
      MAX_RUNS="$2"; shift 2 ;;
    --build-jobs)
      BUILD_JOBS="$2"; shift 2 ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --sequence-length)
      SEQUENCE_LENGTH="$2"; shift 2 ;;
    --eval-sequence-length)
      EVAL_SEQUENCE_LENGTH="$2"; shift 2 ;;
    --chunk-step)
      CHUNK_STEP="$2"; shift 2 ;;
    --eval-chunk-step)
      EVAL_CHUNK_STEP="$2"; shift 2 ;;
    --skip-build)
      SKIP_BUILD="1"; shift ;;
    --force-rebuild)
      FORCE_REBUILD="1"; shift ;;
    --resume-from)
      RESUME_FROM="$2"; shift 2 ;;
    --post-eval)
      POST_EVAL="1"; shift ;;
    --skip-eval)
      POST_EVAL="0"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${SKIP_BUILD}" == "1" && "${FORCE_REBUILD}" == "1" ]]; then
  echo "[error] --skip-build and --force-rebuild cannot be used together" >&2
  exit 2
fi

if [[ -z "${CHUNK_STEP}" ]]; then
  CHUNK_STEP="${SEQUENCE_LENGTH}"
fi

if [[ -z "${EVAL_CHUNK_STEP}" ]]; then
  EVAL_CHUNK_STEP="${EVAL_SEQUENCE_LENGTH}"
fi

if [[ -z "${EVAL_BATCH_SIZE}" ]]; then
  EVAL_BATCH_SIZE="${BATCH_SIZE}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[error] manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[error] base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

OUT_CONFIG="ml_uav_comb/configs/${RUN_NAME}.yaml"
CHECKPOINT_DIR="ml_uav_comb/artifacts/${RUN_NAME}"
mkdir -p "${CHECKPOINT_DIR}"

export MANIFEST_PATH BASE_CONFIG OUT_CONFIG RUN_NAME
export MAX_RUNS BUILD_JOBS NUM_WORKERS BATCH_SIZE EVAL_BATCH_SIZE EPOCHS SEQUENCE_LENGTH EVAL_SEQUENCE_LENGTH CHUNK_STEP EVAL_CHUNK_STEP

python - <<'PY'
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

manifest_path = Path(os.environ["MANIFEST_PATH"]).resolve()
base_config_path = Path(os.environ["BASE_CONFIG"]).resolve()
out_config_path = Path(os.environ["OUT_CONFIG"]).resolve()
run_name = os.environ["RUN_NAME"]
max_runs = os.environ.get("MAX_RUNS", "").strip()

required_cols = {"run_id", "split", "output_wav", "labels_csv"}
df = pd.read_csv(manifest_path)
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"manifest missing columns: {sorted(missing)}")

if "dataset_panel" in df.columns:
    df = df[df["dataset_panel"].astype(str) == "main"].copy()
    if df.empty:
        raise ValueError("manifest has zero main-panel runs after dataset_panel filter")

if "trajectory_type" not in df.columns:
    raise ValueError("manifest is missing trajectory_type; refusing to train from ambiguous dataset")

trajectory_counts = df["trajectory_type"].astype(str).value_counts().to_dict()
unexpected_trajectory_types = sorted(tp for tp in trajectory_counts if tp != "stochastic_smooth")
if unexpected_trajectory_types:
    raise ValueError(
        "manifest contains non-stochastic trajectory_type values: "
        f"{unexpected_trajectory_types}; counts={trajectory_counts}"
    )

if "trajectory_generator" in df.columns:
    generator_counts = df["trajectory_generator"].astype(str).value_counts().to_dict()
    unexpected_generators = sorted(gen for gen in generator_counts if gen != "stochastic_smooth_v4")
    if unexpected_generators:
        raise ValueError(
            "manifest contains unexpected trajectory_generator values: "
            f"{unexpected_generators}; counts={generator_counts}"
        )

df["run_num"] = df["run_id"].astype(str).str.extract(r"(\d+)").astype(int)
df = df.sort_values("run_num")
if max_runs:
    n = int(max_runs)
    if n <= 0:
        raise ValueError("--max-runs must be > 0")
    if n < len(df):
        keep = []
        for split_name in ("train", "val", "test"):
            part = df[df["split"] == split_name].head(1)
            if not part.empty:
                keep.append(part)
        keep_df = pd.concat(keep, axis=0).drop_duplicates(subset=["run_id"]) if keep else df.head(0)
        remaining = max(n - len(keep_df), 0)
        rest = df[~df["run_id"].isin(keep_df["run_id"])].head(remaining)
        df = pd.concat([keep_df, rest], axis=0).sort_values("run_num")
    else:
        df = df.head(n)
if df.empty:
    raise ValueError("manifest produced zero runs after filtering")


def _resolve_distance_bounds_cm(manifest_df):
    train_df = manifest_df[manifest_df["split"].astype(str) == "train"].copy()
    source_df = train_df if not train_df.empty else manifest_df
    distances = []
    for row in source_df.itertuples(index=False):
        label_path = Path(row.labels_csv).resolve()
        labels = pd.read_csv(label_path)
        if "distance_cm" not in labels.columns:
            continue
        dist = pd.to_numeric(labels["distance_cm"], errors="coerce")
        if "distance_valid" in labels.columns:
            valid = pd.to_numeric(labels["distance_valid"], errors="coerce").fillna(0.0) > 0.5
            dist = dist[valid]
        dist = dist[np.isfinite(dist.to_numpy(dtype=float, copy=False))]
        if not dist.empty:
            distances.append(dist.to_numpy(dtype=np.float32, copy=False))
    if not distances:
        raise ValueError("failed to derive distance bounds from labels")
    all_dist = np.concatenate(distances, axis=0).astype(np.float32, copy=False)
    min_cm = max(1.0, float(np.min(all_dist)) - 1.0)
    max_cm = max(min_cm + 1.0, float(np.max(all_dist)) + 1.0)
    return min_cm, max_cm


cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"uav_comb_{run_name}"

recordings = []
for row in df.itertuples(index=False):
    recording_id = f"{row.split}_{row.run_id}"
    recordings.append(
        {
            "recording_id": recording_id,
            "audio_path": str(Path(row.output_wav).resolve()),
            "label_path": str(Path(row.labels_csv).resolve()),
            "label_format": "csv",
            "split_hint": str(row.split),
        }
    )

dist_min_cm, dist_max_cm = _resolve_distance_bounds_cm(df)

cfg["dataset"]["recordings"] = recordings
cfg["dataset"]["build_jobs"] = int(os.environ["BUILD_JOBS"])
cfg["dataset"]["dynamic_epoch_split"] = False
cfg["dataset"]["cache_dir"] = f"ml_uav_comb/cache/{run_name}"
cfg["dataset"]["index_path"] = f"ml_uav_comb/cache/{run_name}/dataset_index.json"
cfg["dataset"]["normalization_path"] = f"ml_uav_comb/cache/{run_name}/normalization_stats.npz"
cfg["dataset"]["meta_path"] = f"ml_uav_comb/cache/{run_name}/dataset_index_meta.json"

cfg["model"]["distance_cm_min"] = float(dist_min_cm)
cfg["model"]["distance_cm_max"] = float(dist_max_cm)

cfg["training"]["sequence_length"] = int(os.environ["SEQUENCE_LENGTH"])
cfg["training"]["eval_sequence_length"] = int(os.environ["EVAL_SEQUENCE_LENGTH"])
cfg["training"]["chunk_step"] = int(os.environ["CHUNK_STEP"])
cfg["training"]["eval_chunk_step"] = int(os.environ["EVAL_CHUNK_STEP"])
cfg["training"]["batch_size"] = int(os.environ["BATCH_SIZE"])
cfg["training"]["eval_batch_size"] = int(os.environ["EVAL_BATCH_SIZE"])
cfg["training"]["num_workers"] = int(os.environ["NUM_WORKERS"])
cfg["training"]["persistent_workers"] = int(os.environ["NUM_WORKERS"]) > 0
cfg["training"]["epochs"] = int(os.environ["EPOCHS"])
cfg["training"]["checkpoint_dir"] = f"ml_uav_comb/artifacts/{run_name}"

cfg["evaluation"]["prediction_csv"] = f"ml_uav_comb/artifacts/{run_name}/eval_predictions.csv"
cfg["inference"]["prediction_csv"] = f"ml_uav_comb/artifacts/{run_name}/infer_predictions.csv"

out_config_path.parent.mkdir(parents=True, exist_ok=True)
out_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

split_counts = df["split"].value_counts().to_dict()
print(f"[config] wrote: {out_config_path}")
print(f"[config] runs: {len(df)} split_counts={split_counts}")
print(f"[config] build_jobs={cfg['dataset']['build_jobs']} num_workers={cfg['training']['num_workers']}")
print(
    f"[config] distance_cm_range=[{cfg['model']['distance_cm_min']:.3f}, "
    f"{cfg['model']['distance_cm_max']:.3f}]"
)
print(
    f"[config] sequence_length={cfg['training']['sequence_length']} "
    f"eval_sequence_length={cfg['training']['eval_sequence_length']} "
    f"batch_size={cfg['training']['batch_size']} "
    f"eval_batch_size={cfg['training']['eval_batch_size']} "
    f"chunk_step={cfg['training']['chunk_step']} "
    f"eval_chunk_step={cfg['training']['eval_chunk_step']} "
    f"epochs={cfg['training']['epochs']}"
)
print(f"[config] dynamic_epoch_split={cfg['dataset']['dynamic_epoch_split']}")
print(f"[config] trajectory_counts={trajectory_counts}")
PY

echo "[train] starting omega training..."
TRAIN_ARGS=(--config "${OUT_CONFIG}" --build-jobs "${BUILD_JOBS}")
if [[ "${SKIP_BUILD}" == "1" ]]; then
  TRAIN_ARGS+=(--skip-build)
fi
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  TRAIN_ARGS+=(--force-rebuild)
fi
if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_ARGS+=(--resume-from "${RESUME_FROM}")
fi
python -m ml_uav_comb.scripts.train_omega "${TRAIN_ARGS[@]}" | tee "${CHECKPOINT_DIR}/train.log"

BEST_CKPT="${CHECKPOINT_DIR}/best.pt"
if [[ "${POST_EVAL}" != "1" ]]; then
  echo "[done] training completed"
  echo "[done] final test metrics were computed inside train_omega.py on the best checkpoint"
  echo "[done] checkpoint: ${BEST_CKPT}"
  exit 0
fi

if [[ -f "${BEST_CKPT}" ]]; then
  echo "[eval] evaluating ${BEST_CKPT} ..."
  python -m ml_uav_comb.scripts.eval_omega_checkpoint --config "${OUT_CONFIG}" --checkpoint "${BEST_CKPT}" --build-jobs "${BUILD_JOBS}" --split test | tee "${CHECKPOINT_DIR}/eval.log"
  echo "[done] omega training + eval completed"
else
  echo "[warn] checkpoint not found: ${BEST_CKPT}"
  echo "[warn] training log: ${CHECKPOINT_DIR}/train.log"
fi
