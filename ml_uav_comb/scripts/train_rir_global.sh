#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
One-click global training for RIR run dataset.

Usage:
  bash ml_uav_comb/scripts/train_rir_global.sh [options]

Options:
  --manifest PATH       run_manifest.csv path
                        (default: /home/lvmingyang/March24/datasets/simulation/tool_box/RIR/rir_run_dataset/run_manifest.csv)
  --run-name NAME       experiment suffix used for config/cache/artifacts
                        (default: rir_runs_global)
  --base-config PATH    base yaml config
                        (default: ml_uav_comb/configs/default.yaml)
  --max-runs N          optional cap on number of runs (for debug)
                        (default: no cap)
  --build-jobs N        parallel jobs for cache building stage
                        (default: auto, all CPU cores)
  --batch-size N        training.batch_size
                        (default: 48)
  --num-workers N       training.num_workers
                        (default: auto, about half CPU cores)
  --stage-a-epochs N    training.stage_a_epochs
                        (default: 10)
  --stage-b-epochs N    training.stage_b_epochs
                        (default: 20)
  --resume-best-stage-a resume from ml_uav_comb/artifacts/<run-name>/best_stage_a.pt
  --skip-build          train directly from existing cache/index/normalization files
  --force-rebuild       rebuild cache/index even if reusable artifacts already exist
  --skip-eval           skip eval_checkpoint after training
  -h, --help            show this message
EOF
}

MANIFEST_PATH="/home/lvmingyang/March24/datasets/simulation/tool_box/RIR/rir_run_dataset/run_manifest.csv"
RUN_NAME="rir_runs_global"
BASE_CONFIG="ml_uav_comb/configs/default.yaml"
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
BATCH_SIZE="48"
NUM_WORKERS="$(( CPU_CORES / 2 ))"
if [[ "${NUM_WORKERS}" -lt 2 ]]; then
  NUM_WORKERS="2"
fi
if [[ "${NUM_WORKERS}" -gt 6 ]]; then
  NUM_WORKERS="6"
fi
STAGE_A_EPOCHS="10"
STAGE_B_EPOCHS="20"
SKIP_BUILD="0"
FORCE_REBUILD="0"
SKIP_EVAL="0"
RESUME_BEST_STAGE_A="0"
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
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --stage-a-epochs)
      STAGE_A_EPOCHS="$2"; shift 2 ;;
    --stage-b-epochs)
      STAGE_B_EPOCHS="$2"; shift 2 ;;
    --resume-best-stage-a)
      RESUME_BEST_STAGE_A="1"; shift ;;
    --skip-build)
      SKIP_BUILD="1"; shift ;;
    --force-rebuild)
      FORCE_REBUILD="1"; shift ;;
    --skip-eval)
      SKIP_EVAL="1"; shift ;;
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
export MAX_RUNS BUILD_JOBS BATCH_SIZE NUM_WORKERS STAGE_A_EPOCHS STAGE_B_EPOCHS

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import yaml
import numpy as np

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

df["run_num"] = df["run_id"].astype(str).str.extract(r"(\d+)").astype(int)
df = df.sort_values("run_num")
if max_runs:
    n = int(max_runs)
    if n <= 0:
        raise ValueError("--max-runs must be > 0")
    if n < len(df):
        seeds = []
        for split_name in ("train", "val", "test"):
            part = df[df["split"] == split_name].head(1)
            if not part.empty:
                seeds.append(part)
        if seeds:
            seed_df = pd.concat(seeds, axis=0).drop_duplicates(subset=["run_id"])
        else:
            seed_df = df.head(0)

        remaining = max(n - len(seed_df), 0)
        rest = df[~df["run_id"].isin(seed_df["run_id"])].head(remaining)
        df = pd.concat([seed_df, rest], axis=0).sort_values("run_num")
    else:
        df = df.head(n)
if df.empty:
    raise ValueError("manifest produced zero runs after filtering")


def _resolve_distance_grid_bounds_cm(manifest_df: pd.DataFrame) -> tuple[float, float, float]:
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
        raise ValueError("failed to derive distance grid bounds from labels: no finite distance_cm values found")
    all_dist = np.concatenate(distances, axis=0).astype(np.float32, copy=False)
    raw_min = float(np.min(all_dist))
    raw_max = float(np.max(all_dist))
    margin_cm = 1.0
    grid_min = max(1.0, raw_min - margin_cm)
    grid_max = max(grid_min + 1.0, raw_max + margin_cm)
    return grid_min, grid_max, float(raw_max - raw_min)

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

cfg["dataset"]["recordings"] = recordings
cfg["dataset"]["build_jobs"] = int(os.environ["BUILD_JOBS"])
cfg["dataset"]["cache_dir"] = f"ml_uav_comb/cache/{run_name}"
cfg["dataset"]["index_path"] = f"ml_uav_comb/cache/{run_name}/dataset_index.csv"
cfg["dataset"]["normalization_path"] = f"ml_uav_comb/cache/{run_name}/normalization_stats.npz"
cfg["dataset"]["meta_path"] = f"ml_uav_comb/cache/{run_name}/dataset_index_meta.json"

grid_min_cm, grid_max_cm, raw_span_cm = _resolve_distance_grid_bounds_cm(df)
target_grid_step_cm = 0.4
required_candidates = int(np.ceil((grid_max_cm - grid_min_cm) / target_grid_step_cm)) + 1
required_candidates = int(np.ceil(max(required_candidates, int(cfg["model"].get("num_candidates", 64))) / 16.0) * 16)
cfg["model"]["num_candidates"] = required_candidates
grid_step_cm = (grid_max_cm - grid_min_cm) / max(required_candidates - 1, 1)
cfg["model"]["distance_grid_cm_min"] = float(grid_min_cm)
cfg["model"]["distance_grid_cm_max"] = float(grid_max_cm)
cfg["model"]["distance_target_sigma_cm"] = float(max(0.8, grid_step_cm * 2.0))

cfg["training"]["checkpoint_dir"] = f"ml_uav_comb/artifacts/{run_name}"
cfg["training"]["batch_size"] = int(os.environ["BATCH_SIZE"])
cfg["training"]["num_workers"] = int(os.environ["NUM_WORKERS"])
cfg["training"]["pin_memory"] = True
cfg["training"]["persistent_workers"] = int(os.environ["NUM_WORKERS"]) > 0
cfg["training"]["prefetch_factor"] = 2
cfg["training"]["non_blocking_to_device"] = True
cfg["training"]["use_amp"] = True
cfg["training"]["amp_dtype"] = "bfloat16"
cfg["training"]["allow_tf32"] = True
cfg["training"]["use_torch_compile"] = True
cfg["training"]["torch_compile_mode"] = "reduce-overhead"
cfg["training"]["cudnn_benchmark"] = True
cfg["training"]["max_cache_files_per_worker"] = 8
cfg["training"]["batch_sampler_mode"] = "contiguous_recording_chunks"
cfg["training"]["chunk_step"] = max(1, int(os.environ["BATCH_SIZE"]) // 2)
cfg["training"]["min_chunk_span_cm"] = 5.0
cfg["training"]["min_chunk_valid_fraction"] = 0.9
cfg["training"]["max_samples_per_recording_per_batch"] = 2
cfg["training"]["group_batches_by_recording"] = False
cfg["training"]["shuffle_within_recording"] = True
cfg["training"]["log_interval_steps"] = 50
cfg["training"]["log_warmup_steps"] = 5
cfg["training"]["stage_a_epochs"] = int(os.environ["STAGE_A_EPOCHS"])
cfg["training"]["stage_b_epochs"] = int(os.environ["STAGE_B_EPOCHS"])

cfg["evaluation"]["prediction_csv"] = f"ml_uav_comb/artifacts/{run_name}/eval_predictions.csv"
cfg["inference"]["prediction_csv"] = f"ml_uav_comb/artifacts/{run_name}/infer_predictions.csv"

out_config_path.parent.mkdir(parents=True, exist_ok=True)
out_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

split_counts = df["split"].value_counts().to_dict()
print(f"[config] wrote: {out_config_path}")
print(f"[config] runs: {len(df)} split_counts={split_counts}")
print(f"[config] build_jobs={cfg['dataset']['build_jobs']} dataloader_workers={cfg['training']['num_workers']}")
print(
    f"[config] distance_grid_cm=[{cfg['model']['distance_grid_cm_min']:.3f}, "
    f"{cfg['model']['distance_grid_cm_max']:.3f}] bins={cfg['model']['num_candidates']} "
    f"step={grid_step_cm:.3f}cm sigma={cfg['model']['distance_target_sigma_cm']:.3f} "
    f"(raw_span={raw_span_cm:.3f}cm)"
)
print(
    f"[config] train_sampler={cfg['training']['batch_sampler_mode']} "
    f"chunk_step={cfg['training'].get('chunk_step', 0)} "
    f"min_chunk_span_cm={cfg['training'].get('min_chunk_span_cm', 0.0)} "
    f"max_cache_files_per_worker={cfg['training']['max_cache_files_per_worker']}"
)
PY

echo "[train] starting full training..."
TRAIN_ARGS=(--config "${OUT_CONFIG}" --build-jobs "${BUILD_JOBS}")
if [[ "${SKIP_BUILD}" == "1" ]]; then
  TRAIN_ARGS+=(--skip-build)
fi
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  TRAIN_ARGS+=(--force-rebuild)
fi
if [[ "${RESUME_BEST_STAGE_A}" == "1" ]]; then
  RESUME_PATH="${CHECKPOINT_DIR}/best_stage_a.pt"
  if [[ ! -f "${RESUME_PATH}" ]]; then
    echo "[error] resume checkpoint not found: ${RESUME_PATH}" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--resume-from "${RESUME_PATH}")
fi
python -m ml_uav_comb.scripts.train_full "${TRAIN_ARGS[@]}" | tee "${CHECKPOINT_DIR}/train.log"

BEST_CKPT="${CHECKPOINT_DIR}/best_stage_b.pt"
if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[done] skip eval (--skip-eval set)"
  echo "[done] checkpoint: ${BEST_CKPT}"
  exit 0
fi

if [[ -f "${BEST_CKPT}" ]]; then
  echo "[eval] evaluating ${BEST_CKPT} ..."
  python -m ml_uav_comb.scripts.eval_checkpoint --config "${OUT_CONFIG}" --checkpoint "${BEST_CKPT}" --build-jobs "${BUILD_JOBS}" | tee "${CHECKPOINT_DIR}/eval.log"
  echo "[done] training + eval completed"
else
  echo "[warn] checkpoint not found: ${BEST_CKPT}"
  echo "[warn] training log: ${CHECKPOINT_DIR}/train.log"
fi
