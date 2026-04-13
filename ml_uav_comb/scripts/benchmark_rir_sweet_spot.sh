#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Benchmark batch_size / num_workers sweet spots for an already-built RIR training config.

Usage:
  bash ml_uav_comb/scripts/benchmark_rir_sweet_spot.sh [options]

Options:
  --config PATH            yaml config path
                           (default: ml_uav_comb/configs/rir_runs_global.yaml)
  --batch-sizes LIST       comma-separated batch sizes
                           (default: 16,24,32,48,64,80,96)
  --num-workers LIST       comma-separated dataloader worker counts
                           (default: 2,4,6)
  --warmup-steps N         steps to discard for warmup/compile
                           (default: 10)
  --measure-steps N        measured train steps per trial
                           (default: 40)
  --build-jobs N           optional dataset build parallelism if artifacts missing
  --force-build            force rebuild dataset artifacts before benchmarking
  -h, --help               show this message
EOF
}

CONFIG="ml_uav_comb/configs/rir_runs_global.yaml"
BATCH_SIZES="16,24,32,48,64,80,96"
NUM_WORKERS="2,4,6"
WARMUP_STEPS="10"
MEASURE_STEPS="40"
BUILD_JOBS=""
FORCE_BUILD="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2 ;;
    --batch-sizes)
      BATCH_SIZES="$2"; shift 2 ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --warmup-steps)
      WARMUP_STEPS="$2"; shift 2 ;;
    --measure-steps)
      MEASURE_STEPS="$2"; shift 2 ;;
    --build-jobs)
      BUILD_JOBS="$2"; shift 2 ;;
    --force-build)
      FORCE_BUILD="1"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

ARGS=(
  --config "${CONFIG}"
  --batch-sizes "${BATCH_SIZES}"
  --num-workers-list "${NUM_WORKERS}"
  --warmup-steps "${WARMUP_STEPS}"
  --measure-steps "${MEASURE_STEPS}"
)
if [[ -n "${BUILD_JOBS}" ]]; then
  ARGS+=(--build-jobs "${BUILD_JOBS}")
fi
if [[ "${FORCE_BUILD}" == "1" ]]; then
  ARGS+=(--force-build)
fi

python -m ml_uav_comb.scripts.benchmark_sweet_spot "${ARGS[@]}"
