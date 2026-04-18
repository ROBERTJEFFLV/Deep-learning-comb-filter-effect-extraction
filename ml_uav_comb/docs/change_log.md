# Change Log

Historical note: `S00` to `S07` record the original v1 bootstrap. The active contract is the `V2 Remediation` section at the end of this file.

## S00

- Created `project_goal.md` with repo-aligned problem statement.
- Created `external_repo_notes.md` with adaptation boundaries for the three external repos.
- Acceptance target: both docs exist and explicitly state that `stpACC` does not replace the lag/rho or comb-motion gate.
- Acceptance command:

```bash
test -f ml_uav_comb/docs/project_goal.md && test -f ml_uav_comb/docs/external_repo_notes.md && rg -n "stpACC.*替代|stpACC.*replace|lag/rho|comb-motion gate" ml_uav_comb/docs/project_goal.md ml_uav_comb/docs/external_repo_notes.md
```

- Acceptance result:
  - Passed.
  - `project_goal.md` and `external_repo_notes.md` both exist.
  - Both files explicitly state that `stpACC` is not a replacement for the lag/rho or comb-motion gate.

## S01

- Created `ml_uav_comb/` directory skeleton.
- Added initial `configs/default.yaml`, `configs/tiny_debug.yaml`, and `README.md`.
- Further step-by-step verification results will be appended below.
- Acceptance command:

```bash
find ml_uav_comb -maxdepth 2 -type d | sort
test -f ml_uav_comb/configs/default.yaml
test -f ml_uav_comb/configs/tiny_debug.yaml
test -f ml_uav_comb/README.md
```

- Acceptance result:
  - Passed.
  - Directory tree exists.
  - `default.yaml`, `tiny_debug.yaml`, and `README.md` exist.

## S02

- Implemented:
  - `features/feature_utils.py`
  - `features/stpacc.py`
  - `data_pipeline/offline_feature_extractor.py`
  - `data_pipeline/export_dataset.py`
- Exported caches from `rec_1.wav` and `recorded_audio.wav` under `ml_uav_comb/cache/tiny_debug/`.
- Acceptance commands:

```bash
python - <<'PY'
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.data_pipeline.export_dataset import build_dataset
cfg = load_yaml_config('ml_uav_comb/configs/tiny_debug.yaml')
summary = build_dataset(cfg)
print(summary)
PY

python -m unittest discover -s ml_uav_comb/tests

python - <<'PY'
import numpy as np
cache = np.load('ml_uav_comb/cache/tiny_debug/rec_1.npz', allow_pickle=True)
for key in ['frame_time_sec','phase_stft','diff_comb','scalar_seq','scalar_mask','stpacc','heuristic_targets','frequencies_hz']:
    print(key, cache[key].shape)
PY
```

- Acceptance result:
  - Passed.
  - `rec_1.npz` exported successfully.
  - Shape summary:
    - `phase_stft`: `(4496, 43, 3)`
    - `diff_comb`: `(4496, 43, 3)`
    - `scalar_seq`: `(4496, 10)`
    - `stpacc`: `(4496, 64)`
    - `heuristic_targets`: `(4496, 7)`
  - `test_feature_shapes.py` passed.

## S03

- Implemented:
  - `data_pipeline/dataset_index.py`
  - `data_pipeline/dataset.py`
  - `data_pipeline/label_schema.md`
- Acceptance commands:

```bash
python - <<'PY'
from ml_uav_comb.data_pipeline.dataset import CachedWindowDataset
ds = CachedWindowDataset('ml_uav_comb/cache/tiny_debug/dataset_index.csv', split='train')
sample = ds[0]
print('len', len(ds))
print('phase', sample['phase'].shape)
print('comb', sample['comb'].shape)
print('scalar', sample['scalar'].shape)
print('stpacc', sample['stpacc'].shape)
PY

python -m unittest discover -s ml_uav_comb/tests
```

- Acceptance result:
  - Passed.
  - Train split length: `190`
  - Sample shapes:
    - `phase`: `(3, 240, 43)`
    - `comb`: `(3, 240, 43)`
    - `scalar`: `(240, 10)`
    - `stpacc`: `(1, 240, 64)`
  - `test_dataset_smoke.py` passed.

## S04

- Implemented:
  - `models/branches.py`
  - `models/heads.py`
  - `models/uav_comb_crnn.py`
- Acceptance command:

```bash
python -m unittest discover -s ml_uav_comb/tests
```

- Acceptance result:
  - Passed.
  - `test_model_forward.py` passed.
  - Forward output includes `distance`, `sign_logits`, and `confidence`.

## S05

- Implemented:
  - `training/losses.py`
  - `training/metrics.py`
  - `training/trainer.py`
  - `scripts/train_debug.py`
  - `scripts/train_full.py`
- Added a small `sys.path` bootstrap to CLI scripts so `python ml_uav_comb/scripts/*.py` works from repo root.
- Acceptance command:

```bash
python ml_uav_comb/scripts/train_debug.py --config ml_uav_comb/configs/tiny_debug.yaml
```

- Acceptance result:
  - Passed.
  - Two-stage training completed.
  - Checkpoints written:
    - `ml_uav_comb/artifacts/tiny_debug/best_stage_a.pt`
    - `ml_uav_comb/artifacts/tiny_debug/best_stage_b.pt`
  - Training history written:
    - `ml_uav_comb/artifacts/tiny_debug/training_history.json`
  - No shape error or NaN observed.

## S06

- Implemented:
  - `training/evaluate.py`
  - `training/infer.py`
  - `scripts/eval_checkpoint.py`
  - `scripts/infer_wav.py`
  - `scripts/run_smoke_test.py`
- Acceptance commands:

```bash
python ml_uav_comb/scripts/eval_checkpoint.py \
  --config ml_uav_comb/configs/tiny_debug.yaml \
  --checkpoint ml_uav_comb/artifacts/tiny_debug/best_stage_b.pt

python ml_uav_comb/scripts/infer_wav.py \
  --config ml_uav_comb/configs/tiny_debug.yaml \
  --checkpoint ml_uav_comb/artifacts/tiny_debug/best_stage_b.pt \
  --wav rec_1.wav

python ml_uav_comb/scripts/run_smoke_test.py --config ml_uav_comb/configs/tiny_debug.yaml
```

- Acceptance result:
  - Passed.
  - Eval metrics were produced and saved to `ml_uav_comb/artifacts/tiny_debug/eval_predictions.csv`.
  - Single-wav inference produced predictions and saved to `ml_uav_comb/artifacts/tiny_debug/infer_predictions.csv`.
  - Smoke test completed full chain: build -> tests -> train -> eval -> infer.

## S07

- Completed docs:
  - `docs/feature_contract.md`
  - `docs/model_design.md`
  - `docs/training_plan.md`
  - `data_pipeline/label_schema.md`
  - `README.md`
- README commands verified:
  - `build_dataset.py`
  - `train_debug.py`
  - `eval_checkpoint.py`
  - `infer_wav.py`
  - `run_smoke_test.py`

## Remaining Risks

- `sign_acc` is low in tiny debug runs. This is expected for the current minimal dataset and GT-sign derivation setup; it needs more balanced real labels or synthetic augmentation.
- `confidence_label` falls back to pseudo labels when explicit `valid_mask` is absent. Current `range_1.csv` does not provide a true `valid_mask`.
- The model uses `nn.LazyLinear` in branch projections. It works in current tests and smoke runs, but it emits PyTorch's standard lazy-module warning.
- The current cache split strategy is time-based per recording. This is appropriate for the current small debug setup, but larger multi-recording experiments may want recording-level split policy refinement.

## V2 Remediation

- Upgraded `ml_uav_comb` to schema version 2.
- Removed heuristic distance / KF state leakage from model inputs.
- Split `scalar_seq` and `teacher_seq`, and added `scalar_observed_mask` plus `scalar_reliable_mask`.
- Excluded `recorded_audio.wav` from supervised train/val/test index while still exporting its cache.
- Implemented time-block split with boundary gap so window overlap does not cross split boundaries.
- Added train-only `normalization_stats.npz` and `dataset_index_meta.json`.
- Implemented `target_space`, `split_hint`, v2 metrics, v2 eval/infer CSV output, and parity tests against legacy `AudioProcessor`.

- Remediation acceptance commands:

```bash
python ml_uav_comb/scripts/build_dataset.py --config ml_uav_comb/configs/tiny_debug.yaml
python -m unittest discover -s ml_uav_comb/tests
python ml_uav_comb/scripts/train_debug.py --config ml_uav_comb/configs/tiny_debug.yaml
python ml_uav_comb/scripts/eval_checkpoint.py --config ml_uav_comb/configs/tiny_debug.yaml --checkpoint ml_uav_comb/artifacts/tiny_debug_v2/best_stage_b.pt
python ml_uav_comb/scripts/infer_wav.py --config ml_uav_comb/configs/tiny_debug.yaml --checkpoint ml_uav_comb/artifacts/tiny_debug_v2/best_stage_b.pt --wav rec_1.wav
python ml_uav_comb/scripts/run_smoke_test.py --config ml_uav_comb/configs/tiny_debug.yaml
```

- Remediation acceptance result:
  - Passed.
  - Build summary:
    - `index_path`: `ml_uav_comb/cache/tiny_debug_v2/dataset_index.csv`
    - `normalization_path`: `ml_uav_comb/cache/tiny_debug_v2/normalization_stats.npz`
    - `supervised_recordings`: `['rec_1']`
    - `excluded_recordings`: `['recorded_audio']`
  - Test suite:
    - `python -m unittest discover -s ml_uav_comb/tests` passed with 9 tests.
  - Training:
    - `best_stage_a.pt` and `best_stage_b.pt` written under `ml_uav_comb/artifacts/tiny_debug_v2/`
  - Eval:
    - `ml_uav_comb/artifacts/tiny_debug_v2/eval_predictions.csv` generated
  - Infer:
    - `ml_uav_comb/artifacts/tiny_debug_v2/infer_predictions.csv` generated
  - Smoke test:
    - Full chain `build -> tests -> train -> eval -> infer` completed successfully.
