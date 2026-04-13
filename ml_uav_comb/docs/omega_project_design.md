# Omega-only LS replacement pipeline

This branch implements the agreed design:

- freeze the legacy front end up to `smooth_d1`
- remove direction vote, gating, and KF from the learning branch
- save frame-level `smooth_d1` plus aligned supervision fields
- build dense `68 x 43` sliding windows
- train a compact CNN with explicit frequency coordinates and frequency-aware aggregation
- predict two outputs at the window end:
  - `pattern_logit` / `pattern_prob`
  - `omega_pred`
- learn `pattern` on all valid supervision frames
- learn `omega` only when the window-end frame is pattern-positive
- derive distance only for metrics / exports from the fixed physics formula
- add adjacent-window delta and second-difference matching only across contiguous pattern-positive windows

## Key files

- `data_pipeline/offline_omega_feature_extractor.py`
- `data_pipeline/omega_dataset_index.py`
- `data_pipeline/omega_normalization.py`
- `data_pipeline/omega_dataset.py`
- `data_pipeline/export_omega_dataset.py`
- `models/uav_comb_omega_net.py`
- `training/omega_losses.py`
- `training/omega_metrics.py`
- `training/omega_trainer.py`
- `scripts/build_omega_dataset.py`
- `scripts/train_omega.py`
- `scripts/eval_omega_checkpoint.py`
- `scripts/infer_omega_wav.py`

## Main commands

```bash
python ml_uav_comb/scripts/build_omega_dataset.py --config ml_uav_comb/configs/omega_default.yaml
python ml_uav_comb/scripts/train_omega.py --config ml_uav_comb/configs/omega_default.yaml
python ml_uav_comb/scripts/eval_omega_checkpoint.py --config ml_uav_comb/configs/omega_default.yaml --checkpoint ml_uav_comb/artifacts/omega_default/best.pt
```
