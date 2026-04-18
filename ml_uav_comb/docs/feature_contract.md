# Feature Contract v2

## Goal

v2 contract 的核心约束是：

- 主模型只看可学的声学量
- heuristic state 只保留为 teacher/debug/export
- 缺失值和不可靠值必须显式区分

## Recording Cache

每个 recording 导出一个 `cache/<recording_id>.npz`，至少包含：

- `schema_version [1] = 2`
- `frame_time_sec [T]`
- `phase_stft [T, 43, 3]`
- `diff_comb [T, 43, 3]`
- `scalar_seq [T, 4]`
- `scalar_field_names [4]`
- `scalar_observed_mask [T, 4]`
- `scalar_reliable_mask [T, 4]`
- `teacher_seq [T, 9]`
- `teacher_field_names [9]`
- `stpacc [T, 64]` if enabled
- `frequencies_hz [43]`

## Branch Inputs

### A. Phase-aware STFT

`phase_stft[t, f, :]` 顺序固定为：

1. `log_mag`
2. `sin_phase`
3. `cos_phase`

### B. Diff-comb

`diff_comb[t, f, :]` 顺序固定为：

1. `smooth_d1`
2. `abs_d1`
3. `delta_d1`

### C. Acoustic Scalar Sequence

`scalar_seq[t, s]` 默认字段顺序固定为：

1. `sum_abs_d1_smooth`
2. `comb_shift_lag`
3. `comb_shift_rho`
4. `is_sound_present`

### D. Teacher-only Sequence

`teacher_seq[t, h]` 默认字段顺序固定为：

1. `heuristic_distance_raw_cm`
2. `heuristic_distance_kf_cm`
3. `heuristic_distance_raw_available`
4. `heuristic_distance_kf_available`
5. `velocity_kf_cm_s`
6. `acceleration_kf_cm_s2`
7. `heuristic_measure_available`
8. `shift_direction_raw`
9. `shift_direction_hysteresis`

这些字段不能进入主模型输入。

### E. STPACC

`stpacc[t, l]` 是单声道 early-reflection 辅助特征，默认 downsample 到 64 lag bins。

`stpACC` 不是 lag/rho gate 的替代品。

## Mask Semantics

### `scalar_observed_mask`

表示这个 scalar 在数值上是否存在/有限。

### `scalar_reliable_mask`

表示这个 scalar 是否通过了当前物理 gate。

对于 `comb_shift_lag / comb_shift_rho`，可靠性的默认判据是：

- `isfinite`
- `abs(lag) >= 1`
- `abs(rho) >= comb_rho_thresh`
- `smooth_amp >= amp_threshold`
- `is_sound_present == 1`

## Dataset Index v2

`dataset_index.csv` 每行至少包含：

- `recording_id`, `split`, `cache_path`
- `start_frame`, `end_frame`, `center_frame`, `center_time_sec`
- `label_source`, `supervision_type`
- `distance_cm`, `distance_target`
- `sign_label`, `confidence_target`
- `valid_dist_gt_mask`, `dist_reliable_mask`, `dist_train_mask`
- `valid_sign_gt_mask`, `sign_train_mask`
- `valid_conf_gt_mask`, `conf_train_mask`
- `heuristic_distance_cm`, `heuristic_distance_target`, `heuristic_distance_available`
- `heuristic_confidence`

## Normalization

`normalization_stats.npz` 由 train split 单独统计：

- phase: per-channel mean/std
- comb: per-channel mean/std
- scalar: median/IQR
- stpacc: per-bin mean/std

训练、评估、推理必须共用同一份统计量。
