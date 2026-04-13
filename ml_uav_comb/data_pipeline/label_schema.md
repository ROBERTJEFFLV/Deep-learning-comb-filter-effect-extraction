# Label Schema (Observer v1)

## Recommended Columns

- `time_sec`
- `distance_cm` (or `distance_mm`)
- `distance_valid`
- `valid_mask` (or `valid` / `reliable` / `confidence_valid`)
- `recording_id`

`time_sec` 是唯一硬约束主键。其它列可缺失。

## Example

```csv
time_sec,distance_cm,distance_valid,valid_mask,recording_id
0.00,95.0,1,1,rec_1
0.10,93.5,1,1,rec_1
0.20,,0,0,rec_1
0.30,91.2,1,1,rec_1
```

## Row Semantics

- `distance_valid=0` 时允许 `distance_cm` 缺失
- 显式 `valid_mask` 优先作为 measurement validity GT
- 若无显式 validity 列，训练时会回退到 `dist_reliable_mask`，并写 `validity_target_source`
- 如果标签文件有 `recording_id` 列，加载时会过滤到当前 recording

## Observer Targets in `dataset_index.csv`

- `target_frame`, `target_time_sec`
- `measurement_distance_target_cm`
- `measurement_distance_train_mask`
- `measurement_validity_target`
- `measurement_validity_train_mask`
- `measurement_target_source`
- `validity_target_source`
- `distance_target_grid` (soft target on candidate `distance_grid_cm`)

兼容字段仍保留（如 `distance_cm`, `dist_train_mask`, `confidence_target`），但主训练路径不再依赖 sign/conf 多头。
