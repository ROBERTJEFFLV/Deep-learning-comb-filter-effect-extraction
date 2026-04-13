# ml_uav_comb

`ml_uav_comb` 目前在这个仓库里有两条并存路线：

- 当前默认训练主线：omega 回归，用 `smooth_d1` 直接回归窗口末端 `omega/distance`
- 并存实验路线：observer/logits，用 distance grid + tracker 做测量建模

这次实际使用和产出缓存/权重的，是 omega 回归主线，不是 observer/logits 主链。

## Current Default Pipeline

当前默认链路：

`audio -> offline omega feature extractor -> cached omega windows -> UAVCombOmegaNet -> omega_pred -> distance_cm`

对应代码：

- 数据构建：`data_pipeline/export_omega_dataset.py`
- 数据集读取：`data_pipeline/omega_dataset.py`
- 模型：`models/uav_comb_omega_net.py`
- 损失与指标：`training/omega_losses.py`、`training/omega_metrics.py`
- 训练：`training/omega_trainer.py`

## What The Model Learns

- 输入：`smooth_d1` 窗口
- 监督：窗口末端 `omega_target`
- 评估：统一解码回 `distance_cm`
- 当前训练日志里的 `train_dist_mae / val_dist_mae / test_dist_mae` 都是距离厘米空间上的指标

## Observer Path Status

`distance logits -> measurement summary -> RangeKF posterior` 这条 observer/logits 路线还保留在仓库里，用于后续实验和对照：

- 模型：`models/uav_comb_observer.py`
- 训练：`training/trainer.py`
- 评估：`training/evaluate.py`

但它不是当前 `omega_*` 配置、缓存、checkpoint 对应的训练路径。

## Main Commands

omega 回归主线常用命令：

```bash
python -m ml_uav_comb.scripts.build_omega_dataset --config ml_uav_comb/configs/omega_default.yaml --build-jobs 4
python -m ml_uav_comb.scripts.train_omega --config ml_uav_comb/configs/omega_default.yaml
python -m ml_uav_comb.scripts.eval_omega_checkpoint --config ml_uav_comb/configs/omega_default.yaml --checkpoint <checkpoint_path> --split test
python -m ml_uav_comb.scripts.infer_omega_wav --config ml_uav_comb/configs/omega_default.yaml --checkpoint <checkpoint_path> --wav rec_1.wav
```

RIR 全量配置示例：

```bash
python -m ml_uav_comb.scripts.build_omega_dataset --config ml_uav_comb/configs/omega_rir_global.yaml --build-jobs 16
python -m ml_uav_comb.scripts.train_omega --config ml_uav_comb/configs/omega_rir_global.yaml
```

## Cache And Artifacts

- 数据集缓存默认在 `ml_uav_comb/cache/<config_name>/`
- 训练产物默认在 `ml_uav_comb/artifacts/<config_name>/`
- cache 里通常包含 `.npz` 特征缓存、`dataset_index.json`、`dataset_index_data/`、`normalization_stats.npz`

## Throughput Notes

- `dataset.build_jobs` / `--build-jobs` 控制离线缓存构建并行度
- `training.num_workers` 只影响 DataLoader
- `train_omega.py` 会优先复用现有 dataset artifacts；如果 cache 被清空，就会重新 build
