# 网络架构与数据集构造说明

本文按当前仓库的机器学习主线来描述：`audio -> omega feature cache -> OmegaWindowDataset -> UAVCombOmegaNet -> omega/distance`。这里的“网络架构”指神经网络结构，不是 WebSocket 或局域网通信结构。

## 1. 版本边界

仓库里同时存在两类资料：

1. 当前源码主线：`ml_uav_comb/data_pipeline/offline_omega_feature_extractor.py` 生成 `schema_version=6` 的 `log_mag_band` 多通道缓存；`OmegaWindowDataset` 和 `omega_normalization.py` 也按这套字段读取。
2. 现存大缓存/部分历史文档：`ml_uav_comb/cache/omega_rir_sweep_real_test/` 里的 `.npz` 是旧版 `schema_version=4` 的 `smooth_d1` 43-bin 缓存，`normalization_stats.npz` 也是旧字段。它对应旧训练产物和旧配置快照，不完全等价于当前源码的 v6 数据读取逻辑。

因此，下面先描述当前源码主线，再单独说明历史缓存形态。

## 2. 总体链路

```text
WAV/label CSV
  -> load_audio_mono + resample
  -> OfflineOmegaFeatureExtractor
  -> per-recording .npz cache
  -> dataset_index.json + dataset_index_data/*.npy
  -> normalization_stats.npz
  -> OmegaWindowDataset
  -> UAVCombOmegaNet
  -> omega_pred, pattern_prob, observability_pred, log_variance
  -> distance_pred_cm = omega_pred * (343 * 100) / (4*pi)
```

模型不直接处理原始音频。音频先被转成窗口化的频谱动态特征，模型只看一个长度固定的滑窗，并预测该窗口末端帧的物理量。

## 3. 输入张量

`OmegaWindowDataset.__getitem__()` 返回的核心输入是：

```text
x: [C, T, F]
```

训练时 DataLoader collate 后进入模型：

```text
x: [B, C, T, F]
```

默认含义：

- `B`: batch 内窗口数。训练采样器会把连续窗口组成 chunk，因此有效步内窗口数通常是 `sequence_length * batch_size`。
- `C`: 输入通道数。当前源码默认 4 个动态通道。
- `T`: 每个窗口的帧数，默认 `window_frames=68`。
- `F`: 频率 bin 数，由 `n_fft/freq_min/freq_max` 决定。

当前 `omega_default.yaml` 的源码默认参数是 `target_sr=48000, n_fft=2048, hop_len=512, freq_min=800, freq_max=8000`，模型计算得到 `F=307`，实际频率范围约为 `820.3125 Hz - 7992.1875 Hz`。每帧 hop 为 `512 / 48000 = 0.0106667 s`，68 帧窗口覆盖约 `0.7253 s`。

历史 `omega_rir_sweep_real_test` 缓存使用 `n_fft=512, hop_len=128, freq_min=1000, freq_max=5000`，对应 `F=43`，频率间隔 `93.75 Hz`。

## 4. UAVCombOmegaNet 网络结构

实现位置：`ml_uav_comb/models/uav_comb_omega_net.py`。

### 4.1 输入准备

`UAVCombOmegaNet._prepare_inputs()` 接收 batch dict 或直接接收张量，要求形状是 `[B, C, T, F]`。如果读到旧 checkpoint 或旧数据的单通道输入 `[B, 1, T, F]`，且当前模型配置 `input_channels > 1`，会把单通道 expand 成多通道以兼容旧模型。

### 4.2 逐频点时间编码器

主编码器是 `DilatedTCNEncoder`。它对每个频率 bin 沿时间轴建模，但实现上使用 `Conv2d(kernel=(k_time, 1_freq))`，避免把 `B * F` 展平导致显存爆炸。

流程：

```text
[B, C, T, F]
  -> 1x1 Conv2d: C -> D
  -> GroupNorm + GELU
  -> 多层 DilatedCausalConv2DPerBin
  -> 1x1 Conv2d + GroupNorm + GELU
  -> [B, D, T, F]
```

每个 `DilatedCausalConv2DPerBin` 的结构：

```text
left causal pad on T
  -> Conv2d(D, D, kernel=(k,1), dilation=(d,1))
  -> GroupNorm
  -> GELU
  -> Dropout
  -> residual add
```

因果 padding 只在时间左侧补零，所以当前时间步不会看到未来帧。感受野为：

```text
RF = 1 + (kernel_size - 1) * sum(dilations)
```

源码默认 dilation 是 `[1, 2, 4, 8, 16]`、`kernel_size=3`，RF 为 63 帧。`omega_rir_sweep_real_test` 的有效配置使用 `[1, 2, 8, 16]`，RF 为 55 帧。

代码还保留两个替代编码器：

- `PerBinGRUEncoder`: 双向 GRU，每个频点一条序列，能看到完整窗口，但显存更重。
- `PerBinTemporalEncoder`: 旧版 per-bin Conv1D residual encoder，用于旧 checkpoint 兼容。

### 4.3 跨频率融合

`CrossFrequencyFusion` 输入 `[B, D, T, F]`，输出 `[B, D, T]`。

它分两步做频率融合：

1. 频率交互：给每个频点加一个可学习 `freq_embed`，与特征拼接成 `[B, D+1, T, F]`，经过 1x1 Conv2d 后残差加回原特征。
2. 频率池化：用 `nn.Linear(F, 1, bias=False)` 在频率维投影到单个时间特征。权重初始化为均值池化 `1/F`，后续训练中可学习不同频率权重。

这不是 softmax attention。代码注释里明确说明 v3 改成线性投影，是为了避免 softmax 均分频率维后削弱梯度。

训练模式下，模型还会计算 `cross_freq_consistency`：

```text
normalize(per_bin_feat)
  -> 跨频均值作为 consensus
  -> mean(1 - cosine_similarity(freq_feat, consensus))
```

这个值进入总损失，鼓励不同频率 bin 的隐特征在可观测模式上保持一致。

### 4.4 可选时间状态聚合

`TemporalStateAggregator` 是一个因果 self-attention 模块，输入输出都是 `[B, D, T]`：

```text
[B, D, T]
  -> permute to [B, T, D]
  -> causal scaled dot-product attention
  -> residual + LayerNorm
  -> FFN
  -> residual + LayerNorm
  -> [B, D, T]
```

是否启用由 `model.use_temporal_state` 控制。`omega_default.yaml` 里设为 `true`；RIR sweep 有效配置里设为 `false`，原因是历史训练记录认为该模块会造成明显梯度衰减，而 TCN 的感受野已经覆盖主要时间上下文。

### 4.5 窗口末端预测头

`EndOfWindowOmegaHead` 只使用窗口最后一个时间步做预测。

结构：

```text
[B, D, T]
  -> Conv1d(D,D,k=5,pad=2) + ReLU
  -> Conv1d(D,D,k=3,pad=1) + ReLU
  -> take last time step: [B, D]
  -> omega_head          -> omega_pred
  -> pattern_head        -> pattern_logit, pattern_prob
  -> observability_head  -> observability_pred
  -> uncertainty_head    -> log_variance
```

`omega_pred` 不是无限范围回归，而是用 `tanh` 限制到配置的距离范围对应的 omega 区间：

```text
omega_min = distance_cm_min * 4*pi / (343*100)
omega_max = distance_cm_max * 4*pi / (343*100)
omega_pred = mid + half_range * tanh(raw)
```

默认 `distance_cm_min=5, distance_cm_max=60`，所以 omega 范围约为 `0.0018318 - 0.0219820`。

### 4.6 输出字段

模型 forward 返回：

- `omega_pred`: 窗口末端帧的 omega 预测。
- `pattern_logit`: 可观测 comb pattern 的分类 logit。
- `pattern_prob`: `sigmoid(pattern_logit)`。
- `observability_logit`: 可观测性辅助分支 logit。
- `observability_pred`: `softplus(observability_logit)`，非负连续分数。
- `log_variance`: 异方差不确定性分支，clamp 到 `[-5, 5]`。
- `cross_freq_consistency`: 跨频一致性正则项。

`return_debug=True` 时还会输出 `cross_freq_attention_weights` 和 `per_bin_feature_map`。

## 5. 损失函数与训练采样

实现位置：`ml_uav_comb/training/omega_losses.py` 和 `ml_uav_comb/training/omega_trainer.py`。

总损失由这些项组成：

```text
total =
  lambda_pattern       * pattern_bce
+ lambda_omega         * omega_huber
+ lambda_delta         * adjacent_delta_huber
+ lambda_acc           * adjacent_second_diff_huber
+ lambda_cross_freq    * cross_freq_consistency
+ lambda_observability * observability_smooth_l1
+ lambda_uncertainty   * heteroscedastic_omega_loss
```

关键权重逻辑：

- `pattern_bce` 在所有 `pattern_target` 有效的帧上训练。
- `omega_huber` 只在 `omega_target` 有限且 `pattern_target` 有效的位置训练。
- omega 权重为 `pattern_target.clamp(0,1)`，所以 pattern 越不可观测，距离回归权重越低。
- 一阶差分和二阶差分损失只在同一 recording、同一 chunk、`sequence_index` 连续的相邻窗口上计算。

训练 DataLoader 不使用普通随机采样，而是 `ContiguousSequenceBatchSampler`。它按 recording 连续切出若干窗口 chunk，让 batch 内存在真实相邻窗口，这样 delta 和 acceleration 损失才有语义。

## 6. 当前源码的数据集构造

### 6.1 配置入口

构建命令：

```bash
python -m ml_uav_comb.scripts.build_omega_dataset --config ml_uav_comb/configs/omega_default.yaml --build-jobs 4
```

训练命令会自动检查数据集产物，缺失时先构建：

```bash
python -m ml_uav_comb.scripts.train_omega --config ml_uav_comb/configs/omega_default.yaml
```

配置中的 `dataset.recordings` 是数据源清单。每条 recording 至少包含：

```yaml
recording_id: rec_1
audio_path: rec_1.wav
label_path: range_1.csv
label_format: csv
split_hint: auto
```

`split_hint` 可以是 `train/val/test/auto/exclude`。

### 6.2 音频读取

`load_audio_mono()` 做这些处理：

1. 使用 `soundfile` 读取 wav。
2. 多声道音频取均值转单声道。
3. 如果采样率不是 `audio.target_sr`，用 `scipy.signal.resample_poly` 重采样。
4. 如果配置了 `max_duration_sec`，截断到指定时长。

### 6.3 离线帧特征

`OfflineOmegaFeatureExtractor.process_audio()` 对每个 STFT 帧执行：

1. 取长度 `n_fft` 的音频帧，乘 Hann window。
2. 计算 `rfft` 幅度谱。
3. 选择 `freq_min - freq_max` 频段。
4. 取 `log(mag + 1e-12)` 得到 `log_mag_band[t, f]`。
5. 对 log 频谱去均值后计算：
   - `smd`: spectral modulation depth。
   - `cpr`: cepstral peak ratio。
   - `cpn`: cepstral peak normalized。
6. 同时保存：
   - `energy_proxy`: 帧 RMS。
   - `snr_proxy`: `20*log10(max(mag_band)/mean(mag_band))`。
   - `frame_time_sec`: `frame_index * hop_sec`。

当前 v6 主输入通道来自 `log_mag_band` 及其时间差分：

```text
log_mag_band              [T, F]
log_mag_band_dt1          [T, F] = x[t] - x[t-dt_short_lag]
log_mag_band_dt_long      [T, F] = x[t] - x[t-dt_long_lag]
log_mag_band_abs_dt1      [T, F] = abs(log_mag_band_dt1)
```

默认 `dt_short_lag=1`，`dt_long_lag=9`。

### 6.4 标签读取与插值

`load_optional_labels()` 支持 CSV 或 JSON。常用字段和别名包括：

- 时间：`time_sec/time/t`。
- 距离：`distance_cm` 或 `distance_mm`。
- 距离有效：`distance_valid`。
- 速度：`v_perp_mps/velocity_mps/speed_mps/radial_velocity_mps/v_mps`。
- 可观测分数：`observability_score_res`。
- pattern 标签：`pattern_label_res`。

标签会按时间排序。帧级标签由这些插值得到：

- `distance_cm`: 线性插值。
- `distance_valid`: 线性插值后按阈值判断是否有效。
- `v_perp_mps`: 线性插值。
- `observability_score_res`: 优先用标签字段，否则由距离和速度推导。
- `pattern_target`: 优先由 observability score 转 soft target；如果没有 score，则使用最近邻 pattern 标签。

omega 标签由距离直接换算：

```text
omega_target = distance_cm * 4*pi / (343*100)
distance_cm  = omega * (343*100) / (4*pi)
```

### 6.5 可观测性 gate

当前特征提取器里有一个经验可观测性 gate，用来避免训练模型在真实不可观测区间硬回归距离。

规则：

```text
hard cutoff: distance_cm >= 25 -> not observable
velocity boundary: v_perp_mps < 3.30 * distance_m - 0.50 -> not observable
```

命中 not observable 后：

```text
pattern_target = 0
pattern_binary_target = 0
omega_target = NaN
```

这意味着分类头仍然学习“不可观测”，但 omega 回归头不会在这些帧上被监督。

### 6.6 每条 recording 的缓存

`export_omega_recording_cache()` 为每条 recording 写一个 `.npz`：

```text
schema_version
frame_time_sec
log_mag_band
log_mag_band_dt1
log_mag_band_dt_long
log_mag_band_abs_dt1
smd, cpr, cpn
energy_proxy, snr_proxy
frequencies_hz
frame_distance_cm
frame_omega_target
frame_v_perp_mps
frame_observability_score_res
frame_pattern_target
frame_pattern_binary_target
```

这些缓存放在 `dataset.cache_dir`，文件名通常是 `<recording_id>.npz`。

### 6.7 compact index 构造

`write_omega_dataset_index()` 不把所有窗口复制成大数组，而是写 compact index：

```text
dataset_index.json
dataset_index_data/
  train_recording_code.npy
  train_start_frame.npy
  train_sequence_index.npy
  val_recording_code.npy
  val_start_frame.npy
  val_sequence_index.npy
  test_recording_code.npy
  test_start_frame.npy
  test_sequence_index.npy
```

窗口生成规则：

```text
start_frame = 0, stride_frames, 2*stride_frames, ...
end_frame = start_frame + window_frames
target_frame = end_frame - 1
```

也就是说，输入窗口覆盖 `[start_frame, end_frame)`，监督目标取窗口最后一帧。

split 规则：

- 如果只有一个 supervised recording 且 `split_hint=auto`，按时间区间切分，并用 `split_margin_sec` 在 split 边界留空隙，避免窗口跨越 train/val/test 边界。
- 如果有多个 recording，`split_hint=train/val/test` 会固定分配；`auto` 则按 recording 级别、随机种子和比例确定 split。
- `split_hint=exclude` 或没有 label 的 recording 不进入监督索引。
- 如果配置 `dynamic_epoch_split=true`，训练时可以按 epoch 重新把 recording code 切到 train/val/test；固定 test recording 可通过 `fixed_test_recording_codes` 保持不变。

### 6.8 归一化统计

`compute_omega_normalization_stats()` 只基于指定 split 统计，默认优先 train split。它对每个动态通道、每个频率 bin 分别计算均值和标准差：

```text
<channel>_mean: [F]
<channel>_std:  [F]
```

为避免真的展开所有窗口，它对每条 recording 的帧级特征先做 prefix sum 和 prefix square sum，再按窗口起点快速累计窗口内统计。

训练、评估和推理必须共用同一份 `normalization_stats.npz`，否则输入分布会不一致。

### 6.9 Dataset 加载

`OmegaWindowDataset` 初始化时读取：

- `dataset_index.json`
- `dataset_index_meta.json`
- `normalization_stats.npz`
- 当前 split 的 memmap 数组

取样时：

1. 根据 `recording_code` 找到 cache 文件。
2. 读取 `[start_frame:end_frame]` 的每个动态通道。
3. 对每个通道按 `[F]` mean/std 标准化。
4. `np.stack(channels, axis=0)` 得到 `[C, T, F]`。
5. 目标取 `target_frame=end_frame-1`。

返回字段：

```text
x
omega_target
pattern_target
observability_score
distance_cm
target_time_sec
sequence_index
recording_id
```

## 7. 现存缓存快照

当前工作区里存在 `ml_uav_comb/cache/omega_rir_sweep_real_test/`，其 manifest 显示：

- `schema_version=2`, `index_format=omega_compact_v1`
- `window_frames=68`, `stride_frames=4`
- recordings: `16237`
- total windows: `7549737`
- train windows: `5474143`
- val windows: `1373702`
- test windows: `701892`
- real-data recordings: `2107`

但这个缓存的 `.npz` 字段是旧版：

```text
schema_version=4
smooth_d1 [T, 43]
sum_abs_d1
energy_proxy
snr_proxy
frame_distance_cm
frame_omega_target
frame_pattern_target
frame_pattern_binary_target
frame_observability_score_res
frame_v_perp_mps
frequencies_hz [43]
```

其 `normalization_stats.npz` 里也是：

```text
smooth_d1_mean/std
smooth_d1_dt1_mean/std
smooth_d1_dt_long_mean/std
smooth_d1_abs_dt1_mean/std
```

如果要用当前源码直接训练，需要重新构建 v6 `log_mag_band` 缓存，或恢复/适配当时的 smooth_d1 数据读取代码。

## 8. 推理流程

`scripts/infer_omega_wav.py` 的推理流程和训练数据构造保持一致：

1. 读取并重采样输入 wav。
2. 用 `process_audio_array()` 生成临时帧级特征。
3. 加载训练时的 `normalization_stats.npz`。
4. 按 `window_frames/stride_frames` 滑窗。
5. 对每个窗口构造 `[1, C, T, F]` 输入模型。
6. 输出 `omega_pred`、`distance_pred_cm`、`pattern_prob`。
7. 如果 `pattern_prob < inference.pattern_threshold`，`distance_pred_cm_gated` 置为 NaN。

推理输出 CSV 中的距离是由 omega 按固定物理公式反解得到，不是模型第二个独立距离回归头。

## 9. 关键设计取舍

- 目标不是通用音频距离估计，而是利用无人机自噪声和近场反射产生的 comb-filter 频谱结构。
- 模型学习的是窗口末端的 omega 和 pattern，而不是整段音频的全局标签。
- 输入保留频率维和时间维，先做 per-bin temporal modeling，再做 cross-frequency fusion，符合 comb pattern “跨频一致、随时间运动”的先验。
- omega 只在 pattern 可观测时训练，避免在物理不可观测区间给模型强行灌入错误监督。
- compact index 避免把几百万滑窗物化成巨大的重复数组，cache 文件保存帧级特征，index 只保存窗口起点和 recording 映射。
