# CombCepstralNet 网络架构文档

> 版本：v1（K+1 倒谱距离分类，22,541 参数）  
> 对应代码：`ml_uav_comb/models/comb_cepstral_net.py`，`ml_uav_comb/data_pipeline/cepstral_dataset.py`

---

## 一、第一性原理

### 物理模型

无人机悬停时，旋翼噪声经地面（或反射面）反射产生延迟副本，与直射声叠加形成梳状滤波器效应（Comb Filter Effect）：

```
H(f) = 1 + α · e^{-j 2πf τ}
```

其中 `τ = 2d / c`（d = 反射距离，c = 343 m/s）。在对数幅度谱上表现为沿频率轴的周期性 ripple：

```
log|Y(f)| = log|X(f)| + log|H(f)|
            ↑ 噪声源谱   ↑ 周期性 ripple，周期 = 1/τ
```

### 倒谱提取距离的原理

对 log 幅度谱的频率轴再做 FFT（即倒谱 cepstrum），周期性 ripple 会在 quefrency = τ 处产生峰值：

```
CEPSTRUM[q] = FFT{ log|Y(f)| }[q]
                                ↑ 在 q ≈ τ 处出现峰
```

距离由 quefrency 峰位置反推：

```
τ = q · Δτ_bin            （Δτ_bin = 139 μs/bin）
d = c · τ / 2 = 343 · τ / 2    （单位：m）
d_cm = d × 100
```

**因此，距离信息完全编码在倒谱搜索区间的峰位置和峰形状中。**

---

## 二、完整数据流水线

### 2.1 音频预处理（v2 最优配置）

离线缓存于 NPZ 文件（`log_mag_preprocessed` 通道），每帧 shape `[F_band]`：

```
原始 STFT 幅度谱（float32，mono，48kHz）
    ↓
log(|STFT|)                         对数幅度谱，F_band=307（800–8000Hz 频段）
    ↓
EMA 时序平滑  α=0.1                  压制电噪声引起的快变尖峰
    ↓
时间差分  Δt=15帧（~160ms）           提取慢变的 comb 状调制，抑制直流漂移
    ↓
频谱高斯平滑  σ=5.0                  平滑谱尖峰，降低宽带噪声干扰
    ↓
log_mag_preprocessed  [T, 307]       ← 缓存于 NPZ，后续 on-the-fly 计算倒谱
```

**为什么用差分**：差分间隔 Δt ≈ 0.07–0.12s 时，STFT 图上 comb filter 效应的视觉对比度显著提升。差分相当于高通时序滤波，把慢变 comb 调制从快变噪声背景中剥离出来。

### 2.2 倒谱 Patch 提取（on-the-fly，无需重建缓存）

```python
# 输入：log_mag_preprocessed [T, 307]

centered = log_mag - log_mag.mean(axis=1, keepdims=True)   # 去均值
cepstrum = |FFT(centered, axis=1)|                          # 沿频率轴 FFT 取模 [T, 307]
patch    = cepstrum[:, 2:30]                                # 截取搜索区间 [T, Q=28]

# 每帧 MAD 归一化（消除能量/增益污染）
med   = median(patch, axis=1, keepdims=True)
mad   = median(|patch - med|, axis=1, keepdims=True)
patch = (patch - med) / (mad + 1e-6)                       # [T, Q=28]
```

**为什么 MAD 归一化**：无人机飞行高度/速度变化导致 RMS 能量大幅波动。全局 RMS 变化在 log 谱上是近似常数偏移，而常数偏移主要落在 cepstrum 的 q=0 附近（DC 项），已被搜索区间截断排除。MAD 归一化进一步消除任何残余尺度污染，使网络只看峰形而不看绝对能量。

---

## 三、Quefrency 搜索几何

| 参数 | 值 | 说明 |
|------|----|------|
| 采样率 | 48000 Hz | |
| FFT 点数 | 2048 | |
| 频段 | 800–8000 Hz | 旋翼噪声主要能量段 |
| 频段 bin 数 `F_band` | 307 | |
| 频率分辨率 `Δf` | 23.44 Hz | 48000/2048 |
| Quefrency 因子 `Δτ_bin` | **139.0 μs/bin** | 1/(307×23.44) |
| 搜索区间起始 `q_min` | 2 | τ_min = 0.25ms → d_min ≈ 4.3cm |
| 搜索区间终止 `q_max` | 30 | τ_max = 4.0ms → d_max ≈ 69cm |
| 搜索区间宽度 **Q** | **28 bins** | |
| 距离 bin 数 **K** | **28** | K = Q |
| bin[0] 中心距离 | **4.77 cm** | |
| bin[27] 中心距离 | **69.12 cm** | |
| bin 宽度 | **2.38 cm** | 均匀分布 |

距离-bin 对应关系：

```
bin_k_center_cm = 343 × (cep_min_bin + k) × Δτ_bin / 2 × 100
                = 343 × (2 + k) × 139e-6 / 2 × 100
```

---

## 四、网络架构：CombCepstralNet

### 4.1 总体设计

```
输入: X ∈ R^(T×Q) = R^(68×28)  ← 最近 T=68 帧的归一化倒谱 patch
         ↓
┌─────────────────────────────────────┐
│  Module 1: Quefrency Conv Encoder   │  ← 学习 quefrency 方向的峰形模板
│  [B*T, 1, Q] → [B*T, 16, Q]        │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Module 2: Per-frame Projection     │  ← 压缩每帧为 D=32 维向量
│  [B*T, 16*28] → [B*T, 32]          │
│  → reshape → [B, T, 32]            │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Module 3: Causal GRU               │  ← 因果时序平滑，抑制 burst 错误
│  [B, T, 32] → [B, T, 32]           │
│  取最后帧: [B, 32]                  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Module 4: K+1 Classifier           │  ← 输出 29 个 logit
│  [B, 32] → [B, 29]                 │
└─────────────────────────────────────┘
         ↓
输出: logits ∈ R^(K+1) = R^29
  class 0: no-pattern（无可靠梳状峰）
  class 1..28: distance bin（距离区间）

距离解码:
  hard:  d = bin_centers_cm[argmax(logits[1:])]
  soft:  d = Σ_k p(k) × bin_centers_cm[k]   (归一化到 pattern 类)
置信度:  confidence = 1 - softmax(logits)[0]
```

### 4.2 各模块详细说明

#### Module 1：Quefrency Conv Encoder

```python
nn.Conv1d(1,  16, kernel_size=5, padding=2)   # [B*T, 1, 28] → [B*T, 16, 28]
nn.ReLU()
nn.Conv1d(16, 16, kernel_size=3, padding=1)   # [B*T, 16, 28] → [B*T, 16, 28]
nn.ReLU()
```

**物理意义**：沿 quefrency 轴的 1D 卷积，近似于匹配滤波（matched filter）。  
- kernel_size=5 覆盖 ±2 个 bin（±4.76cm），学习峰的局部形状
- 对弱峰（远距离）的 SNR 提升约 √5 ≈ 2.2×（相比单点 peak 统计）
- 这正是替代手工 CPR/CPN 标量的关键：**保留局部峰形而不压成标量**

参数量：(1×5×16+16) + (16×3×16+16) = **880**

#### Module 2：Per-frame Projection

```python
nn.Linear(16 * 28, 32)   # 448 → 32
nn.ReLU()
```

把每帧的 conv 特征图展平后投影到 D=32 维。在 reshape 后得到 `[B, T, 32]` 的时序序列。

参数量：448×32+32 = **14,368**

#### Module 3：Causal GRU

```python
nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
```

取最后时间步输出 `h_T ∈ R^32`。

**物理意义**：在线距离不会每帧乱跳，真实 τ 满足 `τ_t ≈ τ_{t-1} + ε`（缓慢漂移）。GRU 通过隐状态实现弱动态先验 `P(bin_t | bin_{t-1})`，等价于一个非常简单的卡尔曼平滑器。  
- 主要作用：抑制 burst 错误（连续帧预测错误）
- 不负责提取峰特征，那是 Conv 的工作

参数量：3×(32×32 + 32×32 + 2×32) = 3×(1024+1024+64) = **6,336**

（GRU 门权重：W_r, W_z, W_n，每个门有 input→hidden 和 hidden→hidden 两个矩阵）

#### Module 4：K+1 Classifier

```python
nn.Linear(32, 29)   # no-pattern + 28 distance bins
```

参数量：32×29+29 = **957**

**Prior 偏置初始化**（防止训练初期崩塌）：

```
softmax(b₀, 0, ..., 0)[0] = exp(b₀) / (exp(b₀) + K) = no_pattern_prior

→ b₀ = log( prior × K / (1 - prior) )
      = log( 0.5 × 28 / 0.5 )
      = log(28) = 3.33
```

设 `classifier.bias[0] = 3.33`，使网络初始即输出 `p(no-pattern) ≈ 50%`，避免 Adam 需要 ~666 步才能从错误初始化中恢复。

### 4.3 参数统计

| 模块 | 参数量 | 占比 |
|------|--------|------|
| Quefrency Conv Encoder | 880 | 3.9% |
| Per-frame Projection | 14,368 | 63.7% |
| Causal GRU | 6,336 | 28.1% |
| K+1 Classifier | 957 | 4.2% |
| **Total** | **22,541** | 100% |

---

## 五、损失函数

### 5.1 Focal Loss + 类别权重

```
L = (1/N) Σ_i Σ_c w_c · (1 - p_{i,c})^γ · (-t_{i,c} · log p_{i,c})
```

- **γ = 2.0**（focal gamma）：easy no-pattern 样本权重 (0.01)² = 0.0001，hard pattern 样本 (0.9)² = 0.81，使梯度主要来自难学样本
- **类别权重 w_c**：逆频率加权，no-pattern（92%）权重≈1，pattern bins 权重≈11.3×（上限 `class_weight_max_ratio=12`）
- **label_smoothing = 0.05**：防止过拟合到 hard label

### 5.2 Soft-bin Targets（可选，消融 A2）

对有 pattern 的帧，将 hard one-hot 目标替换为以真实 bin 为中心的 Gaussian 分布：

```
soft_target[k] = N(k; bin_gt, σ²) / Σ N     k = 1..K
soft_target[0] = 0                           （no-pattern 类保持 hard）
```

σ = 1.5 bins（约 ±3.6cm），为 "检测到 pattern 但 bin 预测偏了 1–2 格" 提供梯度信号，缓解 detect-but-mislocalize 死循环。

### 5.3 损失设计目的映射

| 问题 | 对应设计 |
|------|---------|
| 正负样本比 1:11（8% pattern） | Focal loss + 逆频率类别权重 |
| 训练初期全部预测 no-pattern | Prior bias init (b₀=3.33) |
| 分类器收敛慢（Adam bias 问题） | classifier_lr = 10× backbone_lr |
| Detect-but-mislocalize 循环 | Soft-bin Gaussian targets（A2） |
| 过拟合 | Label smoothing 0.05 + weight decay 1e-4 |

---

## 六、训练配置

| 参数 | 值 | 说明 |
|------|----|----|
| 数据集 | 68 条真实无人机录音 | 698,269 窗口 |
| 训练/验证/测试划分 | 50/9/9 条 | 按 pattern 数量降序轮转分层 |
| 正样本比例 | train 8.1%，val 8.7%，test 7.2% | |
| Optimizer | Adam | backbone lr=5e-4，classifier lr=5e-3 |
| LR Scheduler | Cosine warmup | warmup=5 epochs，min=1e-5 |
| Batch size | 64 chunks × 16 帧 = 1024 窗口 | ContiguousSequenceBatchSampler |
| 输入窗口 T | 68 帧（≈0.73s @ 512 hop） | 来自 NPZ cache |
| Epochs | 50 | |
| Gradient clip | 5.0 | |
| Checkpoint 准则 | `max(val_pat_recall - val_fp_rate)` | 奖励 pattern 检测，惩罚误报 |

---

## 七、消融实验配置

| 配置 | 文件 | 与 A2 的区别 | 验证问题 |
|------|------|------------|---------|
| **A0（基线）** | 旧 v2 标量特征 | 无网络，SMD/CPR/CPN → 阈值规则 | 网络化是否有收益？ |
| **A1（无 GRU）** | `cepstral_a1_no_gru.yaml` | `use_temporal: false` | GRU 对 burst 抑制的贡献？ |
| **A2（完整，默认）** | `cepstral_default.yaml` | — | 基准 |
| **A2+soft**（消融候选） | `cepstral_a2_soft_bins.yaml` | `soft_bin_sigma: 1.5` | Soft target 是否缓解召回震荡？ |
| **A3（无归一化）** | `cepstral_a3_no_norm.yaml` | `use_normalization: false` | MAD 归一化是否消除能量污染？ |

---

## 八、已知问题与当前性能

### 训练结果（50 epochs，cepstral_default）

| 指标 | 最佳 epoch 10 | epoch 49（收敛后） |
|------|-------------|-----------------|
| val_pat_recall | **0.184** | 0.117 |
| val_fp_rate | 0.119 | 0.099 |
| val_combined | **0.0647** | 0.0185 |
| val_dist_mae | 9.56 cm | 6.05 cm |
| train_loss | 0.4279 | 0.3022 |

### 问题诊断

**核心问题：召回率震荡，未收敛**

```
ep00: 0.002 → ep02: 0.043 → ep10: 0.184 → ep11: 0.086 → ep49: 0.117
                              （最高点）    （立刻崩塌）   （低位震荡）
```

根本原因：**Detect-but-mislocalize 循环**  
- 模型开始预测 pattern → 但 distance bin 错误 → focal loss 升高 → 梯度推回 no-pattern → recall 崩塌  
- 这不是收敛慢，是损失函数结构性问题

**数据结构问题：有效 bin 范围太宽**  
- 当前配置 5–60cm，K=28 bins，bin 宽 2.38cm  
- 实际有效范围 5–25cm，只对应 **前 9 个 bin**  
- 19 个无效 bin（>25cm）在稀释梯度，增加分类难度 3×

### 待改进方向

1. **收窄距离范围**：`distance_cm_max: 60 → 25`，K 从 28 降到 9，即可把问题难度降低 3×
2. **Soft-bin targets**（已实现，待训练 A2+soft）：给近邻 bin 梯度信号，打破 mislocalize 循环
3. **两阶段训练**：先训 binary detect（BCE），冻结 Conv+GRU，再 fine-tune K+1 head
4. **可视化验证**：对 positive 帧倒谱 patch 做可视化，确认峰真实存在于搜索区间

---

## 九、推理使用方式

```python
from ml_uav_comb.models.comb_cepstral_net import CombCepstralNet
from ml_uav_comb.features.feature_utils import load_yaml_config

cfg = load_yaml_config("ml_uav_comb/configs/cepstral_default.yaml")
model = CombCepstralNet(cfg)
checkpoint = torch.load("ml_uav_comb/artifacts/cepstral_default/best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# batch["x"]: [B, T=68, Q=28]，已经过 MAD 归一化的倒谱 patch
with torch.inference_mode():
    out = model(batch)
    decoded = model.decode_distance(out["logits"])

distance_cm   = decoded["hard_distance_cm"]   # 0 if no-pattern
confidence    = decoded["pattern_prob"]        # 1 - P(no-pattern)
```

---

## 十、文件结构

```
ml_uav_comb/
├── models/
│   └── comb_cepstral_net.py          # 网络定义 + compute_cepstral_geometry()
├── data_pipeline/
│   └── cepstral_dataset.py           # CepstralBinDataset + _compute_cepstral_patches()
├── training/
│   ├── cepstral_trainer.py           # 完整训练循环
│   ├── cepstral_losses.py            # Focal loss + soft-bin targets
│   └── cepstral_metrics.py           # 每步/每 epoch 指标
├── configs/
│   ├── cepstral_default.yaml         # 主训练配置（A2 完整模型）
│   ├── cepstral_tiny_debug.yaml      # 快速调试（小数据，用于 tests）
│   ├── cepstral_a1_no_gru.yaml       # 消融 A1：无 GRU
│   ├── cepstral_a2_soft_bins.yaml    # 消融 A2+soft：Gaussian bin targets
│   └── cepstral_a3_no_norm.yaml      # 消融 A3：无 MAD 归一化
├── scripts/
│   └── train_cepstral.py             # 训练入口
└── tests/
    └── test_cepstral_model_forward.py # 17 个单元测试
```
