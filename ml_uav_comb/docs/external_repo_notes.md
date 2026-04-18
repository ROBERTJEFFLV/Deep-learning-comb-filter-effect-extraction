# External Repo Notes

## Repo 1: `michaelneri/audio-distance-estimation`

### Purpose

这是一个单通道房间内音频距离估计仓库，目标是从单声道音频中回归连续距离。

### Worth Borrowing

- `model.py` 里的 `log magnitude + sin(phase) + cos(phase)` 输入思想
- CNN + GRU 的时频-时间联合建模
- attention-enhanced CRNN 的轻量主干思路
- 用时间维表示整段内部演化，而不是只看单帧

### Not Suitable To Copy Directly

- 直接对原始整段音频回归单个 scalar
- 通用 speech-distance 假设
- 将输入直接建立在通用 STFT 上而忽略 comb-motion 物理先验
- 依赖 Lightning / torchlibrosa 的训练包装

### How We Adapt It

- 保留 `log_mag + sin_phase + cos_phase` 作为 phase-aware STFT 分支
- 不做整段黑盒回归，而是 sequence-to-center
- 额外加入 diff-comb 分支、scalar heuristic 分支、可选 stpACC 分支
- 让时序主干围绕 comb-motion 而不是 speech reverberation 泛化假设

## Repo 2: `sakshamsingh1/sound_distance_estimation`

### Purpose

这个仓库以 SELDnet 工程组织为基础，提供了一个“特征提取 -> 缓存 -> generator/dataloader -> 训练”的完整训练流水线，并在 only-distance 模式下引入了 mask + distance 联学、预训练和微调协议。

### Worth Borrowing

- `batch_feature_extraction.py` 的离线特征与标签缓存结构
- `cls_data_generator.py` 的序列化 batch 组织思路
- `parameters.py` 的集中配置管理
- `seldnet_model.py` 里 `perm_3` 的两阶段思想：
  - 先学 mask
  - 再 joint 学 mask + distance
- `train_seldnet.py` 的 quick-test / finetune 训练入口结构

### Not Suitable To Copy Directly

- 多通道阵列、SALSA-lite、GCC、SELD、多事件输出
- 默认 event/DOA 标签语义
- 将距离输出强绑定到阵列/SELD 结构

### How We Adapt It

- 借用它的工程结构，不借用它的输入假设
- 把 `mask` 语义改成“reliable comb-motion / reliable reflection cue”
- 把 `distance` 改成单麦克风、近场、UAV ego-noise 场景下的中心窗连续距离
- 两阶段训练改成：
  - Stage A 学 `confidence + sign`
  - Stage B 在可靠窗上强化 `distance`

## Repo 3: `dberghi/SELD-distance-features`

### Purpose

这个仓库实现了基于混响的距离辅助特征提取，重点包括 direct/reverb 分解、DRR 和 `stpACC`。

### Worth Borrowing

- `extractFeatures.py` / `utils.py` 中 `stpACC` 的模块化提取思路
- 将距离相关辅助特征并联到主网络的做法
- 明确把 early reflection 作为辅助观测，而不是唯一依据

### Not Suitable To Copy Directly

- 3D SELD 总体任务包装
- WPE direct/reverb 流水线作为本项目主前端
- 把 `stpACC` 当作跨帧 comb-motion coherence gate 的替代品

### How We Adapt It

- v1 只实现轻量单声道 `stpACC`
- 将其作为辅助分支输入
- 不用它替代 `comb shift lag / rho` 和跨帧相关门控

## Explicit Boundary: Why stpACC Cannot Replace Lag/Rho Gate

`stpACC` 关注的是短时延自相关能量结构，属于 early-reflection 辅助特征。

而当前项目中的 `lag / rho` 逻辑表达的是：

- 当前 comb 模式是否在跨帧稳定移动
- 这个移动是否具备足够相关性
- 该窗口是否存在可靠的 comb-motion coherence

因此：

- `stpACC` 可以辅助网络感知早期反射强弱；
- 但 `stpACC` 不是 residual gate，也不是 `lag / rho gate` 的替代品；
- 本项目里真正决定 “comb-motion 是否可靠” 的核心仍是 diff-comb 与 lag/rho 相关线索。

