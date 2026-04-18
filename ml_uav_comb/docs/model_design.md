# Model Design (Distance-Grid Observer v1)

## Core Pipeline

1. Cached trailing window (`window_anchor=trailing`, `target_position=window_end`)
2. Causal TCN observer
4. Distance logits over candidate grid
5. Measurement summary derived from logits
6. `DistanceGridRangeTracker` (`RangeKF` wrapper)
7. Posterior distance estimate

## Why `distance_logits` Is the Primary Output

- Observer first建模的是“候选距离似然分布”，不是单点回归
- `measurement_distance_cm` 用分布期望得到
- `measurement_logvar/entropy/margin` 直接来自同一分布
- KF 能根据这些不确定性量调节 `R_eff`

## Why v1 Removes Handcrafted Physical Scoring

- 现阶段手写 notch/spacing 规则难以在不同录制条件下稳定泛化
- v1 主路径不再依赖 deterministic physical likelihood scorer
- observer 直接在 candidate distance grid 上学习 logits，保持链路短、可训练、可验证

## Causal Observer

- 复用现有 branch：phase/comb/scalar/stpacc
- 不再使用双向 RNN
- 使用 causal dilated TCN，保证无未来泄漏
- 仅取最后时间步输出 `distance_logits` 与 `measurement_validity_logit`
- `distance_grid_cm` 由配置生成（uniform/log），用于 logits summary 与 soft target 对齐

## Measurement vs Posterior

- measurement：来自 observer 输出分布的瞬时测量
- posterior：KF 融合历史后的状态估计
- 两者必须分开导出，便于分析模型测量质量与滤波增益

## Why KF First, Not UKF

- 当前测量模型是一维距离观测，线性 CV-KF 已足够跑通闭环
- v1 目标是在线主链语义正确，不引入额外状态维和 sigma-point 复杂度
- UKF 接口可在后续版本基于相同 measurement summary 平滑扩展
