# Training Plan v2

## Current Supervision Reality

- `distance`: 真实 GT 来自 `rec_1.wav + range_1.csv`
- `sign`: 当前默认来自局部线性拟合派生，除非未来提供显式 `motion_sign`
- `confidence`: 当前数据里没有可靠 GT，默认还是 physics-derived pseudo supervision

所以 v2 的关键不是“把 pseudo 全删掉”，而是：

- 不再把 pseudo 混进 GT metric
- 不再让 heuristic state 泄漏进主输入
- 不再让 pseudo-only recording 混进评估

## Stage A

目标：

- 先学 `confidence` 和 `sign`
- `distance` 权重默认置 0

checkpoint 选择：

- 按 `sign_acc_train + confidence_acc_train` 选最优

## Stage B

目标：

- 从 Stage A 初始化
- 主攻连续距离回归
- 只在 `dist_train_mask` 上算距离 loss

checkpoint 选择：

- 按 `distance_mae_cm_gt` 选最优

## Target Space

支持：

- `raw`
- `log`
- `inverse`

当前默认：

- `raw`

训练内部在 target space 上做回归，评估和推理统一解码回 `distance_cm`。

## Mask Policy

- `valid_dist_gt_mask`: 距离 GT 是否存在
- `dist_reliable_mask`: 该窗口是否具备可靠 comb cue
- `dist_train_mask = valid_dist_gt_mask & dist_reliable_mask`
- `valid_sign_gt_mask`: 是否有显式 GT sign
- `sign_train_mask`: GT sign 或局部拟合 sign 是否可用于训练
- `valid_conf_gt_mask`: 是否有显式 GT confidence
- `conf_train_mask`: 训练时是否对 confidence 施加监督

## Future Route

- 接入真实 `motion_sign / valid_mask`
- 接入 COMSOL / synthetic 数据，先 synthetic pretrain，再 real fine-tune
- 在更多 recording 上启用 recording-level split，而不是单 recording time split
