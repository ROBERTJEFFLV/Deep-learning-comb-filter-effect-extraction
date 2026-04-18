# Project Goal

## Real Goal

当前仓库的核心目标不是普通音频分类，也不是普通声源定位。

它要解决的是：利用无人机自噪声与近场障碍物反射叠加后，在单麦克风频谱上形成的 comb-filter 干涉结构，做近场障碍物感知，并进一步做连续距离估计。

从现有代码和论文可以确认，真正有价值的线索不是“单帧频谱长什么样”，而是跨时间帧的 comb-motion 变化：

- `processing/audio_processor.py` 先对 1-5 kHz 频带做 STFT 频谱预处理。
- 再计算跨历史帧的差分谱 `diff_amplitude / smooth_d1`，并通过 `sum_abs_d1` 做门控。
- 再通过 `processing/comb_shift.py` 的 lag / rho 相关逻辑，判断 comb notch 的跨帧移动方向与可靠性。
- 再在 68 帧窗口上做余弦拟合，把 comb 周期映射到距离。
- 最后用 `RangeKF + FixedLagRTS` 做连续距离平滑与物理约束。

因此，本项目的学习系统必须复用这些物理前端和启发式状态量，而不是把项目改造成黑盒 raw-audio end-to-end。

## Valuable Priors To Preserve

当前代码中真正值得继承的先验是：

1. band-limited spectrum / phase
2. `d1 / diff-comb` 特征，尤其是 `smooth_d1`
3. `sum_abs_d1` 或其平滑版本
4. comb shift 的 lag / rho 可靠性线索
5. 当前 heuristic 的 `direction_d1 / distance / distance_kf / velocity_kf / acceleration_kf`

这些先验共同描述的是“comb-motion 是否存在、是否稳定、以及其周期与方向如何演化”，而不只是某一帧的静态谱形。

## Why This Is Not Plain Audio Distance Estimation

外部仓库里常见的单通道距离估计，多数默认：

- 目标是通用 speech distance
- 输入是整段音频或通用 STFT
- 输出是单个距离标量

这与当前仓库不同。当前仓库的目标是：

- 单麦克风
- 无人机 ego-noise 主导
- 近场障碍物反射
- comb-filter 干涉
- 跨时间 comb-motion
- 连续距离估计而不是一次性整段回归

## stpACC Boundary

`stpACC` 可以作为 early-reflection / short-delay 的辅助特征，但不能当作 residual gate 的直接替代品。

原因是：

- `stpACC` 更偏向短延迟自相关能量结构；
- 当前项目的关键门控是跨帧 comb-motion coherence，也就是 lag / rho 所编码的“comb 是否在按一致方式移动”；
- 因此 `stpACC` 只能并联辅助，不等价于 `comb shift lag / rho`。

这条边界在后续 `feature_contract.md` 和 `model_design.md` 中会继续保持一致。

## Repo/Code Evidence

- `README.md` 明确把系统定义为基于 comb filter effect 的实时测距系统。
- 论文《Quadrotor Ego-Noise-Based Passive Acoustic Sensing for Obstacle Detection》强调的是 ego-noise reflections、temporal differencing、cross-frame spectral correlation。
- `processing/audio_processor.py` 的主链路围绕 `smooth_d1 -> direction vote -> window cosine fit -> KF/RTS` 展开。
- `processing/comb_shift.py` 直接输出 lag / rho 相关的 comb shift 方向与可靠性。

