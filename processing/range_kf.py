# -*- coding: utf-8 -*-
"""
range_kf.py
-----------------
一维匀速 Kalman 滤波（CV: Constant-Velocity）用于测距平滑。
状态：x = [d, v]^T（d: 距离 cm，v: 速度 cm/s）
观测：z = d + noise

新增（可选）物理约束：
- v_max：速度上限（|v| <= v_max）。设为None或<=0表示禁用限制
- a_max：加速度上限（|v_k - v_{k-1}| <= a_max * dt）。设为None或<=0表示禁用限制
- 基于 v_max 的步长上限：|d_k - d_{k-1}| <= v_max * dt

注意：上述约束在 predict 阶段对 x_pred 进行裁剪；协方差仍按线性模型推进。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
from collections import deque
import numpy as np

__all__ = ["RangeKF", "tuned_params", "FixedLagRTS"]


@dataclass(frozen=True)
class tuned_params:
    R: float = 1194.8190873140381          # cm^2（测量噪声方差）
    sigma_a: float = 16.59745689199247     # cm/s^2（等效加速度标准差）


class RangeKF:
    """
    一维 CV Kalman：x=[d, v]; z=d
    单位：距离 cm；时间 s
    """
    def __init__(
        self,
        R: float = tuned_params.R,
        sigma_a: float = tuned_params.sigma_a,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        v_max: Optional[float] = None,  # 速度上限(cm/s)，None或<=0表示禁用
        a_max: Optional[float] = None,  # 加速度上限(cm/s²)，None或<=0表示禁用
        d_min: Optional[float] = 0.0,   # 距离下限(cm)，None表示不限制，默认不允许负值
    ) -> None:
        self.R = float(R)
        self.sigma_a = float(sigma_a)
        self.x: Optional[np.ndarray] = None if x0 is None else np.asarray(x0, dtype=np.float32).reshape(2,)
        self.P: Optional[np.ndarray] = None if P0 is None else np.asarray(P0, dtype=np.float32).reshape(2,2)
        if (self.x is not None) and (self.P is None):
            # 若给了 x0 未给 P0，则给一个较保守的协方差
            self.P = np.diag([self.R, 1000.0]).astype(np.float32)
        # 速度/加速度上限（可选）
        # 注意：0值表示禁用限制，None也表示禁用限制
        self.v_max: Optional[float] = None if (v_max is None or v_max <= 0) else float(v_max)
        self.a_max: Optional[float] = None if (a_max is None or a_max <= 0) else float(a_max)
        self.d_min: Optional[float] = None if (d_min is None) else float(d_min)
        # 记录最近一次 predict 的步长与预测前状态，用于在 update 后施加加速度约束
        self._last_dt: Optional[float] = None
        self._x_before_predict: Optional[np.ndarray] = None

    # ---------- 内部：过程噪声 Q ----------
    def _Q(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        return np.array([[dt2 * dt2 / 4.0, dt2 * dt / 2.0],
                         [dt2 * dt / 2.0,  dt2           ]], dtype=np.float32) * (self.sigma_a ** 2)

    # ---------- 外部接口 ----------
    def reset(self) -> None:
        self.x, self.P = None, None

    def set_noise(self, R: Optional[float] = None, sigma_a: Optional[float] = None) -> None:
        if R is not None:
            self.R = float(R)
        if sigma_a is not None:
            self.sigma_a = float(sigma_a)

    def predict(self, dt: float) -> None:
        """按时间步长 dt(s) 进行一次状态预测，并对预测值进行限速/限加速度/限步长裁剪。"""
        if self.x is None:
            return
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)

        # 线性预测
        x_prev = self.x.copy()
        x_lin = F @ self.x
        d0, v0 = float(x_prev[0]), float(x_prev[1])
        d1, v1 = float(x_lin[0]),  float(x_lin[1])

        # 限加速度：|v1 - v0| <= a_max * dt
        if (self.a_max is not None) and (dt > 0):
            dv = v1 - v0
            a_lim = float(self.a_max) * float(dt)
            if dv > a_lim:
                v1 = v0 + a_lim
            elif dv < -a_lim:
                v1 = v0 - a_lim

        # 限速度：|v1| <= v_max；限步长：|d1 - d0| <= v_max * dt
        if self.v_max is not None:
            vmax = float(self.v_max)
            if v1 > vmax:
                v1 = vmax
            elif v1 < -vmax:
                v1 = -vmax
            if dt > 0:
                step = d1 - d0
                step_lim = vmax * float(dt)
                if step > step_lim:
                    d1 = d0 + step_lim
                elif step < -step_lim:
                    d1 = d0 - step_lim

        # 距离下限（不允许负数或小于设定下限）
        if self.d_min is not None:
            if d1 < float(self.d_min):
                d1 = float(self.d_min)

        self.x = np.array([d1, v1], dtype=np.float32)
        self.P = F @ self.P @ F.T + self._Q(dt)
        # 保存用于后续加速度约束的信息
        self._last_dt = float(dt)
        self._x_before_predict = x_prev.astype(np.float32)

    def update(self, z: float, R_override: Optional[float] = None) -> None:
        """用一次标量观测 z(cm) 更新；可用 R_override 临时替代测量方差。"""
        R = self.R if (R_override is None) else float(R_override)
        if self.x is None:
            # 首次观测：初始化
            self.x = np.array([float(z), 0.0], dtype=np.float32)
            self.P = np.diag([R, 1000.0]).astype(np.float32)
            return
        H = np.array([[1.0, 0.0]], dtype=np.float32)  # 1x2
        y = float(z) - float(H @ self.x)              # 创新
        S = float(H @ self.P @ H.T) + R               # 标量
        K = (self.P @ H.T) / S                        # 2x1
        self.x = self.x + (K.flatten() * y)
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        # 距离下限（更新后再次裁剪）
        if self.d_min is not None:
            if float(self.x[0]) < float(self.d_min):
                self.x[0] = float(self.d_min)
        # 在测量更新后施加 a_max 与 v_max 物理约束（如配置）
        try:
            if (self.a_max is not None) and (self._last_dt is not None) and (self._last_dt > 0) and (self._x_before_predict is not None):
                v_prev = float(self._x_before_predict[1])
                v_new  = float(self.x[1])
                a_lim  = float(self.a_max) * float(self._last_dt)
                dv     = v_new - v_prev
                if dv > a_lim:
                    v_new = v_prev + a_lim
                elif dv < -a_lim:
                    v_new = v_prev - a_lim
                if self.v_max is not None:
                    vmax = float(self.v_max)
                    if v_new > vmax:
                        v_new = vmax
                    elif v_new < -vmax:
                        v_new = -vmax
                self.x[1] = float(v_new)
        except Exception:
            pass

    # ---------- 便捷访问 ----------
    @property
    def distance(self) -> Optional[float]:
        return None if self.x is None else float(self.x[0])

    @property
    def velocity(self) -> Optional[float]:
        return None if self.x is None else float(self.x[1])

    @property
    def state(self) -> Optional[np.ndarray]:
        return None if self.x is None else self.x.copy()

    @property
    def covariance(self) -> Optional[np.ndarray]:
        return None if self.P is None else self.P.copy()

    # ---------- 可选：一步法（含质量门控） ----------
    def step(self, dt: float, z: Optional[float], quality: Optional[Dict[str, float]] = None) -> None:
        self.predict(dt)
        if z is None:
            return
        rmse = quality.get("rmse") if quality else None

        # 仅软放大 R（可留可删；若不想受 rmse 影响，直接把这段置空并用 self.update(z)）
        R_eff = None
        if rmse is not None:
            scale = 1.0 + max(0.0, (rmse - 8.0) / 8.0)
            scale = float(min(scale, 4.0))
            R_eff = self.R * scale

        self.update(z, R_override=R_eff)



class FixedLagRTS:
    """
    固定时滞 RTS 平滑器（1D CV 模型）：缓存最近 L+1 个时刻的
    F_k、x_filt(k)、P_filt(k)、x_pred(k+1)、P_pred(k+1)。
    在每次推入新点后做一次后向平滑，输出 k-L 时刻的平滑结果。
    """
    def __init__(self, lag: int = 3, v_max=None, a_max=None, dt_window=None, d_min: Optional[float] = 0.0):
        assert lag >= 1
        self.lag = int(lag)
        self.v_max = v_max
        self.a_max = a_max
        self.dt_window = dt_window  # AudioProcessor传递的窗口时间间隔
        self.d_min = None if (d_min is None) else float(d_min)
        self.buf = deque(maxlen=lag + 1)  # 每项: dict(F, x_filt, P_filt, x_pred_next, P_pred_next)
        self._last_smoothed = None

    def reset(self):
        """清空平滑历史，用于新的独立段。"""
        try:
            self.buf.clear()
        except Exception:
            # 兼容性保护
            self.buf = deque(maxlen=self.lag + 1)
        self._last_smoothed = None

    def push(self, F, x_pred_next, P_pred_next, x_filt, P_filt, timestamp=None):
        # 复制以免外部修改
        item = dict(
            F = np.asarray(F, dtype=np.float32),
            x_filt = np.asarray(x_filt, dtype=np.float32).copy(),
            P_filt = np.asarray(P_filt, dtype=np.float32).copy(),
            x_pred_next = np.asarray(x_pred_next, dtype=np.float32).copy(),
            P_pred_next = np.asarray(P_pred_next, dtype=np.float32).copy(),
            timestamp = timestamp,  # 添加时间戳以计算实际间隔
        )
        self.buf.append(item)
        if len(self.buf) < 2:
            self._last_smoothed = None
            return

        # 后向平滑：从最新时刻 n 向前到 0
        n = len(self.buf) - 1
        xs = [None] * (n + 1)
        Ps = [None] * (n + 1)

        xs[n] = self.buf[n]['x_filt']
        Ps[n] = self.buf[n]['P_filt']

        for j in range(n - 1, -1, -1):
            Fj   = self.buf[j]['F']
            xfj  = self.buf[j]['x_filt']
            Pfj  = self.buf[j]['P_filt']
            xpj1 = self.buf[j]['x_pred_next']
            Ppj1 = self.buf[j]['P_pred_next']

            # RTS 增益：Cj = Pfj F_j^T P_pred(j+1)^{-1}
            Cj = Pfj @ Fj.T @ np.linalg.inv(Ppj1)
            xs_raw = xfj + Cj @ (xs[j+1] - xpj1)
            
            # 计算实际的时间间隔用于约束
            if j > 0 and self.buf[j-1].get('timestamp') is not None and self.buf[j].get('timestamp') is not None:
                # 从前一时刻j-1到当前时刻j的时间间隔
                actual_dt = self.buf[j]['timestamp'] - self.buf[j-1]['timestamp']
                actual_dt = max(actual_dt, 1e-6)  # 避免除零，确保为正值
            else:
                # 回退到默认时间间隔
                actual_dt = self.dt_window if self.dt_window is not None else 0.098685
                
            # 对RTS平滑后的状态应用物理约束
            # 需要前一时刻的平滑状态作为参考
            if j > 0 and xs[j-1] is not None:
                x_prev_smoothed = xs[j-1]
            else:
                # 第一个时刻，使用滤波结果作为参考
                x_prev_smoothed = xfj
                
            xs[j] = self._apply_physical_constraints(xs_raw, x_prev_smoothed, actual_dt)
            Ps[j] = Pfj + Cj @ (Ps[j+1] - Ppj1) @ Cj.T

        # 只有当缓存长度达到 L+1 时，才有 k-L 的平滑输出
        self._last_smoothed = (xs[0], Ps[0]) if (len(self.buf) == self.buf.maxlen) else None

    def get_smoothed(self):
        """返回 (x_s, P_s) 或 None（缓存未满 lag+1）。"""
        return self._last_smoothed

    def _apply_physical_constraints(self, x_smoothed, x_prev, dt):
        """对RTS平滑后的状态应用物理约束
        
        Args:
            x_smoothed: RTS平滑后的状态 [距离, 速度]
            x_prev: 前一时刻的滤波状态 [距离, 速度] 
            dt: 实际时间间隔
        """
        if self.v_max is None and self.a_max is None:
            return x_smoothed
        
        # 提取状态变量
        x_constrained = x_smoothed.copy()
        d_smoothed, v_smoothed = float(x_constrained[0]), float(x_constrained[1])
        d_prev, v_prev = float(x_prev[0]), float(x_prev[1])

        # 距离下限先裁剪一次，避免下游步长/速度限制把负值传播
        if self.d_min is not None and d_smoothed < float(self.d_min):
            d_smoothed = float(self.d_min)
        
        # 使用传入的实际时间间隔
        dt = float(dt)
        if dt <= 0:
            return x_smoothed  # 无效时间间隔，不应用约束
        
        # 首先限制步长：|d_smoothed - d_prev| <= v_max * dt
        if self.v_max is not None:
            step = d_smoothed - d_prev
            step_lim = float(self.v_max) * dt
            if step > step_lim:
                d_smoothed = d_prev + step_lim
            elif step < -step_lim:
                d_smoothed = d_prev - step_lim
            
            # 从约束后的距离重新计算速度
            v_from_step = (d_smoothed - d_prev) / dt
        else:
            v_from_step = v_smoothed
        
        # 然后限制加速度：|v_from_step - v_prev| <= a_max * dt
        if self.a_max is not None:
            dv = v_from_step - v_prev
            a_lim = float(self.a_max) * dt
            if dv > a_lim:
                v_constrained = v_prev + a_lim
            elif dv < -a_lim:
                v_constrained = v_prev - a_lim
            else:
                v_constrained = v_from_step
        else:
            v_constrained = v_from_step
        
        # 最后确保速度在绝对限制内：|v_constrained| <= v_max
        if self.v_max is not None:
            vmax = float(self.v_max)
            if v_constrained > vmax:
                v_constrained = vmax
            elif v_constrained < -vmax:
                v_constrained = -vmax
        
        # 根据最终速度重新调整距离（确保一致性）
        if self.v_max is not None and dt > 0:
            d_final = d_prev + v_constrained * dt
        else:
            d_final = d_smoothed

        # 距离下限最终保障
        if self.d_min is not None and d_final < float(self.d_min):
            d_final = float(self.d_min)
        
        # 更新约束后的状态
        x_constrained[0] = d_final
        x_constrained[1] = v_constrained
        return x_constrained
