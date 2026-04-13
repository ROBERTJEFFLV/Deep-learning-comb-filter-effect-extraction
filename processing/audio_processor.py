# src/processing/audio_processor.py
# -*- coding: utf-8 -*-
from queue import Queue, Empty
from collections import deque
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from config import (
    SR, N_FFT, HOP_LEN, FREQ_MIN, FREQ_MAX, HISTORY_FRAMES,
    SMOOTH_SIGMA_1, SMOOTH_SIGMA_2, N_BPF, D1_AMP_SCALE,
    JUMP_T_CM, JUMP_K_CM, JUMP_POWER, JUMP_R_SCALE_MAX, USE_DIRECTION_GATE,
    LS_PARAMS, KF_PARAMS, RTS_PARAMS, OUTPUT_FILTER
)
from data.frequency_bin import FrequencyBin
from processing.comb_shift import detect_comb_direction
import time, threading
from processing.range_kf import RangeKF, FixedLagRTS

# === 引入 sine_fit_22 中的核心函数（φ=0, A≥0, 仅估计 ω 与 A） ===
# 注意：以下函数名以下划线开头，但可直接导入使用
from processing.sine_fit_22 import (
    _make_omega_grid, _select_local_grid,
    _weights_from_amp, _fit_A_phi0_weighted, _coarse_search_fit_phi0
)


# 物理常量（用于距离换算）
C_SPEED = 343.0  # m/s

class AudioProcessor:
    def __init__(
        self,
        process_queue: Queue,
        frame_queue: Queue,
        running_flag: callable,
        ls_params: dict = None,
        kf_params: dict = None,
        rts_params: dict = None,
        output_filter: dict = None
    ):
        """
        process_queue: 每次回调推来的 length=HOP_LEN 的浮点数组
        frame_queue: 处理完的特征帧输出
        ls_params: LS 拟合参数（窗口大小、步长、先验等）
        kf_params: KF 参数（R, sigma_a）
        rts_params: RTS 参数（lag）
        output_filter: 输出过滤参数（min_amplitude）
        """
        self.pq      = process_queue
        self.out_q   = frame_queue
        self.running = running_flag

        # 加载参数（如果未提供则使用默认值）
        self.ls_params = dict(LS_PARAMS) if ls_params is None else dict(ls_params)
        self.kf_params = dict(KF_PARAMS) if kf_params is None else dict(kf_params)
        self.rts_params = dict(RTS_PARAMS) if rts_params is None else dict(rts_params)
        self.output_filter = dict(OUTPUT_FILTER) if output_filter is None else dict(output_filter)

        # FFT 频率与选择的频带（47 个频率点）
        self.fft_freqs = np.fft.rfftfreq(N_FFT, 1/SR)
        self.freq_idx  = np.where(
            (self.fft_freqs >= FREQ_MIN) & (self.fft_freqs <= FREQ_MAX)
        )[0]
        self.selected_freqs = self.fft_freqs[self.freq_idx]
        self.bins    = [FrequencyBin(f) for f in self.selected_freqs]
        self.hop_sec = HOP_LEN / SR

        # 历史缓存
        self.s_hist   = deque(maxlen=150)
        self.h_amp    = deque(maxlen=HISTORY_FRAMES)
        self.d1_hist  = deque(maxlen=150)
        self.amp_hist = deque(maxlen=15)

        # 短窗 & 长窗缓冲 与计数
        self.buf_short   = np.zeros(N_FFT, dtype=np.float32)
        self.buf_long    = np.zeros(N_BPF, dtype=np.float32)
        self.count       = 0
        self.k           = N_BPF // HOP_LEN  # = 2048/128 = 16

        # 方向投票/判定（保留你原有方向检测周边结构，不依赖 SWCLS）
        self.trend_n = 6
        self.trend_min_votes = 1
        self.dir_window = deque(maxlen=20)
        self.cnt_threshold = 10
        self.prev_d1_k = None
        self.k_compare = 30
        self.d1_window = deque(maxlen=self.k_compare)
        self.window_dt = (self.k_compare - 1) * self.hop_sec
        self.bin_width = (FREQ_MAX - FREQ_MIN) / (len(self.selected_freqs) - 1)
        self.amp_threshold = 0.250
        self.sum_abs_d1_hist = deque(maxlen=300)
        self.last_bpf_info = {"f0": None, "fdot": 0.0, "harmonics": [], "conf": 0.0}
        self.prev_d1 = None
        self.sound_threshold = 0.005
        self.last_valid_direction = "None"
        self.none_dir_streak = 0
        self.disable_prev_init = False
        self._last_printed_direction = None  # 上次打印的方向（用于去抖）
        self._last_dir_print_time = 0.0      # 上次打印时间
        self._dir_print_interval = 0.5       # 打印间隔（秒）

        # === 窗口蓄水池（从参数读取或使用默认值） ===
        self.window_size = self.ls_params.get('window_size', 68)
        self.window_hop  = self.ls_params.get('window_hop', 34)
        self.window_pick = self.ls_params.get('window_pick', 11)
        self.window_buffer = deque(maxlen=self.window_size)
        self._frames_since_window = 0

        # === LS 拟合参数 ===
        self.d_min_m = self.ls_params.get('d_min_m', 0.005)
        self.d_min_cm = float(self.d_min_m) * 100.0
        self.oversample = self.ls_params.get('oversample', 8.0)
        self.gaussian_sigma = self.ls_params.get('gaussian_sigma', 6.0)
        self.local_radius_first = self.ls_params.get('local_radius_first', 3)
        self.local_radius_segment = self.ls_params.get('local_radius_segment', 1)     

        # === sine_fit_22 集成状态 ===
        self.prev_window_omega = None
        self.last_fit_result = None

        # 预生成全局 ω 网格
        self.omega_grid_global = _make_omega_grid(
            fmin=float(FREQ_MIN),
            fmax=float(FREQ_MAX),
            n_points=len(self.selected_freqs),
            d_min_m=self.d_min_m,
            c_speed=C_SPEED,
            oversample=self.oversample
        )

        # ===== ① 预计算 COS_GLOBAL：一次性 cos(ω·f)，后续仅切片复用 =====
        # 形状：(G, F)；用 float32 降内存
        Phi = np.outer(self.omega_grid_global, self.selected_freqs).astype(np.float32)
        self.COS_GLOBAL = np.cos(Phi)  # (G, F)

        # ===== ② “就近下标”工具：把实数 ω 映射到最邻近的全局网格下标 =====
        def _nearest_idx_fn(omega: float) -> int:
            G = self.omega_grid_global
            i = int(np.searchsorted(G, omega))
            if i == len(G):
                i -= 1
            if i > 0:
                if abs(G[i] - omega) > abs(G[i - 1] - omega):
                    i -= 1
            return i
        self._nearest_idx = _nearest_idx_fn

        kf_R      = self.kf_params.get('R', 1194.8190873140381)
        kf_sigma_a= self.kf_params.get('sigma_a', 16.59745689199247)
        kf_vmax   = self.kf_params.get('v_max', None)
        kf_amax   = self.kf_params.get('a_max', None)

        self.v_max = kf_vmax
        self.a_max = kf_amax

        self.kf = RangeKF(R=kf_R, sigma_a=kf_sigma_a, v_max=kf_vmax, a_max=kf_amax, d_min=self.d_min_cm)

        self.dt_window = self.window_hop * self.hop_sec
        rts_lag = self.rts_params.get('lag', 3)
        self.rts = FixedLagRTS(lag=rts_lag, v_max=self.v_max, a_max=self.a_max, dt_window=self.dt_window, d_min=self.d_min_cm)

        self._last_distance_kf   = None
        self._last_velocity_kf   = None
        self._last_acceleration  = None
        self._prev_velocity      = None
        self._prev_time          = None
        self._last_measure_time  = None
        self._force_reinit       = False
        self._missing_threshold  = 1.0
        
        # Jump-aware measurement downweight state
        self._last_meas  = None   # 最近一次用于更新的测量 z
        self._last_delta = None   # z 的前向增量，用于"方向一致"惯性判据
        
        # 输出门控：基于振幅的简单阈值（保留你的旧逻辑）
        self.min_amplitude = self.output_filter.get('min_amplitude', 0.25)

        # ===== 噪声门（绝对强度阈值，基于原始音频 RMS） =====
        # 从 config 读取，避免在 main.py 单独处理
        from config import NOISE_GATE_ENABLED, NOISE_GATE_THRESHOLD, NOISE_GATE_SMOOTH
        self.noise_gate_enabled = NOISE_GATE_ENABLED
        self.noise_gate_threshold = NOISE_GATE_THRESHOLD
        self.noise_gate_smooth = NOISE_GATE_SMOOTH
        self._smoothed_rms = 0.0  # 平滑后的绝对 RMS



    # ========== 将 68 帧窗口合成 11 段，并在 φ=0,A≥0 下执行"继承 + 局部搜索"的 ω 估计（向量化版） ==========
    def _fit_window_phi0_inherit(self, block_11xF: np.ndarray) -> dict:
        """
        向量化/去重计算版本：
        - 复用预计算 COS_GLOBAL（避免反复 cos()）
        - 每段 A 的拟合：对局部候选一次性闭式最小二乘（einsum）
        - 共识累计 SSE：J×K×F 三维广播一次完成
        """
        assert block_11xF.ndim == 2
        f_grid = self.selected_freqs
        COS_G  = self.COS_GLOBAL            # (G,F)
        OMG_G  = self.omega_grid_global     # (G,)

        # —— 轻量时间轴平滑（使用配置的 sigma）——
        block_s = gaussian_filter1d(block_11xF, sigma=self.gaussian_sigma, axis=0, mode="reflect")

        omegas = []
        A_bests = []
        omega_prev_seg = None

        use_local_within = (self.prev_window_omega is not None)


        for k, y in enumerate(block_s):
            # ③ 每段只算一次权重（等权就令 w=None）
            w = _weights_from_amp(y)
            sw = np.sqrt(w).astype(np.float32) if w is not None else None

            # ④ 通过"就近下标 + 半径"确定局部切片（使用配置的半径）
            if k == 0 and (self.prev_window_omega is not None):
                i0 = self._nearest_idx(self.prev_window_omega)
                radius = self.local_radius_first
            elif omega_prev_seg is not None:
                i0 = self._nearest_idx(omega_prev_seg)
                radius = self.local_radius_segment
            else:
                i0 = None
                radius = None

            if i0 is None:
                lo, hi = 0, len(OMG_G)
            else:
                lo = max(0, i0 - radius)
                hi = min(len(OMG_G), i0 + radius + 1)

            COS_loc = COS_G[lo:hi, :]            # (G_loc, F)
            OMG_loc = OMG_G[lo:hi]               # (G_loc,)

            # ⑤ 【向量化】一次性闭式最小二乘求所有候选的 A_hat(ω) 与 SSE(ω)
            if sw is None:
                # 等权：A_hat = <cos,y> / <cos,cos>
                denom = np.einsum('gf,gf->g', COS_loc, COS_loc, optimize=True) + 1e-12
                numer = np.einsum('gf,f->g',  COS_loc, y,        optimize=True)
                A_hat = numer / denom
                # 只确保非负幅度
                A_hat_clipped = np.maximum(A_hat, 0.0)
                # SSE
                yhat  = (A_hat_clipped[:, None] * COS_loc)
                r     = (yhat - y)
                sse   = np.einsum('gf,gf->g', r, r, optimize=True)
            else:
                # 加权：把权开根号融入设计矩阵
                COSw  = COS_loc * sw[None, :]
                yw    = y * sw
                denom = np.einsum('gf,gf->g', COSw, COSw, optimize=True) + 1e-12
                numer = np.einsum('gf,f->g',  COSw, yw,   optimize=True)
                A_hat = numer / denom
                A_hat_clipped = np.maximum(A_hat, 0.0)
                yhat  = (A_hat_clipped[:, None] * COS_loc)
                r     = (yhat - y) * sw[None, :]
                sse   = np.einsum('gf,gf->g', r, r, optimize=True)

            best = int(np.argmin(sse))
            omegas.append(float(OMG_loc[best]))
            A_bests.append(float(A_hat_clipped[best]))
            omega_prev_seg = omegas[-1]

        if not omegas:
            return {'omega_win': 0.0, 'A_win': 0.0, 'rmse': float('inf'), 'r2': -1.0,
                    'distance_cm': 0.0, 'success': False}

        # ==================== ⑥ 共识累计 SSE（J×K×F 广播） ====================
        y_rep = np.median(block_s, axis=0).astype(np.float32)  # (F,)
        J = len(omegas)

        # 候选的 cos 行：用就近下标映射到 COS_GLOBAL（复用点①）
        idxJ = np.array([self._nearest_idx(w) for w in omegas], dtype=int)
        COS_J = COS_G[idxJ, :]                     # (J,F)
        A_v   = np.asarray(A_bests, dtype=np.float32)  # (J,)
        YHAT  = A_v[:, None] * COS_J               # (J,F)

        # 每段的权重（与上面保持一致的 _weights_from_amp）
        W_list = [_weights_from_amp(block_s[k]) for k in range(J)]
        if all(wi is None for wi in W_list):
            YK  = block_s.astype(np.float32)       # (K,F) 这里 K=J=11
            RES = (YHAT[:, None, :] - YK[None, :, :])          # (J,K,F)
            SSE = np.einsum('jkf,jkf->jk', RES, RES, optimize=True)
        else:
            WK  = np.stack([np.ones_like(block_s[0], dtype=np.float32) if wi is None else wi.astype(np.float32)
                            for wi in W_list], axis=0)  # (K,F)
            SWK = np.sqrt(WK)                           # (K,F)
            YK  = block_s.astype(np.float32)            # (K,F)
            RES = (YHAT[:, None, :] - YK[None, :, :]) * SWK[None, :, :]  # (J,K,F)
            SSE = np.einsum('jkf,jkf->jk', RES, RES, optimize=True)

        # 排除对角（自身）后，对 K 维求和
        np.fill_diagonal(SSE, 0.0)
        total_sse = SSE.sum(axis=1)               # (J,)
        best_idx  = int(np.argmin(total_sse))

        omega_win = float(omegas[best_idx])
        A_win     = float(A_bests[best_idx])

        # ⑦ 指标（未加权 RMSE/R²；y_rep 已算）
        yhat = A_win * COS_J[best_idx]            # (F,)
        rmse = float(np.sqrt(np.mean((yhat - y_rep) ** 2)))
        ss_res = float(np.sum((y_rep - yhat) ** 2))
        ss_tot = float(np.sum((y_rep - np.mean(y_rep)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        # 距离换算
        distance_cm = C_SPEED * 100.0 * omega_win / (4.0 * math.pi)

        # 更新“上一窗口 ω”
        # self.prev_window_omega = omega_win

        return {
            'omega_win': omega_win,
            'A_win': A_win,
            'rmse': rmse,
            'r2': r2,
            'distance_cm': distance_cm,
            'success': True
        }


    def _run(self):
        # 先填满 N_FFT 的初始缓冲
        filled = 0
        while filled < N_FFT and self.running():
            try:
                block = self.pq.get(timeout=0.1)
                L = min(len(block), N_FFT - filled)
                self.buf_short[filled:filled+L] = block[:L]
                filled += L
            except Empty:
                continue

        hop_n = HOP_LEN
        hop_sec = HOP_LEN / SR

        while self.running():
            try:
                block = self.pq.get(timeout=0.1)
            except Empty:
                continue

            # 绝对强度噪声门开始计时（若你后面有 profiling，可在此记录）
            frame_start = time.perf_counter()
            noise_blocked = False

            # === 1) 绝对强度检测：基于原始 PCM 块的 RMS ===
            if self.noise_gate_enabled:
                rms = float(np.sqrt(np.mean(block ** 2)))
                # 指数平滑，抑制瞬时尖峰
                self._smoothed_rms = (
                    self.noise_gate_smooth * self._smoothed_rms
                    + (1.0 - self.noise_gate_smooth) * rms
                )
                # 若低于阈值：仅标记阻塞，后续跳过 LS/KF 等重逻辑，但保持可视化输出
                if self._smoothed_rms < self.noise_gate_threshold:
                    noise_blocked = True

            # === 2) 正常处理流程（只有在强度足够大时才会走到这里） ===

            # 更新时间计数（无论噪声门是否阻塞，都递增以保持可视化前进）
            self.count += 1
            t = self.count * hop_sec

            # 短窗更新
            self.buf_short[:N_FFT-hop_n] = self.buf_short[hop_n:]
            self.buf_short[N_FFT-hop_n:] = block

            # === 频域与预处理 ===
            win     = self.buf_short * np.hanning(N_FFT)
            mag     = np.abs(np.fft.rfft(win, n=N_FFT))
            data    = mag[self.freq_idx]
            # normalize + smooth
            norm    = data/(data.max()+1e-12)
            smooth  = gaussian_filter1d(norm, sigma=SMOOTH_SIGMA_1)
            centered= smooth - smooth.mean()
            denom   = centered.max() - centered.min()
            current = ((centered-centered.min())/denom if denom>1e-6 else centered)

            # 声音存在性
            is_sound_present = False if noise_blocked else np.any(current > self.sound_threshold)
            if is_sound_present:
                self.count += 1

            # s_hist 平滑
            self.s_hist.append(current)
            if len(self.s_hist) > 1:
                current = np.mean(np.array(self.s_hist), axis=0)

            # d1 特征
            old = self.h_amp[0] if len(self.h_amp) == self.h_amp.maxlen else None
            for i, fb in enumerate(self.bins):
                fb.update(current[i], old[i] if old is not None else None)
            self.h_amp.append(current)

            d1 = np.array([fb.diff_amp for fb in self.bins])
            d1_freq_sm = gaussian_filter1d(d1, sigma=SMOOTH_SIGMA_1)
            self.d1_hist.append(d1_freq_sm)
            smooth_d1 = (gaussian_filter1d(np.array(self.d1_hist), sigma=SMOOTH_SIGMA_2, axis=0)[-1]
                         if len(self.d1_hist) > 1 else d1)

            # 统计 ∑|d1|
            sum_abs_d1 = float(np.sum(np.abs(smooth_d1)))
            self.sum_abs_d1_hist.append(sum_abs_d1)
            abs_arr = np.array(self.sum_abs_d1_hist, dtype=np.float32)
            window = np.ones(len(abs_arr), dtype=np.float32) / max(1, len(abs_arr))
            smooth_amp = np.convolve(abs_arr, window, mode='same')[-1]
            self.d1_window.append(smooth_d1)
            # === 方向判定（保留你原本的机制，但与 ω 拟合无关） ===
            # === 多帧趋势投票：最近 N 帧与当前 new 对比，方向多数投票（兼容 "Left"/"Right"）===
            if smooth_amp > self.amp_threshold and len(self.d1_window) >= self.k_compare:
                new = self.d1_window[-1]

                # 参与投票的参考帧数 N（不含最后一帧）
                n = getattr(self, "trend_n", 6)
                m_min = getattr(self, "trend_min_votes", 1)

                n = min(n, len(self.d1_window) - 1)
                # 从窗口内均匀抽 n 个参考帧索引（覆盖更长时间跨度）
                candidates = list(range(0, len(self.d1_window) - 1))  # 0..len-2
                if n > 0:
                    step = max(1, (len(candidates) // n))
                    ref_idx = candidates[::step][:n]
                else:
                    ref_idx = []

                votes_right = 0
                votes_left  = 0
                valid_pairs = 0

                # 先默认 zeros_d1 用“跨度最大”的一对（最老 vs 最新），位移更可见
                oldest = self.d1_window[ref_idx[0]] if len(ref_idx) > 0 else self.d1_window[0]
                _, zeros_d1 = detect_comb_direction(oldest, new, max_lag=16, rho_thresh=0.5)

                for idx in ref_idx:
                    old = self.d1_window[idx]
                    d_tmp, _ = detect_comb_direction(old, new, max_lag=16, rho_thresh=0.5)
                    if d_tmp == "Left":
                        votes_right += 1
                        valid_pairs += 1
                    elif d_tmp == "Right":
                        votes_left  += 1
                        valid_pairs += 1
                    # "None" 不计票

                if valid_pairs >= m_min:
                    if votes_right > votes_left:
                        direction_d1 = "Left"
                    elif votes_left > votes_right:
                        direction_d1 = "Right"
                    else:
                        direction_d1 = "None"  # 或 "flat"，按你需要
                else:
                    direction_d1 = "None"
            else:
                direction_d1, zeros_d1 = "None", []

            # --- 新增：将本帧方向写入 10 帧滑动窗口，并计算多数结果 ---
            window_direction = "None"
            if direction_d1 in ("Left", "Right"):
                self.dir_window.append(direction_d1)
                cnt_left  = sum(1 for d in self.dir_window if d == "Left")
                cnt_right = sum(1 for d in self.dir_window if d == "Right")
                # 至少 5 张相同方向票，且严格多于另一方向
                if cnt_left >= self.cnt_threshold and cnt_left > cnt_right:
                    window_direction = "Left"
                elif cnt_right >= self.cnt_threshold and cnt_right > cnt_left:
                    window_direction = "Right"

            final_direction = window_direction

            if window_direction == "None":
                self.none_dir_streak += 1
                if self.none_dir_streak > 50:
                    self.disable_prev_init = True
            else:
                self.none_dir_streak = 0
                self.disable_prev_init = False
                self.last_valid_direction = window_direction

                        # === 延时/惯性保持方向 ===
            if final_direction == "None":
                if smooth_amp > self.amp_threshold:
                    # 保持惯性方向
                    final_direction = self.last_valid_direction
                else:
                    # 信号消失，才真正清空
                    self.last_valid_direction = "None"
            else:
                # 有明确方向时，更新last_valid_direction
                self.last_valid_direction = final_direction

            if noise_blocked:
                final_direction = "None"


            # —— 实时打印 window_direction（变化时或每隔一段时间打印一次） ——
            try:
                now_ts = time.perf_counter()
                if (window_direction != self._last_printed_direction) or ((now_ts - self._last_dir_print_time) >= self._dir_print_interval):
                    print(f"[AudioProcessor] window_direction={window_direction} t={self.count * hop_sec:.2f}s window_buf={len(self.window_buffer)} sum_abs_d1={smooth_amp:.3f}")
                    self._last_printed_direction = window_direction
                    self._last_dir_print_time = now_ts
            except Exception:
                pass

            # === 窗口蓄水池推进（68 帧 → 取 11 段 → φ=0, A≥0 拟合 ω） ===
            try:
                # 每帧都推进蓄水池
                if not noise_blocked:
                    self.window_buffer.append(D1_AMP_SCALE * smooth_d1)

                if (not noise_blocked) and (len(self.window_buffer) == self.window_size):
                    # 缓冲已满：先累加 hop 计数器，只有当计数器达到阈值时才执行拟合
                    self._frames_since_window += 1
                    if self._frames_since_window >= self.window_hop:
                        window_block_np = np.stack(list(self.window_buffer), axis=0)  # (68, F)
                        total = self.window_size
                        pick_n = self.window_pick
                        pick_idx = np.linspace(0, total - 1, pick_n, dtype=int)
                        picked_block = window_block_np[pick_idx]              # (11, F)
                        picked_smoothed = gaussian_filter1d(picked_block, sigma=6, axis=0)

                        # —— 执行窗口拟合 ——
                        fit_result = self._fit_window_phi0_inherit(picked_smoothed)
                        F = np.array([[1.0, self.dt_window],[0.0, 1.0]], dtype=np.float32)

                        # 先做一次预测（即使这次不更新观测，也能推进状态）
                        self.kf.predict(self.dt_window)

                        # —— 保存“下一时刻先验”（给 RTS）
                        x_pred_next = None if self.kf.x is None else self.kf.x.copy()
                        P_pred_next = None if self.kf.P is None else self.kf.P.copy()

                        # === 本回合是否“有效测量” ===
                        current_time = self.count * hop_sec
                        apply_gate = (smooth_amp >= self.min_amplitude)
                        has_direction = (final_direction in ("Left", "Right"))
                        has_distance  = bool(fit_result) and (fit_result.get('distance_cm') is not None)
                        effective_measure = apply_gate and has_direction and has_distance

                        if effective_measure:
                            # —— 若上一段标记了强制重起，则先清空 ——
                            if getattr(self, '_force_reinit', False):
                                self.kf.reset()
                                if hasattr(self.rts, 'reset'):
                                    self.rts.reset()
                                self._last_distance_kf = None
                                self._last_velocity_kf = None
                                self._last_acceleration = None
                                self._prev_velocity = None
                                self._prev_time = None
                                self._last_meas = None
                                self._last_delta = None
                                x_pred_next = None
                                P_pred_next = None
                                self._force_reinit = False

                            # —— 更新 KF ——
                            self.last_fit_result = fit_result
                            
                            # === 跳变感知的小幅更新 ===
                            z_meas = float(fit_result['distance_cm'])

                            # 计算"方向一致"标志
                            same_direction = True
                            delta = None

                            if self._last_meas is not None:
                                delta = z_meas - self._last_meas

                                if USE_DIRECTION_GATE:
                                    # 可选：结合声学方向（Left/Right）；请按你的物理语义设定符号
                                    # 这里给出一个保守示例：Right => +1, Left => -1；None/其他 => 不做方向限制
                                    s_dir = None
                                    if final_direction == "Right":
                                        s_dir = +1
                                    elif final_direction == "Left":
                                        s_dir = -1
                                    if s_dir is not None:
                                        same_direction = ( (delta > 0 and s_dir > 0) or (delta < 0 and s_dir < 0) )
                                    else:
                                        same_direction = True
                                else:
                                    # 惯性法：与上一帧增量同号则认为方向一致
                                    if self._last_delta is not None:
                                        same_direction = (delta * self._last_delta > 0.0)
                                    else:
                                        same_direction = True

                            # 计算 R 放大（仅当越过阈值且方向一致）
                            R_override = None
                            if (delta is not None) and same_direction:
                                d = abs(delta)
                                if d > JUMP_T_CM:
                                    scale = 1.0 + ((d - JUMP_T_CM) / max(1e-6, JUMP_K_CM)) ** float(JUMP_POWER)
                                    if scale > float(JUMP_R_SCALE_MAX):
                                        scale = float(JUMP_R_SCALE_MAX)
                                    R_override = float(self.kf.R) * float(scale)

                            # 执行更新（如无 R_override 则正常权重）
                            if R_override is None:
                                self.kf.update(z_meas)
                            else:
                                self.kf.update(z_meas, R_override=R_override)

                            # 维护缓存
                            if self._last_meas is not None:
                                self._last_delta = z_meas - self._last_meas
                            self._last_meas = z_meas

                            # —— 推入 RTS（若可用）——
                            if (x_pred_next is not None) and (self.kf.x is not None):
                                self.rts.push(F, x_pred_next, P_pred_next, self.kf.x, self.kf.P)

                            # —— 取平滑/滤波输出 ——
                            x_s = self.rts.get_smoothed()
                            if x_s is not None:
                                raw_distance = float(x_s[0][0])
                                raw_velocity = float(x_s[0][1])
                            else:
                                raw_distance = float(self.kf.distance) if getattr(self.kf, 'distance', None) is not None else None
                                raw_velocity = float(self.kf.velocity) if getattr(self.kf, 'velocity', None) is not None else None

                            # —— 帧级别速度/步长限制（v_max）和加速度限制（a_max）——
                            if (self._last_distance_kf is not None and self._prev_time is not None and 
                                raw_distance is not None):
                                dt_frame = current_time - self._prev_time
                                if dt_frame > 0:
                                    constrained_distance = raw_distance
                                    constrained_velocity = raw_velocity if raw_velocity is not None else (raw_distance - self._last_distance_kf) / dt_frame
                                    
                                    # 速度限制：|v| <= v_max
                                    if self.v_max is not None:
                                        v_max_abs = float(self.v_max)
                                        max_step = v_max_abs * dt_frame
                                        step = raw_distance - self._last_distance_kf
                                        if step > max_step:
                                            constrained_distance = self._last_distance_kf + max_step
                                            constrained_velocity = max_step / dt_frame
                                        elif step < -max_step:
                                            constrained_distance = self._last_distance_kf - max_step
                                            constrained_velocity = -max_step / dt_frame
                                        
                                        # 确保速度在绝对限制内
                                        if constrained_velocity is not None:
                                            constrained_velocity = min(max(constrained_velocity, -v_max_abs), v_max_abs)
                                    
                                    # 加速度限制：|v_k - v_{k-1}| <= a_max * dt
                                    if (self.a_max is not None) and (self._prev_velocity is not None):
                                        a_lim = float(self.a_max) * dt_frame
                                        dv = float(constrained_velocity) - float(self._prev_velocity)
                                        if dv > a_lim:
                                            constrained_velocity = float(self._prev_velocity) + a_lim
                                        elif dv < -a_lim:
                                            constrained_velocity = float(self._prev_velocity) - a_lim
                                        # 根据最终速度回推一致的距离
                                        constrained_distance = self._last_distance_kf + constrained_velocity * dt_frame
                                    
                                    # 距离下限保障
                                    if constrained_distance is not None:
                                        constrained_distance = max(constrained_distance, self.d_min_cm)

                                    self._last_distance_kf = constrained_distance
                                    self._last_velocity_kf = constrained_velocity
                                else:
                                    self._last_distance_kf = None if raw_distance is None else max(raw_distance, self.d_min_cm)
                                    self._last_velocity_kf = raw_velocity
                            else:
                                self._last_distance_kf = None if raw_distance is None else max(raw_distance, self.d_min_cm)
                                self._last_velocity_kf = raw_velocity

                            # —— 显式计算加速度 ——
                            if (self._prev_velocity is not None and self._prev_time is not None and 
                                self._last_velocity_kf is not None):
                                dt_acc = current_time - self._prev_time
                                if dt_acc > 0:
                                    self._last_acceleration = (self._last_velocity_kf - self._prev_velocity) / dt_acc
                                else:
                                    self._last_acceleration = 0.0
                            else:
                                self._last_acceleration = 0.0

                            # —— 时间/历史更新 ——
                            self._prev_velocity = self._last_velocity_kf
                            self._prev_time = current_time
                            self._last_measure_time = current_time

                        else:
                            # === 无有效测量：按“短缺测/长缺测”处理 ===
                            if (self._last_measure_time is None) or ((current_time - self._last_measure_time) <= self._missing_threshold):
                                # 短缺测：不清空状态，本帧不输出任何 frame
                                self._frames_since_window = 0
                                continue
                            else:
                                # 长缺测：判定当前段结束，重置并标记下次强制重起
                                self.kf.reset()
                                if hasattr(self.rts, 'reset'):
                                    self.rts.reset()
                                self._last_distance_kf = None
                                self._last_velocity_kf = None
                                self._last_acceleration = None
                                self._prev_velocity = None
                                self._prev_time = None
                                self._last_meas = None
                                self._last_delta = None
                                self._force_reinit = True
                                self._frames_since_window = 0
                                continue

                        # 重置 hop 计数
                        self._frames_since_window = 0
            except Exception as e:
                print(f"[AudioProcessor] sine-fit processing failed: {e}")
                import traceback
                traceback.print_exc()

            # —— 构造输出 frame ——  
            # 噪声门阻塞时不输出距离相关信息
            if noise_blocked:
                self.last_fit_result = None
            apply_gate = (not noise_blocked) and (smooth_amp >= self.min_amplitude)
            
            frame = {
                'spectrum':             current,      # 短窗特征
                'diff_amplitude':       smooth_d1,    # d1
                'frequencies':          self.selected_freqs,
                'is_sound_present':     is_sound_present,
                'f0':                   self.last_bpf_info["f0"],
                't':                    self.count * hop_sec,
                'sum_abs_d1':           smooth_amp,

                # === 窗口余弦拟合（替代 SWCLS）最新可用结果 ===
                # 只有当 amplitude >= min_amplitude 时才输出距离信息
                'omega':                (self.last_fit_result['omega_win'] if (self.last_fit_result and apply_gate) else None),
                'A':                    (self.last_fit_result['A_win'] if (self.last_fit_result and apply_gate) else None),
                'r2':                   (self.last_fit_result['r2'] if (self.last_fit_result and apply_gate) else None),
                'rmse':                 (self.last_fit_result['rmse'] if (self.last_fit_result and apply_gate) else None),
                'distance':             (self.last_fit_result['distance_cm'] if (self.last_fit_result and apply_gate) else None),
                
                # === KF / RTS 平滑距离（同样应用门控）===
                'distance_kf':          (getattr(self, '_last_distance_kf', None) if apply_gate else None),
                'velocity_kf':          (getattr(self, '_last_velocity_kf', None) if apply_gate else None),
                'acceleration_kf':      (getattr(self, '_last_acceleration', None) if apply_gate else None),

                # 简化方向输出（如需更精确可在此接回你的方向检测函数）
                'direction_d1':         final_direction,
            }

            try:
                self.out_q.put_nowait(frame)
            except:
                pass

            time.sleep(0)  # 让出线程

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
