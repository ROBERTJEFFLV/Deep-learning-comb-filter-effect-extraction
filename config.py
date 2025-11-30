# src/config.py

# ==== I/O & 频域前端 ====
WS_URI = "ws://localhost:8888"  # WebSocket 音频输入地址 (network/ws_client.py)
SR = 48000  # 采样率 Hz，播放与处理共用 (playback/audio_player.py, processing/audio_processor.py)
N_FFT = 512  # STFT 窗口长度 (processing/audio_processor.py)
HOP_LEN = 128  # 帧步长/回调块大小 (playback/audio_player.py, processing/audio_processor.py)
FREQ_MIN, FREQ_MAX = 1000, 5000  # 处理频带 Hz (processing/audio_processor.py, ui/visualization.py)
HISTORY_FRAMES = int(0.05 * SR / HOP_LEN)  # d1 历史缓存长度 (processing/audio_processor.py)
SMOOTH_SIGMA_1 = 2.2  # 频域短期平滑 sigma (processing/audio_processor.py)
SMOOTH_SIGMA_2 = 30  # 时序平滑 sigma (processing/audio_processor.py)
N_BPF = 4096  # 长窗大小，供包络平滑使用 (processing/audio_processor.py)
D1_AMP_SCALE = 900  # d1 放大系数，用于窗口拟合 (processing/audio_processor.py)

# ==== UI 绘制 ====
WIDTH = 2560  # Pygame 窗口宽度 (ui/visualization.py)
HEIGHT = 1080  # Pygame 窗口高度 (ui/visualization.py)
FPS = 120  # 绘制刷新上限 (ui/visualization.py)
DRAW_STEP = 2  # 频谱抽样步长 (ui/visualization.py)

# ==== Jump-aware measurement downweight ====
JUMP_T_CM = 7.0  # 跳变阈值 cm，超出后放大 R (processing/audio_processor.py)
JUMP_K_CM = 7.0  # 跳变软化尺度 (processing/audio_processor.py)
JUMP_POWER = 1.5  # 跳变惩罚幂次 (processing/audio_processor.py)
JUMP_R_SCALE_MAX = 25.0  # R 放大上限 (processing/audio_processor.py)
USE_DIRECTION_GATE = False  # 是否结合方向 gating (processing/audio_processor.py)

# ==== 噪声门 ====
NOISE_GATE_ENABLED = True  # 绝对强度噪声门开关 (processing/audio_processor.py)
NOISE_GATE_THRESHOLD = 0.01  # RMS 阈值 (processing/audio_processor.py)
NOISE_GATE_SMOOTH = 0.9  # RMS 指数平滑系数 (processing/audio_processor.py)

# ==== LS / KF / RTS / 输出门控（原 kalman_params.json） ====
LS_PARAMS = {
    "window_size": 68,  # 窗口蓄水池长度帧 (processing/audio_processor.py)
    "window_hop": 34,  # 窗口滑动步长帧 (processing/audio_processor.py)
    "window_pick": 11,  # 每次拟合抽取的段数 (processing/audio_processor.py)
    "d_min_m": 0.005,  # ω 网格下限对应的最小距离 m (processing/audio_processor.py)
    "oversample": 8.0,  # ω 网格过采样倍数 (processing/audio_processor.py)
    "gaussian_sigma": 6.0,  # 窗口内 d1 平滑 sigma (processing/audio_processor.py)
    "local_radius_first": 3,  # 首段局部搜索半径 (processing/audio_processor.py)
    "local_radius_segment": 1,  # 后续段局部搜索半径 (processing/audio_processor.py)
}

KF_PARAMS = {
    "R": 4,  # 测量噪声方差 cm^2 (processing/audio_processor.py -> RangeKF)
    "sigma_a": 8,  # 等效加速度标准差 cm/s^2 (processing/audio_processor.py -> RangeKF)
    "v_max": 8,  # 速度上限 cm/s (processing/audio_processor.py -> RangeKF)
    "a_max": 25,  # 加速度上限 cm/s^2 (processing/audio_processor.py -> RangeKF)
}

RTS_PARAMS = {
    "lag": 3,  # RTS 平滑滞后长度 (processing/audio_processor.py -> FixedLagRTS)
}

OUTPUT_FILTER = {
    "min_amplitude": 0.25,  # sum_abs_d1 门控阈值 (processing/audio_processor.py)
}
