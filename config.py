# config.py
# -*- coding: utf-8 -*-

# ==== I/O 基础设施 ====
WS_URI = "ws://localhost:8888"       # WebSocket 音频输入地址 (network/ws_client.py)
SR = 48000                           # 采样率 Hz (全局共用)
PLAYBACK_BLOCKSIZE = 128             # sounddevice 回调块大小 (playback/audio_player.py)

# ==== Comb Filter v2 — 倒谱分析管线 (processing/comb_feature_v2.py) ====
COMB_V2_N_FFT = 2048                 # STFT 窗口 (23.4 Hz 分辨率)
COMB_V2_HOP = 512                    # 帧步长 (~10.7 ms, ~93 fps)
COMB_V2_FREQ_MIN = 800.0             # 分析频带下限 Hz
COMB_V2_FREQ_MAX = 8000.0            # 分析频带上限 Hz
COMB_V2_TAU_MIN = 0.00025            # 搜索延迟下限 s (d≈4.3 cm)
COMB_V2_TAU_MAX = 0.004              # 搜索延迟上限 s (d≈68.6 cm)
COMB_V2_EMA_ALPHA = 0.02             # 运行均值 EMA 系数
COMB_V2_CEP_AVG = 4                  # 倒谱滑动平均帧数
COMB_V2_SMD_THRESHOLD = 0.871        # SMD 检测阈值 (TPR≈92%, FPR≈0%)

# ==== Kalman / RTS 平滑 (processing/range_kf.py) ====
KF_PARAMS = {
    "R": 4,                           # 测量噪声方差 cm²
    "sigma_a": 8,                     # 等效加速度 σ cm/s²
    "v_max": 8,                       # 速度上限 cm/s
    "a_max": 25,                      # 加速度上限 cm/s²
}
RTS_PARAMS = {
    "lag": 3,                         # RTS 固定滞后长度
}

# ==== UI / Pygame ====
WIDTH = 2560                          # 窗口宽度 px
HEIGHT = 1080                         # 窗口高度 px
FPS = 120                             # 刷新上限
DRAW_STEP = 2                         # 频谱抽样步长
