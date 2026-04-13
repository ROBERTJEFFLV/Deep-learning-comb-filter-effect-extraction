# 实时音频测距系统 (Real-time Audio Ranging System)

基于梳状滤波器效应的实时声学测距系统，通过 WebSocket 接收远程音频信号，实现对反射面距离的估计与方向检测。

## 🎯 项目简介

本系统利用**梳状滤波器效应（Comb Filter Effect）**：当声源发出的直达声与反射声在麦克风处叠加时，会在频谱上形成周期性的增强和抵消，其频率间隔与声程差（即距离）直接相关。

**核心原理：**
```
频率间隔 Δf = c / (2d)
其中：c = 343 m/s（声速），d = 反射面距离
```

系统通过拟合频谱上的余弦波纹，估计角频率 ω，进而计算距离：
```
d = c × ω / (4π) × 100  [单位：cm]
```

## ✨ 功能特性

- ✅ **WebSocket 实时音频接收**：支持带时间戳帧头的二进制协议，自动丢弃过时帧
- ✅ **实时音频回放**：低延迟播放接收到的音频信号
- ✅ **WAV 录制**：自动将接收的音频保存为 WAV 文件
- ✅ **频域特征提取**：STFT → 频带选择 → 一阶差分（d1）特征
- ✅ **方向检测**：基于 d1 特征的相位移动，判断反射面在 Left/Right 方向
- ✅ **余弦拟合测距**：φ=0 约束下的 ω 估计 + 窗口共识机制
- ✅ **Kalman 滤波平滑**：一维匀速模型 + RTS 固定时滞平滑器
- ✅ **物理约束**：速度上限、加速度上限、距离下限
- ✅ **Pygame 实时可视化**：频谱曲线、d1 柱状图、方向/距离显示
- ✅ **CSV 日志**：记录时间、方向、距离、拟合指标

## 🏗️ 系统架构

```
┌─────────────────┐
│  ESP32 + Mic    │ ──(串口 PCM)──→ ┌─────────────────┐
└─────────────────┘                 │   RK3588 发送端  │
                                    │  (WebSocket TX)  │
                                    └────────┬────────┘
                                             │ ws://host:8888/audio
                                             ▼
┌────────────────────────────────────────────────────────────────┐
│                         本机接收端                              │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                                              │
│  │  WSClient    │ ──(PCM int16)──→ audio_buffer (deque)        │
│  │  帧头解析     │                        │                     │
│  │  延时过滤     │                        ▼                     │
│  └──────────────┘              ┌──────────────────┐            │
│                                │   AudioPlayer    │            │
│                                │  - 实时播放       │            │
│                                │  - 录制 WAV      │            │
│                                │  - 送处理队列    │            │
│                                └────────┬─────────┘            │
│                                         ▼                      │
│                                ┌──────────────────┐            │
│                                │ AudioProcessor   │            │
│                                │  - STFT 频域分析  │            │
│                                │  - d1 差分特征    │            │
│                                │  - 方向投票检测   │            │
│                                │  - 窗口余弦拟合   │            │
│                                │  - Kalman + RTS  │            │
│                                └────────┬─────────┘            │
│                                         ▼                      │
│                                ┌──────────────────┐            │
│                                │  Visualization   │            │
│                                │  - Pygame 绘制   │            │
│                                │  - CSV 日志      │            │
│                                └──────────────────┘            │
└────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
nov-21/
├── src/
│   ├── main.py                    # 主入口：启动所有模块
│   ├── config.py                  # 全局配置参数
│   │
│   ├── network/
│   │   └── ws_client.py           # WebSocket 客户端（帧头解析、延时过滤）
│   │
│   ├── playback/
│   │   ├── audio_player.py        # 音频播放器（回调模式）
│   │   └── audio_recorder.py      # WAV 录制器（非阻塞队列写入）
│   │
│   ├── processing/
│   │   ├── audio_processor.py     # 核心处理（特征提取、方向检测、测距）
│   │   ├── sine_fit_22.py         # 余弦拟合算法（φ=0, A≥0）
│   │   ├── range_kf.py            # Kalman 滤波器 + RTS 平滑
│   │   └── comb_shift.py          # 梳状滤波器方向检测
│   │
│   ├── data/
│   │   └── frequency_bin.py       # 频率点特征类（幅值、差分）
│   │
│   └── ui/
│       └── visualization.py       # Pygame 可视化 + CSV 记录
│
├── README.md                      # 本文件
├── .gitignore                     # Git 忽略规则
└── requirements.txt               # Python 依赖（可选）
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 支持音频输出的系统（Linux/macOS/Windows）

### 2. 安装依赖

```bash
pip install numpy scipy sounddevice websockets pygame soundfile
```

### 3. 配置参数

编辑 `src/config.py`：

```python
# WebSocket 服务器地址（改为你的发送端 IP）
WS_URI = "ws://192.168.50.171:8888/audio"

# 音频参数
SR = 48000          # 采样率（与发送端一致）
FREQ_MIN = 1000     # 处理频带下限 Hz
FREQ_MAX = 5000     # 处理频带上限 Hz

# Kalman 滤波器参数
KF_PARAMS = {
    "R": 4,         # 测量噪声方差 cm²
    "sigma_a": 8,   # 过程噪声（加速度标准差）cm/s²
    "v_max": 8,     # 速度上限 cm/s
    "a_max": 25,    # 加速度上限 cm/s²
}
```

### 4. 启动发送端

在 RK3588（或其他设备）上运行发送脚本（见下文）。

### 5. 运行接收端

```bash
cd /home/lvmingyang/nov-21
python src/main.py
```

### 6. 操作说明

- **ESC 键** 或 **Ctrl+C**：安全退出并保存文件
- 界面显示：
  - 左侧：频谱曲线
  - 右侧上方：d1 差分柱状图（绿=正，红=负）
  - 右侧下方：方向（Left/Right/None）和距离（cm）

## 📊 输出文件

| 文件 | 说明 |
|------|------|
| `recorded_audio.wav` | 录制的原始音频（int16 单声道） |
| `sine_fit_log.csv` | 测距日志：时间、方向、距离、ω、A、RMSE、R² |

### CSV 格式示例

```csv
time_sec,is_sound_present,direction,distance_cm,omega_rad_per_Hz,A,rmse,r2
12.34,True,Left,45.67,0.00234,12.5,0.8,0.95
12.84,True,Left,46.12,0.00238,11.8,0.7,0.96
```

## 🔧 核心算法

### 1. 方向检测（comb_shift.py）

```python
# 互相关估计频谱位移方向
lag, rho = estimate_comb_filter_shift(frame_old, frame_new)
# lag > 0 → 频谱右移 → 反射面在 Left
# lag < 0 → 频谱左移 → 反射面在 Right
```

### 2. 余弦拟合（sine_fit_22.py）

```python
# 模型：amplitude(f) = A · cos(ω · f)
# 约束：φ = 0，A ≥ 0
# 搜索：粗网格遍历 ω → 闭式解 A_hat → 加权 SSE 选最优
```

### 3. Kalman 滤波（range_kf.py）

```python
# 状态：x = [距离, 速度]ᵀ
# 观测：z = 距离 + 噪声
# 特性：匀速模型 + 物理约束（v_max, a_max, d_min）
# 平滑：固定时滞 RTS 平滑器（lag=3）
```

## 📡 发送端协议

### 帧格式（16 字节头 + PCM payload）

```
┌─────────┬──────────┬─────────┬─────────────┬───────────────┐
│ magic   │ frame_id │ ts_ms   │ payload_len │ PCM payload   │
│ 4 bytes │ 4 bytes  │ 4 bytes │ 4 bytes     │ N bytes       │
└─────────┴──────────┴─────────┴─────────────┴───────────────┘

magic = 0xA55A1234（小端）
frame_id = 帧序号（递增）
ts_ms = 发送时刻的毫秒时间戳（低 32 位）
payload_len = PCM 数据长度（字节）
PCM payload = int16 小端，单声道
```

### 发送端示例（Python）

```python
import struct
import numpy as np

WS_MAGIC = 0xA55A1234
WS_HEADER_FMT = "<IIII"

def send_audio_frame(ws, frame_id, audio_samples):
    """audio_samples: np.ndarray, dtype=int16"""
    payload = audio_samples.tobytes()
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFF
    header = struct.pack(WS_HEADER_FMT, 
                         WS_MAGIC, frame_id, ts_ms, len(payload))
    ws.send(header + payload)
```

## ⚙️ 高级配置

### Kalman 滤波器调参

| 参数 | 含义 | 典型值 | 影响 |
|------|------|--------|------|
| `R` | 测量噪声方差 | 4 cm² | ↑ 更平滑但响应慢 |
| `sigma_a` | 过程噪声 | 8 cm/s² | ↑ 响应快但更抖 |
| `v_max` | 速度上限 | 8 cm/s | 限制最大移动速度 |
| `a_max` | 加速度上限 | 25 cm/s² | 限制急停急起 |

### 噪声门

```python
NOISE_GATE_ENABLED = True    # 开启绝对强度门控
NOISE_GATE_THRESHOLD = 0.01  # RMS 阈值
NOISE_GATE_SMOOTH = 0.9      # 平滑系数（0.9 = 慢响应）
```

## 🐛 故障排除

### 1. 无声音输出

- 检查 `[AudioPlayer] underrun` 计数是否很高
- 确认 WebSocket 地址正确且发送端在运行
- 检查系统音频输出设备

### 2. 延时过大

- 查看 `[WSClient] 丢弃过老帧` 消息
- 调整 `MAX_AGE_MS`（默认 500ms）
- 检查网络延迟

### 3. 距离不稳定

- 增大 `KF_PARAMS["R"]`（更平滑）
- 减小 `OUTPUT_FILTER["min_amplitude"]`
- 确保环境有明确的反射面

### 4. Ctrl+C 无法退出

- 先按 ESC 键
- 或多按几次 Ctrl+C
- 程序会等待 WAV 文件保存完成

## 📜 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Author:** Mingyang Lv  
**Last Updated:** 2024
