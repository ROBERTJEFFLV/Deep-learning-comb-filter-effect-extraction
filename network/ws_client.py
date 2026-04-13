import asyncio
import threading
from collections import deque
import numpy as np
import websockets
from config import WS_URI, SR
import time
import struct

WS_MAGIC = 0xA55A1234
WS_HEADER_FMT = "<IIII"  # magic, frame_id, ts_ms, payload_len
WS_HEADER_SIZE = struct.calcsize(WS_HEADER_FMT)

MAX_AGE_MS = 2000
MAX_BUFFER_SECONDS = 2
CHANNELS = 1
MAX_BUFFER_SAMPLES = int(SR * CHANNELS * MAX_BUFFER_SECONDS)


class WSClient:
    def __init__(self,
                 audio_buffer: deque,
                 running_flag: callable,
                 uri: str = WS_URI):
        """
        audio_buffer: deque to store incoming PCM samples
        running_flag: callable controlling loop continuation; call with stop=True to exit
        uri: WebSocket URI to connect to
        """
        self.buf = audio_buffer
        self.running = running_flag
        self.uri = uri

        self.base_tx_ts = None
        self.base_rx_ts = None
        
        # 统计信息
        self._total_frames_received = 0
        self._total_frames_dropped = 0
        self._total_samples_written = 0

        self._ws = None  # 保存 WebSocket 连接引用

    async def _recv_loop(self):
        """
        建立 WebSocket 连接，接收带 WS 头的音频帧：
        - 帧格式：[WS 头(16B)][payload]
        - WS 头按 WS_HEADER_FMT 解析，校验 magic 和 payload_len
        - 只把 payload 部分按 int16 解码后写入 self.buf
        - 对于 age 过老的帧直接丢弃
        """
        try:
            async with websockets.connect(self.uri, ping_interval=None) as ws:
                self._ws = ws  # 保存引用
                print(f"[WSClient] Connected to {self.uri}")
                while self.running():
                    try:
                        # 添加超时，避免无限阻塞
                        data = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue  # 超时后检查 running 状态
                    except websockets.ConnectionClosed:
                        print("[WSClient] Connection closed by server")
                        break

                    # 只处理二进制帧
                    if not isinstance(data, (bytes, bytearray)):
                        continue

                    buf = bytes(data)
                    total_len = len(buf)
                    
                    if total_len < WS_HEADER_SIZE:
                        # 连头都不够，丢掉
                        print(f"[WSClient] 帧太短 ({total_len} < {WS_HEADER_SIZE})，丢弃")
                        continue

                    # 1) 解析 WS 头：magic, frame_id, ts_ms, payload_len
                    try:
                        magic, frame_id, ts_ms, payload_len = struct.unpack_from(
                            WS_HEADER_FMT, buf, 0
                        )
                    except struct.error as e:
                        # 头格式不对，丢掉
                        print(f"[WSClient] 解析帧头失败: {e}")
                        continue

                    if magic != WS_MAGIC:
                        # 协议不匹配，丢掉
                        print(f"[WSClient] Magic不匹配: 0x{magic:X} != 0x{WS_MAGIC:X}")
                        continue

                    expected_total = WS_HEADER_SIZE + payload_len
                    if total_len < expected_total:
                        # 声称 payload_len=xxx，但实际长度不够，丢掉
                        print(f"[WSClient] 帧不完整: total={total_len} < expected={expected_total}, 丢弃")
                        continue

                    # 2) 取出 payload（纯 PCM 字节，从帧头后开始）
                    payload = buf[WS_HEADER_SIZE:WS_HEADER_SIZE + payload_len]

                    # 3) 计算 age_ms（首帧对齐）
                    now_ms = int(time.time() * 1000) & 0xFFFFFFFF

                    if self.base_tx_ts is None:
                        # 首帧：建立对齐基准
                        self.base_tx_ts = ts_ms
                        self.base_rx_ts = now_ms
                        age_ms = 0
                        print(f"[WSClient] 首帧对齐：frame_id={frame_id}, ts_ms={ts_ms}, now_ms={now_ms}, payload={payload_len}B")
                    else:
                        dt_tx = (ts_ms - self.base_tx_ts) & 0xFFFFFFFF
                        dt_rx = (now_ms - self.base_rx_ts) & 0xFFFFFFFF
                        age_ms = (dt_rx - dt_tx) & 0xFFFFFFFF

                    self._total_frames_received += 1

                    # if age_ms > MAX_AGE_MS:
                    #     # 太老的帧直接丢掉，不进入缓冲区
                    #     self._total_frames_dropped += 1
                    #     if self._total_frames_received % 100 == 0:  # 每100帧打印一次
                    #         print(f"[WSClient] 丢弃过老帧 frame_id={frame_id}, age={age_ms}ms (已丢弃{self._total_frames_dropped}帧)")
                    #     continue

                    # 4) payload → int16 样本（小端）
                    # 注意：发送端是 np.int16.tobytes()，所以这里用 "<i2" 解码
                    try:
                        samples = np.frombuffer(payload, dtype="<i2")
                    except ValueError as e:
                        print(f"[WSClient] payload解析失败: {e}")
                        continue
                    
                    if samples.size == 0:
                        print(f"[WSClient] frame_id={frame_id} payload为空")
                        continue

                    # 5) 写入全局 audio_buffer（这个buffer会被AudioPlayer读取）
                    # 注意：extend 需要可迭代对象，samples是numpy数组，需转list
                    self.buf.extend(samples.tolist())
                    self._total_samples_written += samples.size

                    # 每100帧打印一次统计（避免刷屏）
                    if self._total_frames_received % 100 == 0:
                        audio_duration = self._total_samples_written / SR
                        print(f"[WSClient] 已接收{self._total_frames_received}帧, "
                              f"丢弃{self._total_frames_dropped}帧, "
                              f"音频时长≈{audio_duration:.2f}s, "
                              f"当前buffer={len(self.buf)}samples")

                    # 6) 限制缓冲区长度，只保留最近 MAX_BUFFER_SECONDS 的样本
                    if len(self.buf) > MAX_BUFFER_SAMPLES:
                        extra = len(self.buf) - MAX_BUFFER_SAMPLES
                        for _ in range(extra):
                            self.buf.popleft()

        except Exception as e:
            print(f"[WSClient] Error in recv_loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._ws = None
            print("[WSClient] 接收循环已退出")

    def stop(self):
        """主动关闭 WebSocket 连接"""
        if self._ws is not None:
            try:
                asyncio.create_task(self._ws.close())
            except Exception:
                pass

    def start(self):
        """
        Launch the receive coroutine in a daemon thread.
        """
        thread = threading.Thread(
            target=lambda: asyncio.run(self._recv_loop()),
            daemon=True
        )
        thread.start()
        print(f"[WSClient] 启动接收线程，目标URI: {self.uri}")
