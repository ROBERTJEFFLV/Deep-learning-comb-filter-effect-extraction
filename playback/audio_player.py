# src/playback/audio_player.py

import threading
import time
import numpy as np
import sounddevice as sd
from collections import deque
from queue import Queue, Full
from typing import Optional
from config import SR, PLAYBACK_BLOCKSIZE
from playback.audio_recorder import AudioRecorder


class AudioPlayer:
    def __init__(
        self,
        audio_buffer: deque,
        process_queue: Queue,
        recorder: Optional[AudioRecorder],
        running_flag: callable,
    ):
        """
        audio_buffer: deque[int]，由 WSClient 填入的 int16 PCM 样本
        process_queue: Queue，用于送给 AudioProcessor 的 float32 块
        recorder: 可选的 AudioRecorder，用于写 .wav
        running_flag: callable() -> bool，外部控制整个系统是否继续运行
        """
        self.buf = audio_buffer
        self.p_queue = process_queue
        self.recorder = recorder
        self.running = running_flag
        
        # 统计信息
        self._callback_count = 0
        self._underrun_count = 0
        self._last_print_time = time.time()

    def _callback(self, outdata, frames, time_info, status):
        """
        sounddevice 的回调：
        - 从 self.buf 取出 frames 个 int16 样本（不够则补 0）
        - 写到 outdata[:, 0]（dtype=int16）
        - 同时把归一化后的 float32 块塞进 process_queue
        - 把原始 int16 块交给 AudioRecorder
        """
        if status:
            print(f"[AudioPlayer] Callback status: {status}")

        self._callback_count += 1

        # 1) 从 deque 取样本
        block = np.zeros(frames, dtype=np.int16)
        samples_read = 0
        for i in range(frames):
            if self.buf:
                # self.buf 里是 Python int（int16），逐个弹出
                block[i] = self.buf.popleft()
                samples_read += 1
            else:
                # 不足的部分补 0（欠载）
                break

        # 统计欠载
        if samples_read < frames:
            self._underrun_count += 1

        # 每秒打印一次统计
        now = time.time()
        if now - self._last_print_time >= 1.0:
            print(f"[AudioPlayer] callback={self._callback_count}, "
                  f"underrun={self._underrun_count}, "
                  f"buffer={len(self.buf)}samples, "
                  f"read={samples_read}/{frames}")
            self._last_print_time = now

        # 2) 播放：OutputStream 使用 dtype='int16'
        outdata[:, 0] = block

        # 3) 送给处理线程：转换为 float32，[-1, 1]
        float_block = block.astype(np.float32) / 32768.0
        try:
            self.p_queue.put_nowait(float_block)
        except Full:
            # 队列满了就丢弃本块，避免阻塞回调
            pass

        # 4) 送给录音线程
        if self.recorder is not None:
            self.recorder.write(block)

    def start(self):
        """
        启动回放线程：内部再开一个 sounddevice.OutputStream
        """
        def _run_play():
            while self.running():
                try:
                    with sd.OutputStream(
                        samplerate=SR,
                        channels=1,
                        dtype='int16',
                        blocksize=PLAYBACK_BLOCKSIZE,
                        callback=self._callback,
                        latency='low',
                    ):
                        print(f"[AudioPlayer] Callback playback started at {SR} Hz, blocksize={PLAYBACK_BLOCKSIZE}")
                        # 只要 running，就让回调自己触发
                        while self.running():
                            time.sleep(0.1)
                except Exception as e:
                    print(f"[AudioPlayer] Playback error: {e}")
                    import traceback
                    traceback.print_exc()
                    # 防止疯狂重试，稍微歇一下
                    time.sleep(1.0)

        threading.Thread(target=_run_play, daemon=True).start()
