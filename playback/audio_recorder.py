# src/playback/audio_recorder.py

import threading
import wave
from queue import Queue, Empty, Full
from typing import Optional
import numpy as np


class AudioRecorder:
    """
    非阻塞录音器：
    - 在音频回调里调用 write(block) 推送 int16 单声道块，
    - 后台线程顺序写入 .wav，stop() 保证文件头正确收尾。
    """
    def __init__(self, filepath: str, samplerate: int, queue_size: int = 512):
        self.filepath = filepath
        self.samplerate = int(samplerate)
        self.q: Queue[np.ndarray] = Queue(maxsize=queue_size)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wf: Optional[wave.Wave_write] = None
        
        # 定期刷盘
        self._flush_counter = 0
        self._flush_interval = 100  # 每100块刷盘一次

    def start(self) -> None:
        """
        打开 wav 文件并启动后台写线程。
        """
        self._wf = wave.open(self.filepath, 'wb')
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)  # int16 = 2 bytes
        self._wf.setframerate(self.samplerate)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[AudioRecorder] Start recording to {self.filepath}")

    def write(self, block: np.ndarray) -> None:
        """
        在音频回调中调用：
        - block: np.ndarray，形状 (N,) 或 (N,1)，dtype 任意
        - 内部会转换成一维 int16，并复制一份入队
        """
        if self._wf is None:
            return

        if not isinstance(block, np.ndarray):
            block = np.asarray(block)

        if block.ndim > 1:
            block = block.reshape(-1)

        if block.dtype != np.int16:
            block = block.astype(np.int16)

        try:
            self.q.put_nowait(block.copy())
        except Full:
            # 队列满时直接丢弃，避免卡住音频回调
            pass

    def stop(self) -> None:
        """
        停止录音：等待写线程退出，关闭文件。
        """
        print("[AudioRecorder] 停止信号已发送，等待队列清空...")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)  # 增加超时时间，确保数据写完
            self._thread = None

        if self._wf is not None:
            try:
                self._wf.close()
                print(f"[AudioRecorder] WAV 文件已保存: {self.filepath}")
            except Exception as e:
                print(f"[AudioRecorder] 关闭 WAV 文件错误: {e}")
            self._wf = None
        print("[AudioRecorder] Stopped.")

    def _run(self) -> None:
        assert self._wf is not None, "recorder not started"
        print(f"[AudioRecorder] 写线程已启动，目标文件: {self.filepath}")
        
        while not self._stop.is_set() or not self.q.empty():
            try:
                block = self.q.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._wf.writeframes(block.tobytes())
                self._flush_counter += 1
                
                # 定期刷盘（确保 Ctrl+C 时数据不丢失）
                if self._flush_counter >= self._flush_interval:
                    try:
                        # wave 模块没有 flush()，直接操作底层文件对象
                        if hasattr(self._wf, '_file') and hasattr(self._wf._file, 'flush'):
                            self._wf._file.flush()
                        self._flush_counter = 0
                    except Exception:
                        pass
            except Exception as e:
                print(f"[AudioRecorder] 写入错误: {e}")
        
        # 循环退出后最后一次刷盘
        try:
            if hasattr(self._wf, '_file') and hasattr(self._wf._file, 'flush'):
                self._wf._file.flush()
            print(f"[AudioRecorder] 队列已清空，共写入 {self._flush_counter} 块")
        except Exception:
            pass
