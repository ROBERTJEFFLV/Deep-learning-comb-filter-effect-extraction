from collections import deque
from queue import Queue
from network.ws_client import WSClient
from playback.audio_player import AudioPlayer
from processing.audio_processor import AudioProcessor
from ui.visualization import Visualization
from playback.audio_recorder import AudioRecorder
from config import SR
import signal
import sys
import time


def main():
    raw_buf = deque()
    proc_q  = Queue(maxsize=100)
    frame_q = Queue(maxsize=10)

    recorder = AudioRecorder("recorded_audio.wav", SR)
    recorder.start()

    running_flag = {'run': True}
    def running_fn(stop=False):
        if stop:
            running_flag['run'] = False
        return running_flag['run']

    def signal_handler(sig, frame):
        print("\n[Main] 收到中断信号，正在保存文件...")
        running_fn(stop=True)
        try:
            recorder.stop()
        except Exception as e:
            print(f"[Main] 停止录音错误: {e}")
        time.sleep(0.5)
        print("[Main] 文件保存完成，退出程序")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 1) WebSocket 音频接收
    ws_client = WSClient(raw_buf, running_fn)
    ws_client.start()

    # 2) 播放 + 推送处理队列
    AudioPlayer(raw_buf, proc_q, recorder, running_fn).start()

    # 3) 梳状滤波器特征提取 (v2 倒谱分析)
    audio_processor = AudioProcessor(proc_q, frame_q, running_fn)
    audio_processor.start()

    # 4) 可视化
    viz = Visualization(frame_q, running_fn)
    try:
        viz.start()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt 捕获")
        running_fn(stop=True)
    finally:
        print("[Main] 清理资源...")
        try:
            ws_client.stop()
        except Exception as e:
            print(f"[Main] 停止 WebSocket 错误: {e}")
        try:
            recorder.stop()
        except Exception as e:
            print(f"[Main] 停止录音错误: {e}")
        print("[Main] 退出完成")


if __name__ == "__main__":
    main()
