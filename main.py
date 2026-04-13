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

    # 注册信号处理器，捕获 Ctrl+C
    def signal_handler(sig, frame):
        print("\n[Main] 收到中断信号 (Ctrl+C)，正在保存文件...")
        running_fn(stop=True)
        
        # 立即停止录音（确保 WAV 文件头正确）
        try:
            print("[Main] 正在关闭 WAV 录音...")
            recorder.stop()
        except Exception as e:
            print(f"[Main] 停止录音错误: {e}")
        
        # 给可视化一点时间保存 CSV
        time.sleep(0.5)
        
        print("[Main] 文件保存完成，退出程序")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # 1) 接收 WebSocket 原始音频 → raw_buf
    ws_client = WSClient(raw_buf, running_fn)
    ws_client.start()

    # 2) 播放 + 推送给处理队列
    AudioPlayer(raw_buf, proc_q, recorder, running_fn).start()
    
    # 3) 实例化并启动处理线程
    audio_processor = AudioProcessor(proc_q, frame_q, running_fn)
    audio_processor.start()

    # 4) 可视化：传入 audio_processor.bins
    viz = Visualization(
        frame_q,
        audio_processor.bins,
        running_fn
    )
    try:
        viz.start()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt 捕获（备用）")
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
