# src/ui/visualization_with_record.py

import pygame
from queue import Empty, Queue
import csv
from config import WIDTH, HEIGHT, FPS, DRAW_STEP, FREQ_MIN, FREQ_MAX, HOP_LEN, SR

import numpy as np

def safe_fmt(val, fmt=".1f"):
    return f"{val:{fmt}}" if isinstance(val, (int, float)) else "None"


class Visualization:
    def __init__(self, frame_q: Queue, freq_bins: list, running_flag_fn):
        self.q          = frame_q
        self.bins       = freq_bins
        self.running    = running_flag_fn
        self.sel_freqs  = np.array([fb.freq for fb in freq_bins])

        # —— 颜色 ——  
        self.BLACK  = (0,   0,   0)
        self.WHITE  = (255, 255, 255)
        self.RED    = (255, 0,   0)
        self.BLUE   = (0,   0,   255)
        self.GREEN  = (0,   255, 0)
        self.GRAY   = (200, 200, 200)
        self.YELLOW = (255, 255, 0)

        # —— 频谱绘图区宽度 ——  
        self.spectrum_width = WIDTH // 2 - 100

        # —— 第一组柱状图参数（d1） ——  
        self.CHART1_WIDTH   = 800
        self.BARS1_X        = 200 + self.spectrum_width
        self.BARS1_Y        = 100
        self.BARS1_H        = (HEIGHT - 500) // 2
        self.spacing        = 2
        self.BAR_WIDTH      = max(
            2,
            (self.CHART1_WIDTH - (len(self.bins) + 1) * self.spacing)
            // len(self.bins)
        )
        self.BAR1_MAX_H     = self.BARS1_H // 2 - 50
        self.AMP_SCALE1     = 900

        # —— 第二组柱状图（原先保留位，不再使用 d2） ——  
        self.CHART2_WIDTH   = 800
        self.BARS2_X        = self.BARS1_X
        self.BARS2_Y        = 600
        self.BARS2_H        = (HEIGHT - 500) // 2
        self.BAR2_MAX_H     = self.BARS2_H // 2 - 50
        self.AMP_SCALE2     = 20000

        # —— CSV 记录（改为记录 direction / distance / omega / A / rmse / r2） —— 
        self.csv_file   = open('sine_fit_log.csv', 'w', newline='', buffering=1)  # 行缓冲
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time_sec', 'is_sound_present', 'direction', 'distance_cm',
            'omega_rad_per_Hz', 'A', 'rmse', 'r2'
        ])
        self.csv_file.flush()  # 立即写入表头

        # 定期刷盘计数器
        self._flush_counter = 0
        self._flush_interval = 10  # 每10帧刷盘一次

        # —— 时间戳 —— 
        self.hop_sec    = HOP_LEN / SR

        # —— 运行状态 ——  
        self.last_distance = None    # 如果需要在方向保持时沿用上一距离
        self.frame_count   = 0

        # —— 字体 ——  
        self.label_font = None
        self.score_font = None

    def start(self):
        pygame.init()
        pygame.font.init()

        self.label_font = pygame.font.Font(None, 24)
        self.score_font = pygame.font.Font(None, 48)

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("实时音频处理与抽样绘图")
        self.clock = pygame.time.Clock()
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.running(stop=True)
        finally:
            try:
                self.csv_file.flush()
                self.csv_file.close()
                print("[Visualization] CSV 文件已保存")
            except Exception as e:
                print(f"[Visualization] CSV 关闭错误: {e}")
            pygame.quit()

    def _main_loop(self):
        latest_frame = None

        while self.running():
            # 设置超时，避免无限等待
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running(stop=True)
                    return  # 立即退出
                # 添加 ESC 键退出
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self.running(stop=True)
                        return

            # —— 取队列最新帧 ——  
            try:
                while True:
                    latest_frame = self.q.get_nowait()
            except Empty:
                pass

            # —— 背景与坐标轴 ——  
            self.screen.fill(self.BLACK)
            # 频谱底轴
            pygame.draw.line(
                self.screen, self.GRAY,
                (100, HEIGHT - 100),
                (100 + self.spectrum_width, HEIGHT - 100),
                2
            )
            # 频谱左轴
            pygame.draw.line(
                self.screen, self.GRAY,
                (100, 100),
                (100, HEIGHT - 100),
                2
            )

            if latest_frame:
                # —— 画频谱曲线 ——  
                spec = latest_frame['spectrum'][::DRAW_STEP]
                N = len(spec)
                for i in range(N - 1):
                    x1 = 100 + i * self.spectrum_width / N
                    y1 = int(HEIGHT - 100 - spec[i] * (HEIGHT - 200))
                    x2 = 100 + (i + 1) * self.spectrum_width / N
                    y2 = int(HEIGHT - 100 - spec[i + 1] * (HEIGHT - 200))
                    pygame.draw.line(self.screen, self.WHITE, (x1, y1), (x2, y2), 2)

                # —— 可选：显示若存在的基频标记（若 audio_processor 未提供可忽略） ——  
                f0 = latest_frame.get("f0", None)
                if f0 is not None:
                    n = 1
                    while True:
                        fn = n * f0
                        if fn > FREQ_MAX:
                            break
                        if fn >= FREQ_MIN:
                            frac = (fn - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
                            x = 100 + frac * self.spectrum_width
                            y_top = 95
                            pygame.draw.circle(self.screen, self.YELLOW, (int(x), y_top), 5)
                        n += 1

                # —— 一阶差分柱状图 ——  
                for idx, bin_obj in enumerate(self.bins):
                    bar_x = self.BARS1_X + self.spacing + idx * (self.BAR_WIDTH + self.spacing)
                    diff_val = bin_obj.diff_amp * self.AMP_SCALE1
                    if diff_val >= 0:
                        bar_y   = self.BARS1_Y + self.BARS1_H - diff_val
                        bar_h   = diff_val
                        color   = self.GREEN
                        label_y = bar_y - 20
                    else:
                        bar_y   = self.BARS1_Y + self.BARS1_H
                        bar_h   = -diff_val
                        color   = self.RED
                        label_y = bar_y + bar_h + 5

                    pygame.draw.rect(self.screen, color, (bar_x, bar_y, self.BAR_WIDTH, bar_h))
                    txt = self.label_font.render(f"{bin_obj.diff_amp:.2f}", True, self.WHITE)
                    w, _ = txt.get_size()
                    self.screen.blit(txt, (bar_x + (self.BAR_WIDTH - w) / 2, label_y))

                # —— 数值与文本 ——  
                abs_amp         = latest_frame.get('sum_abs_d1', 0.0)
                t               = latest_frame.get('t', 0.0)
                is_sound_present= latest_frame.get('is_sound_present', None)
                direction_d1    = latest_frame.get('direction_d1', None)
                distance        = latest_frame.get('distance', None)  # 由 audio_processor 的 sine-fit 计算
                omega           = latest_frame.get('omega', None)
                A               = latest_frame.get('A', None)
                rmse            = latest_frame.get('rmse', None)
                r2              = latest_frame.get('r2', None)

                # 顶部两行信息（幅度/基频可根据需要显示）
                amp_txt = self.score_font.render(f"Abs Amplitude: {abs_amp:.3f}", True, self.WHITE)
                self.screen.blit(amp_txt, (self.BARS1_X, self.BARS1_Y + self.BARS1_H + 200))

                # —— 第三行：仅显示 Direction 与 Distance ——  
                # 如需平滑显示，可在方向保持时沿用上一距离
                display_distance = None

                if direction_d1 not in ("Left", "Right"):
                    self.last_distance = None
                    display_distance = None
                else:
                    if distance is not None:
                        display_distance = distance


                line3 = self.score_font.render(
                    f"Dir: {direction_d1}  Dist: {display_distance:.2f} cm" if display_distance is not None else f"Dir: {direction_d1}  Dist: None",
                    True, self.WHITE
                )
                x_text = self.BARS1_X
                y0 = self.BARS1_Y + self.BARS1_H + 250
                self.screen.blit(line3, (x_text, y0))

                # —— CSV 记录（每帧一行，简洁记录当前估计；可按需节流） ——  
                self.frame_count += 1
                self._flush_counter += 1
                # 只有在 display_distance 可用时，才写入这些基于窗口拟合的度量；否则写空字符串（与 distance 的门控保持一致）
                if display_distance is not None:
                    self.csv_writer.writerow([
                        t, is_sound_present, direction_d1, distance,
                        omega if omega is not None else "", A if A is not None else "",
                        rmse if rmse is not None else "", r2 if r2 is not None else ""
                    ])
                else:
                    self.csv_writer.writerow([
                        t, is_sound_present, direction_d1, "",
                        "", "", "", ""
                    ])
                
                # 定期刷盘（每N帧或每秒）
                if self._flush_counter >= self._flush_interval:
                    try:
                        self.csv_file.flush()
                        self._flush_counter = 0
                    except Exception as e:
                        print(f"[Visualization] CSV flush error: {e}")

            # —— 刷新 & 限帧 ——  
            pygame.display.flip()
            self.clock.tick(FPS)

        # 循环退出后立即刷盘并关闭
        try:
            self.csv_file.flush()
            self.csv_file.close()
            print("[Visualization] CSV 文件已保存")
        except Exception as e:
            print(f"[Visualization] CSV 关闭错误: {e}")
        
        pygame.quit()
