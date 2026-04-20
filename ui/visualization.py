# ui/visualization.py
# -*- coding: utf-8 -*-
"""
实时可视化 — 基于梳状滤波器 v2 特征的 Pygame 显示

显示内容:
  左半区: 频谱曲线 (800-8000 Hz)
  右半区: SMD 时间序列 + 检测状态 + 距离估计
  底部: 数值面板 (SMD / CPR / CPN / distance / velocity)

CSV 记录列:
  time_sec, smd, cpr, cpn, nda, obstacle, distance_raw_cm, distance_kf_cm
"""

import pygame
from queue import Empty, Queue
import csv
import numpy as np

from config import (
    WIDTH, HEIGHT, FPS, DRAW_STEP,
    COMB_V2_FREQ_MIN, COMB_V2_FREQ_MAX,
    COMB_V2_HOP, COMB_V2_SMD_THRESHOLD, SR,
)


def safe_fmt(val, fmt=".2f"):
    return f"{val:{fmt}}" if isinstance(val, (int, float)) else "—"


class Visualization:
    # SMD 历史记录长度（绘制时间序列用）
    SMD_HISTORY_LEN = 300

    def __init__(self, frame_q: Queue, running_flag_fn):
        self.q = frame_q
        self.running = running_flag_fn

        # 颜色
        self.BLACK  = (0,   0,   0)
        self.WHITE  = (255, 255, 255)
        self.RED    = (255, 50,  50)
        self.GREEN  = (50,  255, 50)
        self.BLUE   = (80,  140, 255)
        self.GRAY   = (120, 120, 120)
        self.YELLOW = (255, 220, 50)
        self.CYAN   = (0,   220, 220)
        self.DARK_GREEN = (0, 100, 0)

        # 布局
        self.SPEC_X = 80
        self.SPEC_Y = 60
        self.SPEC_W = WIDTH // 2 - 120
        self.SPEC_H = HEIGHT // 2 - 80

        self.SMD_X = WIDTH // 2 + 40
        self.SMD_Y = 60
        self.SMD_W = WIDTH // 2 - 120
        self.SMD_H = HEIGHT // 2 - 80

        self.INFO_Y = HEIGHT // 2 + 40

        # SMD 时间序列
        self.smd_history = []

        # CSV
        self.csv_file = open('comb_v2_log.csv', 'w', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time_sec', 'smd', 'cpr', 'cpn', 'nda',
            'obstacle', 'distance_raw_cm', 'distance_kf_cm',
        ])
        self.csv_file.flush()

        self._flush_counter = 0
        self._flush_interval = 10

        self.hop_sec = COMB_V2_HOP / SR
        self.frame_count = 0

        # 字体（pygame.init() 后设置）
        self.label_font = None
        self.score_font = None
        self.title_font = None

    def start(self):
        pygame.init()
        pygame.font.init()

        self.label_font = pygame.font.Font(None, 24)
        self.score_font = pygame.font.Font(None, 44)
        self.title_font = pygame.font.Font(None, 32)

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Comb Filter v2 — 实时梳状滤波器特征检测")
        self.clock = pygame.time.Clock()
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.running(stop=True)
        finally:
            try:
                self.csv_file.flush()
                self.csv_file.close()
                print("[Visualization] CSV 已保存 → comb_v2_log.csv")
            except Exception as e:
                print(f"[Visualization] CSV 关闭错误: {e}")
            pygame.quit()

    # ------------------------------------------------------------------ #
    #                          主循环                                     #
    # ------------------------------------------------------------------ #
    def _main_loop(self):
        latest = None

        while self.running():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running(stop=True)
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    self.running(stop=True)
                    return

            # 取最新帧
            try:
                while True:
                    latest = self.q.get_nowait()
            except Empty:
                pass

            self.screen.fill(self.BLACK)

            if latest:
                self._draw_spectrum(latest)
                self._draw_smd_timeline(latest)
                self._draw_info_panel(latest)
                self._write_csv(latest)

            pygame.display.flip()
            self.clock.tick(FPS)

    # ------------------------------------------------------------------ #
    #                     频谱显示 (左半区)                                #
    # ------------------------------------------------------------------ #
    def _draw_spectrum(self, frame):
        spec = frame.get('spectrum')
        if spec is None or len(spec) == 0:
            return

        # 坐标轴
        pygame.draw.line(self.screen, self.GRAY,
                         (self.SPEC_X, self.SPEC_Y + self.SPEC_H),
                         (self.SPEC_X + self.SPEC_W, self.SPEC_Y + self.SPEC_H), 1)
        pygame.draw.line(self.screen, self.GRAY,
                         (self.SPEC_X, self.SPEC_Y),
                         (self.SPEC_X, self.SPEC_Y + self.SPEC_H), 1)

        # 标题
        title = self.title_font.render(
            f"Spectrum ({int(COMB_V2_FREQ_MIN)}-{int(COMB_V2_FREQ_MAX)} Hz)",
            True, self.WHITE)
        self.screen.blit(title, (self.SPEC_X, self.SPEC_Y - 30))

        # 归一化绘制
        spec_draw = spec[::DRAW_STEP]
        N = len(spec_draw)
        if N < 2:
            return
        max_val = np.max(spec_draw) if np.max(spec_draw) > 0 else 1.0
        normed = spec_draw / max_val

        points = []
        for i in range(N):
            x = self.SPEC_X + i * self.SPEC_W / N
            y = self.SPEC_Y + self.SPEC_H - normed[i] * self.SPEC_H * 0.9
            points.append((int(x), int(y)))

        if len(points) > 1:
            pygame.draw.lines(self.screen, self.CYAN, False, points, 2)

    # ------------------------------------------------------------------ #
    #                   SMD 时间序列 (右半区)                               #
    # ------------------------------------------------------------------ #
    def _draw_smd_timeline(self, frame):
        smd = frame.get('smd', 0.0)
        detected = frame.get('obstacle_detected', False)

        self.smd_history.append((smd, detected))
        if len(self.smd_history) > self.SMD_HISTORY_LEN:
            self.smd_history = self.smd_history[-self.SMD_HISTORY_LEN:]

        # 坐标轴
        pygame.draw.line(self.screen, self.GRAY,
                         (self.SMD_X, self.SMD_Y + self.SMD_H),
                         (self.SMD_X + self.SMD_W, self.SMD_Y + self.SMD_H), 1)
        pygame.draw.line(self.screen, self.GRAY,
                         (self.SMD_X, self.SMD_Y),
                         (self.SMD_X, self.SMD_Y + self.SMD_H), 1)

        # 标题
        status_color = self.GREEN if detected else self.RED
        status_text = "DETECTED" if detected else "—"
        title = self.title_font.render("SMD Timeline", True, self.WHITE)
        self.screen.blit(title, (self.SMD_X, self.SMD_Y - 30))
        st = self.title_font.render(status_text, True, status_color)
        self.screen.blit(st, (self.SMD_X + 200, self.SMD_Y - 30))

        # 阈值线
        thr = COMB_V2_SMD_THRESHOLD
        smd_max = max(1.5, thr * 1.8)
        thr_y = self.SMD_Y + self.SMD_H - (thr / smd_max) * self.SMD_H
        pygame.draw.line(self.screen, self.YELLOW,
                         (self.SMD_X, int(thr_y)),
                         (self.SMD_X + self.SMD_W, int(thr_y)), 1)
        thr_label = self.label_font.render(f"thr={thr:.3f}", True, self.YELLOW)
        self.screen.blit(thr_label, (self.SMD_X + self.SMD_W + 5, int(thr_y) - 8))

        # 数据点
        n = len(self.smd_history)
        if n < 2:
            return
        for i in range(1, n):
            x0 = self.SMD_X + (i - 1) * self.SMD_W / self.SMD_HISTORY_LEN
            x1 = self.SMD_X + i * self.SMD_W / self.SMD_HISTORY_LEN
            s0, d0 = self.smd_history[i - 1]
            s1, d1 = self.smd_history[i]
            y0 = self.SMD_Y + self.SMD_H - (s0 / smd_max) * self.SMD_H
            y1 = self.SMD_Y + self.SMD_H - (s1 / smd_max) * self.SMD_H
            color = self.GREEN if d1 else self.GRAY
            pygame.draw.line(self.screen, color,
                             (int(x0), int(y0)), (int(x1), int(y1)), 2)

    # ------------------------------------------------------------------ #
    #                      数值面板 (下半区)                                #
    # ------------------------------------------------------------------ #
    def _draw_info_panel(self, frame):
        smd = frame.get('smd', 0.0)
        cpr = frame.get('cpr', 0.0)
        cpn = frame.get('cpn', 0.0)
        dist_raw = frame.get('distance_raw')
        dist_kf = frame.get('distance_kf')
        vel_kf = frame.get('velocity_kf')
        detected = frame.get('obstacle_detected', False)
        t = frame.get('t', 0.0)

        y = self.INFO_Y
        x_left = self.SPEC_X
        x_right = WIDTH // 2 + 40
        line_h = 50

        # 检测状态大字
        if detected:
            det_text = "● OBSTACLE DETECTED"
            det_color = self.GREEN
        else:
            det_text = "○ No obstacle"
            det_color = self.RED

        det_surf = self.score_font.render(det_text, True, det_color)
        self.screen.blit(det_surf, (x_left, y))
        y += line_h + 10

        # 特征数值
        features = [
            ("SMD", smd, self.GREEN if smd > COMB_V2_SMD_THRESHOLD else self.GRAY),
            ("CPR", cpr, self.WHITE),
            ("CPN", cpn, self.WHITE),
        ]
        for name, val, color in features:
            txt = self.score_font.render(f"{name}: {safe_fmt(val, '.4f')}", True, color)
            self.screen.blit(txt, (x_left, y))
            y += line_h

        # 距离信息
        y = self.INFO_Y + line_h + 10
        dist_lines = [
            (f"Distance (raw):  {safe_fmt(dist_raw)} cm", self.CYAN),
            (f"Distance (KF):   {safe_fmt(dist_kf)} cm", self.BLUE),
            (f"Velocity (KF):   {safe_fmt(vel_kf)} cm/s", self.WHITE),
            (f"Time: {safe_fmt(t, '.2f')} s", self.GRAY),
        ]
        for text, color in dist_lines:
            surf = self.score_font.render(text, True, color)
            self.screen.blit(surf, (x_right, y))
            y += line_h

    # ------------------------------------------------------------------ #
    #                            CSV 记录                                  #
    # ------------------------------------------------------------------ #
    def _write_csv(self, frame):
        self.frame_count += 1
        self._flush_counter += 1

        t = frame.get('t', 0.0)
        smd = frame.get('smd', 0.0)
        cpr = frame.get('cpr', 0.0)
        cpn = frame.get('cpn', 0.0)
        nda = frame.get('nda', 0.0)
        detected = 1 if frame.get('obstacle_detected', False) else 0
        dist_raw = frame.get('distance_raw', '')
        dist_kf = frame.get('distance_kf', '')

        self.csv_writer.writerow([
            f"{t:.4f}", f"{smd:.6f}", f"{cpr:.4f}", f"{cpn:.4f}",
            f"{nda:.4f}", detected,
            f"{dist_raw:.2f}" if isinstance(dist_raw, (int, float)) else "",
            f"{dist_kf:.2f}" if isinstance(dist_kf, (int, float)) else "",
        ])

        if self._flush_counter >= self._flush_interval:
            try:
                self.csv_file.flush()
                self._flush_counter = 0
            except Exception as e:
                print(f"[Visualization] CSV flush error: {e}")
