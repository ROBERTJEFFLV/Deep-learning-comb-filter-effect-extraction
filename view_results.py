#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看分析结果的简单脚本
"""

import webbrowser
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# HTML文件路径
html_file = os.path.join(current_dir, "view_results.html")

# 图片文件路径
png_file = os.path.join(current_dir, "window_comparison_analysis.png")

# CSV文件路径
csv_file = os.path.join(current_dir, "window_analysis_summary.csv")

print("时间窗口分析结果文件：")
print(f"1. 可视化图表: {png_file}")
print(f"2. 汇总数据: {csv_file}")
print(f"3. 查看页面: {html_file}")

if os.path.exists(html_file):
    print(f"\n正在打开结果页面...")
    webbrowser.open(f"file://{html_file}")
else:
    print(f"\n未找到HTML文件: {html_file}")

if os.path.exists(png_file):
    print(f"图表文件存在: {png_file}")
else:
    print(f"图表文件不存在: {png_file}")

if os.path.exists(csv_file):
    print(f"CSV文件存在: {csv_file}")
else:
    print(f"CSV文件不存在: {csv_file}")