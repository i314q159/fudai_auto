# -*- coding: utf-8 -*-
"""运行时用到的目录 / 文件路径。

注意：ROOT 是「仓库根目录」，不是包目录——本文件在 fudai_auto/ 里，
所以要往上退一级。main.py 用 fa.ROOT 拼 bootstrap 入口和 config.toml。
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)

LOG_DIR = os.path.join(ROOT, "log")            # 运行日志、事件表、截图
SHOT_DIR = os.path.join(LOG_DIR, "shots")
LOG_CSV = os.path.join(LOG_DIR, "fudai_log.csv")

DATASET_DIR = os.path.join(ROOT, "data", "fudai")
DATASET_ICONS = os.path.join(DATASET_DIR, "icons")     # 采集到的福袋样本
DATASET_LABELS = os.path.join(DATASET_DIR, "labels.csv")

# 运行时就创建 log/，保证第一条日志就能落盘
os.makedirs(LOG_DIR, exist_ok=True)
