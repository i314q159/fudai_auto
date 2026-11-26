# -*- coding: utf-8 -*-
"""训练脚本用到的固定路径与设计基准。

独立成文件，避免训练工具反向依赖运行时包（fudai_auto）而把
adb / OCR 这些重依赖一起加载进来。
"""
import os

# 仓库根目录：本文件在 train_fudai_clf/ 下一层，往上一级就是根
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)

# 训练输入 / 输出
DATA_ICONS = os.path.join(ROOT, "data", "fudai", "icons")   # 运行时采集的福袋样本
SHOT_DIR = os.path.join(ROOT, "log", "shots")               # 全屏截图（取负样本）
MODEL_DIR = os.path.join(ROOT, "data", "fudai", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fudai_clf.npz")

# 设计分辨率与福袋图标所在区域（截图按 1080×2400 设计，运行时等比缩放）
DESIGN_W, DESIGN_H = 1080, 2400
ICON_ROI = (0, 250, 420, 540)
