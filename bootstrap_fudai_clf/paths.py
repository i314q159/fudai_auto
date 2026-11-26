# -*- coding: utf-8 -*-
"""冷启动脚本的路径与设计基准（与运行时保持一致：1080×2400 设计坐标）。"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)

MODEL_DIR = os.path.join(ROOT, "data", "fudai", "model")
OUT_DEFAULT = os.path.join(MODEL_DIR, "fudai_clf_bootstrap.npz")
PROD = os.path.join(MODEL_DIR, "fudai_clf.npz")   # 生产模型（--install 时覆盖）

DESIGN_W, DESIGN_H = 1080, 2400
ICON_ROI = (0, 250, 420, 540)   # 福袋图标所在区域（设计坐标）
