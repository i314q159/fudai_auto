# -*- coding: utf-8 -*-
"""单窗口特征描述子：HOG（梯度方向直方图）+ HSV 颜色直方图。

不依赖 OpenCV 的 HOGDescriptor（OpenCV 5 已移除），纯 numpy 手写，
训练脚本和运行时检测共用这一份实现，保证特征口径完全一致。
"""
import cv2
import numpy as np

from .params import FEA_W, FEA_H, CELL, BINS


def resize_win(win):
    """把任意尺寸的检测窗口统一缩放到特征尺寸（FEA_W × FEA_H）。"""
    return cv2.resize(win, (FEA_W, FEA_H), interpolation=cv2.INTER_AREA)


def _hog(gray):
    """计算灰度图的 HOG：按 CELL 网格统计梯度方向直方图，单元内 L2 归一。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # atan2 得到 -180~180，取模 180 让相反方向合并（无符号梯度）
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    ncy, ncx = FEA_H // CELL, FEA_W // CELL
    hog = np.zeros((ncy, ncx, BINS), np.float32)
    for cy in range(ncy):
        for cx in range(ncx):
            m = mag[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL]
            a = ang[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL]
            hist, _ = np.histogram(a, bins=BINS, range=(0, 180), weights=m)
            hog[cy, cx] = hist / (np.linalg.norm(hist) + 1e-6)
    return hog.flatten()


def _hsv_hist(bgr):
    """HSV 联合直方图（H 12 档 × S 8 档），归一化成概率分布。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
    return hh / (hh.sum() + 1e-6)


def feat(win):
    """窗口 -> 特征向量（HOG 主特征 + HSV 颜色分布，用于区分福袋/券图标/商品卡）。

    传进来的窗口可以是任意分辨率下的实际像素窗口，这里统一 resize 到
    FEA_W×FEA_H 再提特征，所以同一个模型能跨分辨率使用。
    """
    im = resize_win(win)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.concatenate([_hog(gray), _hsv_hist(im).astype(np.float32)])
