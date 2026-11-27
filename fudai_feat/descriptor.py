# -*- coding: utf-8 -*-
"""单窗口特征描述子：HOG（梯度方向直方图）+ HSV 颜色直方图。

不依赖 OpenCV 的 HOGDescriptor（OpenCV 5 已移除），HOG 用 ``fudai_feat.hog``
的向量化实现（训练批量、运行时逐窗共用同一份，口径完全一致）。
"""
import cv2
import numpy as np

from . import accel, hog
from .params import FEA_W, FEA_H, CELL, BINS


def resize_win(win):
    """把任意尺寸的检测窗口统一缩放到特征尺寸（FEA_W × FEA_H）。"""
    return cv2.resize(win, (FEA_W, FEA_H), interpolation=cv2.INTER_AREA)


def _hog(gray):
    """计算灰度图的 HOG：按 CELL 网格统计梯度方向直方图，单元内 L2 归一。"""
    mag, ang = hog.gradients(gray)
    return hog.cells(mag, ang, CELL, BINS)


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


def feat_batch(wins, device="auto"):
    """批量提特征（训练用），结果与逐窗 ``feat`` 一致。

    HOG 交给 ``accel`` 按批算（有 NVIDIA GPU 且装了 torch 时自动用 GPU），
    颜色直方图仍在 CPU 上用 cv2 算。
    """
    if len(wins) == 0:
        dim = (FEA_H // CELL) * (FEA_W // CELL) * BINS + 12 * 8
        return np.zeros((0, dim), np.float32)
    ims = np.stack([resize_win(w) for w in wins])
    grays = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in ims])
    hogs = accel.hog_batch(grays, CELL, BINS, device)
    color = np.stack([_hsv_hist(im).astype(np.float32) for im in ims])
    return np.hstack([hogs, color]).astype(np.float32)
