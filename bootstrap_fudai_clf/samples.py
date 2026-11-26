# -*- coding: utf-8 -*-
"""用一张种子图现场造正/负样本。

正样本：福袋主体居中裁窗口 -> 7 倍平移/亮度增强
负样本：同一张图滑窗采样，排除与福袋框 IoU>0.08 的位置
两张来源都按截图分辨率换算窗口尺寸，最后统一成 120×90，保证与运行时特征同口径。
"""
import cv2
import numpy as np

import fudai_feat as F
from train_fudai_clf import iou, pad_crop

from .paths import DESIGN_W, DESIGN_H


def build_positives(img, box, rx, ry):
    """以福袋框中心裁窗口并做增强；返回增强后的正样本列表。"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = int(round(F.WIN_W * rx)), int(round(F.WIN_H * ry))
    base = pad_crop(img, cx, cy, w, h)
    # 非设计分辨率的种子图：裁完先缩回设计窗口尺寸，再走统一增强
    if base.shape[1] != F.WIN_W or base.shape[0] != F.WIN_H:
        base = cv2.resize(base, (F.WIN_W, F.WIN_H), interpolation=cv2.INTER_AREA)

    pos = []
    for dx, dy, g in (
        (0, 0, 1.0), (0, 0, 0.85), (0, 0, 1.15),
        (-10, 0, 1.0), (10, 0, 1.0), (0, -8, 1.0), (0, 8, 1.0),
    ):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        s = cv2.warpAffine(base, m, (base.shape[1], base.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
        if g != 1.0:
            s = np.clip(s.astype(np.float32) * g, 0, 255).astype(np.uint8)
        pos.append(s)
    return pos


def build_negatives(img, box, rng, limit=120):
    """同图滑窗取负样本：跳过与福袋框重叠的位置，窗口统一 resize 到设计尺寸。"""
    rx, ry = img.shape[1] / DESIGN_W, img.shape[0] / DESIGN_H
    w, h = int(round(F.WIN_W * rx)), int(round(F.WIN_H * ry))
    boxes = []
    for y in range(0, img.shape[0] - h, 30):
        for x in range(0, img.shape[1] - w, 30):
            b = (x, y, x + w, y + h)
            if iou(b, box) > 0.08:
                continue
            boxes.append(b)
    rng.shuffle(boxes)

    neg = []
    for b in boxes[:limit]:
        win = img[b[1]:b[3], b[0]:b[2]].copy()
        if win.size:
            neg.append(cv2.resize(win, (F.WIN_W, F.WIN_H), interpolation=cv2.INTER_AREA))
    return neg
