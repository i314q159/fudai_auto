# -*- coding: utf-8 -*-
"""福袋分类器共用的特征定义（训练与推理必须一致）"""
import cv2
import numpy as np

WIN_W, WIN_H = 120, 90   # 检测窗口（设备像素）
FEA_W, FEA_H = 64, 48    # 特征归一化尺寸
CELL = 8
BINS = 9


def resize_win(win):
    return cv2.resize(win, (FEA_W, FEA_H), interpolation=cv2.INTER_AREA)


def feat(win):
    """HOG（8x8 cell, 9 bin，cell 内 L2 归一）+ HSV 直方图"""
    im = resize_win(win)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    ncy, ncx = FEA_H // CELL, FEA_W // CELL
    hog = np.zeros((ncy, ncx, BINS), np.float32)
    for cy in range(ncy):
        for cx in range(ncx):
            m = mag[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL]
            a = ang[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL]
            hist, _ = np.histogram(a, bins=BINS, range=(0, 180), weights=m)
            hog[cy, cx] = hist / (np.linalg.norm(hist) + 1e-6)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
    hh = hh / (hh.sum() + 1e-6)
    return np.concatenate([hog.flatten(), hh.astype(np.float32)])


def locate(img, roi, w, thresh, stride=8):
    """滑窗打分，返回 (cx, cy, score) 最佳命中"""
    x0r, y0r, x1r, y1r = roi
    sub = img[y0r:y1r, x0r:x1r]
    feats, pos = [], []
    for y in range(0, max(1, sub.shape[0] - w["win_h"] + 1), stride):
        for x in range(0, max(1, sub.shape[1] - w["win_w"] + 1), stride):
            win = sub[y:y + w["win_h"], x:x + w["win_w"]]
            if win.shape[0] != w["win_h"] or win.shape[1] != w["win_w"]:
                continue
            feats.append(feat(win))
            pos.append((x, y))
    if not feats:
        return None
    X = np.stack(feats)
    scores = X @ w["w"][:-1] + w["w"][-1]
    i = int(np.argmax(scores))
    s = float(scores[i])
    if s < thresh:
        return None
    x, y = pos[i]
    cx = x0r + x + w["win_w"] / 2
    cy = y0r + y + w["win_h"] / 2
    return cx, cy, s
