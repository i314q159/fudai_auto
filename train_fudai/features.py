# -*- coding: utf-8 -*-
"""特征与数据增强：手写 HOG + HSV 直方图，以及训练用的平移/缩放/亮度增强。

不依赖 OpenCV 的 HOGDescriptor（OpenCV 5 已移除），只用 numpy + cv2 基础算子。
"""
import cv2
import numpy as np

from .paths import IMG_SIZE


def _l2(x):
    """向量 L2 归一化（加极小值防止除零）。"""
    return x / (np.linalg.norm(x) + 1e-8)


def grad_hist(gray, cell=8, bins=9):
    """手写梯度方向直方图（HOG）：每个 cell 一维 9 bin，逐 cell L2 归一化。"""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # 无符号梯度：角度折到 0~π 再映射到 bin 序号
    bidx = np.minimum(
        (np.arctan2(gy, gx) % np.pi / np.pi * bins).astype(np.int32), bins - 1
    )

    h, w = gray.shape
    ch, cw = h // cell, w // cell
    mag = mag[: ch * cell, : cw * cell].reshape(ch, cell, cw, cell)
    bidx = bidx[: ch * cell, : cw * cell].reshape(ch, cell, cw, cell)

    # 每个 cell 内按 bin 累加梯度幅值
    mag = mag.transpose(0, 2, 1, 3).reshape(ch, cw, cell * cell)
    bidx = bidx.transpose(0, 2, 1, 3).reshape(ch, cw, cell * cell)

    hist = np.zeros((ch, cw, bins), np.float32)
    for b in range(bins):
        hist[..., b] = (mag * (bidx == b)).sum(axis=2)
    hist /= np.linalg.norm(hist, axis=2, keepdims=True) + 1e-6
    return hist.ravel()


def extract_features(img_bgr):
    """BGR 图 -> 特征向量：HOG + HSV 联合直方图 + V 通道直方图（各段分别 L2 归一）。"""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = grad_hist(gray)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    return np.concatenate([_l2(hog), _l2(hs), _l2(v) / 2.0]).astype(np.float32)


def augment(img_bgr, rng, n=8):
    """随机平移/缩放/亮度增强，返回 n 张（第一张是原图）。

    增强参数刻意保守（缩放 0.90~1.10、平移 ±4 像素、亮度 0.85~1.15），
    避免把图标状态里关键的「待参与/已参与/倒计时」文字糊掉。
    """
    out = [img_bgr]
    h, w = img_bgr.shape[:2]
    for _ in range(n - 1):
        s = float(rng.choice([1.0, 0.95, 1.05, 0.90, 1.10]))
        dx = float(rng.uniform(-4, 4))
        dy = float(rng.uniform(-4, 4))
        M = np.array(
            [[s, 0, dx + (1 - s) * w / 2], [0, s, dy + (1 - s) * h / 2]], dtype=np.float32
        )
        warp = cv2.warpAffine(
            img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-15, 15))
        out.append(cv2.convertScaleAbs(warp, alpha=alpha, beta=beta))
    return out
