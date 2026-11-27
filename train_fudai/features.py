# -*- coding: utf-8 -*-
"""特征与数据增强：HOG + HSV 直方图，以及训练用的平移/缩放/亮度增强。

HOG 走 ``fudai_feat.hog`` 的向量化实现（与检测模型同一份代码），
批量训练用 ``extract_features_batch`` 一次算多张，逐张推理用 ``extract_features``。
"""
import cv2
import numpy as np

from fudai_feat import accel, hog

from .paths import IMG_SIZE

CELL = 8        # HOG 单元大小（像素，基于 IMG_SIZE 网格）
BINS = 9        # HOG 方向分箱数量（0~180°）


def _l2(x):
    """向量 L2 归一化（加极小值防止除零）。"""
    return x / (np.linalg.norm(x) + 1e-8)


def grad_hist(gray, cell=CELL, bins=BINS):
    """手写梯度方向直方图（HOG）：每个 cell 一维 9 bin，逐 cell L2 归一化。"""
    mag, ang = hog.gradients(gray)
    return hog.cells(mag, ang, cell, bins)


def _prep(img_bgr):
    """统一缩放到 IMG_SIZE×IMG_SIZE，返回 (BGR, 灰度, HSV)。"""
    im = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return im, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


def _color_hist(hsv):
    """HSV 联合直方图（H 12 × S 8）+ V 通道直方图（8 档），各段分别 L2 归一。"""
    hs = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    return np.concatenate([_l2(hs), _l2(v) / 2.0]).astype(np.float32)


def extract_features(img_bgr):
    """BGR 图 -> 特征向量：HOG + HSV 联合直方图 + V 通道直方图（各段分别 L2 归一）。"""
    _, gray, hsv = _prep(img_bgr)
    return np.concatenate([_l2(grad_hist(gray)), _color_hist(hsv)]).astype(np.float32)


def extract_features_batch(imgs, device="auto"):
    """批量提特征（训练用），结果与逐张 ``extract_features`` 一致。

    HOG 交给 ``accel`` 按批算（有 NVIDIA GPU 且装了 torch 时自动用 GPU）。
    """
    dim = (IMG_SIZE // CELL) ** 2 * BINS + 12 * 8 + 8
    if len(imgs) == 0:
        return np.zeros((0, dim), np.float32)
    prepped = [_prep(i) for i in imgs]
    grays = np.stack([g for _, g, _ in prepped])
    hogs = accel.hog_batch(grays, CELL, BINS, device)
    hogs = hogs / (np.linalg.norm(hogs, axis=1, keepdims=True) + 1e-8)
    color = np.stack([_color_hist(hsv) for _, _, hsv in prepped])
    return np.hstack([hogs, color]).astype(np.float32)


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
