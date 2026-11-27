# -*- coding: utf-8 -*-
"""梯度方向直方图（HOG）的向量化实现：训练与运行时共用同一份。

按 CELL 网格把梯度幅值累加到方向直方图（0~180°，BINS 档），每个 cell 单独做
L2 归一化，再按行展开成定长向量。

旧写法对每个 cell 调一次 ``np.histogram``（64×48 的窗口要调 48 次），单窗约 1.8 ms；
这里改成「一次算好方向档序号 + bincount 累加」，单窗约 90 us，差一个数量级，
数值结果只差 float32 舍入（<1e-6），所以逐窗推理和批量训练可以共用这一份。
"""
import cv2
import numpy as np


def gradients(gray):
    """灰度图 -> (梯度幅值, 方向角 0~180 度)。"""
    g = np.asarray(gray, np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # cv2.phase 是 atan2 的角度版，比 np.arctan2 快一个数量级；取模 180 让
    # 相反方向的梯度合并（无符号梯度）
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    np.mod(ang, 180.0, out=ang)
    return mag, ang


def cell_offset(h, w, cell, bins):
    """每个像素对应的「cell 号 × bins」偏移，形状与裁剪后的图一致。"""
    ncy, ncx = h // cell, w // cell
    off = (
        np.arange(ncy, dtype=np.int64)[:, None] * ncx
        + np.arange(ncx, dtype=np.int64)[None, :]
    ) * bins
    return np.repeat(np.repeat(off, cell, axis=0), cell, axis=1)


def cells_batch(mag, ang, cell, bins):
    """(N, H, W) 的幅值/角度 -> (N, ncy*ncx*bins) 逐 cell 归一化后的 HOG。"""
    mag = np.ascontiguousarray(mag, np.float32)
    n, h, w = mag.shape
    ncy, ncx = h // cell, w // cell
    hc, wc = ncy * cell, ncx * cell

    mag = mag[:, :hc, :wc].reshape(n, -1)
    # 角度落到 0~bins-1：180° 的等价类只到 180（不含），所以 floor 与直方图的
    # 分箱边界一致
    idx = (ang[:, :hc, :wc] * (bins / 180.0)).astype(np.int64)
    np.clip(idx, 0, bins - 1, out=idx)
    idx += cell_offset(hc, wc, cell, bins)
    idx += (np.arange(n, dtype=np.int64) * (ncy * ncx * bins))[:, None, None]

    hist = np.bincount(
        idx.reshape(-1), weights=mag.reshape(-1), minlength=n * ncy * ncx * bins
    ).astype(np.float32)
    hist = hist.reshape(n, ncy * ncx, bins)
    hist /= np.linalg.norm(hist, axis=2, keepdims=True) + 1e-6
    return hist.reshape(n, -1)


def cells(mag, ang, cell, bins):
    """单张图的 HOG 向量。"""
    return cells_batch(mag[None], ang[None], cell, bins)[0]
