# -*- coding: utf-8 -*-
"""训练样本构造：正样本（福袋图标）与负样本（全屏截图里的其它窗口）。

正样本来自运行时采集的 data/fudai/icons/*.png；负样本来自 log/shots 的
全屏截图，排除真实图标框后随机采样，让模型学会区分福袋和券图标/商品大字等。
"""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import fudai_feat as F

from .paths import DATA_ICONS, SHOT_DIR, ICON_ROI, DESIGN_W, DESIGN_H


def pink_box(img, pad=0):
    """返回 (x0, y0, x1, y1) 粉色福袋主体框；找不到返回 None。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (145, 90, 90), (179, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cs:
        a = cv2.contourArea(c)
        if a < 2500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # 福袋主体近似方形：过滤细长的粉色条（按钮/横幅）
        if w < 50 or h < 40 or w / h > 2.6:
            continue
        if best is None or a > best[0]:
            best = (a, x, y, w, h)
    if not best:
        return None
    _, x, y, w, h = best
    return (
        max(0, x - pad), max(0, y - pad),
        min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad),
    )


def pad_crop(img, cx, cy, w=F.WIN_W, h=F.WIN_H):
    """以 (cx, cy) 为中心裁 w×h；越界部分用黑色填充（不改变坐标逻辑）。"""
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    out = np.zeros((h, w, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return out


def iou(a, b):
    """两个 (x0, y0, x1, y1) 框的交并比，用于排除负样本里的真实图标。"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def _augment(base):
    """对一张窗口做 7 倍增强（平移 ±10 / ±8 像素、亮度 0.85 / 1.15 倍）。"""
    out = []
    for dx, dy, g in (
        (0, 0, 1.0), (0, 0, 0.85), (0, 0, 1.15),
        (-10, 0, 1.0), (10, 0, 1.0), (0, -8, 1.0), (0, 8, 1.0),
    ):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        s = cv2.warpAffine(base, m, (base.shape[1], base.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
        if g != 1.0:
            s = np.clip(s.astype(np.float32) * g, 0, 255).astype(np.uint8)
        out.append(s)
    return out


def positives(rng):
    """读取 icons 样本：按粉色主体重新居中裁 120×90，再做增强。

    返回 (增强后的正样本列表, 原始样本文件数)。
    """
    out = []
    files = sorted(glob.glob(os.path.join(DATA_ICONS, "*.png")))
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        b = pink_box(im)
        if b:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        else:
            # 找不到粉色主体就按图片中心裁（样本已统一成 120×90）
            cx, cy = im.shape[1] / 2, im.shape[0] / 2
        out.extend(_augment(pad_crop(im, cx, cy)))
    return out, len(files)


def negatives(rng, per_frame=140, workers=8):
    """逐帧产出负样本窗口（默认每帧 140 个），一帧提完特征就丢，不攒在内存里。

    截图可能来自不同分辨率：图标区与窗口尺寸按截图宽高等比换算，
    再把窗口统一 resize 回 120×90，和正样本口径保持一致。
    """
    # 漏检诊断帧（*_no_icon.png）可能真的含福袋，不能当负样本喂进训练集
    frames = [
        f for f in sorted(glob.glob(os.path.join(SHOT_DIR, "*.png")))
        if "_no_icon" not in os.path.basename(f)
    ]
    for img in _read_frames(frames, workers):
        if img is None or img.shape[0] < 1000:
            continue

        rx = img.shape[1] / DESIGN_W
        ry = img.shape[0] / DESIGN_H
        roi = (
            int(round(ICON_ROI[0] * rx)), int(round(ICON_ROI[1] * ry)),
            int(round(ICON_ROI[2] * rx)), int(round(ICON_ROI[3] * ry)),
        )
        pb = pink_box(img[roi[1]:roi[3], roi[0]:roi[2]], pad=0)
        # 默认排除整块图标区：粉色主体没检出时（例如已参与态图标变色），
        # 也不能把真图标采成负样本
        excl = roi
        if pb:
            excl = (roi[0] + pb[0], roi[1] + pb[1], roi[0] + pb[2], roi[1] + pb[3])

        win_w = max(40, int(round(F.WIN_W * rx)))
        win_h = max(30, int(round(F.WIN_H * ry)))

        boxes = []
        for y in range(0, img.shape[0] - win_h, 30):
            for x in range(0, img.shape[1] - win_w, 30):
                b = (x, y, x + win_w, y + win_h)
                if iou(b, excl) > 0.08:
                    continue
                boxes.append(b)
        rng.shuffle(boxes)
        out = []
        for b in boxes[:per_frame]:
            win = img[b[1]:b[3], b[0]:b[2]].copy()
            if win.size:
                out.append(
                    cv2.resize(win, (F.WIN_W, F.WIN_H), interpolation=cv2.INTER_AREA)
                )
        yield out


def _read_frames(paths, workers=8):
    """并发读全屏截图：PNG 解码在 cv2 里会放 GIL，多线程快数倍（158 帧 4.9s -> 0.9s）。"""
    if not paths:
        return
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths)))) as pool:
        yield from pool.map(cv2.imread, paths)
