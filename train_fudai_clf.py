# -*- coding: utf-8 -*-
"""
训练福袋图标分类器 -> data/fudai/model/fudai_clf.npz

正样本：data/fudai/icons/*.png 里按粉色主体再裁出的图标窗口（含倒计时标签）
负样本：log/shots/*.png 全屏截图，排除真实图标所在框后随机取的窗口（含券图标、商品大字等）
特征：HOG + HSV 直方图；模型：L2 正则线性判别（纯 numpy）
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

import fudai_feat as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ICONS = os.path.join(ROOT, "data", "fudai", "icons")
MODEL_DIR = os.path.join(ROOT, "data", "fudai", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fudai_clf.npz")
REPORT_PATH = os.path.join(MODEL_DIR, "train_report.txt")
SHOT_DIR = os.path.join(ROOT, "log", "shots")
ICON_ROI = (0, 250, 420, 540)


def pink_box(img, pad=0):
    """返回 (x0,y0,x1,y1) 粉色福袋主体框；找不到返回 None"""
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
        if w < 50 or h < 40 or w / h > 2.6:
            continue
        if best is None or a > best[0]:
            best = (a, x, y, w, h)
    if not best:
        return None
    _, x, y, w, h = best
    return (max(0, x - pad), max(0, y - pad),
            min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad))


def pad_crop(img, cx, cy, w=F.WIN_W, h=F.WIN_H):
    """以 (cx,cy) 为中心裁 w*h，越界用边缘填充"""
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    out = np.zeros((h, w, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def positives(rng):
    out = []
    files = sorted(glob.glob(os.path.join(DATA_ICONS, "*.png")))
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        b = pink_box(im)
        if b:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            base = pad_crop(im, cx, cy)
        else:
            c = im.shape
            base = pad_crop(im, c[1] / 2, c[0] / 2)
        for dx, dy, g in [(0, 0, 1.0), (0, 0, 0.85), (0, 0, 1.15),
                          (-10, 0, 1.0), (10, 0, 1.0), (0, -8, 1.0), (0, 8, 1.0)]:
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            s = cv2.warpAffine(base, m, (base.shape[1], base.shape[0]),
                               borderMode=cv2.BORDER_REPLICATE)
            if g != 1.0:
                s = np.clip(s.astype(np.float32) * g, 0, 255).astype(np.uint8)
            out.append(s)
    return out, len(files)


def negatives(rng, per_frame=140):
    out = []
    frames = sorted(glob.glob(os.path.join(SHOT_DIR, "*.png")))
    for f in frames:
        img = cv2.imread(f)
        if img is None or img.shape[0] < 1000:
            continue
        pb = pink_box(img[ICON_ROI[1]:ICON_ROI[3], ICON_ROI[0]:ICON_ROI[2]], pad=0)
        excl = None
        if pb:
            excl = (ICON_ROI[0] + pb[0], ICON_ROI[1] + pb[1],
                    ICON_ROI[0] + pb[2], ICON_ROI[1] + pb[3])
        boxes = []
        for y in range(0, img.shape[0] - F.WIN_H, 30):
            for x in range(0, img.shape[1] - F.WIN_W, 30):
                b = (x, y, x + F.WIN_W, y + F.WIN_H)
                if excl and iou(b, excl) > 0.08:
                    continue
                if not excl and y < 540:
                    continue
                boxes.append(b)
        rng.shuffle(boxes)
        for b in boxes[:per_frame]:
            out.append(img[b[1]:b[3], b[0]:b[2]].copy())
    return out


def main():
    ap = argparse.ArgumentParser(description="训练福袋检测模型")
    ap.add_argument("--model", default=MODEL_PATH, help="模型输出路径")
    ap.add_argument("--dry-run", action="store_true", help="只评估不保存")
    args = ap.parse_args()

    rng = random.Random(20261125)
    pos, n_files = positives(rng)
    neg = negatives(rng)
    print(f"正样本原图 {n_files} -> 增强 {len(pos)}，负样本 {len(neg)}")
    if len(pos) < 30 or len(neg) < 200:
        raise SystemExit("样本不足")

    X = np.stack([F.feat(i) for i in pos + neg]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))]).astype(np.float32)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    Xa = np.hstack([X, np.ones((len(X), 1), np.float32)])

    def fit(lam):
        A = Xa[tr].T @ Xa[tr] + lam * np.eye(Xa.shape[1], dtype=np.float32)
        return np.linalg.solve(A, Xa[tr].T @ y[tr])

    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        w = fit(lam)
        sc = Xa[va] @ w
        for th in np.arange(-0.8, 0.81, 0.05):
            tp = int(((sc >= th) & (y[va] > 0)).sum())
            fp = int(((sc >= th) & (y[va] < 0)).sum())
            fn = int(((sc < th) & (y[va] > 0)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if best is None or f1 > best[0]:
                best = (f1, lam, float(th), w)

    f1, lam, th, w = best
    sc = Xa[va] @ w
    tp = int(((sc >= th) & (y[va] > 0)).sum())
    fp = int(((sc >= th) & (y[va] < 0)).sum())
    tn = int(((sc < th) & (y[va] < 0)).sum())
    fn = int(((sc < th) & (y[va] > 0)).sum())
    acc = (tp + tn) / max(1, len(va))

    os.makedirs(MODEL_DIR, exist_ok=True)
    rep = (f"pos_raw={n_files} pos_aug={len(pos)} neg={len(neg)}\n"
           f"lambda={lam} thresh={th:.2f} val tp={tp} fp={fp} tn={tn} fn={fn} "
           f"acc={acc:.3f} f1={f1:.3f}\n")
    if args.dry_run:
        print(rep + "dry-run：不写模型")
        return 0
    np.savez(args.model, w=w, thresh=np.float32(th),
             win_w=np.int32(F.WIN_W), win_h=np.int32(F.WIN_H),
             fea_w=np.int32(F.FEA_W), fea_h=np.int32(F.FEA_H))
    rep += f"model={args.model}\n"
    report = os.path.join(os.path.dirname(args.model), "train_report.txt")
    open(report, "w", encoding="utf-8").write(rep)
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
