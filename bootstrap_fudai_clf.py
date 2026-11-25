# -*- coding: utf-8 -*-
"""
冷启动：用「一张种子图」训出可运行的福袋检测模型（bootstrap）

正样本：种子图里按粉色主体定位福袋，裁 120×90，做 7 倍平移/亮度增强
负样本：同一张图里滑窗取窗口，排除与福袋框 IoU>0.08 的位置
阈值：训练集自评 F1 最优（标注为 indicative，仅用于点火）
输出：data/fudai/model/fudai_clf_bootstrap.npz（默认不覆盖生产模型）

用法：
  python bootstrap_fudai_clf.py --seed data/fudai/seed/seed.png
  python bootstrap_fudai_clf.py --seed ... --install   # 覆盖 fudai_clf.npz（先备份 .prev）
"""
import argparse
import os
import random
import shutil

import cv2
import numpy as np

import fudai_feat as F
from train_fudai_clf import iou, pad_crop, pink_box

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "data", "fudai", "model")
OUT_DEFAULT = os.path.join(MODEL_DIR, "fudai_clf_bootstrap.npz")
PROD = os.path.join(MODEL_DIR, "fudai_clf.npz")
ICON_ROI = (0, 250, 420, 540)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=os.path.join(ROOT, "data", "fudai", "seed", "seed.png"))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--install", action="store_true", help="覆盖 fudai_clf.npz（先备份 .prev）")
    args = ap.parse_args()

    img = cv2.imread(args.seed)
    if img is None:
        raise SystemExit(f"种子图读取失败: {args.seed}")
    pb = pink_box(img[ICON_ROI[1]:ICON_ROI[3], ICON_ROI[0]:ICON_ROI[2]])
    if not pb:
        raise SystemExit("种子图里没找到粉色福袋主体，换一张图")
    box = (ICON_ROI[0] + pb[0], ICON_ROI[1] + pb[1],
           ICON_ROI[0] + pb[2], ICON_ROI[1] + pb[3])
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    rng = random.Random(20261125)
    base = pad_crop(img, cx, cy)
    pos = []
    for dx, dy, g in [(0, 0, 1.0), (0, 0, 0.85), (0, 0, 1.15),
                      (-10, 0, 1.0), (10, 0, 1.0), (0, -8, 1.0), (0, 8, 1.0)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        s = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
        if g != 1.0:
            s = np.clip(s.astype(np.float32) * g, 0, 255).astype(np.uint8)
        pos.append(s)

    boxes = []
    for y in range(0, img.shape[0] - F.WIN_H, 30):
        for x in range(0, img.shape[1] - F.WIN_W, 30):
            b = (x, y, x + F.WIN_W, y + F.WIN_H)
            if iou(b, box) > 0.08:
                continue
            boxes.append(b)
    rng.shuffle(boxes)
    neg = [img[b[1]:b[3], b[0]:b[2]].copy() for b in boxes[:120]]
    print(f"种子框 {box}  正样本 {len(pos)}  负样本 {len(neg)}")

    X = np.stack([F.feat(i) for i in pos + neg]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))]).astype(np.float32)
    Xa = np.hstack([X, np.ones((len(X), 1), np.float32)])
    lam = 1.0  # 单图 bootstrap，强正则避免记住这一帧
    w = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float32), Xa.T @ y)

    sc_pos = Xa[:len(pos)] @ w
    sc_neg = Xa[len(pos):] @ w
    best = (-1.0, 0.0)
    for th in np.arange(float(sc_neg.max()) - 0.5, float(sc_pos.min()) + 0.5, 0.01):
        tp = int((sc_pos >= th).sum())
        fp = int((sc_neg >= th).sum())
        fn = int((sc_pos < th).sum())
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        if f1 > best[0]:
            best = (f1, float(th))
    f1, th = best
    print(f"分数: 正样本 min/中位 = {sc_pos.min():.2f}/{np.median(sc_pos):.2f}，"
          f"负样本 max = {sc_neg.max():.2f}")
    print(f"自评 F1={f1:.3f} 阈值={th:.2f}（同一帧自评，仅作点火参考）")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, w=w, thresh=np.float32(th),
             win_w=np.int32(F.WIN_W), win_h=np.int32(F.WIN_H),
             fea_w=np.int32(F.FEA_W), fea_h=np.int32(F.FEA_H))
    report = (f"mode=bootstrap seed={args.seed}\nbox={box} pos={len(pos)} neg={len(neg)}\n"
              f"f1_selffit={f1:.3f} thresh={th:.2f} score_pos_min={sc_pos.min():.2f} "
              f"score_neg_max={sc_neg.max():.2f}\nmodel={args.out}\n")
    open(os.path.join(MODEL_DIR, "bootstrap_report.txt"), "w", encoding="utf-8").write(report)
    print(report)

    if args.install:
        if os.path.exists(PROD):
            shutil.copy2(PROD, PROD + ".prev")
        shutil.copy2(args.out, PROD)
        print(f"已安装为生产模型: {PROD}（原模型备份 {PROD}.prev）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
