# -*- coding: utf-8 -*-
"""检测模型训练：L2 正则线性判别 + 验证集阈值搜索（纯 numpy）。

流程：
  1) 构造正样本（图标 + 增强）与负样本（全屏截图随机窗口）
  2) 提特征（HOG + HSV），按 8:2 划分训练/验证
  3) 对若干正则强度 lam 和阈值 th 网格搜索，取验证集 F1 最高的组合
  4) 写出 data/fudai/model/fudai_clf.npz（w / thresh / win_w / win_h / 特征尺寸）
"""
import os

import numpy as np

import fudai_feat as F

from .samples import positives, negatives
from .paths import MODEL_DIR, MODEL_PATH


def _fit(Xa, y, lam):
    """带 L2 正则的最小二乘闭式解（等价于线性判别）。"""
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xa.T @ y)


def _scores(Xa, w):
    return Xa @ w


def train_model(rng, model_path=MODEL_PATH, dry_run=False):
    """训练并保存模型；返回评估报告文本。"""
    pos, n_files = positives(rng)
    neg = negatives(rng)
    print(f"正样本原图 {n_files} -> 增强 {len(pos)}，负样本 {len(neg)}")
    if len(pos) < 30 or len(neg) < 200:
        raise SystemExit("样本不足")

    # 特征矩阵 + 标签（+1 福袋 / -1 非福袋），并加一列偏置
    X = np.stack([F.feat(i) for i in pos + neg]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))]).astype(np.float32)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    Xa = np.hstack([X, np.ones((len(X), 1), np.float32)])

    # 网格搜索：正则强度 × 判别阈值，以验证集 F1 为准
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        w = _fit(Xa[tr], y[tr], lam)
        sc = _scores(Xa[va], w)
        for th in np.arange(-0.8, 0.81, 0.05):
            tp = int(((sc >= th) & (y[va] > 0)).sum())
            fp = int(((sc >= th) & (y[va] < 0)).sum())
            fn = int(((sc < th) & (y[va] > 0)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if best is None or f1 > best[0]:
                best = (f1, lam, float(th), w)

    f1, lam, th, w = best
    sc = _scores(Xa[va], w)
    tp = int(((sc >= th) & (y[va] > 0)).sum())
    fp = int(((sc >= th) & (y[va] < 0)).sum())
    tn = int(((sc < th) & (y[va] < 0)).sum())
    fn = int(((sc < th) & (y[va] > 0)).sum())
    acc = (tp + tn) / max(1, len(va))

    rep = (
        f"pos_raw={n_files} pos_aug={len(pos)} neg={len(neg)}\n"
        f"lambda={lam} thresh={th:.2f} val tp={tp} fp={fp} tn={tn} fn={fn} "
        f"acc={acc:.3f} f1={f1:.3f}\n"
    )
    if dry_run:
        print(rep + "dry-run：不写模型")
        return rep

    # 落盘：win_w/win_h 固定为设计尺寸，运行时按屏幕分辨率等比放大窗口
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(
        model_path, w=w, thresh=np.float32(th),
        win_w=np.int32(F.WIN_W), win_h=np.int32(F.WIN_H),
        fea_w=np.int32(F.FEA_W), fea_h=np.int32(F.FEA_H),
    )
    rep += f"model={model_path}\n"
    report = os.path.join(os.path.dirname(model_path), "train_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(rep)
    print(rep)
    return rep
