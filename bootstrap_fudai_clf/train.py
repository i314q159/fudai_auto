# -*- coding: utf-8 -*-
"""单图 bootstrap 的模型拟合：强正则线性判别 + 同帧自评阈值。

只用于「没有模型、样本又不够」的点火场景，阈值是同一帧自评出来的，
只能保证流程能跑起来，样本攒够后应尽快跑 train_fudai_clf 做全量训练。
"""
import numpy as np

import fudai_feat as F


def fit_bootstrap(pos, neg):
    """返回字典：w（权重）、thresh（阈值）、f1 与正负样本分数统计。

    阈值搜索严格不低于运行时下限 ``fudai_feat.MIN_THRESH``；自评不过关就直接
    SystemExit（不写模型、不安装），避免把不可用的点火模型装成生产模型。
    """
    X = np.stack([F.feat(i) for i in pos + neg]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))]).astype(np.float32)
    Xa = np.hstack([X, np.ones((len(X), 1), np.float32)])

    # 单图 bootstrap：强正则（lam=1.0），避免模型记住这一帧的背景
    lam = 1.0
    w = np.linalg.solve(
        Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float32), Xa.T @ y
    )

    sc_pos = Xa[:len(pos)] @ w
    sc_neg = Xa[len(pos):] @ w

    # 在正负分数之间扫阈值，取 F1 最优（同帧自评，仅作点火参考）；
    # 下界抬到 MIN_THRESH，保证存盘的阈值就是运行时真正生效的阈值
    lo = max(float(sc_neg.max()) - 0.5, float(F.MIN_THRESH))
    hi = float(sc_pos.min()) + 0.5
    best = (-1.0, 0.0)
    for th in np.arange(lo, hi, 0.01):
        tp = int((sc_pos >= th).sum())
        fp = int((sc_neg >= th).sum())
        fn = int((sc_pos < th).sum())
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        if f1 > best[0]:
            best = (f1, float(th))
    f1, th = best
    if f1 < 0.5:
        raise SystemExit(
            f"bootstrap 自评失败：F1={f1:.3f}（正样本最低分 {float(sc_pos.min()):.2f}，"
            f"负样本最高分 {float(sc_neg.max()):.2f}，阈值下限 {F.MIN_THRESH}），"
            "换一张福袋更清晰、背景更干净的种子图"
        )

    return {
        "w": w, "thresh": th, "f1": f1,
        "score_pos_min": float(sc_pos.min()),
        "score_pos_median": float(np.median(sc_pos)),
        "score_neg_max": float(sc_neg.max()),
    }
