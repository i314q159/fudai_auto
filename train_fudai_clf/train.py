# -*- coding: utf-8 -*-
"""检测模型训练：L2 正则线性判别 + 验证集阈值搜索（纯 numpy）。

流程：
  1) 构造正样本（图标 + 增强）与负样本（全屏截图随机窗口），逐帧批量提特征
  2) 按 7:1.5:1.5 划分 训练 / 验证 / holdout
  3) 在验证集上网格搜索正则强度 lam 与阈值 th（th 不低于 fudai_feat.MIN_THRESH，
     与运行时同一个下限），取 F1 最高的组合
  4) 用 holdout 复算 tp/fp/tn/fn 作为报告指标，再写出
     data/fudai/model/fudai_clf.npz（w / thresh / win_w / win_h / 特征尺寸）

特征计算走 ``fudai_feat.accel``：有 NVIDIA GPU 且装了 torch 时自动用 GPU，
否则用 numpy 向量化实现；``device`` 可强制 cpu / cuda。
"""
import os
import time

import numpy as np

import fudai_feat as F
from fudai_feat import accel

from .samples import positives, negatives
from .paths import MODEL_DIR, MODEL_PATH

# 每批提特征多少窗口：太小浪费 Python 开销，太大白占内存
CHUNK = 1024


def _fit(Xa, y, lam):
    """带 L2 正则的最小二乘闭式解（等价于线性判别）。"""
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xa.T @ y)


def _metrics(Xa, y, w, th):
    """按阈值统计 (tp, fp, tn, fn, acc, f1)。"""
    sc = Xa @ w
    pred = sc >= th
    tp = int((pred & (y > 0)).sum())
    fp = int((pred & (y < 0)).sum())
    tn = int(((~pred) & (y < 0)).sum())
    fn = int(((~pred) & (y > 0)).sum())
    acc = (tp + tn) / max(1, len(y))
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return tp, fp, tn, fn, acc, f1


def _rows(wins, device):
    """分批提特征，避免一次把几万个窗口堆进内存。"""
    for s in range(0, len(wins), CHUNK):
        yield F.feat_batch(wins[s:s + CHUNK], device)


def train_model(rng, model_path=MODEL_PATH, dry_run=False, device="auto"):
    """训练并保存模型；返回评估报告文本。"""
    backend = accel.describe(device)
    t0 = time.perf_counter()

    pos, n_files = positives(rng)
    print(f"正样本原图 {n_files} -> 增强 {len(pos)}")
    if len(pos) < 30:
        raise SystemExit("样本不足（正样本 < 30）")

    parts = list(_rows(pos, device))
    n_pos = len(pos)
    del pos

    # 负样本按帧流式采样 + 提特征，单帧窗口提完就丢（全量攒着要几百 MB）
    n_neg, frames = 0, 0
    for wins in negatives(rng):
        frames += 1
        if len(wins) == 0:
            continue
        parts.extend(_rows(wins, device))
        n_neg += len(wins)
    print(f"负样本 {n_neg}（来自 {frames} 帧截图）")
    if n_neg < 200:
        raise SystemExit("样本不足（负样本 < 200）")

    X = np.vstack(parts).astype(np.float32)
    del parts
    print(f"特征后端: {backend}")
    print(f"提特征 {len(X)} 个窗口，耗时 {time.perf_counter() - t0:.1f}s")

    # 特征矩阵 + 标签（+1 福袋 / -1 非福袋），并加一列偏置
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)]).astype(np.float32)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut_va = int(len(idx) * 0.7)
    cut_te = int(len(idx) * 0.85)
    tr, va, te = idx[:cut_va], idx[cut_va:cut_te], idx[cut_te:]
    Xa = np.hstack([X, np.ones((len(X), 1), np.float32)])

    # 网格搜索：正则强度 × 判别阈值，只在验证集上挑；阈值下界与运行时一致
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        w = _fit(Xa[tr], y[tr], lam)
        for th in np.arange(F.MIN_THRESH, 1.0001, 0.05):
            f1 = _metrics(Xa[va], y[va], w, float(th))[5]
            if best is None or f1 > best[0]:
                best = (f1, lam, float(th), w)

    f1_sel, lam, th, w = best
    tp, fp, tn, fn, acc, f1 = _metrics(Xa[va], y[va], w, th)
    htp, hfp, htn, hfn, hacc, hf1 = _metrics(Xa[te], y[te], w, th)

    rep = (
        f"pos_raw={n_files} pos_aug={n_pos} neg={n_neg}\n"
        f"split train={len(tr)} val={len(va)} holdout={len(te)}\n"
        f"lambda={lam} thresh={th:.2f}（下限 {F.MIN_THRESH}，验证集 F1={f1_sel:.3f}）\n"
        f"val     tp={tp} fp={fp} tn={tn} fn={fn} acc={acc:.3f} f1={f1:.3f}（选阈值用）\n"
        f"holdout tp={htp} fp={hfp} tn={htn} fn={hfn} acc={hacc:.3f} f1={hf1:.3f}（报告指标）\n"
        f"feat_backend={backend}\n"
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
