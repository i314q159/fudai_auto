# -*- coding: utf-8 -*-
"""留一交叉验证（Leave-One-Out）。

样本量少时比固定 train/test 划分更稳：每次留一张做验证，其余全部训练。
每张样本的增强用「文件名种子」保证可复现，不会因为进程不同结果漂移。

特征只提一次（``feature_matrix``）：每一折和最终的全量训练都复用这份矩阵。
旧实现是「每个折都把 n-1 张样本的增强特征重提一遍」，样本多了就是 O(n²) 次提特征。
"""
import os

import numpy as np

from .dataset import stable_seed
from .features import augment, extract_features_batch
from .model import SoftmaxClassifier
from .paths import CLASSES


def feature_matrix(samples, aug_n, device="auto"):
    """所有样本（含增强）的特征矩阵，返回 (X, y, owner)，owner 是样本下标。"""
    X, y, owner = [], [], []
    for i, s in enumerate(samples):
        rng = np.random.default_rng(stable_seed(s["path"]))
        feats = extract_features_batch(augment(s["img"], rng, n=aug_n), device)
        X.append(feats)
        y.extend([s["label"]] * len(feats))
        owner.extend([i] * len(feats))
    return np.vstack(X), np.asarray(y), np.asarray(owner, np.int64)


def leave_one_out(samples, X, y, owner, epochs):
    """返回 (准确率, 混淆矩阵 dict, 错误明细 [(文件名, 真值, 预测, 概率)])。"""
    cm = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    wrong = []
    correct = 0

    for i, s in enumerate(samples):
        keep = owner != i
        clf = SoftmaxClassifier().fit(X[keep], y[keep], epochs=epochs)
        pred, proba = clf.predict(s["img"], return_proba=True)
        cm[s["label"]][pred] = cm[s["label"]].get(pred, 0) + 1
        if pred == s["label"]:
            correct += 1
        else:
            wrong.append((os.path.basename(s["path"]), s["label"], pred, proba))

    return correct / max(1, len(samples)), cm, wrong
