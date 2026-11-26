# -*- coding: utf-8 -*-
"""留一交叉验证（Leave-One-Out）。

样本量少时比固定 train/test 划分更稳：每次留一张做验证，其余全部训练。
每张样本的增强用「文件名种子」保证可复现，不会因为进程不同结果漂移。
"""
import os

import numpy as np

from .dataset import stable_seed
from .features import augment, extract_features
from .model import SoftmaxClassifier
from .paths import CLASSES


def leave_one_out(samples, aug_n, epochs):
    """返回 (准确率, 混淆矩阵 dict, 错误明细 [(文件名, 真值, 预测, 概率)])。"""
    n = len(samples)
    cm = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    wrong = []
    correct = 0

    for i in range(n):
        train = [s for j, s in enumerate(samples) if j != i]
        X, y = [], []
        for s in train:
            for v in augment(s["img"], np.random.default_rng(stable_seed(s["path"])), n=aug_n):
                X.append(extract_features(v))
                y.append(s["label"])

        clf = SoftmaxClassifier().fit(X, y, epochs=epochs)
        pred, proba = clf.predict(samples[i]["img"], return_proba=True)
        cm[samples[i]["label"]][pred] = cm[samples[i]["label"]].get(pred, 0) + 1
        if pred == samples[i]["label"]:
            correct += 1
        else:
            wrong.append((os.path.basename(samples[i]["path"]), samples[i]["label"], pred, proba))

    return correct / max(1, n), cm, wrong
