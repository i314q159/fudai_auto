# -*- coding: utf-8 -*-
"""状态分类模型：多项式 softmax 线性分类器（纯 numpy，全批量梯度下降）。

模型文件字段：version / classes / mean / std / W / b / img_size。
训练与推理都走 features.extract_features，保证特征口径一致。
"""
import os

import cv2
import numpy as np

from .features import extract_features
from .paths import IMG_SIZE, MODEL_PATH, MODEL_VERSION


class SoftmaxClassifier:
    """softmax 线性分类器：标准化 + 类别加权 + L2 正则。"""

    def __init__(self, classes=None):
        self.classes = list(classes) if classes else []
        self.mean = None
        self.std = None
        self.W = None
        self.b = None

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(self, X, y, epochs=400, lr=0.08, l2=1e-3, verbose=False):
        """X: (n, d) 特征；y: 类别名列表；返回 self。"""
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        self.classes = sorted(set(y.tolist()))
        n, d = X.shape
        k = len(self.classes)

        # 每维标准化（均值/方差存进模型，推理时用同一套参数）
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        Z = (X - self.mean) / self.std

        # one-hot 标签
        cls2idx = {c: i for i, c in enumerate(self.classes)}
        Y = np.zeros((n, k), np.float32)
        Y[np.arange(n), [cls2idx[v] for v in y]] = 1.0

        # 类别权重：样本少的类给更高权重，避免全被大类吃掉
        counts = Y.sum(axis=0) + 1e-6
        cw = (n / (k * counts)).astype(np.float32)
        sw = (Y * cw).sum(axis=1, keepdims=True)

        rng = np.random.default_rng(0)
        self.W = rng.normal(0, 0.01, (d, k)).astype(np.float32)
        self.b = np.zeros(k, np.float32)

        for ep in range(epochs):
            logits = Z @ self.W + self.b
            logits -= logits.max(axis=1, keepdims=True)   # 防 exp 溢出
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) * sw / sw.sum()
            gW = Z.T @ G + l2 * self.W
            gb = G.sum(axis=0)
            self.W -= lr * gW
            self.b -= lr * gb
            if verbose and (ep + 1) % 100 == 0:
                loss = float(
                    -(sw * (Y * np.log(P + 1e-9)).sum(axis=1, keepdims=True)).mean()
                )
                print(f"  epoch {ep + 1}/{epochs} loss={loss:.4f}")
        return self

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _prep(self, X):
        X = np.asarray(X, np.float32)
        return (X - self.mean) / self.std

    def predict_proba(self, X):
        Z = self._prep(X)
        logits = Z @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, img_bgr, return_proba=False):
        """直接传 BGR 图片（单张 ndarray 或路径），返回类别名（可选概率字典）。"""
        if isinstance(img_bgr, str):
            img_bgr = cv2.imread(img_bgr)
            if img_bgr is None:
                raise ValueError("图片读取失败")
        P = self.predict_proba(extract_features(img_bgr)[None, :])[0]
        idx = int(np.argmax(P))
        best = self.classes[idx]
        if return_proba:
            return best, {c: float(p) for c, p in zip(self.classes, P)}
        return best

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------
    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            version=np.array([MODEL_VERSION]),
            classes=np.array(self.classes),
            mean=self.mean,
            std=self.std,
            W=self.W,
            b=self.b,
            img_size=np.array([IMG_SIZE]),
        )
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型不存在: {path}（先跑一次训练）")
        z = np.load(path, allow_pickle=False)
        m = cls(z["classes"].tolist())
        m.mean, m.std, m.W, m.b = z["mean"], z["std"], z["W"], z["b"]
        return m


def load_model(path=MODEL_PATH):
    """读取状态模型（供外部脚本直接调用：clf = load_model(); clf.predict(img)）。"""
    return SoftmaxClassifier.load(path)
