# -*- coding: utf-8 -*-
"""
福袋图标样本训练脚本（data/fudai -> 分类模型）

数据来源（两个地址）+ 可选追加：
  1. data/fudai/icons/*.png + data/fudai/labels.csv   （fudai_auto.py 采集的样本）
  2. data/fudai/bag_template.png                      （福袋图标模板图，默认只做统计/参考）
  --extra / --extra-label 可再追加任意图片或目录，并指定其类别

标签归一化（labels.csv 里的原始文字很杂，这里统一成 3 类 + 未知）：
  待参与        -> pending     （图标下方显示"待参与"，需要去点）
  已参与        -> joined      （图标下方显示"已参与"，本轮不用再点）
  倒计时 03:17  -> countdown   （图标下方是 mm:ss 倒计时）
  其他/icon_only-> unknown     （不参与训练，只统计）

做法（无新增第三方依赖，只用 numpy + opencv）：
  特征   梯度方向直方图(灰度 8x8 单元, 9 bin) + HSV 颜色直方图，分块 L2 归一化
         （OpenCV 5 已移除 HOGDescriptor，这里用 numpy 手写，无额外依赖）
  增强   随机平移 / 缩放 / 亮度，缓解样本量少
  模型   多项式 softmax 线性分类器，全批量梯度下降 + 类别权重 + L2
  评估   Leave-One-Out（真·留一交叉验证，样本少时比 train/test 划分更稳）
  输出   data/fudai/model/fudai_state.npz（含类别、标准化参数、权重）
         检测模型 fudai_clf.npz 由 train_fudai_clf.py 负责，两者互不覆盖

用法：
  # 训练（默认读 data/fudai，结果存 data/fudai/model/）
  .venv\\Scripts\\python.exe train_fudai.py

  # 只看数据统计和 LOO 结果，不落盘
  .venv\\Scripts\\python.exe train_fudai.py --dry-run

  # 用训好的模型预测单张图 / 整个目录
  .venv\\Scripts\\python.exe train_fudai.py --predict data/fudai/icons/20261125_140117_已参与.png
  .venv\\Scripts\\python.exe train_fudai.py --predict data/fudai/icons

  # 其他代码里直接调
  import train_fudai as tf
  clf = tf.load_model();  clf.predict(img_bgr)
"""

import argparse
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, "data", "fudai")
ICON_DIR = os.path.join(DATASET_DIR, "icons")
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")
MODEL_DIR = os.path.join(DATASET_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fudai_state.npz")   # 状态分类模型（与检测模型分开）
DETECTOR_PATH = os.path.join(MODEL_DIR, "fudai_clf.npz")   # fudai_auto.py 用的检测模型
BAG_TEMPLATE = os.path.join(DATASET_DIR, "bag_template.png")   # 福袋图标模板
EXTRA_SOURCES = [BAG_TEMPLATE]                                 # 默认追加的数据地址

IMG_SIZE = 64
CLASSES = ["pending", "joined", "countdown"]
LABEL_CN = {"pending": "待参与", "joined": "已参与", "countdown": "倒计时", "unknown": "未知"}
TIME_RE = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}")
MODEL_VERSION = 1


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def norm_label(raw):
    """把 labels.csv 的原始标签映射成统一类别"""
    t = (raw or "").strip()
    if "待参与" in t:
        return "pending"
    if "已参与" in t:
        return "joined"
    if TIME_RE.search(t):
        return "countdown"
    return "unknown"


def _expand(paths):
    """把文件/目录展开成图片列表"""
    out = []
    for p in paths:
        if not p:
            continue
        if os.path.isdir(p):
            out += sorted(
                glob.glob(os.path.join(p, "*.png")) + glob.glob(os.path.join(p, "*.jpg"))
            )
        elif os.path.exists(p):
            out.append(p)
        else:
            print(f"[warn] 追加数据地址不存在，已跳过: {p}")
    return out


def load_samples(labels_csv=LABELS_CSV, extra=(), extra_label=None):
    """返回 [{path, raw, label, room, img, src}]，文件缺失或读不出图的直接跳过

    labels.csv 里的样本按标注归一化；extra（含默认 bag_template.png）按文件名
    推断标签，可用 extra_label 强制指定类别。
    """
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"找不到标注文件: {labels_csv}")
    samples, missing, seen = [], [], set()
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = (row.get("file") or "").strip()
            if not fn:                      # labels.csv 里有空行
                continue
            p = fn if os.path.isabs(fn) else os.path.join(ROOT, fn)
            if not os.path.exists(p):
                missing.append(fn)
                continue
            img = cv2.imread(p)
            if img is None:
                missing.append(fn)
                continue
            samples.append(
                {
                    "path": p,
                    "raw": (row.get("label") or "").strip(),
                    "label": norm_label(row.get("label")),
                    "room": (row.get("room") or "").strip(),
                    "img": img,
                    "src": "labels",
                }
            )
            seen.add(os.path.normcase(os.path.abspath(p)))
    for m in missing:
        print(f"[warn] 标注里记录的文件不存在，已跳过: {m}")

    for p in _expand(extra):
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        img = cv2.imread(p)
        if img is None:
            print(f"[warn] 追加图片读取失败，已跳过: {p}")
            continue
        samples.append(
            {
                "path": p,
                "raw": os.path.basename(p),
                "label": extra_label or norm_label(os.path.splitext(os.path.basename(p))[0]),
                "room": "",
                "img": img,
                "src": "extra",
            }
        )
        seen.add(key)
    return samples


# --------------------------------------------------------------------------
# 特征
# --------------------------------------------------------------------------
def _l2(x):
    return x / (np.linalg.norm(x) + 1e-8)


def grad_hist(gray, cell=8, bins=9):
    """手写梯度方向直方图（HOG）：每个 cell 一维 9 bin，逐 cell L2 归一化"""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    bidx = np.minimum((np.arctan2(gy, gx) % np.pi / np.pi * bins).astype(np.int32), bins - 1)

    h, w = gray.shape
    ch, cw = h // cell, w // cell
    mag = mag[: ch * cell, : cw * cell].reshape(ch, cell, cw, cell)
    bidx = bidx[: ch * cell, : cw * cell].reshape(ch, cell, cw, cell)
    mag = mag.transpose(0, 2, 1, 3).reshape(ch, cw, cell * cell)
    bidx = bidx.transpose(0, 2, 1, 3).reshape(ch, cw, cell * cell)

    hist = np.zeros((ch, cw, bins), np.float32)
    for b in range(bins):
        hist[..., b] = (mag * (bidx == b)).sum(axis=2)
    hist /= np.linalg.norm(hist, axis=2, keepdims=True) + 1e-6
    return hist.ravel()


def extract_features(img_bgr):
    """梯度方向直方图 + HSV 颜色直方图，输出 float32 向量"""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = grad_hist(gray)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    return np.concatenate([_l2(hog), _l2(hs), _l2(v) / 2.0]).astype(np.float32)


def augment(img_bgr, rng, n=8):
    """随机平移/缩放/亮度增强，返回 n 张（含原图）"""
    out = [img_bgr]
    h, w = img_bgr.shape[:2]
    for _ in range(n - 1):
        s = float(rng.choice([1.0, 0.95, 1.05, 0.90, 1.10]))
        dx = float(rng.uniform(-4, 4))
        dy = float(rng.uniform(-4, 4))
        M = np.array(
            [[s, 0, dx + (1 - s) * w / 2], [0, s, dy + (1 - s) * h / 2]], dtype=np.float32
        )
        warp = cv2.warpAffine(
            img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-15, 15))
        out.append(cv2.convertScaleAbs(warp, alpha=alpha, beta=beta))
    return out


# --------------------------------------------------------------------------
# 模型：softmax 线性分类器
# --------------------------------------------------------------------------
class SoftmaxClassifier:
    def __init__(self, classes=None):
        self.classes = list(classes) if classes else []
        self.mean = None
        self.std = None
        self.W = None
        self.b = None

    # ---- 训练 ----
    def fit(self, X, y, epochs=400, lr=0.08, l2=1e-3, verbose=False):
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        self.classes = sorted(set(y.tolist()))
        n, d = X.shape
        k = len(self.classes)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        Z = (X - self.mean) / self.std

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
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) * sw / sw.sum()
            gW = Z.T @ G + l2 * self.W
            gb = G.sum(axis=0)
            self.W -= lr * gW
            self.b -= lr * gb
            if verbose and (ep + 1) % 100 == 0:
                loss = float(-(sw * (Y * np.log(P + 1e-9)).sum(axis=1, keepdims=True)).mean())
                print(f"  epoch {ep + 1}/{epochs} loss={loss:.4f}")
        return self

    # ---- 推理 ----
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
        """直接传 BGR 图片（单张 ndarray 或路径），返回类别名"""
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

    # ---- 存取 ----
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
    return SoftmaxClassifier.load(path)


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def leave_one_out(samples, aug_n, epochs):
    """留一交叉验证；返回 (准确率, 混淆矩阵 dict, 错误明细)"""
    n = len(samples)
    cm = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    wrong = []
    correct = 0
    for i in range(n):
        train = [s for j, s in enumerate(samples) if j != i]
        X, y = [], []
        for s in train:
            for v in augment(s["img"], np.random.default_rng(hash(s["path"]) % 2**32), n=aug_n):
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


def print_dataset_stats(samples):
    print(f"样本总数: {len(samples)}  （标注文件: {os.path.relpath(LABELS_CSV, ROOT)}）")
    for c in CLASSES + ["unknown"]:
        files = [os.path.basename(s["path"]) for s in samples
                 if s["label"] == c and s["src"] == "labels"]
        print(f"  {c:<9} {LABEL_CN[c]:<4} {len(files):>2} 张  {files if files else ''}")
    for s in samples:
        if s["src"] != "extra":
            continue
        tag = "参与训练" if s["label"] in CLASSES else "不参与训练"
        print(f"  追加地址 {os.path.relpath(s['path'], ROOT)} -> "
              f"{s['label']}({LABEL_CN.get(s['label'], s['label'])})  {tag}")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def train(args):
    extra = list(EXTRA_SOURCES) + list(getattr(args, "extra", None) or [])
    samples = load_samples(args.labels, extra=extra, extra_label=args.extra_label)
    print_dataset_stats(samples)

    trainable = [s for s in samples if s["label"] in CLASSES]
    if len(trainable) < 3:
        print("[error] 可用于训练的样本不足 3 张，先让 fudai_auto.py 多跑一会儿采集样本")
        return 1
    per_class = {c: sum(1 for s in trainable if s["label"] == c) for c in CLASSES}
    if min(per_class.values()) == 0:
        print(f"[warn] 有空类别 {per_class}，模型会缺类；建议先补齐再训练")

    print(f"\n特征维度: {extract_features(trainable[0]['img']).shape[0]}  "
          f"(HOG + HSV)  增强倍数: {args.aug}")

    print("\n[1/2] Leave-One-Out 交叉验证…")
    acc, cm, wrong = leave_one_out(trainable, args.aug, args.epochs)
    print(f"  LOO 准确率: {acc * 100:.1f}%  (样本 {len(trainable)}，仅作参考)")
    head = "  真实\\预测 " + " ".join(f"{c:>9}" for c in CLASSES)
    print(head)
    for a in CLASSES:
        row = " ".join(f"{cm[a].get(b, 0):>9}" for b in CLASSES)
        print(f"  {a:<9} {row}")
    for name, true, pred, proba in wrong:
        print(f"    误判 {name}: 真={true} 预测={pred} " +
              " ".join(f"{k}:{v:.2f}" for k, v in proba.items()))

    if args.dry_run:
        print("\n--dry-run：不写模型文件")
        return 0

    print("\n[2/2] 全量训练并保存…")
    X, y = [], []
    for s in trainable:
        for v in augment(s["img"], np.random.default_rng(hash(s["path"]) % 2**32), n=args.aug):
            X.append(extract_features(v))
            y.append(s["label"])
    clf = SoftmaxClassifier().fit(X, y, epochs=args.epochs, verbose=True)
    # 训完自检：原图应全部预测正确
    bad = [os.path.basename(s["path"]) for s in trainable if clf.predict(s["img"]) != s["label"]]
    if bad:
        print(f"[warn] 训练集自检有 {len(bad)} 张不符: {bad}")

    p = clf.save(args.model)
    print(f"\n模型已保存: {os.path.relpath(p, ROOT)}")
    print(f"类别: {clf.classes}   训练样本(含增强): {len(X)}")
    print("预测用法: python train_fudai.py --predict <图片或目录>")
    return 0


def predict(args):
    clf = load_model(args.model)
    target = args.predict
    paths = (
        sorted(glob.glob(os.path.join(target, "*.png")) +
               glob.glob(os.path.join(target, "*.jpg")))
        if os.path.isdir(target) else [target]
    )
    if not paths:
        print(f"[error] 没找到图片: {target}")
        return 1
    print(f"模型类别: {clf.classes}\n")
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: 读取失败")
            continue
        label, proba = clf.predict(img, return_proba=True)
        detail = " ".join(f"{k}:{v:.2f}" for k, v in proba.items())
        print(f"{os.path.basename(p):<40} -> {label:<9} {detail}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="训练 data/fudai 福袋图标分类模型")
    ap.add_argument("--labels", default=LABELS_CSV, help="标注 csv 路径")
    ap.add_argument("--model", default=MODEL_PATH, help="模型输出/读取路径")
    ap.add_argument("--aug", type=int, default=8, help="每张样本增强倍数（含原图）")
    ap.add_argument("--epochs", type=int, default=400, help="训练轮数")
    ap.add_argument("--dry-run", action="store_true", help="只评估不保存模型")
    ap.add_argument("--predict", metavar="IMG_OR_DIR", help="用已有模型预测图片或目录")
    ap.add_argument(
        "--extra", nargs="+", default=[], metavar="PATH",
        help="额外数据地址（图片文件或目录），可多个；默认已含 data/fudai/bag_template.png",
    )
    ap.add_argument(
        "--extra-label", choices=CLASSES, default=None,
        help="给额外数据强制指定类别（默认按文件名推断，推断不出则不参与训练）",
    )
    args = ap.parse_args()
    if args.predict:
        return predict(args)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
