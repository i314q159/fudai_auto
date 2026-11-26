# -*- coding: utf-8 -*-
"""命令行：训练 / 评估 / 预测三个子流程，以及参数解析。

用法：
  python -m train_fudai                       # 训练（默认读 data/fudai）
  python -m train_fudai --dry-run             # 只看统计和 LOO 结果，不落盘
  python -m train_fudai --predict <图片/目录>  # 用已有模型预测
"""
import argparse
import glob
import os

import cv2
import numpy as np

from .dataset import load_samples, print_dataset_stats, stable_seed
from .eval import leave_one_out
from .features import augment, extract_features
from .model import SoftmaxClassifier, load_model
from .paths import (
    CLASSES, EXTRA_SOURCES, LABELS_CSV, MODEL_PATH, ROOT,
)


def build_parser():
    """命令行参数（与拆分前的单文件脚本保持一致）。"""
    ap = argparse.ArgumentParser(description="训练 data/fudai 福袋图标分类模型")
    ap.add_argument("--labels", default=LABELS_CSV, help="标注 csv 路径")
    ap.add_argument("--model", default=MODEL_PATH, help="模型输出/读取路径")
    ap.add_argument("--aug", type=int, default=8, help="每张样本增强倍数（含原图）")
    ap.add_argument("--epochs", type=int, default=400, help="训练轮数")
    ap.add_argument("--min-per-class", type=int, default=1,
                    help="每个出现的类别至少要有多少张样本（默认 1，冷启动可调高）")
    ap.add_argument("--dry-run", action="store_true", help="只评估不保存模型")
    ap.add_argument("--predict", metavar="IMG_OR_DIR", help="用已有模型预测图片或目录")
    ap.add_argument("--extra", nargs="+", default=[], metavar="PATH",
                    help="额外数据地址（图片文件或目录），可多个；默认已含 data/fudai/bag_template.png")
    ap.add_argument("--extra-label", choices=CLASSES, default=None,
                    help="给额外数据强制指定类别（默认按文件名推断，推断不出则不参与训练）")
    return ap


def train(args):
    """训练主流程：装载 -> 统计 -> LOO 评估 -> 全量训练 -> 落盘。"""
    extra = list(EXTRA_SOURCES) + list(getattr(args, "extra", None) or [])
    samples = load_samples(args.labels, extra=extra, extra_label=args.extra_label)
    print_dataset_stats(samples)

    trainable = [s for s in samples if s["label"] in CLASSES]
    if len(trainable) < 3:
        print("[error] 可用于训练的样本不足 3 张，先让 main.py（fudai_auto 包）多跑一会儿采集样本")
        return 1

    per_class = {c: sum(1 for s in trainable if s["label"] == c) for c in CLASSES}
    present = [c for c in CLASSES if per_class[c] > 0]
    if len(present) < 2:
        print(f"[error] 状态模型至少需要 2 个类别，当前只有 {present}（{per_class}）")
        print("        冷启动阶段一张种子图无法训练状态模型，先跑采集攒够类别再训")
        return 2
    if min(per_class[c] for c in present) < args.min_per_class:
        print(f"[error] 类别样本不足 {args.min_per_class} 张: {per_class}")
        return 2
    if len(present) < len(CLASSES):
        print(f"[warn] 缺少类别 {[c for c in CLASSES if c not in present]}，模型只覆盖 {present}")

    print(f"\n特征维度: {extract_features(trainable[0]['img']).shape[0]}  "
          f"(HOG + HSV)  增强倍数: {args.aug}")

    print("\n[1/2] Leave-One-Out 交叉验证…")
    acc, cm, wrong = leave_one_out(trainable, args.aug, args.epochs)
    print(f"  LOO 准确率: {acc * 100:.1f}%  (样本 {len(trainable)}，仅作参考)")
    print("  真实\\预测 " + " ".join(f"{c:>9}" for c in CLASSES))
    for a in CLASSES:
        row = " ".join(f"{cm[a].get(b, 0):>9}" for b in CLASSES)
        print(f"  {a:<9} {row}")
    for name, true, pred, proba in wrong:
        print(f"    误判 {name}: 真={true} 预测={pred} " +
              " ".join(f"{k}:{v:.2f}" for k, v in proba.items()))

    if args.dry_run:
        print("\n--dry-run：不写模型文件")
        return 0

    # 全量训练：每张样本用文件名种子做增强，保证结果可复现
    print("\n[2/2] 全量训练并保存…")
    X, y = [], []
    for s in trainable:
        for v in augment(s["img"], np.random.default_rng(stable_seed(s["path"])), n=args.aug):
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
    print("预测用法: python -m train_fudai --predict <图片或目录>")
    return 0


def predict(args):
    """用已有模型预测单张图片或整个目录。"""
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
    """入口：有 --predict 走预测，否则走训练。"""
    args = build_parser().parse_args()
    if args.predict:
        return predict(args)
    return train(args)
