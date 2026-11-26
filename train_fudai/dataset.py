# -*- coding: utf-8 -*-
"""样本装载：读 labels.csv + 追加数据源，统一标签口径。

labels.csv 里的原始标签很杂（「待参与」「已参与」「倒计时 03:17」），
这里统一成 pending / joined / countdown 三类，其它记为 unknown（只统计、不训练）。
"""
import csv
import glob
import os
import zlib

import cv2

from .paths import CLASSES, LABEL_CN, LABELS_CSV, ROOT, TIME_RE


def norm_label(raw):
    """把 labels.csv 的原始标签映射成统一类别。"""
    t = (raw or "").strip()
    if "待参与" in t:
        return "pending"
    if "已参与" in t:
        return "joined"
    if TIME_RE.search(t):
        return "countdown"
    return "unknown"


def _expand(paths):
    """把「文件或目录」列表展开成图片路径列表；不存在的地址打印警告后跳过。"""
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
    """返回 [{path, raw, label, room, img, src}]；文件缺失或读不出图的直接跳过。

    - labels.csv 的样本按标注归一化；
    - extra（含默认 bag_template.png）按文件名推断标签，可用 extra_label 强制指定。
    """
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"找不到标注文件: {labels_csv}")

    samples, missing, seen = [], [], set()
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = (row.get("file") or "").strip()
            if not fn:                      # labels.csv 里可能有空行
                continue
            p = fn if os.path.isabs(fn) else os.path.join(ROOT, fn)
            if not os.path.exists(p):
                missing.append(fn)
                continue
            img = cv2.imread(p)
            if img is None:
                missing.append(fn)
                continue
            samples.append({
                "path": p,
                "raw": (row.get("label") or "").strip(),
                "label": norm_label(row.get("label")),
                "room": (row.get("room") or "").strip(),
                "img": img,
                "src": "labels",
            })
            seen.add(os.path.normcase(os.path.abspath(p)))

    if missing:
        print(f"[warn] 标注里有 {len(missing)} 个文件不存在（多半是数据集被清理过），已跳过")
        for m in missing[:3]:
            print(f"        {m}")

    for p in _expand(extra):
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        img = cv2.imread(p)
        if img is None:
            print(f"[warn] 追加图片读取失败，已跳过: {p}")
            continue
        samples.append({
            "path": p,
            "raw": os.path.basename(p),
            "label": extra_label or norm_label(os.path.splitext(os.path.basename(p))[0]),
            "room": "",
            "img": img,
            "src": "extra",
        })
        seen.add(key)
    return samples


def stable_seed(path):
    """按文件名生成稳定随机种子（不能用内置 hash，Python 每个进程结果都不同）。"""
    return zlib.crc32(os.path.basename(path).encode("utf-8")) % (2 ** 32)


def print_dataset_stats(samples):
    """打印样本统计：每类张数、文件名，以及追加数据源的归类结果。"""
    print(f"样本总数: {len(samples)}  （标注文件: {os.path.relpath(LABELS_CSV, ROOT)}）")
    for c in CLASSES + ["unknown"]:
        files = [
            os.path.basename(s["path"])
            for s in samples
            if s["label"] == c and s["src"] == "labels"
        ]
        print(f"  {c:<9} {LABEL_CN[c]:<4} {len(files):>2} 张  {files if files else ''}")
    for s in samples:
        if s["src"] != "extra":
            continue
        tag = "参与训练" if s["label"] in CLASSES else "不参与训练"
        print(
            f"  追加地址 {os.path.relpath(s['path'], ROOT)} -> "
            f"{s['label']}({LABEL_CN.get(s['label'], s['label'])})  {tag}"
        )
