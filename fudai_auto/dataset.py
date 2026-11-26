# -*- coding: utf-8 -*-
"""福袋样本采集：把识别成功的图标裁成训练样本，写入 icons/ 与 labels.csv。

要点：
  - 只保留「点开面板确认成功」的样本，避免误检（券图标、头像）污染训练集；
  - 同一标签在 throttle_seconds 内不重复保存，画面几乎不变时也跳过（去重）；
  - 裁剪按屏幕缩放换算，但入库统一重采样成 120×90，跨分辨率样本同口径。
"""
import csv
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np

from . import state
from .config import cfg
from .layout import SCALE
from .paths import DATASET_ICONS, DATASET_LABELS, ROOT


def save_dataset(img, cx, cy, label, src):
    """把 (cx, cy) 附近的福袋图标裁成样本；成功返回文件路径，跳过返回 None。"""
    now = time.time()
    st = state.DATASET_CACHE

    # 节流：同一标签在短时间内只存一张
    if st["label"] == label and now - st["ts"] < float(cfg("sample", "throttle_seconds")):
        return None

    # 按设计尺寸（120×90）换算实际裁剪框，保证不同分辨率裁到的范围一致
    x0 = max(0, int(cx) - int(round(60 * SCALE["x"])))
    y0 = max(0, int(cy) - int(round(45 * SCALE["y"])))
    x1 = min(img.shape[1], x0 + int(round(120 * SCALE["x"])))
    y1 = min(img.shape[0], y0 + int(round(90 * SCALE["y"])))
    crop_img = img[y0:y1, x0:x1].copy()
    if crop_img.size == 0:
        return None
    # 统一成设计尺寸 120×90，保证不同分辨率采到的样本进同一套特征
    if crop_img.shape[1] != 120 or crop_img.shape[0] != 90:
        crop_img = cv2.resize(crop_img, (120, 90), interpolation=cv2.INTER_AREA)

    # 去重：和上一张样本几乎一样就不再写盘
    if st["crop"] is not None and st["crop"].shape == crop_img.shape:
        diff = float(np.mean(cv2.absdiff(crop_img, st["crop"])))
        if diff < 2.0 and st["label"] == label:
            return None

    os.makedirs(DATASET_ICONS, exist_ok=True)
    tag = re.sub(r"[^\w\u4e00-\u9fff]", "-", str(label))[:24]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(DATASET_ICONS, f"{ts}_{tag}.png")
    # 文件名只精确到秒：同一秒内不同标签/来源可能撞名，撞到就加序号，不覆盖已有样本
    n = 1
    while os.path.exists(p):
        p = os.path.join(DATASET_ICONS, f"{ts}_{tag}_{n}.png")
        n += 1
    cv2.imwrite(p, crop_img)

    # 记录标签来源：时间 / 房间 / 标签 / 来源（clf/tpl/ocr）/ 实际 bbox / 文件
    new = not os.path.exists(DATASET_LABELS)
    with open(DATASET_LABELS, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "room", "label", "source", "bbox", "file"])
        w.writerow([
            f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            state.ROOM_NAME or "",
            label,
            src,
            f"{x0},{y0},{x1},{y1}",
            os.path.relpath(p, ROOT),
        ])

    st.update(ts=now, label=label, crop=crop_img)
    return p
