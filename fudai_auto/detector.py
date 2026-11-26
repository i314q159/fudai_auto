# -*- coding: utf-8 -*-
"""福袋图标检测：分类模型优先，模板匹配兜底，最后用 OCR 倒计时反推。

模型文件 data/fudai/model/fudai_clf.npz 由训练脚本产出，字段：
  w       线性权重（最后一维为偏置）
  thresh  判定阈值（载入时下限抬到 0.12，防止空模型误命中）
  win_w/h 设计分辨率下的检测窗口（运行时按屏幕缩放）
"""
import os

import cv2
import numpy as np

import fudai_feat

from . import layout, state
from .layout import ICON_ROI, SCALE, crop, sc_roi, sx, sy
from .logbook import log
from .ocrkit import TIME_RE, ocr
from .paths import DATASET_DIR, ROOT

# 备用模板图（模型缺失或打分失败时用）
BAG_TPL_PATH = next(
    (
        p
        for p in (
            os.path.join(DATASET_DIR, "bag_template.png"),
            os.path.join(ROOT, "bag_template.png"),
        )
        if os.path.exists(p)
    ),
    os.path.join(ROOT, "bag_template.png"),
)
BAG_TPL = cv2.imread(BAG_TPL_PATH) if os.path.exists(BAG_TPL_PATH) else None

CLF_PATH = os.path.join(DATASET_DIR, "model", "fudai_clf.npz")


def load_clf():
    """读取检测模型；换机/重训后可直接热重载，无需重启进程。

    载入时校验权重维度与数值：特征口径变了（改 fudai_feat.params）会在这里被拦下，
    按「没有模型」处理并回退模板匹配，避免运行时矩阵乘抛错把主循环打断。
    阈值与训练共用 ``fudai_feat.MIN_THRESH`` 下限，保证线上阈值可复现。
    """
    state.CLF = None
    try:
        if not os.path.exists(CLF_PATH):
            return None
        m = np.load(CLF_PATH, allow_pickle=False)
        w = np.asarray(m["w"], dtype=np.float32)
        expect = fudai_feat.feat(
            np.zeros((fudai_feat.WIN_H, fudai_feat.WIN_W, 3), np.uint8)
        ).shape[0] + 1
        if w.ndim != 1 or w.shape[0] != expect:
            log(f"检测模型维度不符（{w.shape} != ({expect},)），忽略并回退模板匹配")
            return None
        if not np.all(np.isfinite(w)) or not np.any(w):
            log("检测模型权重非法（含 NaN/Inf 或全零），忽略并回退模板匹配")
            return None
        th = float(m["thresh"])
        if not np.isfinite(th):
            th = fudai_feat.MIN_THRESH
        state.CLF = {
            "w": w,
            "thresh": max(th, fudai_feat.MIN_THRESH),
            "win_w": int(m["win_w"]),
            "win_h": int(m["win_h"]),
        }
    except Exception as e:
        log(f"检测模型加载失败，回退模板匹配: {e}")
        state.CLF = None
    return state.CLF


def reload_clf():
    """重新加载模型（训练/冷启动完成后调用）。"""
    return load_clf()


load_clf()


def _clf_window():
    """当前屏幕下的分类器窗口与步长（设计像素按 SCALE 换算）。"""
    return (
        {
            "w": state.CLF["w"],
            "thresh": state.CLF["thresh"],
            "win_w": max(8, int(round(state.CLF["win_w"] * SCALE["x"]))),
            "win_h": max(8, int(round(state.CLF["win_h"] * SCALE["y"]))),
        },
        max(4, int(round(8 * SCALE["x"]))),
    )


def clf_locate(img):
    """用分类模型在福袋区滑窗打分，返回 (cx, cy, 'clf:分数') 或 None。

    窗口尺寸、步长都按当前屏幕缩放，保证换分辨率后窗口仍覆盖图标本体。
    """
    if state.CLF is None:
        return None
    win, stride = _clf_window()
    hit = fudai_feat.locate(
        img, sc_roi(ICON_ROI), win, state.CLF["thresh"], stride=stride,
    )
    if not hit:
        return None
    cx, cy, s = hit
    return cx, cy, f"clf:{s:.2f}"


def clf_best(img):
    """诊断用：不分阈值，返回福袋区里分数最高的窗口 (cx, cy, score)。

    没有模型或区域太小时返回 None。只在检测失败时调用，用来在日志里区分
    「屏上没有福袋（分数极低）」和「有福袋但卡在阈值边缘」。
    """
    if state.CLF is None:
        return None
    win, stride = _clf_window()
    return fudai_feat.locate(img, sc_roi(ICON_ROI), win, -9.9, stride=stride)


def find_bag(img):
    """定位福袋图标：分类模型 -> 模板多尺度匹配 -> OCR 倒计时反推。"""
    hit = clf_locate(img)
    if hit:
        return hit

    roi_px = sc_roi(ICON_ROI)
    roi = crop(img, roi_px)

    # 兜底 1：模板匹配（模板随屏幕缩放，多尺度试 0.85~1.15）
    if BAG_TPL is not None:
        best = (0.0, None, None)
        for s in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
            t = cv2.resize(BAG_TPL, None, fx=s * SCALE["x"], fy=s * SCALE["y"])
            if t.shape[0] >= roi.shape[0] or t.shape[1] >= roi.shape[1]:
                continue
            r = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mxloc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (mx, mxloc, t.shape)
        if best[0] >= 0.75:
            mx, loc, shape = best
            cx = roi_px[0] + loc[0] + shape[1] // 2
            cy = roi_px[1] + loc[1] + int(shape[0] * 0.6)
            return cx, cy, f"tpl:{best[0]:.2f}"

    # 兜底 2：OCR 倒计时贴在最左侧图标下方；过滤掉榜单/商品卡里的其它倒计时
    cands = [i for i in ocr(roi, 0.45) if TIME_RE.search(i["text"])]
    cands.sort(key=lambda i: (i["x"], i["y"]))
    for i in cands:
        if i["x"] >= sx(220):
            continue
        cx = roi_px[0] + i["x"] + i["w"] / 2
        cy = roi_px[1] + i["y"] - sy(25)
        return cx, cy, f"ocr:{i['text']}"
    return None


def pink_blob(roi_img):
    """在 ROI 里找粉色团块（调试/备用特征），返回 (面积, x, y, w, h) 或 None。"""
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (140, 60, 120), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 35:
            continue
        if not (0.5 < w / h < 2.0):
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)
    return best
