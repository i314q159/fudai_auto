# -*- coding: utf-8 -*-
"""直播间相关：读取标题、关注主播。

标题 OCR 做了归一化（只留汉字/字母/数字），避免标点抖动把「同一房间」判成换房。
"""
import re
import time

from .layout import FOLLOW_ROI, NAME_ROI, crop, sc_roi
from .device import tap
from .logbook import log
from .ocrkit import ocr, texts_of


def get_room_name(img):
    """从直播间头部读取标题；失败返回 None。

    取标题行里字高最大的一段（通常是主播名/房间名），并去掉标点，
    连续多次结果不一致时由 main.py 用相似度比对决定是否算换房。
    """
    items = ocr(crop(img, sc_roi(NAME_ROI)), 0.5)
    if not items:
        return None
    items.sort(key=lambda i: i["h"], reverse=True)
    for i in items:
        t = i["text"].strip().replace("\u3000", "")
        t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t)
        if len(t) >= 3:
            return t
    return None


def follow_anchor(img):
    """检查主播「关注」按钮并点一次；已关注返回 'already'，点了返回 'followed'。

    只在头部小区域 OCR，避免把评论区里的「关注」字样误点。
    """
    roi_px = sc_roi(FOLLOW_ROI)
    items = ocr(crop(img, roi_px), 0.45)
    t = texts_of(items)
    if "已关注" in t:
        return "already"
    for i in items:
        if "关注" in i["text"]:
            cx = roi_px[0] + i["x"] + i["w"] / 2
            cy = roi_px[1] + i["y"] + i["h"] / 2
            tap(cx, cy)
            time.sleep(0.6)
            log(f"已点击关注主播 ({int(cx)},{int(cy)})")
            return "followed"
    return None
