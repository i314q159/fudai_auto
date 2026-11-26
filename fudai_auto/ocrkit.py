# -*- coding: utf-8 -*-
"""OCR 封装：RapidOCR 结果转成「文本框字典」，以及倒计时解析。

ocr() 返回的每条结果字段：
  text   识别文字
  score  置信度
  x/y/w/h 该文字块在传入图片里的位置与尺寸（像素，浮点）
"""
import re

import numpy as np
from rapidocr_onnxruntime import RapidOCR

# OCR 引擎是重对象，全进程只初始化一次
engine = RapidOCR()

# 倒计时格式：mm:ss（兼容中文冒号、数字间空格）
TIME_RE = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}")


def ocr(img, min_score=0.5):
    """OCR 一张图，按置信度过滤，把四点框转成 (x, y, w, h) 方便后续定位。

    屏幕截图不会倒置，关掉角度分类器（use_cls）省一点时间。
    """
    res, _ = engine(img, use_cls=False)
    items = []
    if res:
        for box, txt, score in res:
            if float(score) >= min_score:
                pts = np.array(box).reshape(-1, 2)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                items.append({
                    "text": txt,
                    "score": float(score),
                    "x": float(x0),
                    "y": float(y0),
                    "w": float(x1 - x0),
                    "h": float(y1 - y0),
                })
    return items


def texts_of(items):
    """把 ocr() 结果拼成一段文本，供关键词匹配。"""
    return " ".join(i["text"] for i in items)


def parse_countdown(text):
    """从面板文字里解析剩余秒数；兼容 OCR 丢冒号、把数字拆开的情况。"""
    norm = text.replace(" ", "")
    m = re.search(r"(\d{1,2})[:：](\d{1,2})", norm)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if mm <= 60 and ss <= 59:
            return mm * 60 + ss
    # 兜底：OCR 把冒号丢了（如「已参与 0317」）。必须先定位到倒计时前面的标签词，
    # 只解析标签词之后的部分；否则「1234 人观看」这类无关长数字会被读成 12:34
    tail = re.split(r"已参与|剩余|距结束|倒计时", norm)[-1]
    if tail == norm:
        return None
    m = re.search(r"(\d{1,2})\D{0,2}(\d{1,2})(?!\d)", tail)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if mm <= 60 and ss <= 59:
            return mm * 60 + ss
    return None
