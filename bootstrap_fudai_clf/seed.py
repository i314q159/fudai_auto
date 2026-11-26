# -*- coding: utf-8 -*-
"""种子图里定位福袋图标，并把设计坐标换算成种子图自己的分辨率。"""
import cv2

from train_fudai_clf import pink_box

from .paths import ICON_ROI, DESIGN_W, DESIGN_H


def scale_of(img):
    """种子图相对设计分辨率（1080×2400）的缩放系数。"""
    return img.shape[1] / DESIGN_W, img.shape[0] / DESIGN_H


def locate_seed_box(img):
    """返回 (box, rx, ry)：
      box  种子图里福袋主体的 (x0, y0, x1, y1)（实际像素）
      rx/ry  种子图相对设计分辨率的缩放，用于换算窗口尺寸
    找不到粉色主体时抛 SystemExit（提示换一张种子图）。
    """
    rx, ry = scale_of(img)
    roi = (
        int(round(ICON_ROI[0] * rx)), int(round(ICON_ROI[1] * ry)),
        int(round(ICON_ROI[2] * rx)), int(round(ICON_ROI[3] * ry)),
    )
    pb = pink_box(img[roi[1]:roi[3], roi[0]:roi[2]])
    if not pb:
        raise SystemExit("种子图里没找到粉色福袋主体，换一张图")
    box = (roi[0] + pb[0], roi[1] + pb[1], roi[0] + pb[2], roi[1] + pb[3])
    return box, rx, ry


def read_seed(path):
    """读种子图；读不到直接退出并指明路径。"""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"种子图读取失败: {path}")
    return img
