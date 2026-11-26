# -*- coding: utf-8 -*-
"""在指定 ROI 上滑窗打分，返回最佳命中位置。

推理侧（fudai_auto.detector）与训练侧（阈值评估）都走这里，
窗口尺寸由调用方传入（支持按屏幕分辨率等比缩放）。
"""
import numpy as np

from .descriptor import feat


def locate(img, roi, w, thresh, stride=8):
    """滑窗打分，返回 (cx, cy, score) 最佳命中；低于阈值返回 None。

    参数：
      img     整屏 BGR 图
      roi     (x0, y0, x1, y1) 待扫描区域（实际像素）
      w       模型字典，需要键：
                w      权重向量（最后一维是偏置）
                win_w  窗口宽（实际像素）
                win_h  窗口高（实际像素）
                thresh 判定阈值
      thresh  低于该分数直接丢弃（重复传参是为了让调用方显式表达阈值）
      stride  滑窗步长（实际像素，分辨率变化时调用方按比例缩放）
    """
    x0r, y0r, x1r, y1r = roi
    sub = img[y0r:y1r, x0r:x1r]

    feats, pos = [], []
    for y in range(0, max(1, sub.shape[0] - w["win_h"] + 1), stride):
        for x in range(0, max(1, sub.shape[1] - w["win_w"] + 1), stride):
            win = sub[y:y + w["win_h"], x:x + w["win_w"]]
            # 贴边窗口不足窗口尺寸时跳过（不做 padding，避免引入虚假边缘）
            if win.shape[0] != w["win_h"] or win.shape[1] != w["win_w"]:
                continue
            feats.append(feat(win))
            pos.append((x, y))
    if not feats:
        return None

    X = np.stack(feats)
    scores = X @ w["w"][:-1] + w["w"][-1]
    i = int(np.argmax(scores))
    s = float(scores[i])
    if s < thresh:
        return None

    x, y = pos[i]
    # 返回窗口中心（相对整屏的坐标）
    cx = x0r + x + w["win_w"] / 2
    cy = y0r + y + w["win_h"] / 2
    return cx, cy, s
