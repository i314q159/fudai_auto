# -*- coding: utf-8 -*-
"""福袋检测的特征库（训练 / 推理共用）。

对外保持原来的使用方式：``import fudai_feat as F`` 后直接用
``F.WIN_W`` / ``F.feat(win)`` / ``F.locate(img, roi, w, thresh, stride)``；
训练侧另加 ``F.feat_batch(wins, device)`` 批量提特征。

目录结构：
  params.py      窗口与特征尺寸常量
  hog.py         向量化的 HOG（训练批量 / 推理逐窗共用）
  accel.py       计算后端：自动在 NVIDIA GPU（torch CUDA）与 CPU 之间选择
  descriptor.py  单个窗口 -> 特征向量（HOG + HSV 直方图），以及批量版
  sliding.py     ROI 滑窗打分，返回最佳命中
"""
from .params import WIN_W, WIN_H, FEA_W, FEA_H, CELL, BINS, MIN_THRESH
from .descriptor import resize_win, feat, feat_batch
from .sliding import locate

__all__ = [
    "WIN_W", "WIN_H", "FEA_W", "FEA_H", "CELL", "BINS", "MIN_THRESH",
    "resize_win", "feat", "feat_batch", "locate",
]
