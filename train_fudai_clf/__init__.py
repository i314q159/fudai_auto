# -*- coding: utf-8 -*-
"""福袋检测模型（fudai_clf.npz）的训练包。

对外仍然可以 ``from train_fudai_clf import iou, pad_crop, pink_box``
（冷启动脚本 bootstrap_fudai_clf 就复用了这几个几何工具）。

目录结构：
  paths.py    路径与设计基准
  samples.py  正/负样本构造（含粉色主体定位、IoU、裁剪增强）
  train.py    特征 -> 线性判别 -> 阈值搜索 -> 落盘
  __main__.py 命令行入口
"""
from .paths import ROOT, MODEL_PATH
from .samples import pink_box, pad_crop, iou, positives, negatives
from .train import train_model

__all__ = [
    "ROOT", "MODEL_PATH",
    "pink_box", "pad_crop", "iou", "positives", "negatives",
    "train_model",
]
