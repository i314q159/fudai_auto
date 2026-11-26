# -*- coding: utf-8 -*-
"""冷启动包：用一张种子图训出能点火运行的福袋检测模型。

目录结构：
  paths.py    模型路径与设计基准
  seed.py     种子图读取、福袋主体定位（按种子图分辨率换算 ROI）
  samples.py  单图正/负样本构造
  train.py    强正则线性判别 + 同帧自评阈值
  __main__.py 命令行入口（python -m bootstrap_fudai_clf）
"""
from .paths import ROOT, OUT_DEFAULT, PROD
from .seed import read_seed, locate_seed_box, scale_of
from .samples import build_positives, build_negatives
from .train import fit_bootstrap

__all__ = [
    "ROOT", "OUT_DEFAULT", "PROD",
    "read_seed", "locate_seed_box", "scale_of",
    "build_positives", "build_negatives", "fit_bootstrap",
]
