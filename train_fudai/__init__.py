# -*- coding: utf-8 -*-
"""状态模型训练包：图标状态 3 分类（待参与 / 已参与 / 倒计时）。

与检测模型（fudai_clf.npz）分开：这里产出 fudai_state.npz，只做状态分析，
不参与运行时阻断逻辑。对外保留 ``import train_fudai as tf; tf.load_model()`` 用法。

目录结构：
  paths.py     路径与类别常量
  dataset.py   样本装载、标签归一化、统计打印、稳定随机种子
  features.py  HOG + HSV 特征、数据增强
  model.py     softmax 线性分类器与模型存取
  eval.py      留一交叉验证
  cli.py       训练 / 预测命令行流程
  __main__.py  入口（python -m train_fudai）
"""
from .cli import main
from .dataset import load_samples, norm_label
from .features import extract_features
from .model import SoftmaxClassifier, load_model
from .paths import CLASSES, MODEL_PATH

__all__ = [
    "CLASSES", "MODEL_PATH",
    "load_samples", "norm_label", "extract_features",
    "SoftmaxClassifier", "load_model", "main",
]
