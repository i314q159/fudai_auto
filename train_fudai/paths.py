# -*- coding: utf-8 -*-
"""状态模型训练用的路径与常量。

注意：这个训练包只做「图标状态分类」（待参与/已参与/倒计时），
与运行时检测模型 fudai_clf.npz 是两套东西，互不覆盖。
"""
import os
import re

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)

DATASET_DIR = os.path.join(ROOT, "data", "fudai")
ICON_DIR = os.path.join(DATASET_DIR, "icons")          # 运行时采集的图标样本
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")   # 样本标注
MODEL_DIR = os.path.join(DATASET_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fudai_state.npz")     # 状态模型（本包产物）
DETECTOR_PATH = os.path.join(MODEL_DIR, "fudai_clf.npz")    # 检测模型（fudai_auto 用）
BAG_TEMPLATE = os.path.join(DATASET_DIR, "bag_template.png")  # 福袋模板图（默认追加样本）

# 默认追加的数据来源：模板图只做统计/参考，按文件名推断标签
EXTRA_SOURCES = [BAG_TEMPLATE]

IMG_SIZE = 64                  # 特征输入统一缩放到 64×64
CLASSES = ["pending", "joined", "countdown"]
LABEL_CN = {"pending": "待参与", "joined": "已参与", "countdown": "倒计时", "unknown": "未知"}
MODEL_VERSION = 1              # 存盘版本号，方便以后兼容旧模型

# 标签文字里的倒计时格式（mm:ss，兼容中文冒号）
TIME_RE = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}")
