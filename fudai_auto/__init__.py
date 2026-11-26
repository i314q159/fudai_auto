# -*- coding: utf-8 -*-
"""运行时能力库（对外统一入口）。

原来是一个 fudai_auto.py，按功能拆成以下子模块：

  state.py     可变运行状态（设备序列号 / 日志路径 / 房间名 / 检测模型）
  paths.py     目录与文件路径
  config.py    config.toml 读取与默认值
  logbook.py   运行日志 / 事件 CSV / 截图落盘
  layout.py    界面布局常量（1080×2400 设计坐标）与分辨率缩放
  device.py    adb 连接自愈、截图、点击、上滑
  ocrkit.py    OCR 引擎与倒计时解析
  detector.py  福袋图标定位（分类模型 / 模板 / OCR 兜底）
  room.py      直播间标题、关注主播
  panel.py     福袋面板状态与任务点击、样本入库
  result.py    开奖弹层识别与统一关闭、开奖观察窗口
  dataset.py   样本采集

对外仍然按 ``import fudai_auto as fa`` 使用，API 与拆分前一致。
可变状态（SERIAL / RUN_LOG / ROOM_NAME / CLF）做了转发：包级赋值与读取
都会落到 state 模块，保证各子模块看到的是同一份值。
"""
import sys as _sys
import types as _types

from . import state
from .config import CONFIG, DEFAULT_CONFIG, cfg
from .dataset import save_dataset
from .detector import (
    BAG_TPL, BAG_TPL_PATH, CLF_PATH, clf_locate, find_bag, load_clf, pink_blob,
    reload_clf,
)
from .device import adb, device_state, ensure_device, shot, swipe_next_room, tap
from .layout import (
    DESIGN_H, DESIGN_W, FOLLOW_ROI, HEAD_ROI, ICON_ROI, NAME_ROI,
    PANEL_FALLBACK_ROI, PANEL_HEAD_ROI, PANEL_HEAD_ROI_WIDE, PANEL_ROI,
    PAY_ROI, RESULT_BTN_TEXTS, RESULT_KEYS, RESULT_ROI, SCALE,
    WIN_CLOSE_POINTS, crop, sc_pt, sc_roi, sx, sy, update_screen,
)
from .logbook import log, log_csv, save
from .ocrkit import TIME_RE, engine, ocr, parse_countdown, texts_of
from .panel import (
    close_panel, handle_panel, open_bag_panel, panel_state, pink_button_box,
    read_remaining,
)
from .paths import (
    DATASET_DIR, DATASET_ICONS, DATASET_LABELS, LOG_CSV, LOG_DIR, ROOT, SHOT_DIR,
)
from .result import dismiss_result, result_present, watch_result
from .room import follow_anchor, get_room_name

# 这些名字不直接绑到包上，读写都转发给 state（见下方 __getattr__ / __setattr__）
_MUTABLE = {"SERIAL", "RUN_LOG", "ROOM_NAME", "CLF", "DATASET_CACHE"}

_EXTRA_EXPORTS = [
    "DESIGN_W", "DESIGN_H", "SCALE", "sx", "sy", "sc_roi", "sc_pt", "crop",
    "update_screen", "ICON_ROI", "PANEL_ROI", "PANEL_HEAD_ROI",
    "PANEL_HEAD_ROI_WIDE", "HEAD_ROI", "NAME_ROI", "RESULT_ROI", "FOLLOW_ROI",
    "PAY_ROI", "PANEL_FALLBACK_ROI", "RESULT_KEYS", "RESULT_BTN_TEXTS",
    "WIN_CLOSE_POINTS", "ROOT", "LOG_DIR", "SHOT_DIR", "LOG_CSV",
    "DATASET_DIR", "DATASET_ICONS", "DATASET_LABELS", "CONFIG",
    "DEFAULT_CONFIG", "cfg", "log", "log_csv", "save", "adb", "device_state",
    "ensure_device", "shot", "tap", "swipe_next_room", "engine", "ocr",
    "texts_of", "TIME_RE", "parse_countdown", "BAG_TPL", "BAG_TPL_PATH",
    "CLF_PATH", "clf_locate", "find_bag", "load_clf", "reload_clf",
    "pink_blob", "save_dataset", "get_room_name", "follow_anchor",
    "pink_button_box", "read_remaining", "panel_state", "close_panel",
    "handle_panel", "open_bag_panel", "result_present", "dismiss_result",
    "watch_result", "SERIAL", "RUN_LOG", "ROOM_NAME", "CLF",
]


def __getattr__(name):
    """包级读取：可变状态实时取 state 里的值，其余未定义名字正常报错。"""
    if name in _MUTABLE:
        return getattr(state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _CompatModule(_types.ModuleType):
    """包级赋值：把 main.py 里的 fa.SERIAL / fa.RUN_LOG 等写进 state。"""

    def __setattr__(self, name, value):
        if name in _MUTABLE:
            setattr(state, name, value)
        else:
            super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _CompatModule

__all__ = _EXTRA_EXPORTS
