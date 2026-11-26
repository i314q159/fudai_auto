# -*- coding: utf-8 -*-
"""跨模块共享的可变运行状态。

为什么单独放一个模块：
  - 运行时很多模块都要读「当前设备序列号 / 当前日志文件 / 当前房间名 / 检测模型」；
  - main.py 里又有 ``fa.SERIAL = ...``、``fa.RUN_LOG = ...`` 这种包级赋值；
  - 把可变状态集中在这里，包的 __init__.py 负责把包级读写转发过来，
    子模块统一访问 ``state.SERIAL`` 等，避免各模块各存一份副本。
"""
import os

from human import adb_conn

# 设备序列号：优先环境变量，其次由 human.adb_conn 探测当前连接的设备
SERIAL = os.getenv("FUDAI_SERIAL", adb_conn.get_adb_device_serial_via_subprocess())

# 当前运行的日志文件路径（main.py 启动时按时间戳设置；None 表示只打印到控制台）
RUN_LOG = None

# 最近一次 OCR 识别到的直播间标题（用于「没有标题就不点福袋」的保护）
ROOM_NAME = None

# 福袋检测模型：None 或 {"w", "thresh", "win_w", "win_h"}；支持热重载
CLF = None

# 样本采集缓存：记录上一次保存的标签/时间/裁剪图，用于节流和去重
DATASET_CACHE = {"ts": 0.0, "label": None, "crop": None}
