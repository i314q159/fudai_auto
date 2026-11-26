# -*- coding: utf-8 -*-
"""人工辅助工具包：放「给人和其它脚本直接用」的小工具。

目前只有 adb_conn：探测当前连接的 adb 设备序列号。
运行时（fudai_auto）在没设置 FUDAI_SERIAL 时会调用它。
"""
from .adb_conn import get_adb_device_serial_via_subprocess

__all__ = ["get_adb_device_serial_via_subprocess"]
