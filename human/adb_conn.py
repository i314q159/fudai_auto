# -*- coding: utf-8 -*-
"""adb 设备探测：从 `adb devices` 输出里取当前连接的第一台设备序列号。

运行时优先读环境变量 FUDAI_SERIAL；没有环境变量时才调用这里自动探测。
"""
import subprocess


def get_adb_device_serial_via_subprocess():
    """返回 `adb devices` 里第一台设备的序列号；没有设备或命令失败返回 None。

    `adb devices` 的输出形如：
        List of devices attached
        192.168.137.84:44607    device

    第二行按制表符切分，第一段就是序列号。
    """
    try:
        result = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            device_serial = lines[1].split("\t")[0]
            return device_serial
    except subprocess.CalledProcessError as e:
        print(f"Error running adb: {e}")
    return None
