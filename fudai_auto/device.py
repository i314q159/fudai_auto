# -*- coding: utf-8 -*-
"""adb 设备层：连接自愈、截图、点击、上滑。

设计要点：
  - 截图失败时按「exec-out -> 重连 / 重启 adb server -> shell+pull」逐级兜底；
  - ``shot()`` 每次截图都会刷新分辨率缩放系数（layout.update_screen），
    保证后续 ROI / 点击点即使用的是当前屏幕的真实像素；
  - 所有坐标入口只接受实际像素，设计坐标请先用 layout.sc_pt() 换算。
"""
import os
import subprocess
import time

import cv2
import numpy as np

from . import layout, state
from .logbook import log
from .paths import LOG_DIR


def adb(*args, timeout=15):
    """执行 adb 子命令（自动带 -s 序列号），返回 (stdout, stderr)。"""
    if not state.SERIAL:
        raise RuntimeError(
            "未指定 adb 设备：请连接手机并确认 adb devices 已授权，"
            "或用 --serial / 环境变量 FUDAI_SERIAL 指定"
        )
    r = subprocess.run(
        ["adb", "-s", state.SERIAL] + list(args),
        capture_output=True, timeout=timeout,
    )
    return r.stdout, r.stderr


def device_state():
    """返回 'device' / 'offline' / 'absent'：当前序列号在 adb 里的状态。"""
    if not state.SERIAL:
        return "absent"
    try:
        out = subprocess.run(
            ["adb", "devices"], capture_output=True, timeout=10
        ).stdout.decode("utf-8", "ignore")
    except Exception:
        return "absent"
    for line in out.splitlines()[1:]:
        if state.SERIAL in line:
            return "device" if "\tdevice" in line else "offline"
    return "absent"


def ensure_device(wait_s=15):
    """等待设备可用；offline 时尝试重连，absent 时重启 adb server（只重启一次）。"""
    t_end = time.time() + wait_s
    restarted = False
    while time.time() < t_end:
        st = device_state()
        if st == "device":
            return True
        if st == "offline":
            try:
                subprocess.run(["adb", "reconnect", "offline"],
                               capture_output=True, timeout=10)
            except Exception:
                pass
            time.sleep(2)
            continue
        if not restarted:
            restarted = True
            try:
                subprocess.run(["adb", "kill-server"], capture_output=True, timeout=10)
                time.sleep(1)
                subprocess.run(["adb", "start-server"], capture_output=True, timeout=15)
            except Exception:
                pass
        time.sleep(3)
    return device_state() == "device"


def _screencap_via_file():
    """兜底截图：shell screencap 到 /sdcard 再 pull 回来（exec-out 不通时用）。"""
    p = os.path.join(LOG_DIR, "_auto_shot.png")
    subprocess.run(
        ["adb", "-s", state.SERIAL, "shell", "screencap", "-p", "/sdcard/_auto_shot.png"],
        capture_output=True, timeout=20,
    )
    subprocess.run(
        ["adb", "-s", state.SERIAL, "pull", "/sdcard/_auto_shot.png", p],
        capture_output=True, timeout=20,
    )
    if os.path.exists(p):
        img = cv2.imread(p)
        if img is not None and img.shape[0] > 1000:
            return img
    return None


def _shot_raw():
    """带自愈截图：exec-out -> 重连/重启 adb -> shell+pull 兜底。"""
    last_err = ""
    for attempt in range(1, 5):
        try:
            out, err = adb("exec-out", "screencap", "-p", timeout=20)
            if out:
                img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[0] > 1000:
                    return img
            last_err = str(err)[:100]
        except Exception as e:
            last_err = str(e)[:100]

        if not ensure_device(wait_s=8):
            log(f"截图失败#{attempt} 设备不可用: {last_err}")
        else:
            log(f"截图失败#{attempt} 重试: {last_err}")

        try:
            img = _screencap_via_file()
            if img is not None:
                return img
        except Exception as e:
            last_err = str(e)[:100]
        time.sleep(1.5)

    # 最后一招：重启 adb server 再抓一次
    try:
        subprocess.run(["adb", "kill-server"], capture_output=True, timeout=10)
        time.sleep(1)
        subprocess.run(["adb", "start-server"], capture_output=True, timeout=15)
        time.sleep(4)
        out, _ = adb("exec-out", "screencap", "-p", timeout=20)
        if out:
            img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
            if img is not None and img.shape[0] > 1000:
                log("adb server 重启后截图恢复")
                return img
    except Exception as e:
        last_err = str(e)[:100]

    raise RuntimeError(f"screencap failed: {last_err}")


def shot():
    """截图并刷新分辨率缩放系数（后续 ROI / 点击都按当前屏幕换算）。"""
    img = _shot_raw()
    layout.update_screen(img)
    return img


def tap(x, y):
    """点击实际像素坐标。"""
    adb("shell", "input", "tap", str(int(x)), str(int(y)))


def swipe_next_room():
    """在直播画面中部上滑，切到下一个直播间（避开顶部头像和底部评论栏）。"""
    x1, y1 = layout.sc_pt(540, 1950)
    x2, y2 = layout.sc_pt(540, 620)
    adb("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), "300")
    time.sleep(2.2)
