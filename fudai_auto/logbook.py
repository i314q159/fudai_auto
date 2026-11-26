# -*- coding: utf-8 -*-
"""日志三件套：运行日志（run_*.log）、事件表（fudai_log.csv）、事件截图（shots/）。"""
import csv
import os
from datetime import datetime

import cv2

from . import state
from .paths import LOG_CSV, SHOT_DIR


def log(msg):
    """打印并追加到当前运行日志；RUN_LOG 由 main.py 启动时设置。"""
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if state.RUN_LOG:
        try:
            with open(state.RUN_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def log_csv(event, detail):
    """写一行事件表：time,event,detail；文件不存在时自动写表头。"""
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    new = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "event", "detail"])
        w.writerow([f"{datetime.now():%Y-%m-%d %H:%M:%S}", event, detail])


def save(img, tag):
    """把当前截图存到 log/shots/（用于复盘：房间切换、开奖弹层、异常帧）。"""
    os.makedirs(SHOT_DIR, exist_ok=True)
    p = os.path.join(SHOT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{tag}.png")
    cv2.imwrite(p, img)
    return p
