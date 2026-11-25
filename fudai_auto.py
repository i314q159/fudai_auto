# -*- coding: utf-8 -*-
"""
抖音直播间 福袋自动蹲守（adb + OpenCV + OCR）——不区分直播间，跟随当前屏幕直播间

策略（不依赖「待参与」字样，避免卡在 00:00 错过新福袋）：
  每轮截图先 OCR 读取当前直播间标题，切换直播间后自动重置状态继续蹲
  每隔 N 秒打开一次福袋面板：
    - 面板出现「一键发表评论」 -> 点击完成福袋任务，即参与
    - 面板显示「任务已完成 等待开奖」 -> 说明已参与，关闭面板继续蹲
  读取面板倒计时，到点进入开奖观察窗口，识别中奖/未中奖并记录
  任何时刻都停留在直播间，不点关闭/返回
日志与截图统一输出到 log/ 目录
用法:
  python fudai_auto.py --minutes 180 --interval 30
"""

import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

SERIAL = "IB8XCEM7IJBAMZY5"
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "log")
SHOT_DIR = os.path.join(LOG_DIR, "shots")
LOG_CSV = os.path.join(LOG_DIR, "fudai_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

ICON_ROI = (0, 255, 400, 500)
PANEL_ROI = (0, 2150, 1080, 2400)
PANEL_HEAD_ROI = (760, 1560, 1080, 1690)  # 面板倒计时区域
PANEL_HEAD_ROI_WIDE = (0, 1420, 1080, 1760)
HEAD_ROI = (0, 95, 760, 230)
NAME_ROI = (110, 105, 700, 175)  # 直播间标题行
RESULT_ROI = (0, 600, 1080, 2000)

engine = RapidOCR()


RUN_LOG = None


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if RUN_LOG:
        try:
            with open(RUN_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def adb(*args, timeout=15):
    r = subprocess.run(
        ["adb", "-s", SERIAL] + list(args), capture_output=True, timeout=timeout
    )
    return r.stdout, r.stderr


def device_state():
    try:
        out = subprocess.run(
            ["adb", "devices"], capture_output=True, timeout=10
        ).stdout.decode("utf-8", "ignore")
    except Exception:
        return "absent"
    for line in out.splitlines()[1:]:
        if SERIAL in line:
            return "device" if "\tdevice" in line else "offline"
    return "absent"


def ensure_device(wait_s=15):
    t_end = time.time() + wait_s
    restarted = False
    while time.time() < t_end:
        st = device_state()
        if st == "device":
            return True
        if st == "offline":
            try:
                subprocess.run(
                    ["adb", "reconnect", "offline"], capture_output=True, timeout=10
                )
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


def shot():
    """带自愈截图：exec-out -> 重连/重启 adb -> shell+pull 兜底"""
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
            subprocess.run(
                [
                    "adb",
                    "-s",
                    SERIAL,
                    "shell",
                    "screencap",
                    "-p",
                    "/sdcard/_auto_shot.png",
                ],
                capture_output=True,
                timeout=20,
            )
            subprocess.run(
                [
                    "adb",
                    "-s",
                    SERIAL,
                    "pull",
                    "/sdcard/_auto_shot.png",
                    os.path.join(LOG_DIR, "_auto_shot.png"),
                ],
                capture_output=True,
                timeout=20,
            )
            p = os.path.join(LOG_DIR, "_auto_shot.png")
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None and img.shape[0] > 1000:
                    return img
        except Exception as e:
            last_err = str(e)[:100]
        time.sleep(1.5)

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


def tap(x, y):
    adb("shell", "input", "tap", str(int(x)), str(int(y)))


def ocr(img, min_score=0.5):
    res, _ = engine(img)
    items = []
    if res:
        for box, txt, score in res:
            if float(score) >= min_score:
                pts = np.array(box).reshape(-1, 2)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                items.append(
                    {
                        "text": txt,
                        "score": float(score),
                        "x": float(x0),
                        "y": float(y0),
                        "w": float(x1 - x0),
                        "h": float(y1 - y0),
                    }
                )
    return items


def texts_of(items):
    return " ".join(i["text"] for i in items)


def crop(img, roi):
    return img[roi[1] : roi[3], roi[0] : roi[2]]


BAG_TPL_PATH = os.path.join(ROOT, "bag_template.png")
BAG_TPL = cv2.imread(BAG_TPL_PATH) if os.path.exists(BAG_TPL_PATH) else None
TIME_RE = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}")


def find_bag(img):
    """定位福袋图标：模板匹配优先，失败则用 OCR 倒计时标签反推"""
    roi = crop(img, ICON_ROI)
    if BAG_TPL is not None:
        best = (0.0, None, None)
        for s in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
            t = cv2.resize(BAG_TPL, None, fx=s, fy=s)
            if t.shape[0] >= roi.shape[0] or t.shape[1] >= roi.shape[1]:
                continue
            r = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mxloc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (mx, mxloc, t.shape)
        if best[0] >= 0.75:
            mx, loc, shape = best
            cx = ICON_ROI[0] + loc[0] + shape[1] // 2
            cy = ICON_ROI[1] + loc[1] + int(shape[0] * 0.6)
            return cx, cy, f"tpl:{best[0]:.2f}"
    for i in ocr(roi, 0.45):
        if TIME_RE.search(i["text"]):
            cx = ICON_ROI[0] + i["x"] + i["w"] / 2
            cy = ICON_ROI[1] + i["y"] - 25
            return cx, cy, f"ocr:{i['text']}"
    return None


def pink_blob(roi_img):
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (140, 60, 120), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 35:
            continue
        if not (0.5 < w / h < 2.0):
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)
    return best


def pink_button_box(roi_img):
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (165, 120, 170), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 40000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 500:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)
    return best


def save(img, tag):
    os.makedirs(SHOT_DIR, exist_ok=True)
    p = os.path.join(SHOT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{tag}.png")
    cv2.imwrite(p, img)
    return p


def log_csv(event, detail):
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    new = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "event", "detail"])
        w.writerow([f"{datetime.now():%Y-%m-%d %H:%M:%S}", event, detail])


def parse_countdown(text):
    """兼容 OCR 丢失冒号、拆数字的情况"""
    norm = text.replace(" ", "")
    m = re.search(r"(\d{1,2})[:：](\d{1,2})", norm)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if mm <= 60 and ss <= 59:
            return mm * 60 + ss
    # 兜底：截取「已参与」之后的数字（面板右侧倒计时）
    tail = norm.split("已参与")[-1]
    nums = re.findall(r"\d{1,2}", tail)
    if len(nums) >= 2:
        mm, ss = int(nums[0]), int(nums[1])
        if mm <= 60 and ss <= 59:
            return mm * 60 + ss
    return None


def read_remaining(img):
    """两级 ROI 读取倒计时"""
    rem = parse_countdown(texts_of(ocr(crop(img, PANEL_HEAD_ROI), 0.35)))
    if rem is None:
        rem = parse_countdown(texts_of(ocr(crop(img, PANEL_HEAD_ROI_WIDE), 0.35)))
    return rem


def get_room_name(img):
    """从直播间头部读取当前直播间名称"""
    items = ocr(crop(img, NAME_ROI), 0.5)
    if not items:
        return None
    items.sort(key=lambda i: i["h"], reverse=True)
    for i in items:
        t = i["text"].strip().replace("\u3000", "")
        # 归一化：只保留文字/数字，避免 OCR 标点抖动导致误判换房
        t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t)
        if len(t) >= 3:
            return t
    return None


def panel_state(img):
    roi = crop(img, PANEL_ROI)
    t = texts_of(ocr(roi, 0.45))
    if "一键发表评论" in t:
        return "join", t
    if ("请勿离开直播间" in t) or ("等待开奖" in t) or ("任务已完成" in t):
        return "wait", t
    return None, t


def close_panel():
    tap(540, 1150)
    time.sleep(0.8)


def handle_panel(img):
    """处理已打开的福袋面板，返回 (joinstatus, remaining, detail)"""
    roi = crop(img, PANEL_ROI)
    btn = pink_button_box(roi)
    t = texts_of(ocr(roi, 0.45))
    rem = read_remaining(img)

    if btn:
        _, x, y, w, h = btn
        cx, cy = PANEL_ROI[0] + x + w // 2, PANEL_ROI[1] + y + h // 2
        log(f"点「一键发表评论」完成福袋任务 ({cx},{cy})")
        tap(cx, cy)
        time.sleep(1.4)
        p = shot()
        save(p, "joined")
        rem2 = read_remaining(p)
        return "joined_now", rem2 or rem, "comment"
    if "一键发表评论" in t:
        for i in ocr(roi, 0.45):
            if "一键发表评论" in i["text"]:
                tap(
                    PANEL_ROI[0] + i["x"] + i["w"] / 2,
                    PANEL_ROI[1] + i["y"] + i["h"] / 2 + 30,
                )
                time.sleep(1.4)
                p = shot()
                save(p, "joined_text")
                rem2 = read_remaining(p)
                return "joined_now", rem2 or rem, "comment_text"
    if ("请勿离开直播间" in t) or ("等待开奖" in t) or ("任务已完成" in t):
        log(f"该福袋已是参与状态，等待开奖（剩余 {rem}s）")
        close_panel()
        return "already", rem, "wait"
    log(f"面板状态未知: {t[:50]}")
    save(img, "panel_unknown")
    close_panel()
    return "unknown", rem, t[:40]


def open_bag_panel(img):
    """点击福袋图标（袋身上部，避开倒计时条），返回新截图或 None"""
    pos = find_bag(img)
    if not pos:
        return None, "no-icon"
    cx, cy, src = pos
    tap(cx, cy)
    time.sleep(1.4)
    img2 = shot()
    if not panel_state(img2)[0]:
        log(f"首次点击未打开面板({src})，重点一次 ({cx},{cy + 18})")
        tap(cx, cy + 18)
        time.sleep(1.4)
        img2 = shot()
    return img2, f"tap({int(cx)},{int(cy)}) {src}"


def watch_result(seconds=80):
    t_end = time.time() + seconds
    last = 0
    while time.time() < t_end:
        img = shot()
        t = texts_of(ocr(crop(img, RESULT_ROI), 0.45))
        hits = [
            k
            for k in ("恭喜", "中奖", "获得", "很遗憾", "未中奖", "差一点", "已开奖")
            if k in t
        ]
        if hits:
            p = save(img, "result")
            log(f"开奖窗口命中 {hits}: {t[:80]}")
            log_csv("result", f"{hits} | {t[:120]}")
            # 尝试点掉弹窗按钮
            for i in ocr(crop(img, RESULT_ROI), 0.4):
                if i["text"] in ("确定", "知道了", "我知道了", "开心收下", "关闭"):
                    tap(
                        RESULT_ROI[0] + i["x"] + i["w"] / 2,
                        RESULT_ROI[1] + i["y"] + i["h"] / 2,
                    )
                    time.sleep(0.8)
                    break
            return
        if time.time() - last > 20:
            last = time.time()
            log(f"等待开奖中… {t[:40]}")
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--interval", type=float, default=60, help="检查福袋的间隔秒数")
    ap.add_argument(
        "--room",
        default="",
        help="可选：期望的直播间标题关键字，留空则只跟随当前直播间",
    )
    ap.add_argument("--serial", default=None, help="adb 设备序列号")
    args = ap.parse_args()

    global SERIAL, RUN_LOG
    if args.serial:
        SERIAL = args.serial
    RUN_LOG = os.path.join(LOG_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")

    # 单实例锁，防止多个进程抢点击
    import socket

    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 59999))
        lock.listen(1)
    except OSError:
        log("已有实例在运行，本进程退出")
        return

    t_end = time.time() + args.minutes * 60
    bag_count = 0
    cycle_no = 0
    next_draw_ts = 0.0
    draw_watched = True
    cur_room = None
    room_warned = False
    last_room_seen = time.time()
    last_err_log = 0.0
    last_rem = None
    last_rem_ts = 0.0
    cycle_count = 1

    def track_cycle(rem):
        """通过倒计时‘回跳’识别新一轮福袋"""
        nonlocal last_rem, last_rem_ts, cycle_count
        if rem is None:
            return
        now = time.time()
        if last_rem is not None and last_rem_ts:
            expect = max(0, last_rem - (now - last_rem_ts))
            if rem > expect + 90:
                cycle_count += 1
                log(f"检测到新一轮福袋（第 {cycle_count} 轮），剩余 {rem}s 开奖")
                log_csv("new_cycle", f"#{cycle_count} rem={rem}")
        last_rem, last_rem_ts = rem, now

    log(f"启动福袋蹲守 {args.minutes} 分钟，检查间隔 {args.interval}s")
    log_csv("start", f"minutes={args.minutes} interval={args.interval}")

    while time.time() < t_end:
        try:
            img = shot()
        except Exception as e:
            if time.time() - last_err_log > 30:
                last_err_log = time.time()
                log(f"截图异常(自动重试): {e}")
                log_csv("shot_error", str(e)[:80])
            time.sleep(10)
            continue

        cycle_no += 1

        # 抓当前直播间名称：切换房间自动适配
        name = get_room_name(img)
        if name:
            last_room_seen = time.time()
            if cur_room is None:
                cur_room = name
                log(f"当前直播间: {name}")
                log_csv("room", name)
            elif name != cur_room:
                log(
                    f"直播间切换: {cur_room} -> {name}，状态已重置，开始在新直播间蹲福袋"
                )
                log_csv("room_switch", f"{cur_room} -> {name}")
                cur_room = name
                next_draw_ts = 0.0
                draw_watched = True
                last_rem = None
                last_rem_ts = 0.0
                cycle_count = 1
                save(img, "room_switch")
                time.sleep(2)
                continue
            if args.room and args.room not in name and not room_warned:
                log(
                    f"提示: 当前直播间为「{name}」，与 --room「{args.room}」不一致；已按当前直播间继续"
                )
                room_warned = True
        elif time.time() - last_room_seen > 180:
            log("警告: 连续 3 分钟未识别到直播间标题，可能已离开直播间")
            log_csv("warn_room", "no-title")
            last_room_seen = time.time()
            save(img, "no_room_title")

        # 有开奖弹窗先记录
        rt = texts_of(ocr(crop(img, RESULT_ROI), 0.45))
        if any(k in rt for k in ("恭喜", "很遗憾", "未中奖", "中奖")):
            save(img, "result_popup")
            log(f"识别到开奖文案: {rt[:60]}")
            log_csv("result", rt[:120])

        # 面板是否已打开
        st, t = panel_state(img)
        if st:
            status, rem, detail = handle_panel(img)
            if status == "joined_now":
                bag_count += 1
                log(f"第 {bag_count} 次参与成功({detail})")
                log_csv("join", f"{cur_room} #{bag_count} {detail} remaining={rem}")
                if rem is not None:
                    next_draw_ts = time.time() + rem
                    draw_watched = False
            elif rem is not None:
                next_draw_ts = time.time() + rem
                draw_watched = False
            else:
                next_draw_ts = time.time() + 750
                draw_watched = False
            track_cycle(rem)
            time.sleep(1.0)
        else:
            # 打开福袋检查
            img2, info = open_bag_panel(img)
            if img2 is None:
                log(f"未见福袋图标: {info}")
            else:
                st2, _ = panel_state(img2)
                if not st2:
                    log(f"打开福袋后面板未识别，关闭 ({info})")
                    close_panel()
                else:
                    status, rem, detail = handle_panel(img2)
                    if status == "joined_now":
                        bag_count += 1
                        log(f"第 {bag_count} 次参与成功({detail})")
                        log_csv(
                            "join", f"{cur_room} #{bag_count} {detail} remaining={rem}"
                        )
                        if rem is not None:
                            next_draw_ts = time.time() + rem
                            draw_watched = False
                    elif rem is not None:
                        next_draw_ts = time.time() + rem
                        draw_watched = False
                        log(f"当前福袋已参与，剩余 {rem}s 开奖")
                    else:
                        next_draw_ts = time.time() + 750
                        draw_watched = False
                        log("未读到倒计时，按默认 12.5 分钟排开奖")
                    track_cycle(rem)

        # 开奖窗口
        if (not draw_watched) and next_draw_ts and time.time() >= next_draw_ts - 3:
            log(f"进入开奖观察窗口（第 {cycle_count} 轮）")
            watch_result(90)
            draw_watched = True

        # 睡眠：按剩余时间自适应，临近开奖打短，避免错过开奖瞬间
        gap = args.interval
        if (not draw_watched) and last_rem is not None:
            if last_rem > 600:
                gap = 180
            elif last_rem > 240:
                gap = 90
            else:
                gap = max(20, args.interval)
        sleep_s = gap
        if (not draw_watched) and next_draw_ts:
            sleep_s = max(1.5, min(sleep_s, next_draw_ts - time.time() - 2))
        time.sleep(sleep_s)

    log(f"结束，共参与 {bag_count} 次福袋")
    log_csv("end", f"joined={bag_count}")


if __name__ == "__main__":
    main()
