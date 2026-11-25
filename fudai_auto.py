# -*- coding: utf-8 -*-
"""
抖音直播间 福袋自动蹲守（adb + OpenCV + OCR）——不区分直播间，跟随当前屏幕直播间

本文件：底层能力库（截图/OCR/福袋定位/面板处理/样本采集）
主循环与 CLI 已迁到 main.py

策略（不依赖「待参与」字样，避免卡在 00:00 错过新福袋）：
  每轮截图先 OCR 读取当前直播间标题，切换直播间后自动重置状态继续蹲
  每隔 N 秒打开一次福袋面板：
    - 面板出现「一键发表评论」 -> 点击完成福袋任务，即参与
    - 面板显示「任务已完成 等待开奖」 -> 说明已参与，关闭面板继续蹲
  读取面板倒计时，到点进入开奖观察窗口，识别中奖/未中奖并记录
  任何时刻都停留在直播间，不点关闭/返回
日志与截图统一输出到 log/ 目录
用法：
  python main.py --minutes 180 --interval 30
"""

import csv
import os
import re
import subprocess
import time
import tomllib
from datetime import datetime

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

import adb_conn
import fudai_feat

ADB_DEVICE_SERIAL = adb_conn.get_adb_device_serial_via_subprocess()

SERIAL = os.getenv("FUDAI_SERIAL", ADB_DEVICE_SERIAL)
ROOT = os.path.dirname(os.path.abspath(__file__))

# 检测节奏全部集中在 config.toml；这里只是文件缺失/缺键时的兜底
DEFAULT_CONFIG = {
    "run": {"minutes": 180.0, "interval": 60.0},
    "cold_start": {"below": 30},
    "detect": {
        "shot_retry_seconds": 10.0,
        "shot_error_log_every_seconds": 30.0,
        "room_check_seconds": 5.0,
        "room_similarity": 0.86,
        "room_missing_warn_seconds": 180.0,
        "default_draw_seconds": 750.0,
        "new_cycle_jump_seconds": 90.0,
        "watch_seconds": 90.0,
        "draw_margin_seconds": 3.0,
        "panel_settle_seconds": 1.0,
    },
    "pacing": {
        "slow_remaining_seconds": 600.0,
        "slow_interval": 180.0,
        "mid_remaining_seconds": 240.0,
        "mid_interval": 90.0,
        "near_min_seconds": 20.0,
        "min_sleep_seconds": 1.5,
    },
    "sample": {"throttle_seconds": 60.0},
    "watch": {"poll_seconds": 3.0, "log_every_seconds": 20.0},
}
CONFIG_PATH = os.path.join(ROOT, "config.toml")


def _load_config():
    cfg = {sec: dict(vals) for sec, vals in DEFAULT_CONFIG.items()}
    if not os.path.exists(CONFIG_PATH):
        return cfg
    try:
        with open(CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        print(f"[config] 读取 {CONFIG_PATH} 失败，使用内置默认值: {e}")
        return cfg
    for sec, vals in raw.items():
        if sec not in cfg or not isinstance(vals, dict):
            print(f"[config] 忽略未知段 [{sec}]")
            continue
        for key, val in vals.items():
            if key in cfg[sec]:
                cfg[sec][key] = val
            else:
                print(f"[config] 忽略未知键 [{sec}].{key}")
    return cfg


CONFIG = _load_config()


def cfg(section, key):
    """读取配置；缺段/缺键时回落到内置默认值"""
    return CONFIG.get(section, {}).get(key, DEFAULT_CONFIG[section][key])


LOG_DIR = os.path.join(ROOT, "log")
SHOT_DIR = os.path.join(LOG_DIR, "shots")
LOG_CSV = os.path.join(LOG_DIR, "fudai_log.csv")
DATASET_DIR = os.path.join(ROOT, "data", "fudai")
DATASET_ICONS = os.path.join(DATASET_DIR, "icons")
DATASET_LABELS = os.path.join(DATASET_DIR, "labels.csv")
os.makedirs(LOG_DIR, exist_ok=True)

ROOM_NAME = None
_dataset_state = {"ts": 0.0, "label": None, "crop": None}

ICON_ROI = (0, 250, 420, 540)
PANEL_ROI = (0, 2150, 1080, 2400)
PANEL_HEAD_ROI = (760, 1560, 1080, 1690)  # 面板倒计时区域
PANEL_HEAD_ROI_WIDE = (0, 1420, 1080, 1760)
HEAD_ROI = (0, 95, 760, 230)
NAME_ROI = (110, 105, 700, 175)  # 直播间标题行
RESULT_ROI = (0, 600, 1080, 2000)
FOLLOW_ROI = (390, 115, 720, 200)  # 主播「关注」按钮区域
PAY_ROI = (0, 900, 1080, 2200)  # 付费弹窗检测区

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


BAG_TPL_PATH = next(
    (
        p
        for p in (
            os.path.join(DATASET_DIR, "bag_template.png"),
            os.path.join(ROOT, "bag_template.png"),
        )
        if os.path.exists(p)
    ),
    os.path.join(ROOT, "bag_template.png"),
)
BAG_TPL = cv2.imread(BAG_TPL_PATH) if os.path.exists(BAG_TPL_PATH) else None
TIME_RE = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}")
CLF_PATH = os.path.join(DATASET_DIR, "model", "fudai_clf.npz")
CLF = None


def load_clf():
    """读取检测模型；换机/重训后可热重载，无需重启进程"""
    global CLF
    CLF = None
    try:
        if os.path.exists(CLF_PATH):
            _m = np.load(CLF_PATH)
            CLF = {
                "w": _m["w"].astype(np.float32),
                "thresh": max(float(_m["thresh"]), 0.12),
                "win_w": int(_m["win_w"]),
                "win_h": int(_m["win_h"]),
            }
    except Exception:
        CLF = None
    return CLF


def reload_clf():
    return load_clf()


load_clf()


def clf_locate(img):
    """用 data/fudai/model/fudai_clf.npz 在福袋区滑窗分类，返回最佳命中或 None"""
    if CLF is None:
        return None
    hit = fudai_feat.locate(img, ICON_ROI, CLF, CLF["thresh"], stride=8)
    if not hit:
        return None
    cx, cy, s = hit
    return cx, cy, f"clf:{s:.2f}"


def find_bag(img):
    """定位福袋图标：分类模型优先，其次模板匹配，最后用 OCR 倒计时标签反推"""
    hit = clf_locate(img)
    if hit:
        return hit
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
    cands = [i for i in ocr(roi, 0.45) if TIME_RE.search(i["text"])]
    # 福袋倒计时贴在最左侧图标下方；过滤掉榜单/商品卡里的其它倒计时
    cands.sort(key=lambda i: (i["x"], i["y"]))
    for i in cands:
        if i["x"] >= 220:
            continue
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
    if ("加入购物粉丝团" in t) or ("粉丝团并关注" in t) or ("关注主播" in t):
        return "join", t
    if ("请勿离开直播间" in t) or ("等待开奖" in t) or ("任务已完成" in t):
        return "wait", t
    return None, t


def close_panel():
    tap(540, 1150)
    time.sleep(0.8)


def follow_anchor(img):
    """检查主播关注按钮并关注（仅头部小区域，已关注则跳过）"""
    items = ocr(crop(img, FOLLOW_ROI), 0.45)
    t = texts_of(items)
    if "已关注" in t:
        return "already"
    for i in items:
        if "关注" in i["text"]:
            cx = FOLLOW_ROI[0] + i["x"] + i["w"] / 2
            cy = FOLLOW_ROI[1] + i["y"] + i["h"] / 2
            tap(cx, cy)
            time.sleep(1.2)
            log(f"已点击关注主播 ({int(cx)},{int(cy)})")
            return "followed"
    return None


def handle_panel(img):
    """处理福袋面板：可连续完成多步任务（加入粉丝团/关注主播/发表评论）"""
    rem = read_remaining(img)
    clicked = []
    steps = 0
    while steps < 3:
        roi = crop(img, PANEL_ROI)
        btn = pink_button_box(roi)
        t = texts_of(ocr(roi, 0.45))
        rem = read_remaining(img) or rem

        if btn:
            _, x, y, w, h = btn
            cx = PANEL_ROI[0] + x + w // 2
            cy = PANEL_ROI[1] + y + h // 2
            log(f"面板按钮「{t[:24]}」-> 点击 ({cx},{cy})")
            tap(cx, cy)
            time.sleep(1.5)
            img = shot()
            pay = texts_of(ocr(crop(img, PAY_ROI), 0.4))
            if any(k in pay for k in ("抖币", "确认支付", "立即支付", "去支付")):
                log(f"检测到付费确认界面，暂停不自动支付: {pay[:40]}")
                save(img, "need_payment")
                tap(540, 700)
                time.sleep(0.6)
                return "need_payment", rem, "payment"
            steps += 1
            clicked.append(t[:20])
            continue

        if ("请勿离开直播间" in t) or ("等待开奖" in t) or ("任务已完成" in t):
            log(f"该福袋已参与，等待开奖（剩余 {rem}s）")
            close_panel()
            return (
                ("joined_now" if clicked else "already"),
                rem,
                ";".join(clicked) or "wait",
            )

        # 兜底：确认类弹层（文字按钮不在底部粉色条位置）
        wide_roi = (0, 1300, 1080, 2400)
        wide_items = ocr(crop(img, wide_roi), 0.3)
        hit = None
        for kw in (
            "加入购物粉丝团并关注主播",
            "加入购物粉丝团",
            "加入粉丝团",
            "粉丝团并关注",
            "立即加入",
            "确认加入",
            "确认关注",
            "立即关注",
        ):
            hit = next((i for i in wide_items if kw in i["text"]), None)
            if hit:
                break
        if hit:
            cx = hit["x"] + hit["w"] / 2
            cy = wide_roi[1] + hit["y"] + hit["h"] / 2
            log(f"兜底点击「{hit['text']}」({int(cx)},{int(cy)})")
            tap(cx, cy)
            time.sleep(1.5)
            img = shot()
            pay = texts_of(ocr(crop(img, PAY_ROI), 0.4))
            if any(k in pay for k in ("抖币", "确认支付", "立即支付", "去支付")):
                log("确认层触发付费，已暂停")
                tap(540, 700)
                return "need_payment", rem, "payment"
            steps += 1
            clicked.append(hit["text"][:16])
            continue

        log(f"面板已关闭，任务处理结束: {t[:40]}")
        return (
            ("joined_now" if clicked else "unknown"),
            rem,
            ";".join(clicked) or t[:40],
        )

    log("面板任务步骤已达上限，停止继续点击")
    close_panel()
    return ("joined_now" if clicked else "unknown"), rem, ";".join(clicked)


def save_dataset(img, cx, cy, label, src):
    """把成功识别到的福袋图标裁成样本，作为训练数据（节流 + 去重）"""
    now = time.time()
    st = _dataset_state
    if st["label"] == label and now - st["ts"] < float(
        cfg("sample", "throttle_seconds")
    ):
        return None
    x0 = max(0, int(cx) - 60)
    y0 = max(0, int(cy) - 45)
    x1 = min(img.shape[1], x0 + 120)
    y1 = min(img.shape[0], y0 + 90)
    crop_img = img[y0:y1, x0:x1].copy()
    if crop_img.size == 0:
        return None
    if st["crop"] is not None and st["crop"].shape == crop_img.shape:
        diff = float(np.mean(cv2.absdiff(crop_img, st["crop"])))
        if diff < 2.0 and st["label"] == label:
            return None
    os.makedirs(DATASET_ICONS, exist_ok=True)
    tag = re.sub(r"[^\w\u4e00-\u9fff]", "-", str(label))[:24]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(DATASET_ICONS, f"{ts}_{tag}.png")
    cv2.imwrite(p, crop_img)
    new = not os.path.exists(DATASET_LABELS)
    with open(DATASET_LABELS, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "room", "label", "source", "bbox", "file"])
        w.writerow(
            [
                f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                ROOM_NAME or "",
                label,
                src,
                f"{x0},{y0},{x1},{y1}",
                os.path.relpath(p, ROOT),
            ]
        )
    st.update(ts=now, label=label, crop=crop_img)
    return p


def open_bag_panel(img):
    """点击福袋图标（袋身上部，避开倒计时条），返回新截图或 None

    注意：样本只在「点击后面板真正打开」时才写入数据集，
    避免误检（券图标、头像等）污染训练数据。
    """
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
    if not panel_state(img2)[0]:
        log(f"未确认福袋面板，不写入样本 ({src})")
        return img2, f"tap({int(cx)},{int(cy)}) {src} unverified"

    # 面板已确认，才把这张图作为正样本入库
    try:
        roi_txt = texts_of(ocr(crop(img, ICON_ROI), 0.45))
        if "待参与" in roi_txt:
            label = "待参与"
        elif "已参与" in roi_txt:
            label = "已参与"
        else:
            m = TIME_RE.search(roi_txt)
            if m:
                label = m.group(0).replace(" ", "")
            elif src.startswith("ocr:"):
                label = src.split(":", 1)[1].replace(" ", "")
            else:
                label = "icon_only"
        saved = save_dataset(img, cx, cy, label, src)
        if saved:
            log(f"已保存福袋样本: {os.path.basename(saved)} ({label})")
    except Exception as e:
        log(f"样本保存失败(忽略): {e}")
    return img2, f"tap({int(cx)},{int(cy)}) {src}"


def watch_result(seconds=None):
    if seconds is None:
        seconds = float(cfg("detect", "watch_seconds"))
    t_end = time.time() + seconds
    last = 0
    poll = float(cfg("watch", "poll_seconds"))
    log_every = float(cfg("watch", "log_every_seconds"))
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
        if time.time() - last > log_every:
            last = time.time()
            log(f"等待开奖中… {t[:40]}")
        time.sleep(poll)


if __name__ == "__main__":
    # 兼容旧入口：实际主循环已迁移到 main.py
    from main import main as _main

    _main()
