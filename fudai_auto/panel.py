# -*- coding: utf-8 -*-
"""福袋面板：倒计时、状态判定、任务点击、面板开关与样本入库。

典型面板状态：
  join  还能参与（底部有粉色按钮，或正文提示加入粉丝团/关注主播）
  wait  已参与（提示「请勿离开直播间 / 等待开奖 / 任务已完成」）
任务点击最多连做 3 步（加入粉丝团 -> 关注 -> 一键发表评论），
遇到抖币/支付确认立即停手，绝不自动付费。
"""
import os
import time

import cv2
import numpy as np

from .dataset import save_dataset
from .detector import find_bag
from .device import shot, tap
from .layout import (
    ICON_ROI, PANEL_FALLBACK_ROI, PANEL_HEAD_ROI, PANEL_HEAD_ROI_WIDE,
    PANEL_ROI, PAY_ROI, SCALE, crop, sc_pt, sc_roi,
)
from .logbook import log, save
from .ocrkit import TIME_RE, ocr, parse_countdown, texts_of

# 面板「已参与」类文案
WAIT_KEYS = ("请勿离开直播间", "等待开奖", "任务已完成")
# 面板正文里的「可参与」类文案（按钮不在粉色条位置时也能识别）
JOIN_KEYS = ("加入购物粉丝团", "粉丝团并关注", "关注主播")
# 兜底确认按钮的文字（半屏搜索）
CONFIRM_KEYS = (
    "加入购物粉丝团并关注主播",
    "加入购物粉丝团",
    "加入粉丝团",
    "粉丝团并关注",
    "立即加入",
    "确认加入",
    "确认关注",
    "立即关注",
)
# 付费确认关键词：命中即暂停自动点击
PAY_KEYS = ("抖币", "确认支付", "立即支付", "去支付")


def pink_button_box(roi_img):
    """在 ROI 里找底部粉色大按钮，返回 (面积, x, y, w, h) 或 None。"""
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


def read_remaining(img):
    """两级 ROI 读取面板倒计时（秒）；读不到返回 None。"""
    rem = parse_countdown(texts_of(ocr(crop(img, sc_roi(PANEL_HEAD_ROI)), 0.35)))
    if rem is None:
        rem = parse_countdown(
            texts_of(ocr(crop(img, sc_roi(PANEL_HEAD_ROI_WIDE)), 0.35))
        )
    return rem


def panel_state(img):
    """判断福袋面板状态：('join'|'wait'|None, OCR 文本列表)。"""
    roi = crop(img, sc_roi(PANEL_ROI))
    t = texts_of(ocr(roi, 0.45))
    if "一键发表评论" in t:
        return "join", t
    if any(k in t for k in JOIN_KEYS):
        return "join", t
    if any(k in t for k in WAIT_KEYS):
        return "wait", t
    return None, t


def close_panel():
    """点面板外的空白处关闭面板（约屏幕中部）。"""
    tap(*sc_pt(540, 1150))
    time.sleep(0.8)


def handle_panel(img):
    """处理福袋面板：返回 (status, remaining, detail)。

    status:
      joined_now   本次有真实点击（完成任务/加入粉丝团）
      already      打开时已参与，没点任何按钮
      need_payment 触发了付费确认，已停手
      unknown      面板已关闭但没识别到有效状态
    detail 用于日志/CSV，记录点过的按钮文字。
    """
    rem = read_remaining(img)
    clicked = []
    steps = 0
    panel_px = sc_roi(PANEL_ROI)
    pay_px = sc_roi(PAY_ROI)

    while steps < 3:
        roi = crop(img, panel_px)
        btn = pink_button_box(roi)
        t = texts_of(ocr(roi, 0.45))
        rem = read_remaining(img) or rem

        # 情况 1：底部有粉色任务按钮 -> 点一下，再截图检查是否弹付费
        if btn:
            _, x, y, w, h = btn
            cx = panel_px[0] + x + w // 2
            cy = panel_px[1] + y + h // 2
            log(f"面板按钮「{t[:24]}」-> 点击 ({cx},{cy})")
            tap(cx, cy)
            time.sleep(1.5)
            img = shot()
            pay = texts_of(ocr(crop(img, pay_px), 0.4))
            if any(k in pay for k in PAY_KEYS):
                log(f"检测到付费确认界面，暂停不自动支付: {pay[:40]}")
                save(img, "need_payment")
                tap(*sc_pt(540, 700))
                time.sleep(0.6)
                return "need_payment", rem, "payment"
            steps += 1
            clicked.append(t[:20])
            continue

        # 情况 2：显示已参与 -> 关面板，交给主循环排开奖时间
        if any(k in t for k in WAIT_KEYS):
            log(f"该福袋已参与，等待开奖（剩余 {rem if rem is not None else '未知'}s）")
            close_panel()
            return ("joined_now" if clicked else "already"), rem, ";".join(clicked) or "wait"

        # 情况 3：兜底——确认类弹层，文字按钮不在底部粉色条位置
        wide_roi = sc_roi(PANEL_FALLBACK_ROI)
        wide_items = ocr(crop(img, wide_roi), 0.3)
        hit = None
        for kw in CONFIRM_KEYS:
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
            pay = texts_of(ocr(crop(img, pay_px), 0.4))
            if any(k in pay for k in PAY_KEYS):
                log("确认层触发付费，已暂停")
                tap(*sc_pt(540, 700))
                return "need_payment", rem, "payment"
            steps += 1
            clicked.append(hit["text"][:16])
            continue

        # 情况 4：面板已经消失
        log(f"面板已关闭，任务处理结束: {t[:40]}")
        return ("joined_now" if clicked else "unknown"), rem, ";".join(clicked) or t[:40]

    log("面板任务步骤已达上限，停止继续点击")
    close_panel()
    return ("joined_now" if clicked else "unknown"), rem, ";".join(clicked)


def open_bag_panel(img):
    """点福袋图标打开面板，返回 (新截图或 None, 说明)。

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

    # 第一次没打开：图标下半部才是可点区域，往下挪一点再点一次
    if not panel_state(img2)[0]:
        dy = int(round(18 * SCALE["y"]))
        log(f"首次点击未打开面板({src})，重点一次 ({cx},{cy + dy})")
        tap(cx, cy + dy)
        time.sleep(1.4)
        img2 = shot()

    if not panel_state(img2)[0]:
        log(f"未确认福袋面板，不写入样本 ({src})")
        return img2, f"tap({int(cx)},{int(cy)}) {src} unverified"

    # 面板已确认，才把这张图作为正样本入库
    try:
        roi_txt = texts_of(ocr(crop(img, sc_roi(ICON_ROI)), 0.45))
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
