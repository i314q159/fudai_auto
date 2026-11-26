# -*- coding: utf-8 -*-
"""开奖结果弹层：识别、统一关闭、开奖窗口观察。

中奖卡与未中奖卡共用一套关闭逻辑：
  文字按钮（我知道了/确定/开心收下…） -> OCR 找 × -> 固定关闭位（约 (540,1704)）
  -> 都点不掉就等它自己消失（两种卡都会定时自动关闭，**不按返回键**，避免退出直播间）
故意不点「填写地址 / 领取奖品」，避免跳到领奖页离开直播间。
"""
import time

from .config import cfg
from .device import shot, tap
from .layout import (
    RESULT_ACTION_ROI, RESULT_BTN_TEXTS, RESULT_KEYS, RESULT_ROI, WIN_CLOSE_POINTS,
    crop, sc_pt, sc_roi, sx, sy,
)
from .logbook import log, log_csv, save
from .ocrkit import ocr, texts_of


def result_present(text):
    """是否出现开奖/中奖弹层文案（text 可以是字符串或 ocr 文本列表）。"""
    return any(k in text for k in RESULT_KEYS)


def dismiss_result(img, text=None):
    """关闭中奖 / 未中奖弹层，返回 (关闭后的截图, 是否已关闭)。

    最多走 len(WIN_CLOSE_POINTS) 轮：每轮先找文字按钮，再找 × 图标，
    都没有就依次点固定关闭位；还关不掉就等它自动关闭（不按返回键）。
    """
    roi_px = sc_roi(RESULT_ROI)
    if text is None:
        text = texts_of(ocr(crop(img, roi_px), 0.45))

    for step in range(len(WIN_CLOSE_POINTS)):
        if not result_present(text):
            return img, True

        items = ocr(crop(img, roi_px), 0.4)
        hit = None
        # 1) 优先点关闭类文字按钮
        for kw in RESULT_BTN_TEXTS:
            hit = next((i for i in items if kw in i["text"]), None)
            if hit:
                break
        # 2) 再找 × / X（限制尺寸，避免把别的小图标当成关闭按钮）
        if hit is None:
            hit = next(
                (
                    i
                    for i in items
                    if i["text"].strip() in ("×", "X", "x", "✕", "✖")
                    and sx(18) <= i["w"] <= sx(100)
                    and sy(18) <= i["h"] <= sy(100)
                ),
                None,
            )

        if hit is not None:
            tap(
                roi_px[0] + hit["x"] + hit["w"] / 2,
                roi_px[1] + hit["y"] + hit["h"] / 2,
            )
            log(f"开奖弹层点击「{hit['text'][:10]}」")
            # 文字按钮（未中奖卡的「我知道了」）点下去卡片关得快，少等一会儿
            settle = 0.7
        else:
            # 3) 固定关闭位（中奖卡的圆形 × 是图形，OCR 常常读不出文字）
            x, y = sc_pt(*WIN_CLOSE_POINTS[step])
            tap(x, y)
            log(f"开奖弹层点击关闭位 ({x},{y})")
            settle = 1.2

        time.sleep(settle)
        img = shot()
        text = texts_of(ocr(crop(img, roi_px), 0.4))

    if not result_present(text):
        return img, True

    # 4) 点不掉就等它自动关闭：中奖卡/未中奖卡都会自己消失，
    #    按返回键可能直接退出直播间，所以不再兜底返回键
    return _wait_auto_close(img, roi_px)


def _wait_auto_close(img, roi_px):
    """轮询等开奖弹层自己消失，返回 (最后一张截图, 是否已关闭)。"""
    wait_s = float(cfg("detect", "result_auto_close_seconds"))
    end = time.time() + wait_s
    log(f"开奖弹层没点掉，等它自动关闭（最多 {wait_s:g}s）")
    while True:
        if time.time() >= end:
            log("开奖弹层仍未自动关闭，留到下一轮")
            return img, False
        time.sleep(1.5)
        img = shot()
        if not result_present(texts_of(ocr(crop(img, roi_px), 0.4))):
            log("开奖弹层已自动关闭，回到福袋检测")
            return img, True


def watch_result(seconds=None):
    """开奖观察窗口：轮询截图等待中奖/未中奖文案，命中就关闭并记录。

    返回前会把截图和 CSV 事件写好，供复盘（result / result_dismiss）。
    轮询只识别卡片按钮那一条（RESULT_ACTION_ROI）：没有按钮就说明卡片还没出，
    比每次识别整块 RESULT_ROI 快 3~4 倍；命中后再用整块区域去关卡片。
    """
    if seconds is None:
        seconds = float(cfg("detect", "watch_seconds"))
    t_end = time.time() + seconds
    last = 0
    poll = float(cfg("watch", "poll_seconds"))
    log_every = float(cfg("watch", "log_every_seconds"))
    action_px = sc_roi(RESULT_ACTION_ROI)
    last_text = ""

    while time.time() < t_end:
        img = shot()
        last_text = texts_of(ocr(crop(img, action_px), 0.45))
        hits = [k for k in RESULT_KEYS if k in last_text]
        if hits:
            t = texts_of(ocr(crop(img, sc_roi(RESULT_ROI)), 0.45))
            save(img, "result")
            log(f"开奖窗口命中 {hits}: {t[:80]}")
            log_csv("result", f"{hits} | {t[:120]}")
            img2, closed = dismiss_result(img, t)
            log_csv("result_dismiss", "closed" if closed else "still_visible")
            if closed:
                save(img2, "result_closed")
            return
        if time.time() - last > log_every:
            last = time.time()
            log(f"等待开奖中… {last_text[:40]}")
        time.sleep(poll)
