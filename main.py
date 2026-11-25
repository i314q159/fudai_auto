# -*- coding: utf-8 -*-
"""
抖音直播间福袋蹲守：运行时主循环与 CLI

实现细节（adb / OCR / 面板处理 / 样本采集）都在 fudai_auto.py，
本文件只负责参数解析、单实例锁、主循环节奏和状态机。
"""
import argparse
import glob
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher

import fudai_auto as fa


def ensure_model(cold_start_below):
    """样本数低于阈值且没有可用检测模型时，用种子图自动冷启动"""
    n = len(glob.glob(os.path.join(fa.DATASET_ICONS, "*.png")))
    if n >= cold_start_below:
        fa.log(f"样本 {n} ≥ 阈值 {cold_start_below}，使用全量模型")
        return
    if fa.CLF is not None:
        fa.log(f"样本 {n} < 阈值 {cold_start_below}，已加载现有检测模型（冷启动阶段）")
        return
    seed = os.path.join(fa.DATASET_DIR, "seed", "seed_vgn.png")
    if not os.path.exists(seed):
        fa.log(f"样本 {n} < 阈值且没有模型，种子图缺失: {seed}")
        return
    fa.log(f"样本 {n} < 阈值 {cold_start_below}，执行种子图冷启动")
    subprocess.run(
        [sys.executable, os.path.join(fa.ROOT, "bootstrap_fudai_clf.py"),
         "--seed", seed, "--install"],
        cwd=fa.ROOT, check=False,
    )
    fa.reload_clf()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=None,
                    help="运行时长（分钟），默认取 config.toml [run].minutes")
    ap.add_argument("--interval", type=float, default=None,
                    help="检查福袋的基础间隔（秒），默认取 config.toml [run].interval")
    ap.add_argument("--room", default="", help="可选：期望的直播间标题关键字，留空跟随当前直播间")
    ap.add_argument("--serial", default=None, help="adb 设备序列号（默认取 FUDAI_SERIAL 或内置值）")
    ap.add_argument("--cold-start-below", type=int, default=None,
                    help="样本数低于该值时走冷启动，默认取 config.toml [cold_start].below")
    args = ap.parse_args()

    if args.minutes is None:
        args.minutes = float(fa.cfg("run", "minutes"))
    if args.interval is None:
        args.interval = float(fa.cfg("run", "interval"))
    if args.cold_start_below is None:
        args.cold_start_below = int(fa.cfg("cold_start", "below"))

    if args.serial:
        fa.SERIAL = args.serial
    fa.RUN_LOG = os.path.join(
        fa.LOG_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    # 单实例锁，防止多个进程抢点击
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 59999))
        lock.listen(1)
    except OSError:
        fa.log("已有实例在运行，本进程退出")
        return

    ensure_model(args.cold_start_below)

    t_end = time.time() + args.minutes * 60
    bag_count = 0
    cycle_no = 0
    next_draw_ts = 0.0
    next_bag_ts = 0.0
    draw_watched = True
    cur_room = None
    room_warned = False
    last_room_seen = time.time()
    last_err_log = 0.0
    last_rem = None
    last_rem_ts = 0.0
    cycle_count = 1

    def track_cycle(rem):
        """通过倒计时“回跳”识别新一轮福袋"""
        nonlocal last_rem, last_rem_ts, cycle_count
        if rem is None:
            return
        now = time.time()
        if last_rem is not None and last_rem_ts:
            expect = max(0, last_rem - (now - last_rem_ts))
            if rem > expect + float(fa.cfg("detect", "new_cycle_jump_seconds")):
                cycle_count += 1
                fa.log(f"检测到新一轮福袋（第 {cycle_count} 轮），剩余 {rem}s 开奖")
                fa.log_csv("new_cycle", f"#{cycle_count} rem={rem}")
        last_rem, last_rem_ts = rem, now

    room_check = float(fa.cfg("detect", "room_check_seconds"))
    fa.log(f"启动福袋蹲守 {args.minutes} 分钟，检查间隔 {args.interval}s，巡房间隔 {room_check}s")
    fa.log_csv("start", f"minutes={args.minutes} interval={args.interval}")

    while time.time() < t_end:
        try:
            img = fa.shot()
        except Exception as e:
            if time.time() - last_err_log > float(fa.cfg("detect", "shot_error_log_every_seconds")):
                last_err_log = time.time()
                fa.log(f"截图异常(自动重试): {e}")
                fa.log_csv("shot_error", str(e)[:80])
            time.sleep(float(fa.cfg("detect", "shot_retry_seconds")))
            continue

        cycle_no += 1

        # 抓当前直播间名称：切换房间自动适配
        name = fa.get_room_name(img)
        if name:
            fa.ROOM_NAME = name
            last_room_seen = time.time()
            if cur_room is None:
                cur_room = name
                fa.log(f"当前直播间: {name}")
                fa.log_csv("room", name)
                r = fa.follow_anchor(img)
                if r:
                    fa.log_csv("follow", f"{name} {r}")
                    if r == "followed":
                        img = fa.shot()
            elif name != cur_room:
                # OCR 尾字/标点抖动（如「南山先生」→「南山先生m」）不算切房
                ratio = SequenceMatcher(None, name, cur_room).ratio()
                if ratio >= float(fa.cfg("detect", "room_similarity")):
                    cur_room = max((cur_room, name), key=len)
                    fa.ROOM_NAME = cur_room
                else:
                    fa.log(f"直播间切换: {cur_room} -> {name}，状态已重置，开始在新直播间蹲福袋")
                    fa.log_csv("room_switch", f"{cur_room} -> {name}")
                    cur_room = name
                    next_draw_ts = 0.0
                    next_bag_ts = 0.0
                    draw_watched = True
                    last_rem = None
                    last_rem_ts = 0.0
                    cycle_count = 1
                    fa.save(img, "room_switch")
                    time.sleep(float(fa.cfg("detect", "room_switch_settle_seconds")))
                    r = fa.follow_anchor(img)
                    if r:
                        fa.log_csv("follow", f"{name} {r}")
                    continue
            if args.room and args.room not in name and not room_warned:
                fa.log(
                    f"提示: 当前直播间为「{name}」，与 --room「{args.room}」不一致；已按当前直播间继续"
                )
                room_warned = True
        elif time.time() - last_room_seen > float(fa.cfg("detect", "room_missing_warn_seconds")):
            mins = float(fa.cfg("detect", "room_missing_warn_seconds")) / 60
            fa.log(f"警告: 连续 {mins:g} 分钟未识别到直播间标题，可能已离开直播间")
            fa.log_csv("warn_room", "no-title")
            last_room_seen = time.time()
            fa.save(img, "no_room_title")

        now = time.time()
        margin = float(fa.cfg("detect", "draw_margin_seconds"))
        due_watch = (not draw_watched) and bool(next_draw_ts) and now >= next_draw_ts - margin

        # 开奖弹窗先记录：只在整轮检测或临近开奖时扫，快速巡房不重复 OCR 整屏
        if now >= next_bag_ts or due_watch:
            rt = fa.texts_of(fa.ocr(fa.crop(img, fa.RESULT_ROI), 0.45))
            if any(k in rt for k in ("恭喜", "很遗憾", "未中奖", "中奖")):
                fa.save(img, "result_popup")
                fa.log(f"识别到开奖文案: {rt[:60]}")
                fa.log_csv("result", rt[:120])

        bag_touched = False

        # 面板是否已打开：打开着就优先处理，不受巡检节奏限制
        st, t = fa.panel_state(img)
        if st:
            status, rem, detail = fa.handle_panel(img)
            if status == "joined_now":
                bag_count += 1
                fa.log(f"第 {bag_count} 次参与成功({detail})")
                fa.log_csv("join", f"{cur_room} #{bag_count} {detail} remaining={rem}")
                if rem is not None:
                    next_draw_ts = time.time() + rem
                    draw_watched = False
            elif rem is not None:
                next_draw_ts = time.time() + rem
                draw_watched = False
            else:
                next_draw_ts = time.time() + float(fa.cfg("detect", "default_draw_seconds"))
                draw_watched = False
            if status == "need_payment":
                fa.log("该福袋需要付费（抖币），本房间不再自动点击，请手动处理")
                fa.log_csv("need_payment", f"{cur_room} {detail}")
            track_cycle(rem)
            time.sleep(float(fa.cfg("detect", "panel_settle_seconds")))
            bag_touched = True
        else:
            # 未识别到直播间标题时跳过福袋检查，避免在搜索页/桌面误点
            if not fa.ROOM_NAME:
                fa.log("未识别到直播间标题，本轮跳过福袋检查")
                time.sleep(room_check)
                continue

            # 只有到整轮检测时间才点福袋；其余巡检只抓标题，快速发现切房
            if now >= next_bag_ts:
                bag_touched = True
                img2, info = fa.open_bag_panel(img)
                if img2 is None:
                    fa.log(f"未见福袋图标: {info}")
                else:
                    st2, _ = fa.panel_state(img2)
                    if not st2:
                        fa.log(f"打开福袋后面板未识别，关闭 ({info})")
                        fa.close_panel()
                    else:
                        status, rem, detail = fa.handle_panel(img2)
                        if status == "joined_now":
                            bag_count += 1
                            fa.log(f"第 {bag_count} 次参与成功({detail})")
                            fa.log_csv("join", f"{cur_room} #{bag_count} {detail} remaining={rem}")
                            if rem is not None:
                                next_draw_ts = time.time() + rem
                                draw_watched = False
                        elif rem is not None:
                            next_draw_ts = time.time() + rem
                            draw_watched = False
                            fa.log(f"当前福袋已参与，剩余 {rem}s 开奖")
                        else:
                            next_draw_ts = time.time() + float(fa.cfg("detect", "default_draw_seconds"))
                            draw_watched = False
                            mins = float(fa.cfg("detect", "default_draw_seconds")) / 60
                            fa.log(f"未读到倒计时，按默认 {mins:g} 分钟排开奖")
                        if status == "need_payment":
                            fa.log("该福袋需要付费（抖币），已暂停自动点击，请手动处理")
                            fa.log_csv("need_payment", f"{cur_room} {detail}")
                        track_cycle(rem)

        # 开奖窗口（可能正好落在两次整轮检测之间）
        if due_watch:
            fa.log(f"进入开奖观察窗口（第 {cycle_count} 轮）")
            fa.watch_result(float(fa.cfg("detect", "watch_seconds")))
            draw_watched = True
            bag_touched = True

        # 下一轮福袋检测按剩余时间自适应；巡房仍按 room_check 的短节奏
        gap = args.interval
        if (not draw_watched) and last_rem is not None:
            if last_rem > float(fa.cfg("pacing", "slow_remaining_seconds")):
                gap = float(fa.cfg("pacing", "slow_interval"))
            elif last_rem > float(fa.cfg("pacing", "mid_remaining_seconds")):
                gap = float(fa.cfg("pacing", "mid_interval"))
            else:
                gap = max(float(fa.cfg("pacing", "near_min_seconds")), args.interval)
        if bag_touched:
            next_bag_ts = time.time() + gap

        # 睡到最近的事件：下一次整轮检测 / 开奖窗口 / 下一次巡房
        sleep_s = room_check
        if next_bag_ts:
            sleep_s = min(sleep_s, max(0.0, next_bag_ts - time.time()))
        if (not draw_watched) and next_draw_ts:
            sleep_s = min(sleep_s, next_draw_ts - time.time() - margin)
        sleep_s = max(float(fa.cfg("pacing", "min_sleep_seconds")), sleep_s)
        time.sleep(sleep_s)

    fa.log(f"结束，共参与 {bag_count} 次福袋")
    fa.log_csv("end", f"joined={bag_count}")


if __name__ == "__main__":
    main()
