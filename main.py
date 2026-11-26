# -*- coding: utf-8 -*-
"""
抖音直播间福袋蹲守：运行时主循环与 CLI

实现细节（adb / OCR / 面板处理 / 样本采集）都在 fudai_auto/ 包，
本文件只负责参数解析、单实例锁、主循环节奏和状态机。
"""
import argparse
import glob
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher

import fudai_auto as fa

LOCK_PORT = 59999


def _port_listener_pid(port):
    """返回监听该端口的 PID；找不到返回 None"""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                if f":{port}" in line and "LISTENING" in line.upper():
                    return int(line.split()[-1])
        else:
            out = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=10
            ).stdout.strip()
            if out:
                return int(out.splitlines()[0])
            out = subprocess.run(
                ["ss", "-ltnp"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                if f":{port}" in line and "pid=" in line:
                    return int(line.split("pid=")[1].split(",")[0])
    except Exception:
        pass
    return None


def _pid_cmdline(pid):
    try:
        if os.name == "nt":
            cmd = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"
            return subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True, text=True, timeout=10
            ).stdout.strip()
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
    except Exception:
        return ""


def _kill_pid(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, timeout=10)
    else:
        os.kill(pid, signal.SIGKILL)


def acquire_instance_lock(port=LOCK_PORT, attempts=4):
    """端口被旧实例占用时自动结束旧实例并接管；其它程序占用则放弃"""
    for _ in range(attempts):
        lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            lock.bind(("127.0.0.1", port))
            lock.listen(1)
            return lock
        except OSError:
            lock.close()
            pid = _port_listener_pid(port)
            if pid and pid != os.getpid():
                cmd = _pid_cmdline(pid)
                if (not cmd) or ("main.py" in cmd) or ("fudai_auto.py" in cmd):
                    fa.log(f"端口 {port} 被旧实例 PID {pid} 占用，自动结束并接管")
                    _kill_pid(pid)
                    time.sleep(2.0)
                    continue
                fa.log(f"端口 {port} 被 PID {pid} 占用且不是本程序，放弃启动")
                return None
            fa.log(f"端口 {port} 占用进程未知，1 秒后重试")
            time.sleep(1.0)
    fa.log(f"多次尝试仍未取得端口 {port}，放弃启动")
    return None


def same_room(a, b, threshold=0.86):
    """判断两次 OCR 标题是否同一房间：容忍前后缀抖动（如「文英菲克…7PR」）"""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if len(short) >= 4 and short in long:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


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
        [sys.executable, "-m", "bootstrap_fudai_clf",
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

    if not fa.SERIAL:
        fa.log(
            "未找到 adb 设备：请连接手机并确认 adb devices 已授权，"
            "或用 --serial / 环境变量 FUDAI_SERIAL 指定"
        )
        return

    # 单实例锁：旧实例还在时自动结束它并接管
    lock = acquire_instance_lock()
    if lock is None:
        fa.log("无法取得单实例锁，本进程退出")
        return

    ensure_model(args.cold_start_below)

    t_end = time.time() + args.minutes * 60
    bag_count = 0            # 累计参与成功次数（跨直播间）
    next_draw_ts = 0.0
    next_bag_ts = 0.0
    draw_watched = True
    cur_room = None
    room_warned = False
    last_room_seen = time.time()
    last_err_log = 0.0
    cycle_count = 0          # 当前直播间已真实点击参与的轮次
    click_rem = None         # 上次点击参与时的倒计时，用于判断是否同一个福袋
    click_ts = 0.0
    leave_after_draw = False # 本房间已蹲满轮次，等当前轮开奖后切下一间

    max_rounds = int(fa.cfg("room", "max_rounds"))
    switch_after_draw = bool(fa.cfg("room", "switch_after_last_draw"))
    swipe_attempts = int(fa.cfg("room", "swipe_attempts"))

    def track_click(rem):
        """真实点击参与后才算一轮；只查看剩余时间、重复点同一个福袋都不算"""
        nonlocal click_rem, click_ts, cycle_count
        now = time.time()
        jump = float(fa.cfg("detect", "new_cycle_jump_seconds"))
        is_new = True
        if click_ts:
            if rem is None or click_rem is None:
                # 读不到倒计时：只能按「距上次点击是否超过 new_cycle_jump_seconds」兜底，
                # 否则同一个福袋会被反复计轮、提前把房间轮次蹲满
                is_new = now - click_ts >= jump
                fa.log(
                    f"倒计时未读到，按距上次点击 {now - click_ts:.0f}s 判定为"
                    f"{'新' if is_new else '同一'}福袋"
                )
            else:
                expect = max(0.0, click_rem - (now - click_ts))
                is_new = rem > expect + jump
        click_rem, click_ts = rem, now
        if is_new:
            cycle_count += 1
            fa.log(f"第 {cycle_count} 轮福袋点击参与，剩余 {rem if rem is not None else '未知'}s 开奖")
            fa.log_csv("new_cycle", f"#{cycle_count} rem={rem}")
        else:
            fa.log(f"同一福袋内再次点击（剩余 {rem if rem is not None else '未知'}s），不计入新一轮")
            fa.log_csv("click_repeat", f"rem={rem}")
        return is_new

    def plan_switch():
        """本房间蹲满 max_rounds 轮就安排切下一间"""
        nonlocal leave_after_draw, next_draw_ts
        if max_rounds <= 0 or leave_after_draw or cycle_count < max_rounds:
            return
        leave_after_draw = True
        fa.log(f"直播间「{cur_room}」已蹲满 {cycle_count} 轮福袋，准备切下一间")
        fa.log_csv("room_rounds_full", f"{cur_room} rounds={cycle_count}")
        # 等开奖时如果没读到倒计时，按默认时长排期，避免永远等不到观察窗口
        if switch_after_draw and (not next_draw_ts or next_draw_ts <= time.time()):
            next_draw_ts = time.time() + float(fa.cfg("detect", "default_draw_seconds"))

    def switch_room():
        """上滑切下一个直播间；成功后轮次与开奖状态清零"""
        nonlocal cur_room, cycle_count, click_rem, click_ts
        nonlocal last_room_seen, next_draw_ts, next_bag_ts, draw_watched, leave_after_draw
        before = cur_room
        sim = float(fa.cfg("detect", "room_similarity"))
        for attempt in range(1, swipe_attempts + 1):
            try:
                fa.swipe_next_room()
                img = fa.shot()
            except Exception as e:
                fa.log(f"切房第 {attempt}/{swipe_attempts} 次异常: {e}")
                time.sleep(2.0)
                continue
            name = fa.get_room_name(img)
            if name and (not before or not same_room(name, before, sim)):
                fa.log(f"已切到下一间直播间: {before} -> {name}")
                fa.log_csv("switch_room", f"{before} -> {name}")
                cur_room = name
                fa.ROOM_NAME = name
                last_room_seen = time.time()
                cycle_count = 0
                click_rem = None
                click_ts = 0.0
                next_draw_ts = 0.0
                next_bag_ts = 0.0
                draw_watched = True
                leave_after_draw = False
                return name
            fa.log(f"切房未生效（识别到「{name or '无标题'}」），重试 {attempt}/{swipe_attempts}")
            fa.save(img, "switch_retry")
        fa.log(f"连续 {swipe_attempts} 次切房失败，留在当前直播间并重新计轮")
        cycle_count = 0
        click_rem = None
        click_ts = 0.0
        leave_after_draw = False
        return None

    room_check = float(fa.cfg("detect", "room_check_seconds"))
    fa.log(f"启动福袋蹲守 {args.minutes} 分钟，检查福袋开奖间隔 {args.interval}s，巡房间隔 {room_check}s")
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
                # 暂时只检测标题变化，不自动切房：不重置轮次、不关注新主播
                if same_room(name, cur_room, float(fa.cfg("detect", "room_similarity"))):
                    cur_room = max((cur_room, name), key=len)
                    fa.ROOM_NAME = cur_room
                else:
                    fa.log(f"直播间标题变化: {cur_room} -> {name}，重置状态并重新开始检测福袋")
                    fa.log_csv("room_title_change", f"{cur_room} -> {name}")
                    cur_room = name
                    next_draw_ts = 0.0
                    next_bag_ts = 0.0
                    draw_watched = True
                    cycle_count = 0
                    click_rem = None
                    click_ts = 0.0
                    leave_after_draw = False
                    fa.save(img, "room_title_change")
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

        # 开奖/中奖弹层：识别到就用统一逻辑关掉，先回到福袋检测
        if now >= next_bag_ts or due_watch:
            rt = fa.texts_of(fa.ocr(fa.crop(img, fa.sc_roi(fa.RESULT_ROI)), 0.45))
            if fa.result_present(rt):
                fa.save(img, "result_popup")
                fa.log(f"识别到开奖文案: {rt[:60]}")
                fa.log_csv("result", rt[:120])
                img, closed = fa.dismiss_result(img, rt)
                fa.log_csv("result_dismiss", "closed" if closed else "still_visible")
                if closed:
                    fa.save(img, "result_closed")
                    draw_watched = True   # 开奖结果已处理，避免再进一次观察窗口
                    continue
                # 关不掉时不要在弹层上继续找福袋/乱点：跳过本轮，下一轮再试关
                draw_watched = True
                next_bag_ts = time.time() + args.interval
                fa.save(img, "result_stuck")
                fa.log("开奖弹层未关闭，本轮跳过福袋检查，下一轮再试")
                continue

        bag_touched = False

        # 面板是否已打开：打开着就优先处理，不受巡检节奏限制
        st, t = fa.panel_state(img)
        if st:
            status, rem, detail = fa.handle_panel(img)
            if status == "joined_now":
                if track_click(rem):
                    bag_count += 1
                    fa.log(f"第 {bag_count} 次参与成功({detail})")
                    fa.log_csv("join", f"{cur_room} #{bag_count} {detail} remaining={rem}")
                if rem is not None:
                    next_draw_ts = time.time() + rem
                    draw_watched = False
                plan_switch()
            elif rem is not None:
                next_draw_ts = time.time() + rem
                draw_watched = False
            else:
                next_draw_ts = time.time() + float(fa.cfg("detect", "default_draw_seconds"))
                draw_watched = False
            if status == "need_payment":
                fa.log("该福袋需要付费（抖币），本房间不再自动点击，请手动处理")
                fa.log_csv("need_payment", f"{cur_room} {detail}")
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
                            if track_click(rem):
                                bag_count += 1
                                fa.log(f"第 {bag_count} 次参与成功({detail})")
                                fa.log_csv("join", f"{cur_room} #{bag_count} {detail} remaining={rem}")
                            if rem is not None:
                                next_draw_ts = time.time() + rem
                                draw_watched = False
                            plan_switch()
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

        # 配置为“点完第 5 轮立即切房”时，不等开奖直接上滑
        if leave_after_draw and not switch_after_draw:
            switch_room()
            continue

        # 开奖窗口（可能正好落在两次整轮检测之间）
        if due_watch:
            if cycle_count:
                fa.log(f"进入开奖观察窗口（第 {cycle_count} 轮）")
            else:
                fa.log("进入开奖观察窗口（本轮未点击参与）")
            fa.watch_result(float(fa.cfg("detect", "watch_seconds")))
            draw_watched = True
            # 本房间蹲满轮次：看完这一轮开奖就切下一间
            if leave_after_draw:
                switch_room()
                continue
            bag_touched = True

        # 福袋面板检查固定按 [run].interval（默认 60s = 1 分钟一次），不再按剩余时间自适应
        gap = args.interval
        if bag_touched:
            # 从本轮检查起点计时，保证两次面板检查的间隔稳定在 gap 左右
            next_bag_ts = now + gap

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
