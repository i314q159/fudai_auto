# -*- coding: utf-8 -*-
"""
抖音直播间福袋蹲守：运行时主循环与 CLI

实现细节（adb / OCR / 面板处理 / 样本采集）都在 fudai_auto/ 包，
本文件只负责参数解析、单实例锁、主循环节奏和状态机。
"""
import argparse
import glob
import os
import random
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


def pick_rounds(room, lo, hi):
    """进入直播间时定本次蹲几轮：在 [lo, hi] 内随机（含端点）。

    hi <= 0 表示不切房（沿用 max_rounds = 0 的约定）；返回该房间的目标轮数。
    """
    if hi <= 0:
        return 0
    lo = max(1, min(lo, hi))
    n = random.randint(lo, hi)
    fa.log(f"直播间「{room}」本次蹲守 {n} 轮（{lo}~{hi} 随机）")
    fa.log_csv("room_plan", f"{room} rounds={n}")
    return n


def is_new_bag(now, rem, last_rem, last_ts, jump):
    """这次「一键发表评论」是否属于一个新福袋，返回 (是否新袋, 判定依据)。

    - 没有上次记录 -> 新袋；
    - 上一个福袋已过开奖时刻（留 5s「读倒计时->点下去」的滞后）-> 新袋，
      这样只剩几十秒的新袋也不会被误判成同一个；
    - 两边都有倒计时 -> 比「上次剩余 - 已过时间」回跳超过 jump 才算新袋；
    - 任一侧读不到倒计时 -> 距上次点击超过 jump 才算新袋。
    """
    if not last_ts:
        return True, "首次点击"
    if last_rem is not None and now >= last_ts + last_rem + 5:
        return True, f"上一袋已开奖 {now - last_ts - last_rem:.0f}s"
    if rem is None or last_rem is None:
        gap = now - last_ts
        return gap >= jump, f"倒计时缺失，距上次点击 {gap:.0f}s"
    expect = max(0.0, last_rem - (now - last_ts))
    return rem > expect + jump, f"剩余 {rem}s / 预期 {expect:.0f}s"


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

    rounds_min = int(fa.cfg("room", "min_rounds"))
    rounds_max = int(fa.cfg("room", "max_rounds"))
    if rounds_max > 0 and rounds_min > rounds_max:
        fa.log(f"配置异常：min_rounds={rounds_min} > max_rounds={rounds_max}，按 {rounds_max}~{rounds_min} 处理")
        rounds_min, rounds_max = rounds_max, rounds_min
    if rounds_max <= 0:
        fa.log("max_rounds ≤ 0：不自动切房")
    room_rounds = 0          # 当前直播间的蹲守轮数目标（进入房间时随机定）
    switch_after_draw = bool(fa.cfg("room", "switch_after_last_draw"))
    swipe_attempts = int(fa.cfg("room", "swipe_attempts"))

    followed_room = None     # 已经点过关注的直播间：没检测到福袋的房间不点关注

    def maybe_follow(found_img):
        """确认本房间有福袋后才点一次关注；同一房间只尝试一次。

        传进来的是「命中福袋、还没点开面板」的那张图，头部按钮位置是准的。
        """
        nonlocal followed_room
        if followed_room == cur_room:
            return
        followed_room = cur_room
        r = fa.follow_anchor(found_img)
        if r:
            fa.log_csv("follow", f"{cur_room} {r}")

    def track_click(rem):
        """点到「一键发表评论」后才算一轮（main 里只有 commented 才会调到这里）。

        返回 True 表示这是一个新的福袋；同一个福袋重复点评论不计轮。
        判定规则见 is_new_bag()。
        """
        nonlocal click_rem, click_ts, cycle_count
        now = time.time()
        is_new, why = is_new_bag(
            now, rem, click_rem, click_ts,
            float(fa.cfg("detect", "new_cycle_jump_seconds")),
        )
        click_rem, click_ts = rem, now
        left = rem if rem is not None else "未知"
        if is_new:
            cycle_count += 1
            fa.log(f"第 {cycle_count} 轮福袋点击「一键发表评论」，剩余 {left}s 开奖（{why}）")
            fa.log_csv("new_cycle", f"#{cycle_count} rem={rem}")
        else:
            fa.log(f"同一福袋内再次点评论（剩余 {left}s，{why}），不计入新一轮")
            fa.log_csv("click_repeat", f"rem={rem}")
        return is_new

    def plan_switch():
        """本房间蹲满本次目标轮数就安排切下一间"""
        nonlocal leave_after_draw, next_draw_ts
        if room_rounds <= 0 or leave_after_draw or cycle_count < room_rounds:
            return
        leave_after_draw = True
        fa.log(f"直播间「{cur_room}」已蹲满 {cycle_count} 轮（本次目标 {room_rounds}），准备切下一间")
        fa.log_csv("room_rounds_full", f"{cur_room} rounds={cycle_count} goal={room_rounds}")

    def switch_room():
        """上滑切下一个直播间；成功后轮次与开奖状态清零"""
        nonlocal cur_room, cycle_count, click_rem, click_ts, room_rounds
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
                room_rounds = pick_rounds(name, rounds_min, rounds_max)
                return name
            fa.log(f"切房未生效（识别到「{name or '无标题'}」），重试 {attempt}/{swipe_attempts}")
            fa.save(img, "switch_retry")
        fa.log(f"连续 {swipe_attempts} 次切房失败，留在当前直播间并重新计轮")
        cycle_count = 0
        click_rem = None
        click_ts = 0.0
        leave_after_draw = False
        room_rounds = pick_rounds(cur_room, rounds_min, rounds_max)
        return None

    room_check = float(fa.cfg("detect", "room_check_seconds"))
    scan_interval = min(args.interval, float(fa.cfg("detect", "scan_interval_seconds")))
    fa.log(
        f"启动福袋蹲守 {args.minutes} 分钟，检查福袋开奖间隔 {args.interval}s，"
        f"未跟到福袋时 {scan_interval}s 巡一次，巡房间隔 {room_check}s"
    )
    fa.log_csv("start", f"minutes={args.minutes} interval={args.interval} scan={scan_interval}")

    def bag_gap():
        """下一轮福袋检查的间隔：等开奖时按 interval，本房间还没袋时用更快的 scan_interval。"""
        waiting = (not draw_watched) and next_draw_ts > time.time()
        return args.interval if waiting else scan_interval

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
                room_rounds = pick_rounds(name, rounds_min, rounds_max)
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
                    room_rounds = pick_rounds(name, rounds_min, rounds_max)
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
            status, rem, detail, commented = fa.handle_panel(img, panel_text=t)
            if status == "joined_now":
                if not commented:
                    fa.log(f"本次只点了「{detail}」，没点到「一键发表评论」，不计轮次")
                    fa.log_csv("no_comment", f"{cur_room} {detail}")
                elif track_click(rem):
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
                # 读不到倒计时就不排期：不编造开奖时间，靠每轮巡检抓开奖弹层
                draw_watched = True
                fa.log("未读到倒计时，不排开奖时间")
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
                img2, info = fa.open_bag_panel(img, before_tap=maybe_follow)
                if img2 is None:
                    fa.log(f"未见福袋图标: {info}")
                else:
                    st2, t2 = fa.panel_state(img2)
                    if not st2:
                        fa.log(f"打开福袋后面板未识别，关闭 ({info})")
                        fa.close_panel()
                    else:
                        status, rem, detail, commented = fa.handle_panel(img2, panel_text=t2)
                        if status == "joined_now":
                            if not commented:
                                fa.log(f"本次只点了「{detail}」，没点到「一键发表评论」，不计轮次")
                                fa.log_csv("no_comment", f"{cur_room} {detail}")
                            elif track_click(rem):
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
                            draw_watched = True
                            fa.log("未读到倒计时，不排开奖时间")
                        if status == "need_payment":
                            fa.log("该福袋需要付费（抖币），已暂停自动点击，请手动处理")
                            fa.log_csv("need_payment", f"{cur_room} {detail}")

        # 配置为“点完最后一轮立即切房”，或已经没有可等的开奖时间时，不等开奖直接上滑
        if leave_after_draw and not (
            switch_after_draw and not draw_watched and next_draw_ts > time.time()
        ):
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

        # 等开奖时按 [run].interval，本房间还没跟到袋时用更快的 [detect].scan_interval_seconds
        gap = bag_gap()
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
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C 是正常停止方式，不甩 traceback；记一行便于复盘
        fa.log("收到 Ctrl+C，已停止")
        fa.log_csv("end", "interrupted")
