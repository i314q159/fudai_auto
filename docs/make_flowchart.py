# -*- coding: utf-8 -*-
"""生成 README 用流程图（PIL 绘制，中文用微软雅黑）

两张图：
  docs/pipeline.png      采集 → 训练（冷启动 / 全量）→ 模型装配
  docs/runtime_loop.png  main.py 入口与单轮巡检循环
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs")
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONTB = r"C:\Windows\Fonts\msyhbd.ttc"

BG = (247, 248, 250)
INK = (33, 37, 41)
MUTED = (90, 98, 112)
BLUE = (47, 111, 235)
BLUE_BG = (232, 240, 254)
GREEN = (31, 157, 85)
GREEN_BG = (230, 244, 234)
ORANGE = (217, 119, 6)
ORANGE_BG = (254, 243, 226)
GRAY = (120, 128, 140)
GRAY_BG = (240, 242, 245)
RED = (190, 55, 55)
RED_BG = (252, 235, 235)
LINE = (70, 78, 90)


def f(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def text_size(d, s, font):
    b = d.textbbox((0, 0), s, font=font)
    return b[2] - b[0], b[3] - b[1]


def wrap(d, s, font, max_w):
    out = []
    for para in s.split("\n"):
        line = ""
        for ch in para:
            if text_size(d, line + ch, font)[0] <= max_w:
                line += ch
            else:
                out.append(line)
                line = ch
        out.append(line)
    return out


def box(d, xy, title, body, accent, bg, title_font, body_font):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=18, fill=bg, outline=accent, width=3)
    lines = wrap(d, body, body_font, x1 - x0 - 36)
    th = text_size(d, title, title_font)[1]
    bh = text_size(d, "汉", body_font)[1] + 10
    total = th + 12 + len(lines) * bh
    cy = (y0 + y1 - total) / 2
    tw = text_size(d, title, title_font)[0]
    d.text(((x0 + x1 - tw) / 2, cy), title, font=title_font, fill=accent)
    for i, ln in enumerate(lines):
        w = text_size(d, ln, body_font)[0]
        d.text(((x0 + x1 - w) / 2, cy + th + 14 + i * bh), ln, font=body_font, fill=INK)


def diamond(d, cx, cy, w, h, title, body, title_font, body_font, accent=GRAY, bg=GRAY_BG):
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    d.polygon(pts, fill=bg, outline=accent, width=3)
    tw = text_size(d, title, title_font)[0]
    d.text((cx - tw / 2, cy - 38), title, font=title_font, fill=INK)
    bw = text_size(d, body, body_font)[0]
    d.text((cx - bw / 2, cy + 8), body, font=body_font, fill=MUTED)


def arrow(d, p0, p1, color=LINE, width=4, dashed=False, head=14):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / L, dy / L
    ex, ey = x1 - ux * head, y1 - uy * head
    if dashed:
        step = 18
        t = 0
        while t < max(0, L - head):
            a = t / L
            b = min((t + 10) / L, max(0, (L - head) / L))
            d.line([(x0 + dx * a, y0 + dy * a), (x0 + dx * b, y0 + dy * b)],
                   fill=color, width=width)
            t += step
    else:
        d.line([(x0, y0), (ex, ey)], fill=color, width=width)
    nx, ny = -uy, ux
    d.polygon([(x1, y1), (ex + nx * head * 0.55, ey + ny * head * 0.55),
               (ex - nx * head * 0.55, ey - ny * head * 0.55)], fill=color)


def label(d, xy, s, font, color=MUTED, anchor="mm"):
    d.text(xy, s, font=font, fill=color, anchor=anchor)


def pipeline():
    W, H = 2400, 1840
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ft = f(50, True)
    fs = f(27)
    fb = f(26, True)
    fbd = f(23)
    label(d, (60, 55), "福袋模型：采集、训练与装配", ft, INK, "la")
    label(d, (60, 125), "采集在运行主循环里自动完成；训练统一入口 train.ps1，样本不足走种子图冷启动", fs, MUTED, "la")

    # 采集
    box(d, (700, 180, 1700, 340), "采集：main.py 主循环 → fudai_auto 包",
        "adb 截图定位福袋 → 打开面板 → 确认参与成功才采集\n裁 120×90 图标，同状态 60s 节流 + 像素差去重",
        BLUE, BLUE_BG, fb, fbd)
    # 训练入口
    box(d, (1840, 180, 2340, 340), "train.ps1 统一入口",
        "默认冷启动阈值 30\n-ColdStartBelow / -Seed\n-DryRun / -SkipState", GREEN, GREEN_BG, fb, fbd)
    # 数据
    box(d, (620, 430, 1780, 660), "data/fudai + log/shots",
        "icons/*.png  120×90 福袋图标（正样本）\nlabels.csv  时间 / 房间 / 标签 / 来源 / bbox\nlog/shots/*.png  关键节点截图（负样本来源）",
        ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (760, 340), (760, 430))
    label(d, (784, 375), "识别即入集", fs, MUTED, "lm")
    arrow(d, (1780, 545), (1900, 545), GRAY, 3, dashed=True, head=12)
    d.line([(1900, 545), (1900, 340)], fill=GRAY, width=3)
    d.polygon([(1900, 340), (1893, 354), (1907, 354)], fill=GRAY)
    label(d, (1822, 522), "读取", f(23), MUTED)

    # 决策
    diamond(d, 1200, 830, 560, 210, "样本数 ≥ 阈值？", "icons 计数（默认 30）", fb, fbd)
    arrow(d, (1200, 660), (1200, 725))
    label(d, (1224, 692), "分类决策", fs, MUTED, "lm")

    # 冷启动分支
    box(d, (160, 1010, 920, 1250), "bootstrap_fudai_clf 冷启动",
        "种子图 → 粉色主体定位 → 7 正 + 120 负\n强正则线性判别，自评 F1 选阈值 → --install\n输出 fudai_clf.npz，旧模型备份 .prev",
        ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (920, 830), (540, 830), GRAY, 4)
    arrow(d, (540, 830), (540, 1010), GRAY, 4)
    label(d, (740, 800), "否（样本太少）", fs, MUTED)

    # 全量分支
    box(d, (1000, 1010, 1500, 1210), "train_fudai_clf 检测模型",
        "HOG + HSV 特征 → 线性判别\n正样本：重裁 + 增强\n负样本：shots 滑窗避开真实框",
        GREEN, GREEN_BG, fb, fbd)
    box(d, (1560, 1010, 2160, 1210), "train_fudai 状态模型",
        "softmax 3 分类\n待参与 / 已参与 / 倒计时\n类别 < 2 自动跳过",
        GREEN, GREEN_BG, fb, fbd)
    arrow(d, (1480, 830), (1860, 830))
    d.line([(1860, 830), (1860, 960), (1250, 960)], fill=LINE, width=4)
    arrow(d, (1250, 960), (1250, 1010))
    arrow(d, (1860, 960), (1860, 1010))
    label(d, (1600, 800), "是（全量训练）", fs, MUTED)

    # 模型产物
    box(d, (1000, 1340, 1500, 1500), "data/fudai/model/fudai_clf.npz",
        "w / thresh / win_w / win_h\n120×90 滑窗打分，运行时唯一主检测器", BLUE, BLUE_BG, fb, fbd)
    box(d, (1560, 1340, 2160, 1500), "data/fudai/model/fudai_state.npz",
        "W / b / mean / std / classes\n只做状态分析，不阻断检测", GRAY, GRAY_BG, fb, fbd)
    arrow(d, (1250, 1210), (1250, 1340))
    arrow(d, (1860, 1210), (1860, 1340))

    # 装配
    box(d, (700, 1580, 2100, 1760), "main.py 启动装配",
        "load_clf() 加载检测模型；无模型且样本 < 阈值 → 自动 bootstrap + reload_clf()\n检测优先，失败回退模板匹配 / OCR；换机：git clone + 放回 data（样本 / 模型不入库）",
        BLUE, BLUE_BG, fb, fbd)
    arrow(d, (540, 1250), (540, 1610))
    arrow(d, (540, 1610), (700, 1610))
    label(d, (566, 1430), "安装 / 热加载", fs, MUTED, "lm")
    arrow(d, (1250, 1500), (1250, 1580))
    arrow(d, (1860, 1500), (1860, 1580))

    label(d, (60, 1800), "实线 = 数据 / 产物    虚线 = 依赖 / 复用    橙色 = 冷启动路径（样本不足时）", f(23), MUTED, "la")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "pipeline.png")
    img.save(p)
    return p


def runtime_loop():
    W, H = 2400, 2150
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ft = f(50, True)
    fs = f(27)
    fb = f(26, True)
    fbd = f(23)
    label(d, (60, 45), "运行时：入口与单轮巡检循环（main.py + fudai_auto 包）", ft, INK, "la")
    label(d, (60, 115), "标题变化 = 重置并重新检测福袋（不自动关注）；标题不变按预定节奏蹲守", fs, MUTED, "la")

    # 入口
    box(d, (900, 160, 1700, 320), "main.py 启动",
        "venv 启动 · 单实例锁 127.0.0.1:59999\n--minutes / --interval / --cold-start-below 默认读 config.toml",
        BLUE, BLUE_BG, fb, fbd)
    box(d, (900, 400, 1700, 610), "加载检测模型（ensure_model）",
        "load_clf() 加载 fudai_clf.npz\n无模型且样本 < 阈值（默认 30）→ bootstrap + reload_clf()\n日志标明当前模型来源，失败回退模板匹配",
        BLUE, BLUE_BG, fb, fbd)
    arrow(d, (1300, 320), (1300, 400))

    # 每轮
    box(d, (900, 690, 1700, 860), "每轮：截图",
        "adb exec-out screencap；异常 → 重连 adb / shell+pull 兜底\n每 5s 巡房认标题；点福袋仍按 [run].interval",
        BLUE, BLUE_BG, fb, fbd)
    arrow(d, (1300, 610), (1300, 690))
    label(d, (1324, 650), "开始巡检", fs, MUTED, "lm")

    box(d, (900, 930, 1700, 1150), "1. OCR 直播间标题",
        "首见 → 点一次主播「关注」\n标题变化 → 重置轮次 / 开奖时间，立即重新检测（不自动关注）\n标题不变 → 每 60 秒检查一次福袋；无标题超阈值则跳过福袋检查",
        ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (1300, 860), (1300, 930))

    # 面板判定
    diamond(d, 1300, 1330, 520, 180, "福袋面板已打开？", "底部面板 / 福袋任务文案", fb, fbd)
    arrow(d, (1300, 1150), (1300, 1240))

    # 是
    box(d, (300, 1520, 1150, 1770), "分支 A：handle_panel() 连续参与",
        "最多 3 步，每次点完重新截图再判定\n加入购物粉丝团并关注 → 一键发表评论\n出现抖币 / 确认支付 → 停手，记 need_payment",
        RED, RED_BG, fb, fbd)
    arrow(d, (1040, 1330), (725, 1330))
    d.line([(725, 1330), (725, 1520)], fill=LINE, width=4)
    label(d, (880, 1300), "是", fs, MUTED)

    # 否
    box(d, (1450, 1520, 2300, 1790), "分支 B：open_bag_panel() 找新福袋",
        "模型定位图标 ≥ 0.12 → 点击 → 面板确认\n面板未识别 → close_panel()；确认成功才裁 120×90 入库\n未见图标 → no-icon，等下一轮，不卡 00:00",
        GREEN, GREEN_BG, fb, fbd)
    arrow(d, (1560, 1330), (1875, 1330))
    d.line([(1875, 1330), (1875, 1520)], fill=LINE, width=4)
    label(d, (1710, 1300), "否", fs, MUTED)
    arrow(d, (1450, 1655), (1150, 1655), GRAY, 3, dashed=True, head=12)
    label(d, (1300, 1618), "复用 handle_panel()", f(22), MUTED)

    # 收尾
    box(d, (500, 1860, 1900, 2050), "汇总：读倒计时 → 排开奖 → 固定 1 分钟巡检",
        "rem → next_draw_ts；真实点击参与才计一轮（只看剩余时间不计）\n满 max_rounds 轮上滑切下一间（默认等第 5 轮开奖）\n巡检节奏固定 60s / 开奖窗口 / 截图重试等待读 config.toml\n结果与参与记录写 log/fudai_log.csv",
        ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (725, 1770), (725, 1860))
    arrow(d, (1875, 1790), (1875, 1860))

    # 回环
    d.line([(500, 1955), (60, 1955), (60, 240), (900, 240)], fill=GRAY, width=3)
    d.polygon([(900, 240), (886, 233), (886, 247)], fill=GRAY)
    label(d, (78, 700), "下一轮（满轮次则切房）", f(24), MUTED, "lm")

    # 侧注
    box(d, (150, 400, 780, 610), "运行前提",
        "手机 1080×2400，抖音 com.ss.android.ugc.aweme\nadb 已授权；串口可用 FUDAI_SERIAL 覆盖",
        GRAY, GRAY_BG, fb, fbd)
    box(d, (150, 930, 780, 1150), "产物目录 log/",
        "run_*.log 运行日志\nfudai_log.csv 参与 / 标题变化 / 开奖记录\nshots/*.png 关键节点截图",
        GRAY, GRAY_BG, fb, fbd)
    box(d, (1800, 690, 2340, 880), "开奖/中奖弹层：统一关闭",
        "OCR 到「恭喜抽中 / 未中奖 / 领取奖品」\n→ 文字按钮 / 卡片 × / 返回键\n→ 关掉后立刻回到福袋检测",
        GRAY, GRAY_BG, fb, fbd)
    arrow(d, (1700, 775), (1800, 775), GRAY, 3, dashed=True, head=12)

    label(d, (60, 2110), "蓝色 = 入口 / 采集    橙色 = 状态与调度    红色 = 参与动作    绿色 = 搜索与打开    虚线 = 复用 / 旁路", f(23), MUTED, "la")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "runtime_loop.png")
    img.save(p)
    return p


if __name__ == "__main__":
    print(pipeline())
    print(runtime_loop())


