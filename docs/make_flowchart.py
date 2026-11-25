# -*- coding: utf-8 -*-
"""生成 README 用流程图（PIL 绘制，中文用微软雅黑）"""
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


def box(d, xy, title, body, accent, bg, title_font, body_font, dashed=False):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=18, fill=bg, outline=accent, width=3)
    if dashed:
        step = 14
        for x in range(x0 + 8, x1 - 8, step):
            d.line([(x, y0), (min(x + 7, x1), y0)], fill=accent, width=3)
            d.line([(x, y1), (min(x + 7, x1), y1)], fill=accent, width=3)
        for y in range(y0 + 8, y1 - 8, step):
            d.line([(x0, y), (x0, min(y + 7, y1))], fill=accent, width=3)
            d.line([(x1, y), (x1, min(y + 7, y1))], fill=accent, width=3)

    lines = wrap(d, body, body_font, x1 - x0 - 40)
    th = text_size(d, title, title_font)[1]
    bh = text_size(d, "汉", body_font)[1] + 10
    total = th + 12 + len(lines) * bh
    cy = (y0 + y1 - total) / 2
    tw = text_size(d, title, title_font)[0]
    d.text(((x0 + x1 - tw) / 2, cy), title, font=title_font, fill=accent)
    for i, ln in enumerate(lines):
        w = text_size(d, ln, body_font)[0]
        d.text(((x0 + x1 - w) / 2, cy + th + 14 + i * bh), ln, font=body_font, fill=INK)


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
    W, H = 2200, 1420
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ft = f(52, True)
    fs = f(28)
    fb = f(26, True)
    fbd = f(24)
    label(d, (60, 60), "福袋模型：训练数据流与运行时加载", ft, INK, "la")
    label(d, (60, 132), "采集在 fudai_auto.py 运行时自动完成；训练统一走 train.ps1", fs, MUTED, "la")

    box(d, (740, 190, 1460, 320), "fudai_auto.py 运行时主程序",
        "adb 截图 / 点击 · OCR · 开奖识别 · 样本采集", BLUE, BLUE_BG, fb, fbd)
    box(d, (600, 420, 1600, 610), "data/fudai + log/shots",
        "icons/*.png  120×90 图标样本\nlabels.csv  时间 / 房间 / 标签 / 来源 / bbox\nshots/*.png  关键节点截图（负样本来源）",
        ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (1100, 320), (1100, 420))
    label(d, (1124, 370), "每识别到一次福袋", fs, MUTED, "lm")

    box(d, (500, 760, 1040, 960), "train_fudai_clf.py 检测模型",
        "HOG + HSV 特征 → 线性判别\n正样本：icons 按粉色主体重裁 + 增强\n负样本：shots 滑窗，避开真实福袋框",
        GREEN, GREEN_BG, fb, fbd)
    box(d, (1160, 760, 1700, 960), "train_fudai.py 状态模型",
        "softmax 3 分类：待参与 / 已参与 / 倒计时\n增强 8×，类别加权\nLeave-One-Out 交叉验证评估",
        GREEN, GREEN_BG, fb, fbd)
    arrow(d, (840, 610), (840, 760))
    label(d, (864, 685), "正样本 + 增强", fs, MUTED, "lm")
    arrow(d, (1420, 610), (1420, 760))
    label(d, (1444, 685), "截图当负样本", fs, MUTED, "lm")

    box(d, (500, 1090, 1040, 1260), "data/fudai/model/fudai_clf.npz",
        "w / thresh / win_w / win_h\n运行时检测福袋位置（唯一在用模型）", BLUE, BLUE_BG, fb, fbd)
    box(d, (1160, 1090, 1700, 1260), "data/fudai/model/fudai_state.npz",
        "W / b / mean / std / classes\n仅做状态分析，不参与运行", GRAY, GRAY_BG, fb, fbd)
    arrow(d, (770, 960), (770, 1090))
    arrow(d, (1430, 960), (1430, 1090))

    box(d, (60, 790, 430, 940), "fudai_feat.py（共用）",
        "HOG + HSV 特征\n滑窗打分 locate()\n训练与推理同一套实现", GRAY, GRAY_BG, fb, fbd, dashed=True)
    arrow(d, (430, 865), (500, 865), GRAY, 3, dashed=True, head=12)

    pts = [(1000, 1260), (1000, 1320), (2010, 1320), (2010, 255), (1460, 255)]
    for a, b in zip(pts[:-1], pts[1:]):
        arrow(d, a, b, GRAY, 3, dashed=True, head=12)
    label(d, (1750, 1295), "启动时加载，重训后需重启", fs, MUTED, "mm")

    label(d, (60, 1380), "实线 = 数据流    虚线 = 依赖 / 复用 / 加载关系", f(24), MUTED, "la")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "pipeline.png")
    img.save(p)
    return p


def runtime_loop():
    W, H = 2200, 1560
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ft = f(52, True)
    fs = f(28)
    fb = f(27, True)
    fbd = f(25)
    label(d, (60, 60), "运行时：单轮循环（fudai_auto.py）", ft, INK, "la")

    box(d, (640, 150, 1300, 270), "截图", "adb exec-out screencap，失败自动重连 / 重启 adb", BLUE, BLUE_BG, fb, fbd)
    box(d, (640, 340, 1300, 500), "OCR 直播间标题", "房间变了 → 重置轮次、开奖时间、关注状态\n新房间点一次主播「关注」", ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (970, 270), (970, 340))

    cx, cy, dw, dh = 970, 620, 420, 160
    d.polygon([(cx, cy - dh / 2), (cx + dw / 2, cy), (cx, cy + dh / 2), (cx - dw / 2, cy)],
              fill=GRAY_BG, outline=GRAY, width=3)
    label(d, (cx, cy - 22), "面板已打开？", fb, INK)
    label(d, (cx, cy + 26), "福袋面板 / 福袋任务", fbd, MUTED)
    arrow(d, (970, 500), (970, 540))

    box(d, (1500, 540, 2080, 700), "检测福袋浮标",
        "fudai_clf.npz 滑窗打分 ≥ 0.12\n失败回退模板匹配 / OCR 倒计时", GREEN, GREEN_BG, fb, fbd)
    arrow(d, (1180, 620), (1500, 620))
    label(d, (1340, 592), "否", fs, MUTED)

    box(d, (1500, 780, 2080, 920), "点开福袋面板", "点图标 → 面板从底部弹出", GREEN, GREEN_BG, fb, fbd)
    arrow(d, (1790, 700), (1790, 780))

    box(d, (520, 780, 1180, 960), "连续点面板粉色按钮",
        "加入购物粉丝团并关注主播 / 一键发表评论\n出现抖币、支付字样 → 停手并记 need_payment", BLUE, BLUE_BG, fb, fbd)
    arrow(d, (760, 700), (760, 780))
    label(d, (700, 726), "是", fs, MUTED)
    arrow(d, (1500, 850), (1180, 850))

    box(d, (520, 1040, 1180, 1200), "读剩余秒数 + 采集样本",
        "倒计时排下次开奖时间\n每次命中裁 120×90 存入 data/fudai/icons", ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (850, 960), (850, 1040))

    box(d, (520, 1280, 1180, 1430), "到点进开奖观察窗口",
        "识别「中奖 / 未中奖 / 很遗憾」\n结果写入 log/fudai_log.csv", ORANGE, ORANGE_BG, fb, fbd)
    arrow(d, (850, 1200), (850, 1280))

    pts = [(520, 1355), (300, 1355), (300, 210), (640, 210)]
    for a, b in zip(pts[:-1], pts[1:]):
        arrow(d, a, b, GRAY, 3, dashed=True, head=12)
    label(d, (310, 1040), "继续下一轮", f(26), MUTED, "lm")
    label(d, (310, 1080), "不离开直播间", f(24), MUTED, "lm")

    label(d, (60, 1510), "蓝色 = 交互动作    绿色 = 检测 / 打开面板    橙色 = 状态记录", f(24), MUTED, "la")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "runtime_loop.png")
    img.save(p)
    return p


if __name__ == "__main__":
    print(pipeline())
    print(runtime_loop())
