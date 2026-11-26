# -*- coding: utf-8 -*-
"""离线校验分辨率自适应：把同一张 1080×2400 截图缩放到其它尺寸，
检查福袋定位 / 房间名 / 倒计时 / 开奖弹层识别是否仍然成立。

放在 tests/ 下，不参与运行时，也不影响 main.py / 训练脚本；
脚本自己把仓库根目录加进 sys.path，任意工作目录都能直接跑。

用法: python tests/check_resolution.py [截图路径]
默认用 log/shots 下最新的两张全屏截图。
"""
import glob
import os
import sys

import cv2

# 允许从任意目录运行：先找到仓库根（本文件在 tests/ 下一层）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fudai_auto as fa

SIZES = [(1080, 2400), (1440, 3200), (720, 1600), (1080, 2340), (1260, 2800)]


def pick_shot(keyword=None):
    files = sorted(glob.glob(os.path.join(fa.SHOT_DIR, "*.png")))
    if keyword:
        hit = [f for f in files if keyword in os.path.basename(f)]
        if hit:
            return hit[-1]
    return files[-1] if files else None


def run(path, tag):
    img0 = cv2.imread(path)
    if img0 is None:
        print(f"[SKIP] 读不到 {path}")
        return
    print(f"\n=== {tag}: {os.path.basename(path)} ===")
    for w, h in SIZES:
        img = cv2.resize(img0, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (
            img0.shape[1],
            img0.shape[0],
        ) else img0
        fa.update_screen(img)
        bag = fa.find_bag(img)
        room = fa.get_room_name(img)
        rem = fa.read_remaining(img) if fa.panel_state(img)[0] else None
        txt = fa.texts_of(fa.ocr(fa.crop(img, fa.sc_roi(fa.RESULT_ROI)), 0.45))
        print(
            f"{w}x{h}: 福袋={bag} 房间={room} 倒计时={rem} "
            f"开奖文案={fa.result_present(txt)}"
        )
    print(
        "中奖关闭位映射:",
        [fa.sc_pt(*p) for p in fa.WIN_CLOSE_POINTS],
    )


def main():
    args = sys.argv[1:]
    if args:
        run(args[0], "指定图")
    else:
        run(pick_shot("result_popup"), "开奖弹层")
        run(pick_shot("result_closed"), "直播间画面")


if __name__ == "__main__":
    main()
