# -*- coding: utf-8 -*-
"""命令行入口：``python -m bootstrap_fudai_clf --seed 种子图 [--install]``。

--install 会先备份现有生产模型（fudai_clf.npz -> .prev），再覆盖。
"""
import argparse
import os
import random
import shutil

import numpy as np

import fudai_feat as F

from .paths import MODEL_DIR, OUT_DEFAULT, PROD
from .samples import build_positives, build_negatives
from .seed import read_seed, locate_seed_box
from .train import fit_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "fudai", "seed", "seed.png",
    ))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--install", action="store_true",
                    help="覆盖 fudai_clf.npz（先备份 .prev）")
    args = ap.parse_args()

    img = read_seed(args.seed)
    box, rx, ry = locate_seed_box(img)

    rng = random.Random(20261125)
    pos = build_positives(img, box, rx, ry)
    neg = build_negatives(img, box, rng)
    print(f"种子框 {box}  正样本 {len(pos)}  负样本 {len(neg)}")

    model = fit_bootstrap(pos, neg)
    print(
        f"分数: 正样本 min/中位 = {model['score_pos_min']:.2f}/"
        f"{model['score_pos_median']:.2f}，负样本 max = {model['score_neg_max']:.2f}"
    )
    print(f"自评 F1={model['f1']:.3f} 阈值={model['thresh']:.2f}（同一帧自评，仅作点火参考）")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out, w=model["w"], thresh=np.float32(model["thresh"]),
        win_w=np.int32(F.WIN_W), win_h=np.int32(F.WIN_H),
        fea_w=np.int32(F.FEA_W), fea_h=np.int32(F.FEA_H),
    )
    report = (
        f"mode=bootstrap seed={args.seed}\n"
        f"box={box} pos={len(pos)} neg={len(neg)}\n"
        f"f1_selffit={model['f1']:.3f} thresh={model['thresh']:.2f} "
        f"score_pos_min={model['score_pos_min']:.2f} "
        f"score_neg_max={model['score_neg_max']:.2f}\nmodel={args.out}\n"
    )
    with open(os.path.join(MODEL_DIR, "bootstrap_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    if args.install:
        if os.path.exists(PROD):
            shutil.copy2(PROD, PROD + ".prev")
        shutil.copy2(args.out, PROD)
        print(f"已安装为生产模型: {PROD}（原模型备份 {PROD}.prev）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
