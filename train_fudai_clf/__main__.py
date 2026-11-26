# -*- coding: utf-8 -*-
"""命令行入口：``python -m train_fudai_clf [--model 路径] [--dry-run]``。"""
import argparse
import random

from .paths import MODEL_PATH
from .train import train_model


def main():
    ap = argparse.ArgumentParser(description="训练福袋检测模型")
    ap.add_argument("--model", default=MODEL_PATH, help="模型输出路径")
    ap.add_argument("--dry-run", action="store_true", help="只评估不保存")
    args = ap.parse_args()

    # 固定随机种子，保证同一批样本每次训练结果可复现
    rng = random.Random(20261125)
    train_model(rng, args.model, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
