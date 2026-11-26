# -*- coding: utf-8 -*-
"""命令行入口：``python -m train_fudai [--dry-run] [--predict 图片/目录]``。"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
