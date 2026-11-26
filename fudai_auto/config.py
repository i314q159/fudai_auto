# -*- coding: utf-8 -*-
"""配置读取：config.toml 覆盖内置默认值，缺段/缺键自动回落。

所有节奏参数都集中在 config.toml（见 README 配置表），这里只是兜底默认值。
"""
import os
import tomllib

from .paths import ROOT

DEFAULT_CONFIG = {
    "run": {"minutes": 180.0, "interval": 60.0},
    "room": {"min_rounds": 3, "max_rounds": 9, "switch_after_last_draw": True, "swipe_attempts": 3},
    "cold_start": {"below": 30},
    "detect": {
        "shot_retry_seconds": 10.0,
        "shot_error_log_every_seconds": 30.0,
        "room_check_seconds": 5.0,
        "scan_interval_seconds": 20.0,
        "room_similarity": 0.86,
        "room_missing_warn_seconds": 180.0,
        "default_draw_seconds": 750.0,
        "new_cycle_jump_seconds": 90.0,
        "watch_seconds": 90.0,
        "result_auto_close_seconds": 12.0,
        "draw_margin_seconds": 3.0,
        "tap_wait_seconds": 6.0,
        "panel_open_settle_seconds": 3.0,
        "panel_settle_seconds": 0.5,
    },
    "pacing": {"min_sleep_seconds": 1.5},
    "sample": {"throttle_seconds": 60.0},
    "watch": {"poll_seconds": 3.0, "log_every_seconds": 20.0},
}

CONFIG_PATH = os.path.join(ROOT, "config.toml")


def _keep_type(section, key, val, default):
    """按内置默认值的类型校验配置值；类型不符时打印提示并沿用默认值。

    bool 单独判断（Python 里 True 是 int 的子类，不单独处理会把 1/0 当开关值）；
    浮点键允许写整数（toml 里 60 和 60.0 等价），统一转成 float。
    """
    if isinstance(default, bool):
        ok, want = isinstance(val, bool), "true/false"
    elif isinstance(default, int):
        ok, want = isinstance(val, int) and not isinstance(val, bool), "整数"
    elif isinstance(default, float):
        ok, want = isinstance(val, (int, float)) and not isinstance(val, bool), "数字"
        if ok:
            val = float(val)
    else:
        ok, want = isinstance(val, type(default)), type(default).__name__
    if not ok:
        print(f"[config] [{section}].{key} 期望 {want}，实际 {val!r}，已按默认值 {default!r} 处理")
        return default
    return val


def _load_config():
    """读 config.toml：已知段/键才覆盖，未知段/键打印提示并忽略（防止拼错静默生效）。"""
    cfg = {sec: dict(vals) for sec, vals in DEFAULT_CONFIG.items()}
    if not os.path.exists(CONFIG_PATH):
        return cfg
    try:
        with open(CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        print(f"[config] 读取 {CONFIG_PATH} 失败，使用内置默认值: {e}")
        return cfg
    for sec, vals in raw.items():
        if sec not in cfg or not isinstance(vals, dict):
            print(f"[config] 忽略未知段 [{sec}]")
            continue
        for key, val in vals.items():
            if key in cfg[sec]:
                cfg[sec][key] = _keep_type(sec, key, val, cfg[sec][key])
            else:
                print(f"[config] 忽略未知键 [{sec}].{key}")
    return cfg


CONFIG = _load_config()


def cfg(section, key):
    """读取配置；缺段/缺键时回落到内置默认值。"""
    return CONFIG.get(section, {}).get(key, DEFAULT_CONFIG[section][key])
