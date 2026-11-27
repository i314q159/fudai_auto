# -*- coding: utf-8 -*-
"""特征计算的加速后端：自动在 NVIDIA GPU（torch CUDA）与 CPU 之间选择。

模型本身只是几百维的线性打分，numpy 就够快；真正费时间的是「批量提特征」
（检测模型一次要提几万个窗口）。所以 GPU 只接管 HOG 这一段，颜色直方图每窗
只要几微秒，留在 CPU 上用 cv2 算更快。

torch 是可选依赖，不写进 requirements：
  - 装了且检测到 CUDA 设备 -> ``auto`` 自动用 GPU，样本越多收益越明显
  - 没装 / 没有 GPU -> 全部走 numpy 向量化实现，结果在 float32 精度内一致
首次上 GPU 时会抽几窗和 numpy 版对一下，偏差过大就退回 CPU，避免静默跑偏。
"""
import numpy as np

from . import hog

DEVICES = ("auto", "cpu", "cuda")

_CACHE = {}
_WARNED = set()
_VERIFIED = False
_TORCH = None
_TORCH_TRIED = False


def _torch():
    """惰性导入 torch（没装返回 None，不抛异常）。"""
    global _TORCH, _TORCH_TRIED
    if not _TORCH_TRIED:
        _TORCH_TRIED = True
        try:
            import torch
        except Exception:
            torch = None
        _TORCH = torch
    return _TORCH


def _cuda_devices():
    """可用 CUDA 设备数：没装 torch / 驱动或 CUDA 不可用时返回 0，不抛异常。"""
    torch = _torch()
    if torch is None:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def resolve(device="auto"):
    """决定用哪个后端，返回 {"device": "cpu"/"cuda", "name": 设备名, "why": 原因}。"""
    device = (device or "auto").strip().lower()
    if device not in DEVICES:
        raise ValueError(f"未知 device: {device}（可选 {'/'.join(DEVICES)}）")
    if device in _CACHE:
        return _CACHE[device]

    torch = _torch()
    count = _cuda_devices()
    if device == "cuda" and count == 0:
        hint = "未安装 torch（uv pip install torch）" if torch is None else "没有可用的 CUDA 设备"
        raise SystemExit(f"--device cuda 不可用：{hint}")

    if device == "cuda" or (device == "auto" and count > 0):
        info = {
            "device": "cuda",
            "name": str(torch.cuda.get_device_name(0)),
            "why": f"CUDA 设备 {count} 个",
        }
    elif torch is None:
        info = {"device": "cpu", "name": "CPU", "why": "未安装 torch，GPU 加速可选（uv pip install torch）"}
    else:
        info = {"device": "cpu", "name": "CPU", "why": "torch 没检测到 CUDA 设备"}
    _CACHE[device] = info
    return info


def describe(device="auto"):
    """一行说明当前后端，训练日志里打印。"""
    info = resolve(device)
    if info["device"] == "cuda":
        return f"NVIDIA GPU {info['name']}（{info['why']}）"
    return f"CPU（{info['why']}）"


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[warn] {msg}")


def hog_batch(grays, cell, bins, device="auto"):
    """(N, H, W) 灰度图 -> (N, ncy*ncx*bins) 的 HOG；device 取 auto/cpu/cuda。"""
    grays = np.ascontiguousarray(grays)
    if len(grays) == 0:
        return np.zeros((0, 0), np.float32)
    if resolve(device)["device"] == "cuda":
        try:
            out = _hog_torch(grays, cell, bins, "cuda")
            if _matches_numpy(grays, cell, bins, out):
                return out
        except Exception as e:                      # CUDA OOM / 驱动异常等
            _warn_once("cuda", f"GPU 提特征失败，改走 CPU：{e}")
    return _hog_numpy(grays, cell, bins)


def _hog_numpy(grays, cell, bins, chunk=512):
    """分块算 HOG：一次只留几百张的幅值/角度，避免几万个窗口把内存吃满。"""
    n, h, w = grays.shape
    out = np.empty((n, (h // cell) * (w // cell) * bins), np.float32)
    for s in range(0, n, chunk):
        part = grays[s:s + chunk]
        mag = np.empty(part.shape, np.float32)
        ang = np.empty(part.shape, np.float32)
        for i in range(len(part)):
            mag[i], ang[i] = hog.gradients(part[i])
        out[s:s + len(part)] = hog.cells_batch(mag, ang, cell, bins)
    return out


def _hog_torch(grays, cell, bins, dev):
    """torch 版 HOG：Sobel -> 幅值/角度 -> 按 cell 累加，与 numpy 版逐项对应。"""
    torch = _torch()
    device = torch.device(dev)
    g = torch.from_numpy(np.ascontiguousarray(grays, np.float32)).to(device)
    n, h, w = g.shape
    ncy, ncx = h // cell, w // cell

    # 与 cv2.Sobel(ksize=3) 同一套核；边界用 reflect（等价 BORDER_REFLECT_101）
    kx = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        dtype=torch.float32, device=device,
    ).view(1, 1, 3, 3)
    src = torch.nn.functional.pad(
        g[:, :ncy * cell, :ncx * cell].unsqueeze(1), (1, 1, 1, 1), mode="reflect"
    )
    gx = torch.nn.functional.conv2d(src, kx)
    gy = torch.nn.functional.conv2d(src, kx.transpose(2, 3).contiguous())
    mag = torch.sqrt(gx * gx + gy * gy)
    ang = torch.rad2deg(torch.atan2(gy, gx)) % 180.0

    shape = (n, ncy, cell, ncx, cell)
    mag = mag.view(shape).permute(0, 1, 3, 2, 4).reshape(n, ncy, ncx, cell * cell)
    idx = (ang * (bins / 180.0)).long().clamp_(0, bins - 1)
    idx = idx.view(shape).permute(0, 1, 3, 2, 4).reshape(n, ncy, ncx, cell * cell)

    hist = torch.zeros((n, ncy, ncx, bins), dtype=torch.float32, device=device)
    hist.scatter_add_(3, idx, mag)
    hist /= hist.norm(dim=3, keepdim=True) + 1e-6
    return hist.reshape(n, -1).cpu().numpy()


def _matches_numpy(grays, cell, bins, gpu_out):
    """首次用 GPU 时抽几窗和 numpy 版对一下；偏差过大就放弃 GPU。

    GPU 与 CPU 的 Sobel/atan2 只在最后几位浮点上不同，个别像素会落到相邻的
    方向档（单窗 HOG 每档最多差一个格的幅值），所以看平均偏差而不是最大值：
    实现正确时实测平均偏差 ~1e-5，检测分数差 <0.01。
    """
    global _VERIFIED
    if _VERIFIED:
        return True
    _VERIFIED = True
    sample = grays[:16]
    ref = _hog_numpy(sample, cell, bins)
    diff = np.abs(ref - gpu_out[:len(sample)])
    mean, worst = float(diff.mean()), float(diff.max())
    # 实测：真实窗口平均偏差 ~1e-5，纯噪声最坏 ~1.4e-4，卡在 2e-3 仍有十倍余量
    if mean > 2e-3:
        _warn_once("mismatch", f"GPU 特征与 CPU 平均偏差 {mean:.2g}（最大 {worst:.2g}），改走 CPU")
        return False
    print(f"[info] GPU 特征已对齐 CPU 版（平均偏差 {mean:.2g}，最大 {worst:.2g}），GPU 加速启用")
    return True
