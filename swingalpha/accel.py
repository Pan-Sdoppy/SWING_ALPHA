# -*- coding: utf-8 -*-
"""
计算后端：自动检测 torch+CUDA（RTX 4070 Ti Super 等），用 GPU 做面板滚动统计；
不可用时回退 numpy/pandas（CPU）。两条路径口径一致（因果滚动、min_periods=1），
GPU 路径出现任何异常都会自动退回 CPU，保证结果正确优先。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger()
_BACKEND = None


def detect_backend(force_cpu: bool = False) -> str:
    global _BACKEND
    if _BACKEND and not force_cpu:
        return _BACKEND
    if force_cpu:
        _BACKEND = "numpy"
        return _BACKEND
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log.info(f"启用 GPU 加速：{name}（torch {torch.__version__}）")
            _BACKEND = "torch-cuda"
        else:
            log.info("已安装 torch 但无可用 CUDA，使用 CPU(numpy) 后端")
            _BACKEND = "numpy"
    except Exception:  # noqa
        log.info("未安装 torch，使用 CPU(numpy/pandas) 后端（安装 CUDA 版 torch 可启用 GPU）")
        _BACKEND = "numpy"
    return _BACKEND


# ---------------- 因果滚动（对齐到当行，min_periods=1） ----------------
def _roll_pandas(x: np.ndarray, w: int, how: str) -> np.ndarray:
    # pandas3 移除 rolling(axis=)；转置为“时间在行、个股在列”，沿行滚动后再转回
    df = pd.DataFrame(np.asarray(x).T)
    r = df.rolling(window=w, min_periods=1)
    if how == "mean":
        out = r.mean()
    elif how == "max":
        out = r.max()
    elif how == "min":
        out = r.min()
    elif how == "std":
        out = r.std()
    else:
        raise ValueError(how)
    return out.values.T


def _roll_torch(x: np.ndarray, w: int, how: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
                        dtype=torch.float32, device="cuda")
    valid = (~torch.isnan(torch.as_tensor(x, dtype=torch.float32, device="cuda"))).float()
    T = t.shape[1]
    pad = w - 1

    def causal_pool(v, kernel, mode):
        v = F.pad(v.unsqueeze(1), (pad, 0), mode="constant",
                  value=0.0 if mode == "sum" else -np.inf).squeeze(1)
        if mode == "max":
            v = F.max_pool1d(v.unsqueeze(1), kernel_size=kernel, stride=1).squeeze(1)
        else:
            v = F.avg_pool1d(v.unsqueeze(1), kernel_size=kernel, stride=1,
                             padding=0, ceil_mode=False).squeeze(1) * min(w, T)
        return v[:, :T]

    cnt = F.avg_pool1d(F.pad(valid.unsqueeze(1), (pad, 0)).squeeze(1).unsqueeze(1),
                       kernel_size=w, stride=1).squeeze(1)[:, :T] * w
    if how == "mean":
        # 窗口和 / 有效个数
        ts = t.unsqueeze(1)
        s = F.avg_pool1d(F.pad(ts, (pad, 0), value=0.0), kernel_size=w,
                         stride=1).squeeze(1)[:, :T] * w
        out = s / torch.clamp(cnt, min=1.0)
    elif how == "max":
        neg = torch.as_tensor(np.nan_to_num(x, nan=-np.inf), dtype=torch.float32, device="cuda")
        neg = F.pad(neg.unsqueeze(1), (pad, 0), value=-np.inf).squeeze(1)
        out = F.max_pool1d(neg.unsqueeze(1), kernel_size=w, stride=1).squeeze(1)[:, :T]
    elif how == "min":
        pos = torch.as_tensor(np.nan_to_num(x, nan=np.inf), dtype=torch.float32, device="cuda")
        pos = F.pad(pos.unsqueeze(1), (pad, 0), value=np.inf).squeeze(1)
        out = -F.max_pool1d((-pos).unsqueeze(1), kernel_size=w, stride=1).squeeze(1)[:, :T]
    elif how == "std":
        raise RuntimeError("std 在 CPU 路径计算")
    out = torch.where(cnt >= 1.0, out, torch.full_like(out, np.nan))
    return out.cpu().numpy()


def rolling(x: np.ndarray, w: int, how: str = "mean", backend: str = None) -> np.ndarray:
    """二维面板 (n_codes × n_dates) 的因果滚动。"""
    backend = backend or detect_backend()
    if how == "std" or backend != "torch-cuda":
        return _roll_pandas(x, w, how)
    try:
        return _roll_torch(x, w, how)
    except Exception as e:  # noqa
        log.warning(f"GPU 滚动计算失败，回退 CPU: {e}")
        return _roll_pandas(x, w, how)
