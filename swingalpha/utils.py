# -*- coding: utf-8 -*-
"""日志与通用工具。"""
from __future__ import annotations
import logging
import sys
from datetime import datetime
from pathlib import Path
from .config import LOG_DIR

_LOGGER = None


def get_logger(name: str = "swingalpha") -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(Path(LOG_DIR) / f"run_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    _LOGGER = logger
    return logger


def code_to_bs(code: str) -> str:
    """600000 -> sh.600000 ; 000001 -> sz.000001。兼容已带前缀。"""
    c = code.replace("sh.", "").replace("sz.", "").strip()
    if c.startswith(("60", "68", "9")):
        return f"sh.{c}"
    return f"sz.{c}"


def code_to_plain(code: str) -> str:
    """sh.600000 / sh600000 / 600000 -> 600000（提取6位数字代码）。"""
    import re
    if code is None:
        return ""
    m = re.search(r"(\d{6})", str(code))
    return m.group(1) if m else str(code).strip()


def code_to_ak(code: str) -> str:
    """600000 -> sh600000；000001 -> sz000001（akshare/新浪分钟线格式）。"""
    c = code_to_plain(code)
    return ("sh" if c.startswith(("60", "68", "9")) else "sz") + c


def is_main_board(code: str) -> bool:
    """仅沪深主板：60 / 000 / 001 / 002 / 003。"""
    c = code_to_plain(code)
    return c.startswith(("60", "000", "001", "002", "003"))


def is_clean_name(name: str) -> bool:
    """名称不得含 ST / *ST / 退。"""
    if not isinstance(name, str):
        return False
    return ("ST" not in name.upper()) and ("退" not in name)
