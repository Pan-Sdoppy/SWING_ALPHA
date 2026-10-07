# -*- coding: utf-8 -*-
"""
akshare(新浪) 数据源：盘中全市场实时快照、股票列表、候选股分钟线。
- 全市场快照 stock_zh_a_spot（新浪，约14秒，单次调用，无 V8 多线程问题）。
- 在线获取与本地缓存读入都强制执行：剔 ST/退市 + 仅主板(60/00) 过滤。
"""
from __future__ import annotations
import socket
from typing import Optional

import pandas as pd

from ..utils import get_logger, is_main_board, is_clean_name, code_to_plain

log = get_logger()


def _norm_spot(df: pd.DataFrame) -> pd.DataFrame:
    """把新浪快照归一化为统一列（兼容中英文两套表头）。"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    # 别名 -> 标准列（同时兼容新版中文表头与旧版英文表头）
    alias = {
        "代码": "raw_code", "code": "raw_code", "symbol": "raw_code",
        "名称": "name", "name": "name",
        "最新价": "price", "trade": "price", "current": "price", "price": "price", "最新": "price",
        "今开": "open", "open": "open",
        "最高": "high", "high": "high",
        "最低": "low", "low": "low",
        "昨收": "preclose", "settlement": "preclose", "pre_close": "preclose",
        "成交量": "volume", "volume": "volume",
        "成交额": "amount", "amount": "amount",
        "涨跌幅": "pctChg", "changepercent": "pctChg",
        "换手率": "turn", "turnoverratio": "turn",
        "时间戳": "ticktime", "ticktime": "ticktime",
    }
    colmap = {c: alias[str(c).lower()] for c in df.columns
              if str(c).lower() in alias}
    df = df.rename(columns=colmap)

    # 新版中文新浪快照成交量单位为“股”；旧版英文接口为“手”(×100)
    is_cn = any(c in df.columns for c in ("代码", "名称")) or "成交量" in colmap
    if "raw_code" in df.columns:
        df["code"] = df["raw_code"].map(code_to_plain)
    keep = ["code", "name", "price", "open", "high", "low", "preclose",
            "volume", "amount", "pctChg", "turn", "ticktime"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep]
    for c in ["price", "open", "high", "low", "preclose", "volume",
              "amount", "pctChg", "turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not is_cn:
        df["volume"] = df["volume"] * 100.0   # 手 → 股
    return df


def apply_universe_filter(df: pd.DataFrame) -> pd.DataFrame:
    """强制：仅主板 + 剔 ST/退市（在线与缓存读入均调用，杜绝脏缓存）。"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    out = out[out["code"].map(is_main_board)]
    if "name" in out.columns:
        out = out[out["name"].map(is_clean_name)]
    return out.reset_index(drop=True)


def fetch_spot(timeout: int = 30, retries: int = 3) -> pd.DataFrame:
    """获取新浪全市场实时快照并归一化、过滤主板非ST。失败自动重试（间隔2s递增）。"""
    socket.setdefaulttimeout(timeout)
    import akshare as ak
    last_err = None
    for attempt in range(retries):
        try:
            log.info(f"拉取新浪全市场实时快照 stock_zh_a_spot（第{attempt+1}次）…")
            raw = ak.stock_zh_a_spot()
            df = _norm_spot(raw)
            if df is None or len(df) == 0:
                raise RuntimeError("快照返回为空")
            n_all = len(df)
            df = apply_universe_filter(df)
            log.info(f"快照获取完成：全市场 {n_all} 只，主板非ST {len(df)} 只")
            return df
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = 2 * (attempt + 1)
                log.warning(f"快照拉取失败（{e}），{wait}s 后重试…")
                import time as _t; _t.sleep(wait)
    raise RuntimeError(f"新浪快照连续{retries}次失败: {last_err}")


def fetch_stock_list(timeout: int = 30) -> pd.DataFrame:
    """返回当前主板非ST股票列表 DataFrame[code,name]（来源新浪快照）。"""
    df = fetch_spot(timeout=timeout)
    if len(df) == 0:
        return pd.DataFrame(columns=["code", "name"])
    return df[["code", "name"]].drop_duplicates("code").reset_index(drop=True)


def fetch_index_spot(timeout: int = 20) -> dict:
    """新浪指数实时快照，返回 {sh000001: {'price':..,'pct':..}, ...}（盘中门控用）。"""
    socket.setdefaulttimeout(timeout)
    import akshare as ak
    out = {}
    try:
        df = ak.stock_zh_index_spot_sina()
        if df is None or len(df) == 0:
            return out
        code_col = "代码" if "代码" in df.columns else "code"
        pct_col = "涨跌幅" if "涨跌幅" in df.columns else "changepercent"
        px_col = "最新价" if "最新价" in df.columns else "trade"
        for _, r in df.iterrows():
            c = str(r[code_col])
            if c in ("sh000001", "sh000300", "sh000852"):
                out[c] = {"price": float(r[px_col]), "pct": float(r[pct_col])}
    except Exception as e:  # noqa
        log.debug(f"指数快照获取失败: {e}")
    return out


def fetch_minute(plain_code: str, period: str = "5", timeout: int = 20) -> Optional[pd.DataFrame]:
    """
    单只股票新浪分钟线（仅对候选股增强使用）。
    period: 1/5/15/30/60。返回 day,open,high,low,close,volume（数值化）。
    """
    socket.setdefaulttimeout(timeout)
    import akshare as ak
    from ..utils import code_to_ak
    sym = code_to_ak(plain_code)
    try:
        df = ak.stock_zh_a_minute(symbol=sym, period=period, adjust="")
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={"day": "datetime"})
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.reset_index(drop=True)
    except Exception as e:  # noqa
        log.debug(f"分钟线获取失败 {plain_code}: {e}")
        return None
