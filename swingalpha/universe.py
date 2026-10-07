# -*- coding: utf-8 -*-
"""股票池与日线清洗。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import DAILY_COLS, UNIVERSE
from .utils import get_logger, is_main_board, is_clean_name

log = get_logger()

NUM_COLS = ["open", "high", "low", "close", "preclose", "volume",
            "amount", "turn", "pctChg"]


def normalize_daily(df: pd.DataFrame, code: str, amount_scale: float = 1000.0) -> pd.DataFrame:
    """baostock 原始日线 → 干净的标准日线（去停牌/价格异常，类型转换）。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=DAILY_COLS)
    out = df.copy()
    for c in NUM_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["date"] = out["date"].astype(str)
    if "code" not in out.columns:
        out["code"] = code
    else:
        out["code"] = out["code"].str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False)

    # 停牌：成交量为 0 / 字段缺失
    if UNIVERSE["drop_zero_volume"]:
        out = out[out["volume"].fillna(0) > 0]
    # 价格异常
    out = out[(out["close"] > UNIVERSE["price_min"]) &
              (out["close"] < UNIVERSE["price_max"])]
    out = out.dropna(subset=["close", "high", "low", "open"])
    # amount 千元 → 元
    out["amount"] = out["amount"] * amount_scale
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    cols = [c for c in DAILY_COLS if c in out.columns]
    return out[cols]


def listing_trade_days(df: pd.DataFrame) -> int:
    """已上市交易日数（用本地有效 K 线根数近似，首下载即从 start_date 起）。"""
    return 0 if df is None or len(df) == 0 else len(df)


def merge_industry(stock_df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
    """把 baostock 行业分类合并进股票列表。"""
    out = stock_df.copy()
    if ind_df is None or len(ind_df) == 0:
        out["industry"] = ""
        return out
    # 先丢弃列表里的占位/空 industry 列，避免 merge 时同名列变成 industry_x/industry_y
    if "industry" in out.columns:
        out = out.drop(columns=["industry"])
    ind = ind_df.copy()
    ind["code"] = ind["code"].str.replace("sh.", "", regex=False)\
                             .str.replace("sz.", "", regex=False)
    if "industry" not in ind.columns:
        out["industry"] = ""
        return out
    ind = ind[["code", "industry"]].drop_duplicates("code")
    out = out.merge(ind, on="code", how="left")
    # 去掉证监会门类代码前缀（如 "C35专用设备制造业" → "专用设备制造业"）
    out["industry"] = (out["industry"].fillna("")
                       .astype(str)
                       .str.replace(r"^[A-Z]\d{1,2}", "", regex=True)
                       .str.strip())
    return out


def filter_pool(df: pd.DataFrame) -> pd.DataFrame:
    """最终强制过滤：主板 + 名称非ST/退（在线与缓存统一入口）。"""
    out = df.copy()
    out = out[out["code"].astype(str).map(is_main_board)]
    if "name" in out.columns:
        out = out[out["name"].fillna("").map(is_clean_name)]
    return out.reset_index(drop=True)
