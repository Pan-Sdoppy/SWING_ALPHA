# -*- coding: utf-8 -*-
"""
盘中实时服务（predict.py 盘中模式专用，update 不调用）：
- 拉取新浪全市场快照，给每只股票在内存中追加“当日临时K线”，不写入 data/raw 正式增量；
  快照另存 data/intraday/_spot_latest.parquet 仅作临时缓存。
- 仅对日线初筛后的少量候选股拉新浪分钟线，判定封板时间/开板次数/红旗飘飘/竞价高开。
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import storage
from .config import DAILY_COLS, LIMIT, INTRADAY_DIR
from .datasource import akshare_source as aksrc
from .utils import get_logger

log = get_logger()


def trading_minutes_elapsed(now: datetime) -> float:
    """已交易分钟数（9:30-11:30, 13:00-15:00）。"""
    t = now.hour * 60 + now.minute
    morning = max(0, min(t, 11 * 60 + 30) - (9 * 60 + 30))
    afternoon = max(0, min(t, 15 * 60) - 13 * 60)
    return float(max(1, morning + afternoon))


TOTAL_MIN = 240.0


def load_spot(use_cache: bool = False) -> Optional[pd.DataFrame]:
    if use_cache and storage.SPOT_CACHE.exists():
        return pd.read_parquet(storage.SPOT_CACHE)
    try:
        spot = aksrc.fetch_spot()
        if spot is not None and len(spot):
            spot.to_parquet(storage.SPOT_CACHE, index=False)
        return spot
    except Exception as e:  # noqa
        log.error(f"实时快照获取失败: {e}")
        if storage.SPOT_CACHE.exists():
            log.warning("回退到最近一次盘中快照缓存")
            return pd.read_parquet(storage.SPOT_CACHE)
        return None


def build_intraday_dfs(stock_dfs: Dict[str, pd.DataFrame], spot: pd.DataFrame,
                       now: Optional[datetime] = None,
                       asof: str = None) -> Dict[str, pd.DataFrame]:
    """在内存中给每只股票追加当日临时 bar（投影全天量能），不修改原始 df 文件。"""
    now = now or datetime.now()
    asof = asof or now.strftime("%Y-%m-%d")
    elapsed_ratio = min(1.0, trading_minutes_elapsed(now) / TOTAL_MIN)
    spot = spot.set_index("code")
    out = dict(stock_dfs)  # 浅拷贝
    n_added = 0
    for code, df in stock_dfs.items():
        if code not in spot.index:
            continue
        r = spot.loc[code]
        price = r.get("price")
        if pd.isna(price) or price <= 0:
            continue
        last = df.iloc[-1]
        preclose = r.get("preclose")
        if pd.isna(preclose) or preclose <= 0:
            preclose = float(last["close"])
        op = r.get("open"); hi = r.get("high"); lo = r.get("low")
        if pd.isna(op) or op <= 0:
            op = price
        if pd.isna(hi) or hi <= 0:
            hi = max(op, price)
        if pd.isna(lo) or lo <= 0:
            lo = min(op, price)
        pct = r.get("pctChg")
        if pd.isna(pct):
            pct = (price / preclose - 1) * 100
        # 新浪快照：volume 单位股、amount 单位元
        cum_vol = float(r.get("volume")) if pd.notna(r.get("volume")) else np.nan
        cum_amt = float(r.get("amount")) if pd.notna(r.get("amount")) else np.nan
        # 换手率：快照缺失时，用近60日 volume/(turn/100) 估流通股本，反推累计换手
        turn = r.get("turn")
        if pd.isna(turn):
            hist = df.tail(60)
            est = hist["volume"] / (hist["turn"] / 100.0)
            est = est.replace([np.inf, -np.inf], np.nan).dropna()
            float_shares = float(est.median()) if len(est) else np.nan
            turn = (cum_vol / float_shares * 100.0) if (
                pd.notna(cum_vol) and pd.notna(float_shares) and float_shares > 0) else np.nan
        # 按时间进度投影全天量能，供量比等因子使用；换手用累计值
        vol = cum_vol / elapsed_ratio if pd.notna(cum_vol) else np.nan
        amt = cum_amt / elapsed_ratio if pd.notna(cum_amt) else np.nan
        bar = pd.DataFrame([{
            "date": asof, "code": code, "open": float(op), "high": float(hi),
            "low": float(lo), "close": float(price), "preclose": float(preclose),
            "volume": vol, "amount": amt,
            "turn": float(turn) if pd.notna(turn) else np.nan,
            "pctChg": float(pct),
        }])
        # 若同日已存在（重复运行）则替换
        d = df[df["date"] != asof]
        out[code] = pd.concat([d, bar], ignore_index=True)
        n_added += 1
    log.info(f"已为 {n_added} 只股票追加当日({asof})盘中临时K线（投影进度 "
             f"{elapsed_ratio*100:.0f}%，不落正式库）")
    return out


def minute_seal_info(code: str, preclose: float) -> Optional[dict]:
    """用5分钟线判定封板时间/开板次数/红旗/竞价高开。best-effort。"""
    df = aksrc.fetch_minute(code, period="5")
    if df is None or not len(df) or preclose is None:
        return None
    limitpx = round(preclose * 1.10, 2)
    eps = LIMIT["limit_eps"]
    dt = pd.to_datetime(df["datetime"])
    sealed = (df["close"] >= limitpx - eps).values
    high = df["high"].values
    times = dt.dt.strftime("%H:%M").values
    # 竞价高开（用当日第一根开盘 vs 昨收）
    gap = (float(df["open"].iloc[0]) / preclose - 1) * 100
    touch_idx = np.where(high >= limitpx - eps)[0]
    if len(touch_idx) == 0:
        return {"gap_pct": gap, "seal_state": "未触板", "reopens": 0}
    first_touch = times[touch_idx[0]]
    # 首次封板后，封→开 切换次数
    reopens = 0
    prev = sealed[touch_idx[0]] if touch_idx[0] < len(sealed) else False
    final_seal_time = first_touch
    for k in touch_idx:
        cur = bool(sealed[k])
        if prev and not cur:
            reopens += 1
        if cur:
            final_seal_time = times[k]
        prev = cur
    seal_after_1130 = final_seal_time > "11:30"
    # 红旗飘飘：14:30 后仍开板，或反复开板≥3
    red_flag = (len(times) and np.any((times[touch_idx] >= "14:30") &
                                      (~sealed[touch_idx]))) or reopens >= 3
    state = "一字/T字" if df["low"].iloc[0] >= limitpx - eps else \
            ("回封" if reopens else "首封")
    return {"gap_pct": float(gap), "first_touch": str(first_touch),
            "final_seal": str(final_seal_time), "seal_after_1130": bool(seal_after_1130),
            "reopens": int(reopens), "red_flag": bool(red_flag),
            "seal_state": state}
