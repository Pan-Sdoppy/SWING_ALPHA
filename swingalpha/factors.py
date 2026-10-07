# -*- coding: utf-8 -*-
"""
因子引擎：把全部个股日线对齐到统一交易日历，构建宽面板，向量化计算因子。
滚动类统计走 accel（CUDA 可选，否则 CPU）；连板等状态机按 numpy 向量化推进。
因子结果缓存 data/processed/factor_data.parquet，数据未变直接复用。
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import accel, storage
from .config import LIMIT, INDICES
from .utils import get_logger

log = get_logger()

ROLL_FACTORS = ["ma5", "ma10", "ma20", "ma60", "vma5", "vma20", "vma60",
                "high20", "high60", "high250", "low20", "low60", "amtma60"]


def _prev(x: np.ndarray) -> np.ndarray:
    """右移一列（用上一交易日数据），首列 NaN。"""
    p = np.empty_like(x)
    p[:, 0] = np.nan
    p[:, 1:] = x[:, :-1]
    return p


class Panel:
    def __init__(self):
        self.codes = None
        self.dates = None
        self.name_map: Dict[str, str] = {}
        self.ind_map: Dict[str, str] = {}
        self.f: Dict[str, np.ndarray] = {}
        self.index_close: Dict[str, pd.Series] = {}
        self.index_pct: Dict[str, pd.Series] = {}
        self.broken_rate: Optional[pd.Series] = None
        self.asof: Optional[str] = None

    def row_of(self, code: str) -> Optional[int]:
        idx = np.where(self.codes == code)[0]
        return int(idx[0]) if len(idx) else None

    def col_of(self, date: str) -> Optional[int]:
        idx = np.where(self.dates == date)[0]
        return int(idx[0]) if len(idx) else None


def _wide(long: pd.DataFrame, dates: np.ndarray, codes: np.ndarray,
          col: str) -> np.ndarray:
    pv = long.pivot_table(index="code", columns="date", values=col, aggfunc="last")
    pv = pv.reindex(index=codes, columns=dates)
    return pv.values.astype(np.float32)


def _board_count(sealed: np.ndarray) -> np.ndarray:
    """截至当日的连续封板天数（向量化按日推进）。"""
    n, T = sealed.shape
    cnt = np.zeros((n, T), dtype=np.float32)
    # NaN（缺失/停牌）必须视为未封板：NaN.astype(bool) 会误为 True
    s = np.nan_to_num(sealed, nan=0.0).astype(bool)
    run = np.zeros(n, dtype=np.float32)
    for t in range(T):
        run = np.where(s[:, t], run + 1.0, 0.0)
        cnt[:, t] = run
    return cnt


def build_panel(stock_dfs: Dict[str, pd.DataFrame], name_map: Dict[str, str],
                industry_map: Dict[str, str], backend: str = None,
                asof: Optional[str] = None) -> Panel:
    backend = backend or accel.detect_backend()
    p = Panel()
    p.name_map, p.ind_map = name_map, industry_map

    # 主日历：用上证指数，缺失则并集
    sh = storage.load_index("sh000001")
    if sh is not None and len(sh):
        dates = np.sort(sh["date"].astype(str).unique())
    else:
        dates = np.array(sorted({d for df in stock_dfs.values() for d in df["date"].astype(str)}))
    if asof and asof not in dates:
        dates = np.sort(np.append(dates, asof))
    codes = np.array(sorted(stock_dfs.keys()))
    p.codes, p.dates, p.asof = codes, dates, asof

    frames = []
    for c, df in stock_dfs.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d["code"] = c
        d["date"] = d["date"].astype(str)
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    for col in ["open", "high", "low", "close", "preclose", "volume",
                "amount", "turn", "pctChg"]:
        long[col] = pd.to_numeric(long[col], errors="coerce")

    O = _wide(long, dates, codes, "open")
    H = _wide(long, dates, codes, "high")
    L = _wide(long, dates, codes, "low")
    C = _wide(long, dates, codes, "close")
    VOL = _wide(long, dates, codes, "volume")
    AMT = _wide(long, dates, codes, "amount")
    TURN = _wide(long, dates, codes, "turn")
    PCT = _wide(long, dates, codes, "pctChg")
    PREC = _wide(long, dates, codes, "preclose")
    np.nan_to_num(PREC, nan=np.nan)

    # 滚动因子（用上一交易日口径，避免当日未来函数）
    Hp, Lp, Cp, Vp, Ap = _prev(H), _prev(L), _prev(C), _prev(VOL), _prev(AMT)
    f = {}
    f["ma5"] = accel.rolling(Cp, 5, "mean", backend)
    f["ma10"] = accel.rolling(Cp, 10, "mean", backend)
    f["ma20"] = accel.rolling(Cp, 20, "mean", backend)
    f["ma60"] = accel.rolling(Cp, 60, "mean", backend)
    f["vma5"] = accel.rolling(Vp, 5, "mean", backend)
    f["vma20"] = accel.rolling(Vp, 20, "mean", backend)
    f["vma60"] = accel.rolling(Vp, 60, "mean", backend)
    f["high20"] = accel.rolling(Hp, 20, "max", backend)
    f["high60"] = accel.rolling(Hp, 60, "max", backend)
    f["high250"] = accel.rolling(Hp, 250, "max", backend)
    f["low20"] = accel.rolling(Lp, 20, "min", backend)
    f["low60"] = accel.rolling(Lp, 60, "min", backend)
    f["amtma60"] = accel.rolling(Ap, 60, "mean", backend)

    # 涨跌停
    limitpx = np.round(PREC * (1 + LIMIT["pct"]), 2)
    eps = LIMIT["limit_eps"]
    sealed = (C >= limitpx - eps) & (PCT >= LIMIT["eps_pct"])
    touched = H >= limitpx - eps
    broken = touched & ~sealed
    oneword = sealed & (L >= limitpx - eps)
    limitdown = PCT <= LIMIT["limit_down_pct"]
    f["limitpx"] = limitpx
    f["sealed"] = sealed.astype(np.float32)
    f["touched"] = touched.astype(np.float32)
    f["broken"] = broken.astype(np.float32)
    f["oneword"] = oneword.astype(np.float32)
    f["limitdown"] = limitdown.astype(np.float32)
    f["boardcnt"] = _board_count(sealed)

    # 量比（对前20日均量）、20日动能、距年高回撤
    C20 = np.full_like(C, np.nan)
    C20[:, 20:] = C[:, :-20]
    with np.errstate(invalid="ignore", divide="ignore"):
        f["vr20"] = (VOL / f["vma20"]).astype(np.float32)
        f["vr5"] = (VOL / f["vma5"]).astype(np.float32)
        f["mom20"] = (C / C20 - 1.0).astype(np.float32)
        f["dd250"] = (C / f["high250"] - 1.0).astype(np.float32)
    f.update({"open": O, "high": H, "low": L, "close": C, "volume": VOL,
              "amount": AMT, "turn": TURN, "pctChg": PCT, "preclose": PREC})
    p.f = f

    # 指数序列
    for idx in INDICES:
        idf = storage.load_index(idx)
        if idf is None or not len(idf):
            continue
        idf = idf.copy()
        idf["date"] = idf["date"].astype(str)
        idf["close"] = pd.to_numeric(idf["close"], errors="coerce")
        idf["pctChg"] = pd.to_numeric(idf["pctChg"], errors="coerce")
        s = idf.set_index("date")["close"].reindex(dates)
        pp = idf.set_index("date")["pctChg"].reindex(dates)
        p.index_close[idx] = s
        p.index_pct[idx] = pp

    # 市场炸板率（按日横截面）
    touched_n = np.nansum(touched, axis=0)
    broken_n = np.nansum(broken, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(touched_n > 0, broken_n / touched_n * 100.0, np.nan)
    p.broken_rate = pd.Series(rate, index=dates)
    return p


def load_all_stock_dfs(codes) -> Dict[str, pd.DataFrame]:
    out = {}
    for c in codes:
        df = storage.load_daily(c)
        if df is not None and len(df):
            out[c] = df
    return out


# 参与缓存的因子列（boardcnt 等状态机每次重建）
_CACHE_RAW = ["open", "high", "low", "close", "preclose", "volume",
              "amount", "turn", "pctChg"]
_CACHE_FAC = ROLL_FACTORS + ["vr20", "vr5", "mom20", "dd250", "limitpx",
                             "sealed", "touched", "broken", "oneword", "limitdown"]


def export_long(panel: Panel) -> pd.DataFrame:
    """宽面板 → 长表（含原始价量与滚动/涨跌停因子），用于 parquet 缓存。"""
    n, T = len(panel.codes), len(panel.dates)
    code_col = np.repeat(panel.codes, T)
    date_col = np.tile(panel.dates, n)
    data = {"code": code_col, "date": date_col}
    for col in _CACHE_RAW + _CACHE_FAC:
        if col in panel.f:
            data[col] = panel.f[col].reshape(-1)
    out = pd.DataFrame(data)
    return out.dropna(subset=["close"])


def build_panel_from_long(long: pd.DataFrame, name_map, industry_map,
                          asof: Optional[str] = None) -> Panel:
    """由缓存长表快速重建宽面板（仅重算连板状态机、指数、炸板率）。"""
    p = Panel()
    p.name_map, p.ind_map, p.asof = name_map, industry_map, asof
    codes = np.array(sorted(long["code"].unique()))
    sh = storage.load_index("sh000001")
    if sh is not None and len(sh):
        dates = np.sort(sh["date"].astype(str).unique())
    else:
        dates = np.array(sorted(long["date"].unique()))
    if asof and asof not in dates:
        dates = np.sort(np.append(dates, asof))
    p.codes, p.dates = codes, dates
    for col in _CACHE_RAW + _CACHE_FAC:
        if col in long.columns:
            p.f[col] = _wide(long, dates, codes, col)
    p.f["boardcnt"] = _board_count(p.f["sealed"])
    for idx in INDICES:
        idf = storage.load_index(idx)
        if idf is None or not len(idf):
            continue
        idf = idf.copy()
        idf["date"] = idf["date"].astype(str)
        for c in ("close", "pctChg"):
            idf[c] = pd.to_numeric(idf[c], errors="coerce")
        s = idf.set_index("date")
        p.index_close[idx] = s["close"].reindex(dates)
        p.index_pct[idx] = s["pctChg"].reindex(dates)
    touched = np.nansum(p.f["touched"], axis=0)
    broken = np.nansum(p.f["broken"], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(touched > 0, broken / touched * 100.0, np.nan)
    p.broken_rate = pd.Series(rate, index=dates)
    return p
