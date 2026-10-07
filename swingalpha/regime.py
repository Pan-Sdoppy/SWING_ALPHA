# -*- coding: utf-8 -*-
"""
市场环境（天时）四周期与门控。
夏：收盘创20日新高；秋：1~3日未新高；深秋：≥4日未新高但守20日线；
冬：≥4日未新高且破20日线。指数级环境对全市场生效；板块相位数据可用时在个股层叠加。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import REGIME, BREAKFAST, BOTTOM_CLUSTER
from .utils import get_logger

log = get_logger()


def index_phase(close: pd.Series, at: str, ma: int = REGIME["ma"]) -> str:
    s = close.loc[:at].dropna()
    if len(s) < ma + 1:
        return "unknown"
    ma_s = s.rolling(ma, min_periods=ma).mean()
    today_close = s.iloc[-1]
    high20 = s.iloc[-ma:].max()
    # 距今天数未创20日新高
    days_no_new = 0
    for v in s.iloc[::-1]:
        if v >= high20 - 1e-9:
            break
        days_no_new += 1
    above_ma = today_close >= ma_s.iloc[-1]
    is_new_high = today_close >= high20 - 1e-9
    if is_new_high:
        return "summer"
    if days_no_new <= REGIME["autumn_days"]:
        return "autumn"
    if above_ma:
        return "deep_autumn"
    return "winter"


def market_snapshot(panel, at: str) -> dict:
    """返回当日市场环境快照（门控用）。"""
    snap = {"date": at, "phase": "unknown", "csi1000_pct": np.nan,
            "broken_rate": np.nan, "sh_pct": np.nan, "sh_5d": np.nan,
            "extreme_panic": False}
    if "sh000300" in panel.index_close:
        snap["phase"] = index_phase(panel.index_close["sh000300"], at)
    elif "sh000001" in panel.index_close:
        snap["phase"] = index_phase(panel.index_close["sh000001"], at)

    if "sh000852" in panel.index_pct:
        s = panel.index_pct["sh000852"].loc[:at].dropna()
        if len(s):
            snap["csi1000_pct"] = float(s.iloc[-1])
    if "sh000001" in panel.index_close:
        c = panel.index_close["sh000001"].loc[:at].dropna()
        pp = panel.index_pct.get("sh000001")
        if pp is not None:
            pp = pp.loc[:at].dropna()
            if len(pp):
                snap["sh_pct"] = float(pp.iloc[-1])
                snap["sh_5d"] = float(pp.iloc[-5:].sum()) if len(pp) >= 1 else np.nan
        # 极端情绪日：上证单日≤-1.5% 或 5日累计≤-5%
        if snap["sh_pct"] <= -1.5 or (not np.isnan(snap["sh_5d"]) and snap["sh_5d"] <= -5.0):
            snap["extreme_panic"] = True

    if panel.broken_rate is not None and at in panel.broken_rate.index:
        snap["broken_rate"] = float(panel.broken_rate.loc[at])
    return snap


def breakfast_gate(snap: dict) -> tuple:
    """早餐策略环境门控：非冬天 + 中证1000为正 + 炸板率<35%。返回(通过, 原因)。"""
    reasons = []
    if snap.get("phase") == "winter":
        reasons.append("市场处于冬天(主跌段)")
    if BREAKFAST["require_index_up"]:
        if np.isnan(snap.get("csi1000_pct", np.nan)):
            reasons.append("中证1000数据缺失")
        elif snap["csi1000_pct"] <= 0:
            reasons.append(f"中证1000 {snap['csi1000_pct']:.2f}%≤0")
    br = snap.get("broken_rate", np.nan)
    if not np.isnan(br) and br >= BREAKFAST["max_broken_board_rate"]:
        reasons.append(f"炸板率 {br:.0f}%≥{BREAKFAST['max_broken_board_rate']:.0f}%")
    if not np.isnan(br) and br < BREAKFAST["min_broken_board_rate"]:
        reasons.append(f"炸板率 {br:.0f}%<{BREAKFAST['min_broken_board_rate']:.0f}%(死水日无溢价)")
    return (len(reasons) == 0, "；".join(reasons))


def bottom_cluster_gate(snap: dict) -> tuple:
    """底部簇环境闸门：熊市主跌段（冬天）禁用。"""
    if BOTTOM_CLUSTER["require_not_bear"] and snap.get("phase") == "winter":
        return False, "市场冬天(主跌段)，底部簇策略禁用"
    return True, ""
