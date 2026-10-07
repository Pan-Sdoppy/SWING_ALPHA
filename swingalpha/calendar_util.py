# -*- coding: utf-8 -*-
"""
交易日历与运行时段判定。
- 预期最新交易日：周一~周五 15:00 后=当天，15:00 前=上一交易日；周末=上周五。
- 节假日以 baostock 交易日历为准（离线时回退到朴素工作日规则）。
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Optional, List

from .utils import get_logger

log = get_logger()


def _naive_prev_workday(d: date) -> date:
    """不考虑节假日的上一个工作日（周一→上周五）。"""
    while d.weekday() >= 5:  # 5=周六 6=周日
        d -= timedelta(days=1)
    return d


def _is_trading_day_bs(d: date, bs=None) -> Optional[bool]:
    """用 baostock 交易日历判断；失败返回 None。
    baostock 为全局有状态会话，直接导入已登录的同一模块即可（bs 参数仅占位）。"""
    try:
        import baostock as _bs
        rs = _bs.query_trade_dates(start_date=d.strftime("%Y-%m-%d"),
                                   end_date=d.strftime("%Y-%m-%d"))
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        return rows[0][1] == "1"
    except Exception as e:  # noqa
        log.warning(f"交易日历查询失败 {d}: {e}")
        return None


def _trade_calendar(start: date, end: date):
    """一次区间查询交易日历，返回 [(date, is_trading_bool)]；失败返回 []。"""
    try:
        import baostock as _bs
        rs = _bs.query_trade_dates(start_date=start.strftime("%Y-%m-%d"),
                                   end_date=end.strftime("%Y-%m-%d"))
        out = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            out.append((datetime.strptime(row[0], "%Y-%m-%d").date(),
                        row[1] == "1"))
        return out
    except Exception as e:  # noqa
        log.warning(f"交易日历区间查询失败 {start}~{end}: {e}")
        return []


def latest_trading_day(bs=None, now: Optional[datetime] = None) -> date:
    """
    返回“数据应当更新到的”最新交易日。
    盘后(≥15:00) 才把当天计入；盘中/盘前计上一交易日；周末取上周五。
    节假日以 baostock 交易日历为准（一次区间查询），离线回退朴素工作日规则。
    """
    now = now or datetime.now()
    today = now.date()

    if today.weekday() >= 5:           # 周末 → 上周五（朴素）
        candidate = today
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
    elif now.hour < 15:                # 工作日盘前/盘中
        candidate = _naive_prev_workday(today - timedelta(days=1))
    else:                              # 工作日盘后
        candidate = today

    cal = _trade_calendar(candidate - timedelta(days=20), candidate)
    if cal:
        tdays = [d for d, ok in cal if ok and d <= candidate]
        if tdays:
            return max(tdays)
    return candidate


def trading_days_between(start: date, end: date, bs=None) -> List[date]:
    """区间内交易日列表（bs 可用时为真实日历，否则朴素工作日）。"""
    out, d = [], start
    while d <= end:
        if bs is None:
            if d.weekday() < 5:
                out.append(d)
        else:
            if _is_trading_day_bs(d, bs) is True:
                out.append(d)
        d += timedelta(days=1)
    return out


def session_state(now: Optional[datetime] = None) -> str:
    """
    运行时段：
      intraday  工作日 09:30~15:00（盘中，需拉实时快照）
      preopen   工作日 09:30 前（按上一交易日存量）
      afterclose工作日 15:00 后（按当日存量）
      weekend   周末
    节假日不在此判定，调用方应结合最新交易日比较。
    """
    now = now or datetime.now()
    wd = now.weekday()
    if wd >= 5:
        return "weekend"
    t = now.hour * 60 + now.minute
    if t < 9 * 60 + 30:
        return "preopen"
    if t < 15 * 60:
        return "intraday"
    return "afterclose"
