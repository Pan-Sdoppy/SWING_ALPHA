# -*- coding: utf-8 -*-
"""
回测引擎（严格无未来函数）
--------------------------------------------------------------------
数据隔离铁律：
  · 信号：逐日 t 调用与 predict 完全相同的扫描函数；因子为因果滚动（rolling 用上一
    交易日口径），扫描在第 t 列只读取 ≤t 的数据；按当日评分取 topN。
  · 超短(隔日套利)：t 日涨停价(=收盘价，模拟排板成交)买入，t+1 开盘卖出；门控/真空
    否决/一字板均与实盘一致（一字板买不进，剔除）。
  · 波段(A杀低吸)：t 收盘出信号，t+1 开盘买入；之后只用入场之后的行情记账——
    盘中触止损按止损出，否则按”放量大阳线收盘”退出（涨幅≥5%且量≥5日均量2倍，最长20日兜底）。
  · 成本：买卖佣金、卖出印花税、滑点均计入；t+1 开盘停牌则顺延，仍停牌则撤单。
写死最优参数：超短 top1，波段 bigbar/top3（经全网格验证后锁定，不再做参数寻优）。
输出：【超短】【波段】各一张总表 + 年度(2017-2025)与 2026 逐月明细
      （交易数/胜率/平均收益/最大回撤/平均持仓日）。CSV 留档 data/processed/backtests/。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import accel, factors, storage, report
from .config import BACKTEST as BT, BACKTEST_DIR
from .regime import market_snapshot
from .strategies.ultra_short import scan_ultra_short
from .strategies.swing import scan_swing
from .utils import get_logger

log = get_logger()

BUY_COST = BT["commission"] + BT["slippage"]
SELL_COST = BT["commission"] + BT["stamp_tax"] + BT["slippage"]
TRADING_DAYS = 244


# ============================================================
# 信号生成（逐日，只用 ≤t 数据）
# ============================================================
def _maps():
    name_map, ind_map = {}, {}
    cached, _ = storage.get_stock_list()
    if cached is not None and len(cached):
        name_map = dict(zip(cached["code"].astype(str),
                            cached.get("name", pd.Series([""] * len(cached)))))
        if "industry" in cached.columns:
            ind_map = dict(zip(cached["code"].astype(str),
                               cached["industry"].fillna("")))
    for c, rec in storage.load_meta().items():
        name_map.setdefault(c, rec.get("name", ""))
    return name_map, ind_map


def build_signals(backend: str, start: str, codes: Optional[List[str]] = None,
                  bypass_season: bool = False):
    """全历史构建一次面板（因果因子），逐日扫描，返回 panel 与 {date: [(code,score,...)]}。
    bypass_season=True 时绕过冬天/门控过滤（实验用：不区分大盘四季每天都买）。"""
    name_map, ind_map = _maps()
    use_codes = codes if codes else sorted(name_map.keys())
    dfs = factors.load_all_stock_dfs(use_codes)
    if not dfs:
        raise RuntimeError("本地无个股日线，请先运行 update.py 与 update.py --backfill 2020-07-01")
    log.info(f"载入本地日线 {len(dfs)} 只，构建全历史因果因子面板…")
    panel = factors.build_panel(dfs, name_map, ind_map, backend=backend, asof=None)

    dates = list(panel.dates.astype(str))
    trade_dates = [d for d in dates if d >= start]
    code_i = {c: i for i, c in enumerate(panel.codes)}
    gidx = {d: k for k, d in enumerate(dates)}

    ultra: Dict[str, List[Tuple]] = {}
    swing: Dict[str, List[Tuple]] = {}
    # 回测逐日扫描期间屏蔽策略层的每日 INFO 日志（漏斗/闸门/出票），仅保留进度条
    _prev_level = log.level
    log.setLevel(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in tqdm(trade_dates, ncols=88, desc="逐日信号"):
            panel.asof = d
            snap = market_snapshot(panel, d)
            du = scan_ultra_short(panel, snap, mode="after", minute_info={}, show_all=bypass_season)
            if len(du):
                u = du[(du["信号"] == "早餐·压力位")
                       & (du["A1压力位"].astype(str).str.startswith("命中"))
                       & (~du["建议"].astype(str).str.contains("一字"))]
                if len(u):
                    ultra[d] = sorted(zip(u["代码"], u["评分"]),
                                      key=lambda z: -float(z[1]))
            ds = scan_swing(panel, snap, mode="after", show_all=bypass_season)
            if len(ds):
                s = ds[ds["建议"].astype(str).str.contains("可低吸|深跌低吸")]
                if len(s):
                    swing[d] = sorted(zip(s["代码"], s["评分"], s["止损"]),
                                      key=lambda z: -float(z[1]))
    log.setLevel(_prev_level)
    panel.asof = None
    return panel, dates, trade_dates, gidx, code_i, ultra, swing


# ============================================================
# 价格工具
# ============================================================
def _net(entry: float, exit_px: float) -> float:
    return (exit_px / entry) * (1 - SELL_COST) / (1 + BUY_COST) - 1.0


def _next_open(panel, i, k, last):
    O = panel.f["open"]
    for j in range(k + 1, min(k + 1 + BT["entry_roll_days"], last + 1)):
        px = O[i, j]
        if np.isfinite(px) and px > 0:
            return j, float(px)
    return None


# ============================================================
# 超短：t 收盘(涨停价)买，t+1 开盘卖（持仓不隔夜重叠）
# ============================================================
def simulate_ultra(panel, dates, trade_dates, gidx, code_i, signals, topn):
    C = panel.f["close"]
    last = len(dates) - 1
    trades = []
    daily_ret = []
    for d in trade_dates:
        t = gidx[d]
        picks = signals.get(d, [])[:topn]
        rets = []
        for code, score in picks:
            i = code_i.get(code)
            if i is None:
                continue
            entry = C[i, t]
            if not np.isfinite(entry) or entry <= 0:
                continue
            nxt = _next_open(panel, i, t, last)
            if nxt is None:
                continue
            j, ex = nxt
            r = _net(float(entry), ex)
            rets.append(r)
            trades.append(dict(code=code, signal_date=d, entry_date=d,
                               exit_date=dates[j], entry_px=float(entry),
                               exit_px=ex, hold_days=1, ret=r,
                               score=float(score), reason="次日开盘卖"))
        daily_ret.append(np.mean(rets) if rets else 0.0)
    eq = [1.0]
    for r in daily_ret:
        eq.append(eq[-1] * (1 + r))
    curve_dates = [trade_dates[0]] + trade_dates if trade_dates else []
    return trades, curve_dates, np.array(eq)


# ============================================================
# 波段：N 个独立分槽，t+1 开盘买，止损/持有期/放量大阳退出
# ============================================================
@dataclass
class _Slot:
    cash: float
    init: float
    pos: Optional[dict] = None
    pending: Optional[dict] = None


def _manage_position(panel, slot: _Slot, k: int, last: int, variant: str,
                     first_day_loss_exit: bool = False):
    """对持仓在第 k 天做止损/退出判定；返回当日平仓的 trade 片段或 None。
    first_day_loss_exit=True 时，入场当天若收盘<入场价，当天平仓（实验用）。"""
    pos = slot.pos
    if pos is None:
        return None
    i = pos["i"]
    H, Lw, C, O = (panel.f["high"][i], panel.f["low"][i],
                   panel.f["close"][i], panel.f["open"][i])
    VOL, VMA5, PCT = panel.f["volume"][i], panel.f["vma5"][i], panel.f["pctChg"][i]
    exit_px, reason = None, None

    # 0) 首日亏损即平仓（实验：入场当天收盘<入场价则当天走）
    if first_day_loss_exit and k == pos["entry_idx"] and np.isfinite(C[k]) and C[k] < pos["entry_px"]:
        exit_px, reason = float(C[k]), "首日亏损平仓"

    # 更新持仓峰值（用于移动止损）
    if np.isfinite(H[k]) and H[k] > 0:
        pos["peak"] = max(pos.get("peak", pos["entry_px"]), float(H[k]))

    # 1) 盘中止损（开盘已跳空低于止损则按开盘出）
    stop = pos["stop"]
    if stop and np.isfinite(Lw[k]) and Lw[k] <= stop:
        if np.isfinite(O[k]) and O[k] < stop:
            exit_px, reason = float(O[k]), "止损(跳空)"
        else:
            exit_px, reason = float(stop), "止损"
    # 2) 固定持有期，收盘退出（停牌日收盘为 NaN，顺延到可成交日）
    if exit_px is None and variant.startswith("hold"):
        H_days = int(variant.replace("hold", ""))
        if k - pos["entry_idx"] >= H_days and np.isfinite(C[k]):
            exit_px, reason = float(C[k]), f"持有{H_days}日"
    # 3) 放量大阳线收盘退出；最长持有兜底
    if exit_px is None and variant in ("bigbar", "bigbar_trail10", "bigbar_target15"):
        held = k - pos["entry_idx"]
        big_up = np.isfinite(PCT[k]) and PCT[k] >= BT["bigbar_pct"]
        big_vol = (np.isfinite(VOL[k]) and np.isfinite(VMA5[k])
                   and VMA5[k] > 0 and VOL[k] >= BT["bigbar_vol_ratio"] * VMA5[k])
        if held >= 1 and big_up and big_vol and np.isfinite(C[k]):
            exit_px, reason = float(C[k]), "放量大阳"
        elif held >= BT["bigbar_max_hold"] and np.isfinite(C[k]):
            exit_px, reason = float(C[k]), "最长持有到期"
    # 4) 移动止损（从峰值回撤X%）
    if exit_px is None and variant in ("trail10", "trail12", "bigbar_trail10"):
        trail_pct = 0.10 if "trail10" in variant else 0.12
        peak = pos.get("peak", pos["entry_px"])
        trail_line = peak * (1 - trail_pct)
        held = k - pos["entry_idx"]
        if held >= 1 and np.isfinite(Lw[k]) and Lw[k] <= trail_line and peak > pos["entry_px"]:
            if np.isfinite(O[k]) and O[k] < trail_line:
                exit_px, reason = float(O[k]), f"移动止损(跳空,{int(trail_pct*100)}%)"
            else:
                exit_px, reason = float(trail_line), f"移动止损({int(trail_pct*100)}%)"
        elif held >= BT["bigbar_max_hold"] and np.isfinite(C[k]):
            exit_px, reason = float(C[k]), "最长持有到期"
    # 5) 目标止盈（相对entry涨幅X%）
    if exit_px is None and variant in ("target15", "target20", "bigbar_target15"):
        target_pct = 0.20 if variant == "target20" else 0.15
        target_px = pos["entry_px"] * (1 + target_pct)
        held = k - pos["entry_idx"]
        if held >= 1 and np.isfinite(H[k]) and H[k] >= target_px:
            exit_px, reason = float(target_px), f"目标止盈(+{int(target_pct*100)}%)"
        elif held >= BT["bigbar_max_hold"] and np.isfinite(C[k]):
            exit_px, reason = float(C[k]), "最长持有到期"
    # 期末强平（若末日停牌，向前取最近有效收盘）
    if exit_px is None and k == last:
        j = k
        while j >= pos["entry_idx"] and not np.isfinite(C[j]):
            j -= 1
        if j >= pos["entry_idx"]:
            exit_px, reason = float(C[j]), "期末强平"

    if exit_px is None:
        return None
    proceeds = pos["shares"] * exit_px * (1 - SELL_COST)
    r = proceeds / pos["entry_cash"] - 1.0
    slot.cash = proceeds
    slot.pos = None
    return dict(code=pos["code"], signal_date=pos["signal_date"],
                entry_date=pos["entry_date"], exit_date=panel.dates[k],
                entry_px=pos["entry_px"], exit_px=exit_px, stop=pos["stop"],
                hold_days=k - pos["entry_idx"] + 1, ret=r,
                score=pos["score"], reason=reason)


def simulate_swing(panel, dates, trade_dates, gidx, code_i, signals,
                   topn: int, variant: str, first_day_loss_exit: bool = False):
    last = len(dates) - 1
    g0 = gidx[trade_dates[0]]
    slots = [_Slot(cash=1.0 / topn, init=1.0 / topn) for _ in range(topn)]
    trades = []
    eq_curve = []
    cancelled = 0

    def equity_at(k):
        tot = 0.0
        for s in slots:
            if s.pos is not None:
                px = panel.f["close"][s.pos["i"], k]
                tot += s.pos["shares"] * px if np.isfinite(px) else s.pos["entry_cash"]
            else:
                tot += s.cash
        return tot

    for k in range(g0, last + 1):
        d = str(dates[k])
        # 1) 执行待成交买单（t+1 开盘）；pending 记录创建日，超期撤单
        for s in slots:
            if s.pending is not None:
                p = s.pending
                i = p["i"]
                px = panel.f["open"][i, k]
                if np.isfinite(px) and px > 0:
                    if p["stop"] > 0 and px <= p["stop"]:
                        # t+1 开盘已跌破止损=支撑破位，低吸不成立，放弃成交（不接飞刀）
                        s.pending = None
                        cancelled += 1
                    elif BT.get("entry_skip_gap_down", 0) > 0 and k > 0:
                        # t+1 开盘相对 t 收盘低开超过阈值→仍在下跌，放弃（不接飞刀）
                        prev_c = panel.f["close"][i, k - 1]
                        if np.isfinite(prev_c) and prev_c > 0:
                            gap = px / prev_c - 1.0
                            if gap < -BT["entry_skip_gap_down"]:
                                s.pending = None
                                cancelled += 1
                                continue
                        shares = s.cash * (1 - BUY_COST) / px
                        s.pos = dict(code=p["code"], i=i, entry_idx=k, entry_date=str(dates[k]),
                                     entry_px=float(px),
                                     stop=p["stop"], shares=shares, entry_cash=s.cash,
                                     signal_date=p["signal_date"], score=p["score"])
                        s.cash = 0.0
                        s.pending = None
                    else:
                        shares = s.cash * (1 - BUY_COST) / px
                        s.pos = dict(code=p["code"], i=i, entry_idx=k, entry_date=str(dates[k]),
                                     entry_px=float(px),
                                     stop=p["stop"], shares=shares, entry_cash=s.cash,
                                     signal_date=p["signal_date"], score=p["score"])
                        s.cash = 0.0
                        s.pending = None
                elif k - p["created_idx"] >= BT["entry_roll_days"]:
                    s.pending = None  # 连续停牌，撤单
        # 2) 持仓管理（含当日新开仓）
        for s in slots:
            tr = _manage_position(panel, s, k, last, variant,
                                   first_day_loss_exit=first_day_loss_exit)
            if tr is not None:
                trades.append(tr)
        # 3) 收盘出信号，分配给空闲分槽（次日开盘成交）
        if d in signals:
            held = {x.pos["code"] for x in slots if x.pos is not None}
            held |= {x.pending["code"] for x in slots if x.pending is not None}
            free = [x for x in slots if x.pos is None and x.pending is None]
            for code, score, stop in signals[d]:
                if not free:
                    break
                i = code_i.get(code)
                if i is None or code in held:
                    continue
                stv = stop if isinstance(stop, (int, float)) and np.isfinite(stop) else 0.0
                s = free.pop(0)
                s.pending = dict(code=code, i=i, stop=float(stv), score=float(score),
                                 signal_date=d, created_idx=k)
                held.add(code)
        eq_curve.append(equity_at(k))

    curve_dates = [str(dates[k]) for k in range(g0, last + 1)]
    return trades, curve_dates, np.array(eq_curve), cancelled


# ============================================================
# 绩效统计
# ============================================================
def _curve_metrics(curve_dates, eq):
    eq = np.asarray(eq, dtype=float)
    if len(eq) < 2:
        return dict(total=0.0, cagr=0.0, maxdd=0.0, sharpe=0.0)
    r = np.diff(eq) / eq[:-1]
    years = len(eq) / TRADING_DAYS
    total = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.nanmin((eq - peak) / peak))
    sharpe = float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(TRADING_DAYS))
    return dict(total=float(total), cagr=float(cagr), maxdd=maxdd, sharpe=sharpe)


def _period_curve(curve_dates, eq, key):
    idx = [k for k, d in enumerate(curve_dates) if d.startswith(key)]
    if not idx:
        return None
    a, b = idx[0], idx[-1]
    base = eq[a - 1] if a > 0 else eq[a]
    seg = np.concatenate([[base], eq[a:b + 1]])
    ret = seg[-1] / seg[0] - 1
    peak = np.maximum.accumulate(seg)
    dd = float(np.nanmin((seg - peak) / peak))
    return float(ret), dd


def _trade_stats(df: pd.DataFrame) -> dict:
    if not len(df):
        return dict(n=0, win=0.0, avg=0.0, median=0.0, pf=0.0, hold=0.0)
    wins = df[df["ret"] > 0]["ret"]
    losses = df[df["ret"] < 0]["ret"]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf
    return dict(n=len(df), win=float((df["ret"] > 0).mean() * 100),
                avg=float(df["ret"].mean() * 100),
                median=float(df["ret"].median() * 100),
                pf=float(pf) if np.isfinite(pf) else 999.0,
                hold=float(df["hold_days"].mean()))


def _breakdown(strategy, param, trades, curve_dates, eq, periods):
    rows = []
    df = pd.DataFrame(trades)
    for key, label in periods:
        sub = df[df["entry_date"].str.startswith(key)] if len(df) else df
        st = _trade_stats(sub)
        pc = _period_curve(curve_dates, eq, key)
        rows.append(dict(策略=strategy, 参数=param, 区间=label, 交易数=st["n"],
                         胜率pct=st["win"], 平均收益pct=st["avg"],
                         区间最大回撤pct=(pc[1] * 100 if pc else np.nan),
                         平均持仓日=st["hold"]))
    return rows


def _periods(latest: str):
    latest_y = int(latest[:4])
    years = [str(y) for y in range(2017, latest_y + 1)]
    periods = [(y, f"{y}年") for y in years if y != str(latest_y)]
    if latest >= f"{latest_y}-01":
        for m in range(1, int(latest[5:7]) + 1):
            periods.append((f"{latest_y}-{m:02d}", f"{latest_y}年{m}月"))
    return periods


def run_backtest(force_cpu: bool = False, codes: Optional[List[str]] = None,
                 start: str = BT["start"], bypass_season: bool = False,
                 first_day_loss_exit: bool = False):
    backend = accel.detect_backend(force_cpu=force_cpu)
    panel, dates, trade_dates, gidx, code_i, ultra_sig, swing_sig = build_signals(
        backend, start, codes, bypass_season=bypass_season)
    if not trade_dates:
        raise RuntimeError("回测区间内无交易日（是否忘了回填历史数据？）")
    latest = trade_dates[-1]
    periods = _periods(latest)
    log.info(f"回测区间 {trade_dates[0]} ~ {latest}，共 {len(trade_dates)} 个交易日；"
             f"超短信号日 {len(ultra_sig)}，波段信号日 {len(swing_sig)}")

    # ---- 写死最优参数：超短 top1（次日开盘卖），波段 bigbar/top3（放量大阳退出） ----
    ULTRA_PARAM = "top1"
    SWING_PARAM = "bigbar/top3"

    breakdown_rows = []
    trade_dump = []

    # 超短
    trades_u, cd_u, eq_u = simulate_ultra(panel, dates, trade_dates, gidx,
                                            code_i, ultra_sig, 1)
    for tr in trades_u:
        tr.update(strategy="超短", param=ULTRA_PARAM)
    trade_dump += [{**tr, "策略": "超短", "参数": ULTRA_PARAM} for tr in trades_u]
    breakdown_rows += _breakdown("超短", ULTRA_PARAM, trades_u, cd_u, eq_u, periods)

    # 波段
    trades_s, cd_s, eq_s, cancelled = simulate_swing(panel, dates, trade_dates, gidx,
                                                       code_i, swing_sig, 3, "bigbar",
                                                       first_day_loss_exit=first_day_loss_exit)
    for tr in trades_s:
        tr.update(strategy="波段", param=SWING_PARAM)
    trade_dump += [{**tr, "策略": "波段", "参数": SWING_PARAM} for tr in trades_s]
    breakdown_rows += _breakdown("波段", SWING_PARAM, trades_s, cd_s, eq_s, periods)

    bd_df = pd.DataFrame(breakdown_rows)
    trades_df = pd.DataFrame(trade_dump)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    bp = BACKTEST_DIR / f"年度月度_{latest}_{ts}.xlsx"
    tp = BACKTEST_DIR / f"全部交易_{latest}_{ts}.xlsx"
    bd_df.to_excel(bp, index=False)
    trades_df.to_excel(tp, index=False)

    # ---- 总体汇总（全区间） ----
    def _overall(strategy, param, trades, curve_dates, eq):
        st = _trade_stats(pd.DataFrame(trades))
        cm = _curve_metrics(curve_dates, eq)
        return dict(策略=strategy, 参数=param, 交易数=st["n"],
                    胜率pct=round(st["win"], 2), 平均收益pct=round(st["avg"], 2),
                    总收益率pct=round(cm["total"] * 100, 2),
                    年化pct=round(cm["cagr"] * 100, 2) if np.isfinite(cm["cagr"]) else "-",
                    最大回撤pct=round(cm["maxdd"] * 100, 2),
                    盈亏比=round(st["pf"], 2) if np.isfinite(st["pf"]) else "-",
                    夏普=round(cm["sharpe"], 2),
                    平均持仓日=round(st["hold"], 1))

    overall_rows = [
        _overall("超短", ULTRA_PARAM, trades_u, cd_u, eq_u),
        _overall("波段", SWING_PARAM, trades_s, cd_s, eq_s),
    ]
    overall_df = pd.DataFrame(overall_rows)

    # ---- 控制台输出 ----
    exp_tag = []
    if bypass_season:
        exp_tag.append("无季节过滤(每天买)")
    if first_day_loss_exit:
        exp_tag.append("波段首日亏损即平仓")
    exp_str = f"｜实验:{' + '.join(exp_tag)}" if exp_tag else ""
    print("\n" + "#" * 100)
    print(f"# 回测 {trade_dates[0]} ~ {latest}｜成本(买{BUY_COST*100:.2f}%/卖{SELL_COST*100:.2f}%)"
          f"｜超短={ULTRA_PARAM} 次日开盘卖，波段={SWING_PARAM} 放量大阳退出{exp_str}")
    print("#" * 100)
    log.info(f"波段 t+1 开盘破位放弃成交 {cancelled} 次（低吸条件破坏，不进场）")
    report.show(overall_df, f"【总体汇总】全区间 {trade_dates[0]} ~ {latest}", n=10)
    report.show(bd_df[bd_df["策略"] == "超短"],
                f"【超短·隔日套利】{ULTRA_PARAM} 年度与2026逐月明细", n=40)
    report.show(bd_df[bd_df["策略"] == "波段"],
                f"【波段·A杀低吸】{SWING_PARAM} 年度与2026逐月明细", n=40)
    log.info(f"回测结果已留档：{bp.name} / {tp.name}")
    return dict(breakdown=bd_df, trades=trades_df, overall=overall_df,
                paths=dict(breakdown=str(bp), trades=str(tp)))
