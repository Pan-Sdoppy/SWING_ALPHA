# -*- coding: utf-8 -*-
"""
波段（超跌拐头）选股，含四类候选，按《板块相位与优质个股推荐》评分规则：
1. A杀低吸（连板被套回救，置信度A，全体系最强）
2. 底部爆量簇启动（命题3，牛市/震荡市有效，熊市加闸门）
3. 趋势波段（夏天机构缩量推升延续）
4. 箱体回踩（zigzag 摆动底±4%，观察型）
核心买点纪律：买“拐头阳”，不买下跌阴线；A杀目标=封板价，深跌30%为第二买点。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import (AKILL as AK, BOTTOM_CLUSTER as BC, TREND as TR,
                     BOX_PULLBACK as BX, UNIVERSE, OUTPUT)
from ..regime import bottom_cluster_gate
from ..utils import get_logger

log = get_logger()

COLS = ["代码", "名称", "现价", "涨跌幅", "类型", "评分", "封板价(目标)",
        "目标空间%", "板数", "高位红盘量比", "出货空间", "闷杀跌停",
        "杀跌幅度%", "已历经交易日", "拐头状态", "止损", "建议", "所属板块"]


def _sl(v, i, t):
    x = v[i, t]
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)


def _board_runs(sealed_row, start, end):
    """返回 [(run_len, last_idx)]，索引区间 [start,end] 内所有连板段。"""
    runs = []
    i = start
    while i <= end:
        if sealed_row[i]:
            j = i
            while j + 1 <= end and sealed_row[j + 1]:
                j += 1
            runs.append((j - i + 1, j))
            i = j + 1
        else:
            i += 1
    return runs


def _cleanliness_score(r: float) -> float:
    if r <= AK["red_volume_clean"]:
        return 30.0
    if r <= 0.2:
        return 24.0
    if r <= AK["red_volume_trapped"]:
        return 18.0
    if r <= 0.5:
        return 10.0
    if r <= AK["red_volume_distributed"]:
        return 5.0
    return 0.0


def _classify(space, ratio, smother, boards):
    if (space >= AK["space_distribute_min"] and ratio >= AK["red_volume_distributed"]
            and smother == 0):
        return "出货完成"
    if ratio <= AK["red_volume_trapped"] or smother >= AK["smother_limitdowns"]:
        return "爆头被套"
    if space < AK["space_distribute_min"]:
        return "被套(空间不足)"
    return "部分出货(仍被套)"


def _analyze_akill(panel, i, t, code):
    """检测该股截至 t 的最优（最近）未解救 A 杀事件，返回事件 dict 或 None。"""
    lo = max(0, t - AK["lookback_days"])
    sealed = panel.f["sealed"][i].astype(bool)
    O = panel.f["open"][i]; H = panel.f["high"][i]; L = panel.f["low"][i]
    C = panel.f["close"][i]; VOL = panel.f["volume"][i]; PCT = panel.f["pctChg"][i]
    LD = panel.f["limitdown"][i].astype(bool)
    runs = _board_runs(sealed, lo, t)
    best = None
    for boards, e in runs:
        if boards < AK["min_boards"]:
            continue
        w_end = min(t, e + AK["drawdown_window"])
        if w_end <= e:
            continue
        seal = C[e]
        if np.isnan(seal):
            continue
        win_lo_idx = e + 1
        low_seg = L[win_lo_idx:w_end + 1]
        ok = low_seg[~np.isnan(low_seg)]
        if len(ok) == 0:
            continue
        kill_idx = int(np.nanargmin(low_seg)) + win_lo_idx
        kill_low = L[kill_idx]
        drawdown = seal / kill_low - 1.0 if kill_low > 0 else 0
        if drawdown < AK["drawdown_min"]:
            continue
        # 至今未回封板价（杀跌最低点之后最高仍 < 封板价，且现价 < 封板价）
        after = H[kill_idx + 1:t + 1]
        after = after[~np.isnan(after)]
        price = C[t]
        if np.isnan(price):
            continue
        if len(after) and np.nanmax(after) >= seal:
            continue
        if price >= seal:
            continue
        # 连板期 VWAP（成本）
        bv = VOL[e - boards + 1:e + 1]
        tp = (O[e - boards + 1:e + 1] + H[e - boards + 1:e + 1] +
              C[e - boards + 1:e + 1]) / 3.0
        m = ~np.isnan(bv) & (bv > 0)
        vwap = float(np.sum(tp[m] * bv[m]) / np.sum(bv[m])) if m.any() else np.nan
        # 派发窗（最后封板日→杀跌最低点）
        ds, de = e + 1, kill_idx
        if de <= ds:
            space = seal / vwap if vwap else np.nan
            eff_ratio = 0.0
            smother = 0
            window_high = seal
        else:
            window_high = np.nanmax(H[ds:de + 1])
            space = window_high / vwap if vwap else np.nan
            red = (PCT[ds:de + 1] > 0) & (C[ds:de + 1] >= 0.85 * window_high)
            eff_vol = np.nansum(np.where(red, VOL[ds:de + 1], 0.0))
            eff_ratio = float(eff_vol / np.nansum(bv)) if np.nansum(bv) > 0 else np.nan
            smother = int(np.nansum(LD[ds:de + 1]))
        cls = _classify(space if not np.isnan(space) else 1.0,
                        eff_ratio if not np.isnan(eff_ratio) else 1.0, smother, boards)
        if cls == "出货完成" and AK["score_avoid_distributed"]:
            continue
        elapsed = t - e
        # 回测验证硬过滤：板结束后<30天不做（坑）；闷杀跌停>1次不做（胜率仅35%）
        if elapsed < AK["min_days_since_board"]:
            continue
        if smother > AK["max_smother_allow"]:
            continue
        cand = dict(boards=boards, seal=float(seal), kill_idx=kill_idx,
                    kill_low=float(kill_low), drawdown=float(drawdown),
                    vwap=vwap, space=float(space) if not np.isnan(space) else np.nan,
                    ratio=float(eff_ratio) if not np.isnan(eff_ratio) else np.nan,
                    smother=smother, cls=cls, elapsed=int(elapsed),
                    price=float(price))
        # 取最近的合格事件（elapsed 最小）
        if best is None or cand["elapsed"] < best["elapsed"]:
            best = cand
    return best


def _turning_state(panel, i, t, ev, snap):
    """判定拐头状态与建议。"""
    O = panel.f["open"][i]; H = panel.f["high"][i]; L = panel.f["low"][i]
    C = panel.f["close"][i]; VOL = panel.f["volume"][i]
    price = ev["price"]
    deep_line = ev["seal"] * (1 - AK["deep_oversold"])
    notes = []
    # 拐头阳：今日阳线且收盘高于昨收，且近5日不再创新低（高于杀跌最低点）
    today_up = (not np.isnan(O[i] if False else C[t])) and C[t] > O[t] and C[t] > C[t - 1]
    recent_low = np.nanmin(L[max(0, t - AK["platform_days"]):t + 1])
    holds_low = recent_low >= ev["kill_low"] * 0.99
    # 小平台
    seg_h = np.nanmax(H[max(0, t - AK["platform_days"]):t + 1])
    seg_l = np.nanmin(L[max(0, t - AK["platform_days"]):t + 1])
    platform = (seg_h / seg_l - 1) <= AK["platform_range"] if seg_l > 0 else False

    if price <= deep_line:
        state = "深跌30%买点区"
        if snap.get("extreme_panic"):
            notes.append("大盘极端情绪日(加分)")
        action = "深跌低吸(试错仓)"
        bonus = 8 + (5 if snap.get("extreme_panic") else 0)
    elif today_up and holds_low:
        state = "拐头阳" + ("(平台后)" if platform else "(企稳后)")
        action = "可低吸(试错20%)"
        bonus = 12
    elif holds_low and platform:
        state = "横盘企稳(待拐头阳)"
        action = "等拐头阳确认"
        bonus = 4
    else:
        state = "回落途中(等拐头)"
        action = "等拐头阳,勿接飞刀"
        bonus = 0
    return state, action, bonus, "；".join(notes)


def _zigzag_last_low(H, L, pct):
    """简易 zigzag，返回最近一个已确认的摆动低点价格。"""
    n = len(H)
    last_low = np.nan
    trend = 0
    piv_h = piv_l = H[0]
    piv_li = piv_hi = 0
    for k in range(n):
        h, l = H[k], L[k]
        if np.isnan(h) or np.isnan(l):
            continue
        if trend <= 0:  # 找低点
            if l <= piv_l:
                piv_l, piv_li = l, k
            if h >= piv_l * (1 + pct):
                last_low = piv_l
                piv_h, piv_hi, trend = h, k, 1
        if trend >= 0 and trend != 0:
            if h >= piv_h:
                piv_h, piv_hi = h, k
            if l <= piv_h * (1 - pct):
                piv_l, piv_li, trend = l, k, -1
    return last_low


def scan_swing(panel, snap: dict, mode: str = "after", show_all: bool = False) -> pd.DataFrame:
    t = panel.col_of(panel.asof) if panel.asof else panel.dates.shape[0] - 1
    if t is None:
        t = panel.dates.shape[0] - 1
    cluster_gate, cluster_gate_reason = bottom_cluster_gate(snap)
    rows = []
    akill_total = 0
    akill_winter_filtered = 0

    for i, code in enumerate(panel.codes):
        price = _sl(panel.f["close"], i, t)
        if price is None:
            continue
        valid_bars = int(np.sum(~np.isnan(panel.f["close"][i, :t + 1])))
        if valid_bars < UNIVERSE["min_list_days_swing"]:
            continue
        amt60 = _sl(panel.f["amtma60"], i, t)
        if amt60 is None or amt60 < UNIVERSE["swing_min_amount60"]:
            continue
        pct = _sl(panel.f["pctChg"], i, t)
        base = dict(代码=code, 名称=panel.name_map.get(code, ""), 现价=round(price, 2),
                    涨跌幅=round(pct, 2) if pct is not None else None,
                    所属板块=panel.ind_map.get(code, ""))

        # ---------- 1. A杀低吸 ----------
        ev = _analyze_akill(panel, i, t, code)
        if ev is not None:
            akill_total += 1
            state, action, turn_bonus, extra = _turning_state(panel, i, t, ev, snap)
            sc = AK["score_base"]
            sc += _cleanliness_score(ev["ratio"])
            if ev["boards"] in AK["prefer_boards"]:
                sc += AK["score_23_board"]
            if ev["ratio"] <= AK["red_volume_clean"]:
                sc += AK["score_clean_head"]
            if ev["elapsed"] > AK["stale_days_penalty"]:
                sc += AK["score_stale_penalty"]
            if ev["boards"] >= AK["avoid_boards"]:
                sc -= 15
            sc += turn_bonus
            # 回测验证硬过滤：市场冬天A杀继续阴跌，排除后回撤从-43%降至-19%（show_all时绕过，仅供观察）
            if AK["avoid_winter"] and snap.get("phase") == "winter" and not show_all:
                akill_winter_filtered += 1
            else:
                upside = (ev["seal"] / price - 1) * 100
                stop = round(ev["kill_low"] * 0.99, 2)
                rows.append({**base,
                    "类型": "A杀低吸", "评分": round(sc, 1),
                    "封板价(目标)": round(ev["seal"], 2),
                    "目标空间%": round(upside, 1), "板数": ev["boards"],
                    "高位红盘量比": round(ev["ratio"], 2),
                    "出货空间": round(ev["space"], 2) if not np.isnan(ev["space"]) else None,
                    "闷杀跌停": ev["smother"], "杀跌幅度%": round(ev["drawdown"] * 100, 1),
                    "已历经交易日": ev["elapsed"], "拐头状态": state,
                    "止损": stop, "建议": action + (f"（{extra}）" if extra else ""),
                })

        # ---------- 2. 底部爆量簇 ----------
        lo = max(0, t - BC["lookback"])
        vr = panel.f["vr20"][i, lo:t + 1]
        vol = panel.f["volume"][i, lo:t + 1]
        tp = ((panel.f["open"][i, lo:t + 1] + panel.f["high"][i, lo:t + 1] +
               panel.f["close"][i, lo:t + 1]) / 3.0)
        spike = vr >= BC["spike_ratio"]
        n_spike = int(np.nansum(spike))
        dd = _sl(panel.f["dd250"], i, t)
        if n_spike >= BC["cluster_min"] and dd is not None and dd <= -BC["drawdown_from_high"]:
            sv = vol[spike]; stp = tp[spike]
            m = ~np.isnan(sv) & (sv > 0)
            if m.any():
                cvwap = float(np.sum(stp[m] * sv[m]) / np.sum(sv[m]))
                if (not BC["require_above_cluster_vwap"]) or price >= cvwap:
                    if cluster_gate:
                        mom = _sl(panel.f["mom20"], i, t) or 0.0
                        sc = BC["score_base"] + min(BC["score_count_max"],
                                                    (n_spike - 3) * 3)
                        sc += BC["score_above_vwap"] if price >= cvwap else 0
                        sc += float(np.clip(mom * 100 * 1.5, 0, BC["score_momentum_max"]))
                        rows.append({**base, "类型": "底部簇启动", "评分": round(sc, 1),
                            "封板价(目标)": None, "目标空间%": None, "板数": n_spike,
                            "高位红盘量比": None, "出货空间": None, "闷杀跌停": None,
                            "杀跌幅度%": round(dd * 100, 1), "已历经交易日": None,
                            "拐头状态": f"{n_spike}根爆量K,簇VWAP{cvwap:.2f}",
                            "止损": round(cvwap * (1 - BC["stop_below_vwap"]), 2),
                            "建议": "簇上拐头阳可试错(试错仓)"})
                    else:
                        rows.append({**base, "类型": "底部簇(禁用)", "评分": 40.0,
                            "封板价(目标)": None, "目标空间%": None, "板数": n_spike,
                            "高位红盘量比": None, "出货空间": None, "闷杀跌停": None,
                            "杀跌幅度%": round(dd * 100, 1), "已历经交易日": None,
                            "拐头状态": cluster_gate_reason, "止损": None,
                            "建议": "市场冬天,底部簇策略禁用"})

        # ---------- 3. 趋势波段（夏天延续） ----------
        c = panel.f["close"][i, t]; ma20 = panel.f["ma20"][i, t]
        ma60 = panel.f["ma60"][i, t]; h20 = panel.f["high20"][i, t]
        mom = _sl(panel.f["mom20"], i, t)
        if (not np.isnan(h20) and c >= h20 and not np.isnan(ma20) and c > ma20
                and not np.isnan(ma60) and c > ma60 and mom is not None
                and mom > TR["mom_min"]):
            shrink = panel.f["vma5"][i, t] < panel.f["vma20"][i, t]
            sc = TR["score_base"] + float(np.clip(mom * 100 * 1.5, 0,
                                                  TR["score_momentum_max"]))
            sc += TR["score_shrink_push"] if shrink else 0
            sc += TR["score_above_ma60"] if c > ma60 else 0
            rows.append({**base, "类型": "趋势波段", "评分": round(sc, 1),
                "封板价(目标)": None, "目标空间%": None, "板数": None,
                "高位红盘量比": None, "出货空间": None, "闷杀跌停": None,
                "杀跌幅度%": None, "已历经交易日": None,
                "拐头状态": f"20日新高,20日涨{mom*100:.1f}%",
                "止损": round(float(ma20), 2), "建议": "夏天延续,回踩5/10日线低吸"})

        # ---------- 4. 箱体回踩（观察） ----------
        lx = max(0, t - BX["swing_lookback"])
        sw_low = _zigzag_last_low(panel.f["high"][i, lx:t + 1],
                                  panel.f["low"][i, lx:t + 1], BX["zigzag_pct"])
        if (not np.isnan(sw_low)) and sw_low > 0:
            dist = price / sw_low - 1
            if abs(dist) <= BX["near_low_pct"]:
                rows.append({**base, "类型": "箱体回踩", "评分": float(BX["score_base"]),
                    "封板价(目标)": None, "目标空间%": None, "板数": None,
                    "高位红盘量比": None, "出货空间": None, "闷杀跌停": None,
                    "杀跌幅度%": None, "已历经交易日": None,
                    "拐头状态": f"距摆动底{dist*100:.1f}%",
                    "止损": round(sw_low * 0.97, 2),
                    "建议": "观察:回踩摆动底,待拐头阳"})

    # A杀透明日志：让用户知道今天有多少A杀候选、是否被冬天过滤
    if akill_total > 0:
        if akill_winter_filtered == akill_total:
            log.info(f"波段·A杀：候选{akill_total}只全部被冬天过滤（市场处于{AK.get('phase','winter')}），本日不出A杀票")
        elif akill_winter_filtered > 0:
            log.info(f"波段·A杀：候选{akill_total}只，冬天过滤{akill_winter_filtered}只，出票{akill_total - akill_winter_filtered}只")

    df = pd.DataFrame(rows)
    if len(df):
        for c in COLS:
            if c not in df.columns:
                df[c] = None
        df = df[COLS]
        df = df.sort_values(["类型", "评分"], ascending=[True, False])
        df = df[df["评分"] >= OUTPUT["min_score_show"]].head(OUTPUT["swing_top"])
    return df.reset_index(drop=True)
