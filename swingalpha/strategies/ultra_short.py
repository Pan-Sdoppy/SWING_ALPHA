# -*- coding: utf-8 -*-
"""
超短（隔日套利）选股，含两套子策略：
A. 寻找早餐（师傅原始方法，用户裁定以此为准；权重 A1 70 / A3 15 / A2 15）：
   A1 压力位【决定性】：往左60日找“爆量K”(量≥前5日均量2倍)，在其最低价(下沿)画
   水平线，今日涨停价正好压在下沿下方0~3%即命中，可在压力位下排板/炸板回封时去。
   A3 板型：急拉、远离分时均价=好；贴均价横盘“红旗飘飘”=出货；围均价炸了又封=主力花钱。
   A2 关键位：起涨点(近30日最低)的×1.5/×2整数关口要避开；与A1重合时以爆量K下沿为准。
   备胎(仅炸板时去，不如A1)：B1 补缺口、B2 新高诱多。
   一票否决：X4 左侧真空、X7 涨停价钻进爆量K实体/影线、X2 二顶、X1 无压力又无缺口/诱多、
   顶部首板、老庄股、坏行情(门控不过)炸板不回封；一字板买不进仅记录。
   门控：中证1000为正 + 全市场炸板率<35% + 非冬天。
B. 缩量连板追板（打连板 S1，置信度B，彩票仓≤20%）：
   2~3板、连板期平均量比<1.5、低位(封板价<起涨点×1.5)、市场非冬天、竞价高开>2%才追。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import BREAKFAST as BF, SHRINK_BOARD as SB, UNIVERSE, OUTPUT
from ..regime import breakfast_gate
from ..utils import get_logger

log = get_logger()


def _sl(v, i, t):
    x = v[i, t]
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)


# ============================================================
# A. 寻找早餐：A1 压力位 / A2 关键位 / B1 缺口
# ============================================================
def _a1_pressure(panel, i, t, limitpx) -> dict:
    """返回今日涨停价相对左侧爆量K的位置关系。
    kind:
      hit      —— 涨停价压在某根爆量K最低价(下沿)下方0~3%，A1命中（首选）
      drilled  —— 涨停价钻进爆量K实体（X7，剔除）
      shadow   —— 涨停价越过下沿、钻进下影线（不干净，剔除）
      far      —— 上方有爆量K但下沿离涨停价>3%，没顶到（不确认）
      none     —— 上方没有任何爆量K压力（结合真空/二顶判定，剔除）
    """
    res = {"kind": "none", "edge": None, "gap": None}
    lo = max(0, t - BF["spike_lookback"])
    if lo >= t or limitpx is None or np.isnan(limitpx):
        return res
    O = panel.f["open"][i, lo:t]
    H = panel.f["high"][i, lo:t]
    Lw = panel.f["low"][i, lo:t]
    C = panel.f["close"][i, lo:t]
    V = panel.f["volume"][i, lo:t]
    vm = panel.f["vma5"][i, lo:t]
    with np.errstate(invalid="ignore"):
        spk = (V >= BF["spike_ratio"] * vm) & ~np.isnan(vm)
    if not spk.any():
        return res
    body_lo = np.minimum(O, C)
    body_hi = np.maximum(O, C)
    tol = BF["pressure_tol"] / 100.0

    def _edge(k):
        res["edge"] = float(Lw[k])
        res["gap"] = (float(Lw[k]) / limitpx - 1.0) * 100.0

    si = np.where(spk)[0]
    # X7：涨停价钻进上方爆量K的实体（收盘已越过实体下沿）
    overhead = body_hi >= limitpx * 0.99
    drill = si[overhead[si] & (limitpx > body_lo[si])]
    if len(drill):
        k = drill[np.argmin(np.abs(body_lo[drill] - limitpx))]
        _edge(k)
        res["kind"] = "drilled"
        return res
    # 越过下沿、钻进下影线（介于最低价与实体下沿之间）——同样不干净
    shadow = si[overhead[si] & (limitpx > Lw[si]) & (limitpx <= body_lo[si])]
    if len(shadow):
        k = shadow[np.argmin(np.abs(Lw[shadow] - limitpx))]
        _edge(k)
        res["kind"] = "shadow"
        return res
    # A1命中：爆量K最低价在涨停价上方0~3%，且涨停价未越过该下沿
    hit = si[(Lw[si] >= limitpx) & (Lw[si] <= limitpx * (1 + tol))]
    if len(hit):
        k = hit[np.argmin(Lw[hit])]      # 取离涨停价最近的那根压力K
        _edge(k)
        res["kind"] = "hit"
        return res
    # 上方有爆量K但下沿还差>3%
    far = si[Lw[si] >= limitpx]
    if len(far):
        k = far[np.argmin(Lw[far])]
        _edge(k)
        res["kind"] = "far"
        return res
    return res


def _a2_level(panel, i, t, limitpx) -> tuple:
    """A2 关键位：起涨点(近30日最低)的×1.5/×2 是否正好卡住涨停价。
    返回(得分0~15, 说明)。未卡位给15，卡位给5。"""
    lo = max(0, t - BF["a2_lookback"])
    if lo >= t or limitpx is None:
        return 10, ""
    start = np.nanmin(panel.f["low"][i, lo:t])
    if not np.isfinite(start) or start <= 0:
        return 10, ""
    for m in BF["a2_multiples"]:
        lvl = start * m
        if abs(limitpx / lvl - 1.0) * 100.0 <= BF["a2_block_tol"]:
            return 5, f"卡在起涨点×{('%g' % m)}整数位"
    return 15, ""


def _b1_gap(panel, i, t, limitpx) -> bool:
    """B1 补缺口（备胎）：近60日存在尚未回补的向下跳空缺口，其缺口下沿就在涨停价
    上方0~3%（炸板时才去）。向下缺口=当日最高<前一日最低，且其后最高始终未补缺口。"""
    if t < 1 or limitpx is None:
        return False
    Hh = panel.f["high"][i]
    Ll = panel.f["low"][i]
    a = max(1, t - BF["b1_gap_lookback"])
    tol = BF["pressure_tol"] / 100.0
    for k in range(t - 1, a - 1, -1):
        if not (np.isfinite(Hh[k]) and np.isfinite(Ll[k - 1])):
            continue
        if Hh[k] < Ll[k - 1]:                    # 当日最高<前日最低 → 向下缺口
            gap_bottom = Hh[k]
            later = Hh[k + 1:t]
            if later.size and np.nanmax(later) >= gap_bottom:
                continue                         # 已回补
            if limitpx <= gap_bottom <= limitpx * (1 + tol):
                return True
    return False


def _a3_score(minfo: dict) -> tuple:
    """A3 板型（分时，权重15）。无分钟数据给中性8分并提示盘中确认。"""
    if not minfo:
        return 8, ["板型待盘中确认(要急拉远离均价、无红旗飘飘)"]
    score = 10
    notes = []
    if minfo.get("red_flag"):
        score -= 10
        notes.append("红旗飘飘(贴均价横盘出货)")
    if minfo.get("seal_after_1130"):
        score -= 5
        notes.append("封板偏晚")
    if minfo.get("reopens", 0) > 0:
        score = min(15, score + 3)
        notes.append("炸板回封(主力花钱,非出货)")
    if score >= 13:
        notes.append("板型强(急拉远离均价)")
    return max(0, min(15, score)), notes


def _shared_factors(panel, i, t):
    """提取当日共用字段；左侧真空(X4)。"""
    price = _sl(panel.f["close"], i, t)
    turn = _sl(panel.f["turn"], i, t)
    pct = _sl(panel.f["pctChg"], i, t)
    limitpx = _sl(panel.f["limitpx"], i, t)
    oneword = bool(panel.f["oneword"][i, t])
    vacuum = None
    if limitpx is not None:
        h60 = panel.f["high60"][i, t]
        vacuum = bool(np.isnan(h60) or h60 < limitpx)
    return dict(price=price, turn=turn, pct=pct, limitpx=limitpx,
                oneword=oneword, vacuum=vacuum)


def _breakfast_row(panel, i, t, bc, gate_ok, mode, minfo, sh):
    """A. 寻找早餐（A1压力位）。门控不过 / 非A1且无备胎 → 返回 None（不影响S1）。"""
    if bc not in BF["boards_allowed"] or not gate_ok:
        return None
    code = panel.codes[i]
    price, turn, pct, limitpx, oneword = (sh["price"], sh["turn"], sh["pct"],
                                          sh["limitpx"], sh["oneword"])
    vacuum = sh["vacuum"]

    a1 = _a1_pressure(panel, i, t, limitpx)
    lo30 = max(0, t - BF["a2_lookback"])
    start30 = np.nanmin(panel.f["low"][i, lo30:t]) if lo30 < t else np.nan
    near_mult = False
    if np.isfinite(start30) and start30 > 0 and limitpx:
        for m in BF["a2_multiples"]:
            if abs(limitpx / (start30 * m) - 1.0) * 100.0 <= BF["pressure_tol"]:
                near_mult = True
                break
    b2 = bool(vacuum and near_mult)        # B2 新高诱多：真空但恰在倍数位（仅炸板去）
    b1 = (a1["kind"] != "hit") and _b1_gap(panel, i, t, limitpx)

    if a1["kind"] == "hit":
        kind = "A1"
        a1col = f"命中 {a1['gap']:+.1f}%"
    elif b1:
        kind = "B1"
        a1col = "B1补缺口(备胎)"
    elif b2:
        kind = "B2"
        a1col = "B2新高诱多(备胎)"
    else:
        # X4真空 / X7钻进实体 / 影线 / X2二顶 / X1无压力 —— 不入选（仅本分支，不牵连S1）
        return None

    notes = []
    a3s, a3notes = _a3_score(minfo)
    notes += a3notes
    a2s, a2note = _a2_level(panel, i, t, limitpx)
    if a2note:
        notes.append(a2note)
    board_bonus = {1: 3, 2: 2, 3: 0}.get(bc, 0)

    if kind == "A1":
        score = BF["w_a1"] + a3s + a2s + board_bonus
        if turn is not None and turn > BF["turnover_avoid_min"]:
            score -= 10
            notes.append(f"换手{turn:.0f}%过高")
        if a1["gap"] is not None:
            notes.append(f"涨停价顶到爆量K下沿({a1['gap']:+.1f}%)")
    else:
        score = 52 + (3 if b1 else 0)      # 备胎封顶，明显低于A1
        notes.append("备胎仅在炸板/回封瞬间去，不回封不去")
    score = float(min(100, max(0, round(score, 1))))

    seal_state = minfo.get("seal_state", "一字" if oneword else "封板")
    if oneword:
        action = "一字板(买不进,仅记录)"
        score = 50.0
        notes.append("一字板排不进")
    elif kind in ("B1", "B2"):
        action = "备胎:仅炸板回封时去" if mode == "intraday" else "备胎(盘后标记,次日不追)"
    elif mode == "intraday":
        action = "压力位下排板/炸板回封去"
    else:
        action = "盘后标记:复盘A1,次日不追高"

    return dict(
        代码=code, 名称=panel.name_map.get(code, ""), 现价=price,
        涨跌幅=pct, 换手率=turn, 连板数=bc, 信号="早餐·压力位",
        **{"A1压力位": a1col},
        封板状态=seal_state,
        评分=score, 建议=action,
        备注="；".join(notes), 所属板块=panel.ind_map.get(code, ""),
    )


def _s1_row(panel, i, t, bc, phase, mode, minfo, sh, bypass_winter=False):
    """B. 缩量连板追板（独立子策略，自带非冬天门控，不被早餐分支屏蔽）。"""
    if not (SB["min_boards"] <= bc <= SB["max_boards"]):
        return None
    if phase == "winter" and not bypass_winter:
        return None                            # 市场冬天不做连板
    code = panel.codes[i]
    price, turn, pct, limitpx, vacuum = (sh["price"], sh["turn"], sh["pct"],
                                        sh["limitpx"], sh["vacuum"])

    lo = max(0, t - bc)
    vr_window = (panel.f["volume"][i, lo:t + 1] /
                 panel.f["vma20"][i, lo:t + 1])
    vr_window = vr_window[~np.isnan(vr_window)]
    avg_vr = float(np.nanmean(vr_window)) if len(vr_window) else np.nan
    run_low = _sl(panel.f["low60"], i, t)
    pos_mult = (limitpx / run_low) if (limitpx and run_low) else np.nan
    cond_shrink = (not np.isnan(avg_vr)) and avg_vr < SB["board_vol_ratio_max"]
    cond_low = (not np.isnan(pos_mult)) and pos_mult < SB["position_multiple_max"]
    cond_phase = phase in ("summer", "deep_autumn", "autumn", "unknown")
    gap = minfo.get("gap_pct")
    if gap is None:
        o = _sl(panel.f["open"], i, t)
        pc = _sl(panel.f["preclose"], i, t)
        gap = (o / pc - 1) * 100 if (o and pc) else None

    score = 55
    score += 12 if cond_shrink else 0
    score += 10 if cond_low else 0
    score += 8 if phase == "summer" else 0
    score += 5 if vacuum else 0
    notes = []
    if not cond_shrink:
        notes.append(f"连板量比{avg_vr:.2f}≥1.5")
    if not cond_low:
        notes.append(f"非低位({pos_mult:.2f}x)")
    if phase == "winter":
        notes.append("市场冬天")

    if mode == "intraday":
        if gap is not None and gap >= SB["auction_gap_min"] and cond_shrink and cond_low and cond_phase:
            action = "可追板(彩票仓≤20%)"
        elif gap is None:
            action = "观察(待竞价)"
        else:
            action = "回避(竞价不足/条件不符)"
    else:
        if cond_shrink and cond_low and cond_phase:
            action = "次日观察:竞价高开>2%才追"
        else:
            action = "不追"

    return dict(
        代码=code, 名称=panel.name_map.get(code, ""), 现价=price,
        涨跌幅=pct, 换手率=turn, 连板数=bc, 信号="缩量连板",
        **{"A1压力位": "-"},
        封板状态=f"均量比{avg_vr:.2f}" if not np.isnan(avg_vr) else "-",
        评分=round(score, 1), 建议=action,
        备注="；".join(notes), 所属板块=panel.ind_map.get(code, ""),
    )


def scan_ultra_short(panel, snap: dict, mode: str = "after",
                     minute_info: dict = None, show_all: bool = False) -> pd.DataFrame:
    minute_info = minute_info or {}
    gate_ok, gate_reason = breakfast_gate(snap)
    phase = snap.get("phase", "unknown")
    if show_all:
        gate_ok = True  # predict模式：绕过早餐门控，全部展示
    t = panel.col_of(panel.asof) if panel.asof else panel.dates.shape[0] - 1
    if t is None:
        t = panel.dates.shape[0] - 1

    boardcnt = panel.f["boardcnt"]
    rows = []

    # —— 漏斗统计（用于在结果为空时把“为什么没有票”打到日志） ——
    close_v = panel.f["close"][:, t]
    vol_v = panel.f["volume"][:, t]
    valid = ~np.isnan(close_v) & (np.nan_to_num(vol_v) > 0)
    touched = np.nan_to_num(panel.f["touched"][:, t], nan=0) == 1
    sealed = np.nan_to_num(panel.f["sealed"][:, t], nan=0) == 1
    n_touch = int((touched & valid).sum())
    n_seal = int((sealed & valid).sum())
    n_broken = n_touch - n_seal
    geo = {"hit": 0, "drilled": 0, "shadow": 0, "far": 0, "none": 0}
    n_allowed = 0
    n_vacuum = 0

    for i, code in enumerate(panel.codes):
        bc = _sl(boardcnt, i, t)
        if bc is None or bc < 1:
            continue
        # 上市天数（有效K线根数）
        valid_bars = int(np.sum(~np.isnan(panel.f["close"][i, :t + 1])))
        if valid_bars < UNIVERSE["min_list_days_pool"]:
            continue
        bc = int(bc)
        sh = _shared_factors(panel, i, t)
        minfo = minute_info.get(code, {})

        # A1 几何漏斗（无论门控是否通过都统计，便于解释“为什么不能做”）
        if bc in BF["boards_allowed"]:
            n_allowed += 1
            geo[_a1_pressure(panel, i, t, sh["limitpx"])["kind"]] += 1
            n_vacuum += int(bool(sh["vacuum"]))

        # A. 寻找早餐（A1压力位）——门控不过/非A1只跳过本分支，不牵连 B
        row_a = _breakfast_row(panel, i, t, bc, gate_ok, mode, minfo, sh)
        if row_a is not None:
            rows.append(row_a)

        # B. 缩量连板追板（独立门控：非冬天；show_all时绕过）
        row_b = _s1_row(panel, i, t, bc, phase, mode, minfo, sh, bypass_winter=show_all)
        if row_b is not None:
            rows.append(row_b)

    # —— 打印门控结论与漏斗 ——
    broken_pct = (n_broken / n_touch * 100.0) if n_touch else 0.0
    log.info(
        f"超短漏斗[{panel.asof}] 触板{n_touch}/封板{n_seal}/炸板{n_broken}"
        f"(炸板率{broken_pct:.0f}%)；1~3板封板{n_allowed}只 → "
        f"A1命中{geo['hit']}、钻进实体{geo['drilled']}、影线{geo['shadow']}、"
        f"离压力位>3% {geo['far']}、无压力{geo['none']}(其中左侧真空{n_vacuum})")
    if not gate_ok:
        log.info(f"早餐·压力位：天时闸门关闭，本日不出早餐票（{gate_reason}）")
    if phase == "winter":
        log.info("缩量连板：市场处于冬天，按规则不做连板追板")

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["信号", "评分"], ascending=[True, False])
        df = df[df["评分"] >= 50].head(OUTPUT["ultra_top"] * 2)
        n_bf = int((df["信号"] == "早餐·压力位").sum())
        n_s1 = int((df["信号"] == "缩量连板").sum())
        log.info(f"超短出票：早餐·压力位 {n_bf} 只、缩量连板 {n_s1} 只")
    else:
        log.info("超短今日无可下手标的，空仓观望（坏行情不做就是赚）")
    return df.reset_index(drop=True)
