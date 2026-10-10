# -*- coding: utf-8 -*-
"""
SWING_ALPHA · 全市场主力分析引擎（八步法）
================================================================
基于《代理交易体系完整手册》八步法，对3052只主板股票逐只分析：
  步骤一~三：主力前世（筹码链）—— 爆量簇识别、VWAP成本、出货成败、未解救筹码
  步骤四：箱体定位（双轨汇聚）—— 级联箱体法 + 单锚点倍数法
  步骤五~六：结构与性格 —— zigzag(±15%)底部计数、主力性格分类
  步骤七：方向判断（三步法）—— 出货完了吗？到箱体顶了吗？在顶部做什么？
  步骤八：输出信号卡 —— 买点/减仓点/止损点/主力是否需自救

用法:
  python -m swingalpha.stock_analysis                    # 全市场3052只
  python -m swingalpha.stock_analysis --codes 001696    # 指定股票
  python -m swingalpha.stock_analysis --top 50           # 只输出最需自救的前50只
  python -m swingalpha.stock_analysis --cpu              # 强制CPU
输出: data/processed/stock_analysis/主力分析_YYYYMMDD.xlsx
"""
from __future__ import annotations
import argparse
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# 支持直接运行: python swingalpha/stock_analysis.py
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from swingalpha import factors, storage
from swingalpha.utils import get_logger
from swingalpha.config import PROCESSED_DIR

log = get_logger()

# ============================================================
# 参数（手册口径）
# ============================================================
VOL_CLUSTER_MIN_BARS = 3          # 爆量簇最少根数
VOL_CLUSTER_WINDOW = 10           # 爆量簇合并窗口（交易日）
VOL_SPIKE_RATIO = 2.0             # 爆量K = 量 ≥ 前5日均量 × 此倍数
DIST_SPACE_MIN = 1.5              # 出货空间下限（距成本≥1.5倍才可能是出货）
ZIGZAG_THRESHOLD = 0.15           # zigzag ±15%
DROP70 = 0.70                      # 70%下跌规律
ANALYSIS_DIR = PROCESSED_DIR / "stock_analysis"


# ============================================================
# 步骤一~三：主力前世（筹码链）
# ============================================================
@dataclass
class VolumeCluster:
    """一轮爆量簇（建仓/出货）。"""
    start_idx: int
    end_idx: int
    vwap: float                      # 成交均价 = sum(close*vol)/sum(vol)
    total_vol: float                 # 簇总成交量
    avg_price: float                 # 簇平均收盘价
    max_price: float                 # 簇内最高价
    min_price: float                 # 簇内最低价
    n_bars: int                      # 簇内K线数


def find_volume_clusters(close: np.ndarray, volume: np.ndarray) -> List[VolumeCluster]:
    """识别爆量簇：量≥前5日均量×2的K线，3根以上且间距≤10日合并为一簇。"""
    n = len(close)
    if n < 20:
        return []
    # 5日均量（前向，不含当日）
    vma5 = np.full(n, np.nan)
    for i in range(5, n):
        vma5[i] = np.nanmean(volume[i - 5:i])
    # 爆量K标记
    is_spike = (volume >= VOL_SPIKE_RATIO * vma5) & np.isfinite(vma5) & (volume > 0)
    spike_idx = np.where(is_spike)[0]
    if len(spike_idx) < VOL_CLUSTER_MIN_BARS:
        return []
    # 合并为簇（间距≤WINDOW）
    clusters = []
    cur_start = spike_idx[0]
    cur_end = spike_idx[0]
    for idx in spike_idx[1:]:
        if idx - cur_end <= VOL_CLUSTER_WINDOW:
            cur_end = idx
        else:
            if cur_end - cur_start + 1 >= VOL_CLUSTER_MIN_BARS:
                clusters.append((cur_start, cur_end))
            cur_start = idx
            cur_end = idx
    if cur_end - cur_start + 1 >= VOL_CLUSTER_MIN_BARS:
        clusters.append((cur_start, cur_end))
    # 计算每簇VWAP等
    result = []
    for s, e in clusters:
        seg_v = volume[s:e + 1]
        seg_c = close[s:e + 1]
        total_v = float(np.nansum(seg_v))
        vwap = float(np.nansum(seg_c * seg_v) / total_v) if total_v > 0 else float(np.nanmean(seg_c))
        result.append(VolumeCluster(
            start_idx=s, end_idx=e, vwap=vwap, total_vol=total_v,
            avg_price=float(np.nanmean(seg_c)),
            max_price=float(np.nanmax(seg_c)),
            min_price=float(np.nanmin(seg_c)),
            n_bars=e - s + 1,
        ))
    return result


def analyze_chains(close: np.ndarray, volume: np.ndarray,
                   clusters: List[VolumeCluster]) -> dict:
    """逐轮追踪筹码链：判断每轮建仓后是否出货成功，建立未解救筹码清单。"""
    n = len(close)
    rounds = []
    for ci, cl in enumerate(clusters):
        # 建仓后区间
        after_start = cl.end_idx + 1
        after_end = clusters[ci + 1].start_idx if ci + 1 < len(clusters) else n - 1
        if after_start >= after_end:
            rounds.append(dict(cluster=cl, distributed=False, dist_vol=0,
                                dist_price_max=0, trapped_high=cl.max_price))
            continue
        seg_c = close[after_start:after_end + 1]
        seg_v = volume[after_start:after_end + 1]
        # 出货判断：出货量≥建仓量 且 最高价≥成本×1.5 且 有红盘
        dist_vol = float(np.nansum(seg_v))
        dist_price_max = float(np.nanmax(seg_c)) if len(seg_c) else 0
        has_red = bool(np.any(seg_c > np.roll(seg_c, 1))) if len(seg_c) > 1 else False
        distributed = (dist_vol >= cl.total_vol and
                       dist_price_max >= cl.vwap * DIST_SPACE_MIN and
                       has_red)
        rounds.append(dict(cluster=cl, distributed=distributed,
                            dist_vol=dist_vol, dist_price_max=dist_price_max,
                            trapped_high=cl.max_price if not distributed else 0))
    # 未解救筹码：所有未出货成功的簇的最高价
    trapped = [r for r in rounds if not r["distributed"]]
    trapped_highs = [r["trapped_high"] for r in trapped if r["trapped_high"] > 0]
    latest_cluster = clusters[-1] if clusters else None
    latest_distributed = rounds[-1]["distributed"] if rounds else False
    return dict(
        n_rounds=len(rounds),
        n_distributed=sum(1 for r in rounds if r["distributed"]),
        n_trapped=len(trapped),
        trapped_max=max(trapped_highs) if trapped_highs else 0,
        trapped_min=min(trapped_highs) if trapped_highs else 0,
        latest_cost=latest_cluster.vwap if latest_cluster else 0,
        latest_vol=latest_cluster.total_vol if latest_cluster else 0,
        latest_distributed=latest_distributed,
        rounds=rounds,
    )


# ============================================================
# 步骤四：箱体定位（双轨汇聚）
# ============================================================
def identify_box(close: np.ndarray, low: np.ndarray, current_price: float,
                 effective_start_idx: int) -> dict:
    """级联箱体法 + 单锚点倍数法，双轨汇聚。"""
    n = len(close)
    # 有效起点 = 有效起始日后的最低价
    seg_low = low[effective_start_idx:]
    rel_idx = int(np.argmin(seg_low)) if len(seg_low) else 0
    seg_idx = rel_idx + effective_start_idx  # 绝对索引
    anchor = float(seg_low[rel_idx]) if rel_idx < len(seg_low) else float(np.nanmin(low))
    # 级联箱体：底×2=顶，新箱体底=上箱体顶
    cascade_levels = []
    b = anchor
    for _ in range(6):
        top = b * 2
        mid = b * 1.5
        cascade_levels.append(dict(bottom=b, mid=mid, top=top))
        b = top
    # 单锚点倍数
    anchor_multiples = {f"×{m}": anchor * m for m in [2, 3, 4, 6, 8, 10]}
    # 找当前所在箱体
    current_box = None
    for i, lv in enumerate(cascade_levels):
        if lv["bottom"] <= current_price <= lv["top"]:
            current_box = i
            break
    if current_box is None:
        current_box = len(cascade_levels) - 1 if current_price > cascade_levels[-1]["top"] else 0
    box = cascade_levels[current_box]
    # 现价在箱体内的位置（0=底，1=顶）
    box_pos = (current_price - box["bottom"]) / (box["top"] - box["bottom"]) if box["top"] > box["bottom"] else 0
    # 双轨汇聚：级联箱体关键位 vs 锚点倍数，容差±2%
    convergences = []
    key_levels = []
    for lv in cascade_levels:
        for label, px in [("底", lv["bottom"]), ("中继", lv["mid"]), ("顶", lv["top"])]:
            key_levels.append((f"箱体{label}", px))
    for label, px in anchor_multiples.items():
        key_levels.append((f"锚点{label}", px))
    # 找汇聚对（差距≤2%）
    for i in range(len(key_levels)):
        for j in range(i + 1, len(key_levels)):
            l1, p1 = key_levels[i]
            l2, p2 = key_levels[j]
            if p1 > 0 and p2 > 0 and abs(p1 - p2) / max(p1, p2) <= 0.02:
                convergences.append(dict(level1=l1, level2=l2, price=round((p1 + p2) / 2, 2)))
    return dict(
        anchor=round(anchor, 2),
        anchor_date_idx=int(seg_idx),
        cascade=box,
        cascade_idx=current_box,
        box_position_pct=round(box_pos * 100, 1),
        anchor_multiples={k: round(v, 2) for k, v in anchor_multiples.items()},
        convergences=convergences[:5],  # 最多5个汇聚区
        near_top=box_pos >= 0.9,
        near_bottom=box_pos <= 0.1,
    )


# ============================================================
# 步骤五~六：zigzag结构 + 主力性格
# ============================================================
def zigzag(close: np.ndarray, threshold: float = ZIGZAG_THRESHOLD) -> Tuple[np.ndarray, np.ndarray]:
    """标准zigzag：返回拐点索引和拐点价格。"""
    n = len(close)
    if n < 3:
        return np.array([0, n - 1]), np.array([close[0], close[-1]])
    pivots_idx = [0]
    pivots_px = [close[0]]
    trend = 0  # 0=unknown, 1=up, -1=down
    last_extreme_idx = 0
    last_extreme_px = close[0]
    for i in range(1, n):
        if not np.isfinite(close[i]):
            continue
        if trend == 0:
            if close[i] >= last_extreme_px * (1 + threshold):
                trend = 1
                last_extreme_idx = i
                last_extreme_px = close[i]
            elif close[i] <= last_extreme_px * (1 - threshold):
                trend = -1
                last_extreme_idx = i
                last_extreme_px = close[i]
        elif trend == 1:
            if close[i] > last_extreme_px:
                last_extreme_idx = i
                last_extreme_px = close[i]
            elif close[i] <= last_extreme_px * (1 - threshold):
                pivots_idx.append(last_extreme_idx)
                pivots_px.append(last_extreme_px)
                trend = -1
                last_extreme_idx = i
                last_extreme_px = close[i]
        elif trend == -1:
            if close[i] < last_extreme_px:
                last_extreme_idx = i
                last_extreme_px = close[i]
            elif close[i] >= last_extreme_px * (1 + threshold):
                pivots_idx.append(last_extreme_idx)
                pivots_px.append(last_extreme_px)
                trend = 1
                last_extreme_idx = i
                last_extreme_px = close[i]
    pivots_idx.append(last_extreme_idx)
    pivots_px.append(last_extreme_px)
    return np.array(pivots_idx), np.array(pivots_px)


def analyze_structure(close: np.ndarray, box: dict) -> dict:
    """zigzag结构分析 + 底部计数 + 主力性格分类。"""
    pidx, ppx = zigzag(close)
    if len(pidx) < 3:
        return dict(n_pivots=len(pidx), n_bottoms=0, current_range_low=0,
                    current_range_high=0, personality="数据不足", structure="未知")
    # 识别拐点类型：局部极小=底，局部极大=顶
    is_bottom = np.zeros(len(ppx), dtype=bool)
    is_top = np.zeros(len(ppx), dtype=bool)
    for i in range(1, len(ppx) - 1):
        if ppx[i] < ppx[i - 1] and ppx[i] < ppx[i + 1]:
            is_bottom[i] = True
        elif ppx[i] > ppx[i - 1] and ppx[i] > ppx[i + 1]:
            is_top[i] = True
    # 端点也标记（与邻居比较）
    if len(ppx) >= 2:
        if ppx[0] < ppx[1]:
            is_bottom[0] = True
        else:
            is_top[0] = True
        if ppx[-1] < ppx[-2]:
            is_bottom[-1] = True
        else:
            is_top[-1] = True
    # 当前区间（最后一个完整zigzag段）
    bottom_indices = np.where(is_bottom)[0]
    top_indices = np.where(is_top)[0]
    range_low = float(ppx[bottom_indices[-1]]) if len(bottom_indices) else float(np.nanmin(close))
    range_high = float(ppx[top_indices[-1]]) if len(top_indices) else float(np.nanmax(close))
    # 统计当前箱体内的底部数（zigzag低点在箱体内下半区）
    box_bottom = box["cascade"]["bottom"]
    box_top = box["cascade"]["top"]
    bottoms_in_box = 0
    for bi in bottom_indices:
        if box_bottom <= ppx[bi] <= box_top * 0.6:
            bottoms_in_box += 1
    # 主力性格分类（启发式）
    personality = classify_personality(close, ppx, bottoms_in_box, box)
    # 结构判断
    if bottoms_in_box >= 7:
        structure = "七底极限区(高位分歧巨大)"
    elif bottoms_in_box >= 4:
        structure = "四底(四渡赤水，最后机会)"
    elif bottoms_in_box >= 3:
        structure = "三底(正常结构)"
    else:
        structure = f"底部数{bottoms_in_box}(结构未完成)"
    return dict(
        n_pivots=len(pidx),
        n_bottoms=bottoms_in_box,
        current_range_low=round(range_low, 2),
        current_range_high=round(range_high, 2),
        personality=personality,
        structure=structure,
    )


def classify_personality(close: np.ndarray, ppx: np.ndarray,
                         n_bottoms: int, box: dict) -> str:
    """主力性格启发式分类（手册案例口径）。"""
    n = len(close)
    if n < 60:
        return "数据不足"
    # 计算整体涨幅（从有效起点到现价）
    total_gain = close[-1] / np.nanmin(close) - 1 if np.nanmin(close) > 0 else 0
    # 波动率
    returns = np.diff(close[close > 0]) / close[close > 0][:-1]
    vol = float(np.nanstd(returns)) if len(returns) > 1 else 0
    # 连板特征（连续涨停>9.5%）
    big_up = returns > 0.095
    consecutive_big = 0
    max_consecutive = 0
    for b in big_up:
        if b:
            consecutive_big += 1
            max_consecutive = max(max_consecutive, consecutive_big)
        else:
            consecutive_big = 0
    # 分类
    if total_gain >= 5 and vol < 0.03:
        return "机构票(稳步推升，高位大阴=见顶)"
    if max_consecutive >= 3 and n_bottoms >= 3:
        return "标准游资(三底节奏，连板+炸板出货)"
    if max_consecutive >= 2 and vol > 0.04:
        return "穷主力(缩量连板不花钱，控盘率低)"
    if n_bottoms >= 5:
        return "不肯吃亏型(每败必救，七底通道)"
    if total_gain >= 8:
        return "有情怀型(十倍情节)"
    if n_bottoms >= 3:
        return "标准游资(箱体推进)"
    if vol > 0.05:
        return "废柴主力(反复被爆头，出货困难)"
    return "普通混庄(特征不明显)"


# ============================================================
# 步骤七：方向判断（三步法）
# ============================================================
def judge_direction(chain: dict, box: dict, structure: dict,
                    current_price: float) -> dict:
    """三步法：出货完了吗？到箱体顶了吗？在顶部做什么？"""
    # 第一步：出货完了吗
    if chain["latest_distributed"] and chain["n_trapped"] == 0:
        return dict(direction="主跌浪(出货完成，不碰)",
                    can_buy="不买", reason="主力已全部出货")
    # 第二步：到箱体顶了吗（≥90%按顶部处理）
    if not box["near_top"]:
        return dict(direction="主升浪中(未到箱体顶)",
                    can_buy="可按买点试错",
                    reason=f"现价在箱体{box['box_position_pct']}%位置，未到顶")
    # 第三步：在顶部做什么
    if structure["n_bottoms"] >= 7:
        return dict(direction="顶部七底(极限分歧，方向待定)",
                    can_buy="只做小波段",
                    reason="七底极限区，高位分歧巨大")
    if structure["n_bottoms"] >= 4:
        return dict(direction="顶部四底(四渡赤水)",
                    can_buy="观望，过不去就崩",
                    reason="四底是最后机会")
    return dict(direction="顶部出货区(钩子/双顶/突破待定)",
                can_buy="只做小波段，不追高",
                reason=f"现价在箱体{box['box_position_pct']}%位置，已到顶")


# ============================================================
# 单只股票完整八步法分析
# ============================================================
def analyze_one(code: str, name: str, industry: str, df: pd.DataFrame, dates: np.ndarray) -> dict:
    """对单只股票执行八步法完整分析。"""
    close = df["close"].values.astype(float)
    low = df["low"].values.astype(float)
    high = df["high"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)
    if n < 60:
        return None
    current_price = float(close[-1])
    # 步骤一：有效起点（上市<10年从上市开始，10~20年从最近大底，>20年从最近两大底）
    # 简化：取最近3年内的最低点作为有效起点
    lookback = min(n, 750)  # ~3年
    start_idx = n - lookback
    # 步骤一~三：筹码链
    clusters = find_volume_clusters(close, volume)
    chain = analyze_chains(close, volume, clusters)
    # 步骤四：箱体
    box = identify_box(close, low, current_price, start_idx)
    # 步骤五~六：结构与性格
    structure = analyze_structure(close, box)
    # 步骤七：方向判断
    direction = judge_direction(chain, box, structure, current_price)
    # 步骤八：关键节点
    need_rescue = (not chain["latest_distributed"]) and chain["trapped_max"] > current_price
    first_buy = round(box["cascade"]["bottom"] * 1.02, 2) if box["near_bottom"] else round(
        structure["current_range_low"] * 1.02, 2)
    second_buy = round(box["cascade"]["mid"], 2)
    reduce_point = round(chain["trapped_max"], 2) if chain["trapped_max"] > 0 else round(
        box["cascade"]["top"], 2)
    stop_loss = round(structure["current_range_low"] * 0.97, 2)
    return dict(
        代码=code, 名称=name, 现价=round(current_price, 2),
        有效起点=box["anchor"],
        建仓轮次=chain["n_rounds"],
        已出货轮次=chain["n_distributed"],
        未解救轮次=chain["n_trapped"],
        最新建仓成本=round(chain["latest_cost"], 2),
        最新建仓量=round(chain["latest_vol"] / 1e8, 2),  # 亿股
        最新出货成功="是" if chain["latest_distributed"] else "否",
        未解救最高价=round(chain["trapped_max"], 2),
        主力需自救="是" if need_rescue else "否",
        箱体底=round(box["cascade"]["bottom"], 2),
        箱体中继=round(box["cascade"]["mid"], 2),
        箱体顶=round(box["cascade"]["top"], 2),
        箱体位置pct=box["box_position_pct"],
        锚点x2=box["anchor_multiples"].get("×2", 0),
        锚点x4=box["anchor_multiples"].get("×4", 0),
        锚点x6=box["anchor_multiples"].get("×6", 0),
        汇聚区数=len(box["convergences"]),
        zigzag底部数=structure["n_bottoms"],
        当前区间低=structure["current_range_low"],
        当前区间高=structure["current_range_high"],
        主力性格=structure["personality"],
        结构=structure["structure"],
        方向=direction["direction"],
        能否买=direction["can_buy"],
        第一买点=first_buy,
        第二买点=second_buy,
        减仓点=reduce_point,
        止损点=stop_loss,
        所属板块=industry or "未知",
    )


# ============================================================
# 主流程
# ============================================================
def run_analysis(codes: Optional[List[str]] = None, top_n: Optional[int] = None,
                 force_cpu: bool = False) -> pd.DataFrame:
    """全市场主力分析。"""
    from swingalpha.accel import detect_backend
    backend = detect_backend(force_cpu=force_cpu)
    log.info(f"计算后端: {backend}")
    # 载入股票列表
    sl_df, sl_src = storage.get_stock_list()
    if sl_df is None or len(sl_df) == 0:
        raise RuntimeError("股票列表为空，请先运行 update.py")
    name_map = dict(zip(sl_df["code"].astype(str), sl_df["name"].astype(str)))
    industry_map = dict(zip(sl_df["code"].astype(str), sl_df["industry"].astype(str))) if "industry" in sl_df.columns else {}
    use_codes = codes if codes else sorted(name_map.keys())
    log.info(f"待分析股票: {len(use_codes)} 只")
    # 批量载入日线
    dfs = factors.load_all_stock_dfs(use_codes)
    log.info(f"成功载入 {len(dfs)} 只股票日线")
    results = []
    for code in tqdm(use_codes, desc="八步法分析", ncols=88):
        if code not in dfs:
            continue
        df = dfs[code]
        if df is None or len(df) < 60:
            continue
        name = name_map.get(code, code)
        industry = industry_map.get(code, "")
        try:
            r = analyze_one(code, name, industry, df, df["date"].values)
            if r:
                results.append(r)
        except Exception as e:
            log.debug(f"{code} 分析失败: {e}")
            continue
    if not results:
        raise RuntimeError("无有效分析结果")
    df_out = pd.DataFrame(results)
    # 排序：主力需自救优先，未解救最高价与现价差距大优先
    df_out["自救空间pct"] = np.where(
        df_out["主力需自救"] == "是",
        (df_out["未解救最高价"] / df_out["现价"] - 1) * 100,
        0)
    df_out = df_out.sort_values(["主力需自救", "自救空间pct"],
                                 ascending=[False, False]).reset_index(drop=True)
    # 保存
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = ANALYSIS_DIR / f"主力分析_{ts}.xlsx"
    df_out.to_excel(out_path, index=False)
    log.info(f"分析完成：{len(df_out)} 只，已保存 {out_path}")
    # 控制台输出
    print("\n" + "=" * 100)
    print(f"【主力分析】全市场 {len(df_out)} 只主板股票｜按自救优先级排序")
    print("=" * 100)
    show_cols = ["代码", "名称", "现价", "主力需自救", "未解救最高价", "自救空间pct",
                 "最新建仓成本", "箱体位置pct", "主力性格", "方向", "第一买点", "止损点", "所属板块"]
    show_n = top_n if top_n else min(30, len(df_out))
    from swingalpha.report import show as report_show
    report_show(df_out[show_cols].head(show_n), f"TOP {show_n}（最需自救）", n=show_n)
    print(f"\n完整结果已保存: {out_path}")
    return df_out


def main():
    ap = argparse.ArgumentParser(description="SWING_ALPHA 全市场主力八步法分析")
    ap.add_argument("--codes", nargs="*", default=None, help="指定股票代码")
    ap.add_argument("--top", type=int, default=None, help="只输出前N只最需自救")
    ap.add_argument("--cpu", action="store_true", help="强制CPU")
    args = ap.parse_args()
    try:
        run_analysis(codes=args.codes, top_n=args.top, force_cpu=args.cpu)
    except Exception as e:
        log.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
