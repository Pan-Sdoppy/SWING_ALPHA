# -*- coding: utf-8 -*-
"""
预测编排：
- 盘后(含盘前/周末/节假日)：基于存量日线 + 因子缓存，直接扫描。
- 盘中：拉新浪快照在内存追加当日临时K线（不做 update 的增量落库），重算因子后扫描，
        并对当日触板候选拉分钟线做封板质量增强。
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd

from . import accel, storage, factors, report, intraday
from .calendar_util import session_state
from .strategies.ultra_short import scan_ultra_short
from .strategies.swing import scan_swing
from .regime import market_snapshot
from .utils import get_logger

log = get_logger()


def _maps():
    name_map, ind_map = {}, {}
    cached, _ = storage.get_stock_list()
    if cached is not None and len(cached):
        name_map = dict(zip(cached["code"].astype(str),
                            cached.get("name", pd.Series([""] * len(cached)))))
        if "industry" in cached.columns:
            ind_map = dict(zip(cached["code"].astype(str),
                               cached["industry"].fillna("")))
    meta = storage.load_meta()
    for c, rec in meta.items():
        name_map.setdefault(c, rec.get("name", ""))
    return name_map, ind_map


def _latest_data_date() -> str:
    sh = storage.load_index("sh000001")
    if sh is not None and len(sh):
        return str(sh["date"].iloc[-1])
    meta = storage.load_meta()
    lasts = [v.get("last_date") for v in meta.values() if v.get("last_date")]
    return max(lasts) if lasts else datetime.now().strftime("%Y-%m-%d")


def _build_after_panel(backend, name_map, ind_map, asof):
    long, cm = storage.load_factor_cache()
    if long is not None and cm.get("asof") == asof:
        log.info(f"命中因子缓存（asof={asof}），直接复用")
        return factors.build_panel_from_long(long, name_map, ind_map, asof=asof)
    log.info("因子缓存未命中，从原始 parquet 计算因子…")
    codes = sorted(name_map.keys())
    dfs = factors.load_all_stock_dfs(codes)
    panel = factors.build_panel(dfs, name_map, ind_map, backend=backend, asof=asof)
    try:
        storage.save_factor_cache(factors.export_long(panel), asof)
    except Exception as e:  # noqa
        log.warning(f"因子缓存写入失败(不影响结果): {e}")
    return panel


def run_predict(force_cpu: bool = False, spot_cache: bool = False,
                codes: Optional[List[str]] = None, no_minute: bool = False,
                save: bool = True):
    backend = accel.detect_backend(force_cpu=force_cpu)
    name_map, ind_map = _maps()

    state = session_state()
    spot = None
    mode = "after"
    asof = _latest_data_date()

    # 盘中：尝试实时快照，并用 ticktime 确认今天确为交易日，否则回退盘后
    if state == "intraday" and not codes:
        spot = intraday.load_spot(use_cache=spot_cache)
        if spot is not None and len(spot) and "ticktime" in spot.columns:
            # 时间戳仅 HH:MM:SS；交易日盘中应接近当前时刻，节假日则停在上一交易日15:00
            tt = (pd.to_datetime(spot["ticktime"].dropna().astype(str).str[-8:],
                                 format="%H:%M:%S", errors="coerce"))
            now_min = datetime.now().hour * 60 + datetime.now().minute
            # 午休时段(11:30-13:00)：快照时间戳停在11:30属正常，以此为基准判断新鲜度
            if 11 * 60 + 30 <= now_min < 13 * 60:
                expected_min = 11 * 60 + 30
            else:
                expected_min = now_min
            fresh = False
            if len(tt):
                hhmm = tt.dt.hour * 60 + tt.dt.minute
                med = int(hhmm.median())
                # 盘中（上午9:30-11:30/下午13:00-15:00）且时间戳与期望时刻相差≤20分钟
                fresh = abs(med - expected_min) <= 20
            if fresh:
                mode = "intraday"
                asof = datetime.now().strftime("%Y-%m-%d")
            else:
                log.warning(f"快照时间戳非盘中实时（中位数{med//60:02d}:{med%60:02d}，"
                            f"期望{expected_min//60:02d}:{expected_min%60:02d}），按盘后存量数据处理")
                spot = None
        elif spot is not None and len(spot):
            mode = "intraday"
            asof = datetime.now().strftime("%Y-%m-%d")
        else:
            log.warning("盘中实时快照不可用，回退盘后存量数据")

    log.info(f"运行模式：{ '盘中实时' if mode=='intraday' else '盘后存量' }，asof={asof}")

    # 载入个股日线
    use_codes = codes if codes else sorted(name_map.keys())
    dfs = factors.load_all_stock_dfs(use_codes)
    log.info(f"载入本地日线 {len(dfs)} 只")
    if not dfs:
        raise RuntimeError("本地无任何个股日线数据，请先运行 python update.py 完成（增量）下载。")

    minute_info = {}
    if mode == "intraday" and spot is not None:
        dfs = intraday.build_intraday_dfs(dfs, spot, asof=asof)
        panel = factors.build_panel(dfs, name_map, ind_map, backend=backend, asof=asof)
        # 注入当日指数实时涨跌（门控用，baostock 盘后才有当日指数）
        try:
            from .datasource import akshare_source as aksrc
            ispot = aksrc.fetch_index_spot()
            for idx, kv in ispot.items():
                if idx in panel.index_close:
                    panel.index_close[idx].loc[asof] = kv["price"]
                    panel.index_pct[idx].loc[asof] = kv["pct"]
        except Exception as e:  # noqa
            log.warning(f"盘中指数快照获取失败: {e}")
        # 仅对当日触板候选拉分钟线（少量，best-effort）
        if not no_minute:
            t = panel.col_of(asof)
            cand = [panel.codes[i] for i in range(len(panel.codes))
                    if t is not None and panel.f["touched"][i, t] == 1]
            cand = cand[:60]
            log.info(f"对 {len(cand)} 只当日触板候选拉取分钟线做封板质量判定…")
            for c in cand:
                i = np.where(panel.codes == c)[0]
                if not len(i):
                    continue
                i = int(i[0])
                pc = panel.f["preclose"][i, t]
                if np.isnan(pc):
                    continue
                try:
                    info = intraday.minute_seal_info(c, float(pc))
                    if info:
                        minute_info[c] = info
                except Exception:  # noqa
                    continue
    else:
        if codes:
            # 小样本调试：直接重算，不读写全市场因子缓存
            panel = factors.build_panel(dfs, name_map, ind_map, backend=backend, asof=asof)
        else:
            panel = _build_after_panel(backend, name_map, ind_map, asof)

    snap = market_snapshot(panel, asof)
    csi = f"{snap['csi1000_pct']:.2f}" if not np.isnan(snap['csi1000_pct']) else "-"
    brk = f"{snap['broken_rate']:.2f}" if not np.isnan(snap['broken_rate']) else "-"
    log.info(f"市场环境：相位={snap['phase']}，中证1000={csi}%，炸板率={brk}%，极端情绪={snap['extreme_panic']}")

    df_ultra = scan_ultra_short(panel, snap, mode=mode, minute_info=minute_info, show_all=True)
    df_swing = scan_swing(panel, snap, mode=mode, show_all=True)

    report.show(df_ultra, f"【超短·隔日套利】选股结果  {asof}  {mode}", n=40)
    report.show(df_swing, f"【波段·超跌拐头】选股结果  {asof}  {mode}", n=50)

    # —— 统一市场环境选股建议 ——
    phase = snap.get("phase", "unknown")
    csi_val = snap.get("csi1000_pct", 0)
    brk_val = snap.get("broken_rate", 0)
    suggestions = []
    if phase == "winter":
        suggestions.append("市场处于冬天（主跌段），以上标的仅供观察，建议空仓或轻仓≤20%")
    elif phase == "deep_autumn":
        suggestions.append("市场处于深秋（高位滞涨≥4日），逢高减仓，仓位≤30%")
    elif phase == "autumn":
        suggestions.append("市场处于秋天（1~3日未新高），逐步减仓，仓位≤50%")
    elif phase == "summer":
        suggestions.append("市场处于夏天（主升段，创20日新高），可积极做多，仓位≤80%")
    if not np.isnan(csi_val) and csi_val <= 0:
        suggestions.append(f"中证1000 {csi_val:.2f}%≤0，中小盘偏弱，回避追高")
    if not np.isnan(brk_val) and brk_val >= 35:
        suggestions.append(f"炸板率 {brk_val:.2f}%≥35%，情绪极端恶劣，禁止开新仓")
    print("=" * 80)
    print("【选股建议】" + "；".join(suggestions) if suggestions else "【选股建议】市场环境正常，按策略执行")
    print("=" * 80)

    paths = {}
    if save:
        paths = report.save(df_ultra, df_swing, mode, asof)
    return {"mode": mode, "asof": asof, "snap": snap,
            "ultra": df_ultra, "swing": df_swing, "paths": paths}
