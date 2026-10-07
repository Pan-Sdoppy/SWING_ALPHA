# -*- coding: utf-8 -*-
"""
baostock 行业映射构建（单连接，禁止并发）。
用法: python -m swingalpha.datasource.industry_builder
优先用 query_all_industry() 单次全量查询（仅1次API请求）；
失败时回退单线程逐只查询（约3052次请求，仍远低于日限5万）。
输出 data/raw/_industry_map.json，供 storage.get_stock_list 合并。
"""
import json, re, time
from pathlib import Path

import pandas as pd

from ..storage import RAW_DIR, load_stock_list_cache
from .baostock_source import BaoStockSession
from ..utils import get_logger

log = get_logger()
OUT = RAW_DIR / "_industry_map.json"


def _clean_industry(raw: str) -> str:
    """baostock返回如 'J66货币金融服务'，去掉前缀代码保留 '货币金融服务'。"""
    if not raw:
        return ""
    m = re.match(r"^[A-Z]\d+", raw)
    if m:
        return raw[m.end():]
    return raw


def build() -> dict:
    """构建全量行业映射并保存到 _industry_map.json。单连接，禁止并发。"""
    cached, _ = load_stock_list_cache()
    if cached is None or len(cached) == 0:
        raise RuntimeError("股票列表缓存为空，请先运行 update.py")
    universe = set(cached["code"].astype(str).str.zfill(6).tolist())

    # 合并已有映射（增量补拉）
    existing = {}
    if OUT.exists():
        with open(OUT, "r", encoding="utf-8") as f:
            existing = json.load(f)

    sess = BaoStockSession()
    if not sess.login():
        raise RuntimeError("baostock 登录失败")

    ind_map = dict(existing)
    try:
        # —— 方案A：单次全量查询（仅1次请求，优先）——
        log.info("尝试 query_all_industry() 单次全量查询…")
        t0 = time.time()
        df_all = sess.query_all_industry()
        if df_all is not None and len(df_all) > 0:
            log.info(f"全量查询返回 {len(df_all)} 行，耗时 {time.time()-t0:.1f}s")
            # df_all 列: updateDate, code, code_name, industry, industryClassification
            code_col = "code" if "code" in df_all.columns else df_all.columns[1]
            ind_col = "industry" if "industry" in df_all.columns else df_all.columns[3]
            for _, row in df_all.iterrows():
                raw_code = str(row[code_col])
                # baostock code 格式如 sh.600000 → 提取6位
                m = re.search(r"(\d{6})", raw_code)
                if m:
                    code = m.group(1)
                    if code in universe:
                        ind_map[code] = _clean_industry(str(row[ind_col]))
            n_has = sum(1 for c in universe if ind_map.get(c))
            log.info(f"行业映射完成：{n_has}/{len(universe)} 只有行业")
        else:
            log.warning("全量查询返回为空，回退逐只查询…")
            _build_per_stock(sess, universe, ind_map)
    finally:
        sess.logout()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(ind_map, f, ensure_ascii=False, indent=2)
    n_has = sum(1 for c in universe if ind_map.get(c))
    log.info(f"已保存 {OUT}，覆盖率 {n_has}/{len(universe)} ({n_has/len(universe)*100:.1f}%)")
    return ind_map


def _build_per_stock(sess, universe, ind_map):
    """回退方案：单线程逐只查询（禁止多进程/并发）。"""
    missing = [c for c in sorted(universe) if c not in ind_map]
    log.info(f"逐只查询 {len(missing)} 只（单线程，预计 {len(missing)*0.5/60:.0f} 分钟）…")
    t0 = time.time()
    for idx, code in enumerate(missing):
        prefix = "sh" if code.startswith("6") else "sz"
        try:
            df_ind = sess.query_stock_industry(f"{prefix}.{code}")
            if df_ind is not None and len(df_ind) > 0:
                ind_col = "industry" if "industry" in df_ind.columns else df_ind.columns[-1]
                for _, row in df_ind.iterrows():
                    if str(row[ind_col]):
                        ind_map[code] = _clean_industry(str(row[ind_col]))
                        break
        except RuntimeError as e:
            # baostock_source 的日请求上限守卫抛出，停止查询
            log.error(str(e))
            break
        except Exception as e:
            log.warning(f"  {code} 行业查询失败: {e}")
        if (idx + 1) % 200 == 0:
            log.info(f"  进度 {idx+1}/{len(missing)}，耗时 {(time.time()-t0)/60:.1f} 分钟")
    log.info(f"逐只查询完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    build()
