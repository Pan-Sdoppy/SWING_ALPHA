# -*- coding: utf-8 -*-
"""
落盘存储：
- data/raw/{code}.parquet        每只股票独立日线（前复权），单股失败互不影响
- data/raw/_meta.json            每只股票最新日期/行数/名称/行业，秒级判断增量；可从文件重建
- data/raw/_stock_list.csv       股票列表缓存(7天)，API失败回退过期缓存，读入仍强制过滤
- data/indices/{idx}.parquet     指数日线
- data/processed/factor_data.parquet  因子计算缓存
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .config import (RAW_DIR, INDEX_DIR, META_PATH, STOCK_LIST_PATH,
                     FACTOR_CACHE, SPOT_CACHE, DAILY_COLS)
from .utils import get_logger, is_main_board, is_clean_name
from .datasource import akshare_source as aksrc

log = get_logger()


# ---------------- 个股 parquet ----------------
def raw_path(code: str) -> str:
    return str(RAW_DIR / f"{code}.parquet")


def load_daily(code: str) -> Optional[pd.DataFrame]:
    p = raw_path(code)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p)
        return df
    except Exception as e:  # noqa
        log.warning(f"读取 {code} parquet 失败: {e}")
        return None


def save_daily(code: str, df: pd.DataFrame):
    """整表覆盖写（增量已在下载时拼接完成）。"""
    if df is None:
        return
    p = raw_path(code)
    tmp = p + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


# ---------------- 元数据 _meta.json ----------------
def load_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception as e:  # noqa
            log.warning(f"_meta.json 损坏，将从 parquet 重建: {e}")
    # 缺失/损坏但已下载过 parquet（如上次中途被 Ctrl+C）：自动重建，保证重跑秒级 skip
    if list(RAW_DIR.glob("*.parquet")):
        return rebuild_meta()
    return {}


def save_meta(meta: dict):
    tmp = str(META_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp, META_PATH)


def rebuild_meta() -> dict:
    """从 data/raw 下所有 parquet 重建元数据（秒级，无需逐股联网）。"""
    meta = {}
    files = sorted(RAW_DIR.glob("*.parquet"))
    for fp in files:
        code = fp.stem
        if not is_main_board(code):
            continue
        try:
            df = pd.read_parquet(fp, columns=["date", "code"])
            if len(df) == 0:
                continue
            name_col = "name" if "name" in df.columns else None
            meta[code] = {
                "last_date": str(df["date"].iloc[-1]),
                "rows": int(len(df)),
                "name": df["name"].iloc[-1] if name_col else "",
            }
        except Exception:  # noqa
            continue
    save_meta(meta)
    log.info(f"已从 {len(files)} 个 parquet 重建 _meta.json，有效 {len(meta)} 只")
    return meta


def update_meta(meta: dict, code: str, df: pd.DataFrame, name: str = ""):
    if df is None or len(df) == 0:
        return
    meta[code] = {
        "last_date": str(df["date"].iloc[-1]),
        "rows": int(len(df)),
        "name": name or meta.get(code, {}).get("name", ""),
    }


# ---------------- 股票列表 7 天缓存 ----------------
def load_stock_list_cache():
    """返回 (df, age_days)；缓存不存在返回 (None, None)。读入强制主板非ST过滤。"""
    if not STOCK_LIST_PATH.exists():
        return None, None
    try:
        df = pd.read_csv(STOCK_LIST_PATH, dtype={"code": str})
        mtime = datetime.fromtimestamp(STOCK_LIST_PATH.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 86400.0
        df = df[df["code"].map(is_main_board)]
        if "name" in df.columns:
            df = df[df["name"].fillna("").map(is_clean_name)]
        return df.reset_index(drop=True), age
    except Exception as e:  # noqa
        log.warning(f"股票列表缓存读取失败: {e}")
        return None, None


def save_stock_list(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return
    tmp = str(STOCK_LIST_PATH) + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, STOCK_LIST_PATH)


def get_stock_list(cache_days: int = 7, timeout: int = 30):
    """
    获取主板非ST列表（含行业）：缓存新鲜(<cache_days天且有industry列)直接用，否则在线刷新。
    行业数据从本地 _industry_map.json 合并（baostock多进程一次性构建，东财在线已不可用）。
    在线失败时回退过期缓存（不因其过期而报错）。
    返回 (df, source)；两者都失败返回 (None, 'failed')。
    """
    # 1. 缓存新鲜且有行业数据 → 直接用
    cached, age = load_stock_list_cache()
    if (cached is not None and len(cached) > 0 and age is not None
            and age <= cache_days and "industry" in cached.columns
            and cached["industry"].astype(bool).any()):
        return cached, "cache"

    # 2. 在线刷新股票列表（新浪快照，稳定可用）
    try:
        df = aksrc.fetch_stock_list(timeout=timeout)
        if df is not None and len(df) > 0:
            # 3. 从本地行业映射JSON合并（baostock多进程预构建）
            df["industry"] = ""
            ind_path = RAW_DIR / "_industry_map.json"
            if ind_path.exists():
                try:
                    with open(ind_path, "r", encoding="utf-8") as f:
                        ind_map = json.load(f)
                    df["industry"] = df["code"].astype(str).map(ind_map).fillna("")
                    n_has = df["industry"].astype(bool).sum()
                    log.info(f"行业映射（本地）：{n_has}/{len(df)} 只有行业")
                except Exception as e:
                    log.warning(f"行业映射读取失败（不阻塞）: {e}")
            else:
                log.warning("未找到 _industry_map.json，行业列将为空（运行 _fetch_industry.py 构建）")
            save_stock_list(df)
            log.info(f"在线获取股票列表成功：{len(df)} 只（已刷新缓存）")
            return df, "online"
        raise RuntimeError("在线列表为空")
    except Exception as e:  # noqa
        log.warning(f"在线获取股票列表失败: {e}")
        if cached is not None and len(cached) > 0:
            log.warning(f"回退到本地股票列表缓存（{age:.1f} 天前，{len(cached)} 只）")
            return cached, "stale_cache"
        return None, "failed"


# ---------------- 指数 ----------------
def index_path(idx: str) -> str:
    return str(INDEX_DIR / f"{idx}.parquet")


def save_index(idx: str, df: pd.DataFrame):
    if df is None or len(df) == 0:
        return
    p = index_path(idx)
    df.to_parquet(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)


def load_index(idx: str) -> Optional[pd.DataFrame]:
    p = index_path(idx)
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


# ---------------- 因子缓存 ----------------
def save_factor_cache(long_df: "pd.DataFrame", asof: str):
    """缓存因子长表 parquet 并记录 asof。"""
    if long_df is None or len(long_df) == 0:
        return
    long_df.to_parquet(FACTOR_CACHE, index=False)
    meta = {"asof": asof, "rows": int(len(long_df)),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(str(FACTOR_CACHE) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_factor_cache():
    if not FACTOR_CACHE.exists():
        return None, None
    try:
        big = pd.read_parquet(FACTOR_CACHE)
        meta = {}
        mp = str(FACTOR_CACHE) + ".meta.json"
        if os.path.exists(mp):
            meta = json.loads(open(mp, encoding="utf-8").read())
        return big, meta
    except Exception as e:  # noqa
        log.warning(f"因子缓存读取失败: {e}")
        return None, None
