# -*- coding: utf-8 -*-
"""全局路径与常量。"""
from __future__ import annotations
import os
from pathlib import Path

# 项目根目录（本文件位于 <root>/swingalpha/config.py）
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # 每只股票独立 parquet：{code}.parquet
INDEX_DIR = DATA_DIR / "indices"      # 指数日线
PROCESSED_DIR = DATA_DIR / "processed"
INTRADAY_DIR = DATA_DIR / "intraday"  # 盘中临时快照（不写入正式增量）
PRED_DIR = PROCESSED_DIR / "predictions"
BACKTEST_DIR = PROCESSED_DIR / "backtests"
LOG_DIR = ROOT / "logs"

META_PATH = RAW_DIR / "_meta.json"
STOCK_LIST_PATH = RAW_DIR / "_stock_list.csv"
FACTOR_CACHE = PROCESSED_DIR / "factor_data.parquet"
SPOT_CACHE = INTRADAY_DIR / "_spot_latest.parquet"

for _d in (DATA_DIR, RAW_DIR, INDEX_DIR, PROCESSED_DIR, INTRADAY_DIR,
           PRED_DIR, BACKTEST_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 标准日线列（parquet 落库 schema）
DAILY_COLS = [
    "date", "code", "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "pctChg",
]

# 统一并入顶层 config/strategy_params.py 的策略/数据参数（单一事实源）。
# 用文件路径直接加载，避免与包名 swingalpha.config 冲突，且不依赖运行目录。
import importlib.util as _ilu

_pf = ROOT / "config" / "strategy_params.py"
_spec = _ilu.spec_from_file_location("strategy_params", _pf)
_sp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sp)
for _k in dir(_sp):
    if _k.isupper():
        globals()[_k] = getattr(_sp, _k)
del _ilu, _pf, _spec, _sp, _k
