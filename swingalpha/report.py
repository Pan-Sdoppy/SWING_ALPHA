# -*- coding: utf-8 -*-
"""结果打印与留档：所属板块列永远在最右边；按策略类型分表打印；自定义等宽对齐。"""
from __future__ import annotations
import unicodedata

import pandas as pd

from .config import PRED_DIR
from .utils import get_logger

log = get_logger()


CJK_WIDTH = 1.5  # 终端里汉字约占1.5个数字宽度


def _display_width(s) -> float:
    """字符串显示宽度：中文/全角字符按 CJK_WIDTH，其余按1。"""
    w = 0.0
    for ch in str(s):
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += CJK_WIDTH
        else:
            w += 1.0
    return w


def _pad(s, width: float) -> str:
    """按显示宽度左填充空格（标准四舍五入到整数空格数）。"""
    s = str(s)
    n = int(width - _display_width(s) + 0.5)
    return s + " " * max(0, n)


def _sector_last(df: pd.DataFrame) -> pd.DataFrame:
    if "所属板块" in df.columns:
        cols = [c for c in df.columns if c != "所属板块"] + ["所属板块"]
        return df[cols]
    return df


def _format_floats(df: pd.DataFrame, as_str: bool) -> pd.DataFrame:
    """所有浮点列统一保留两位小数（打印时转字符串，留档时四舍五入）。"""
    out = df.copy()
    for c in out.select_dtypes(include="float").columns:
        if as_str:
            out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
        else:
            out[c] = out[c].round(2)
    return out


def _print_one_table(show_df: pd.DataFrame, title: str, total: int, n: int):
    """打印单张等宽表格（已格式化、已fillna）。"""
    cols = list(show_df.columns)
    widths = []
    for c in cols:
        w = _display_width(c)
        for v in show_df[c]:
            w = max(w, _display_width(v))
        widths.append(w)
    total_w = int(round(sum(widths) + (len(cols) - 1) * 2))
    print("\n" + "=" * total_w)
    print(title + f"（共 {total} 只，展示前 {min(n, total)}）")
    print("=" * total_w)
    print("  ".join(_pad(c, widths[i]) for i, c in enumerate(cols)))
    for _, row in show_df.iterrows():
        print("  ".join(_pad(row[c], widths[i]) for i, c in enumerate(cols)))


def _drop_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
    """丢弃全为"-"的列（保留"类型""所属板块"）。"""
    keep = []
    for c in df.columns:
        if c in ("类型", "所属板块"):
            keep.append(c)
        elif (df[c] == "-").all():
            continue
        else:
            keep.append(c)
    return df[keep]


def show(df: pd.DataFrame, title: str, n: int = 50):
    """按策略类型分表打印：A杀/趋势波段/底部簇/箱体回踩 各自一张表，只显示有数据的列。"""
    df = _sector_last(df)
    if df.empty:
        print("\n" + "=" * 60)
        print(title + "（共 0 只）")
        print("=" * 60)
        print("（无符合条件标的）")
        return

    # 有"类型"列且多种类型 → 分表打印
    if "类型" in df.columns and df["类型"].nunique() > 1:
        for typ in df["类型"].unique():
            sub_raw = df[df["类型"] == typ]
            sub = _format_floats(sub_raw.head(n), as_str=True).fillna("-")
            sub = _drop_empty_cols(sub)
            _print_one_table(sub, f"{title} · {typ}", len(sub_raw), n)
    else:
        show_df = _format_floats(df.head(n), as_str=True).fillna("-")
        show_df = _drop_empty_cols(show_df)
        _print_one_table(show_df, title, len(df), n)


def save(df_ultra: pd.DataFrame, df_swing: pd.DataFrame, mode: str, asof: str) -> dict:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p1 = PRED_DIR / f"{asof}_{mode}_超短_{ts}.csv"
    p2 = PRED_DIR / f"{asof}_{mode}_波段_{ts}.csv"
    _format_floats(_sector_last(df_ultra), as_str=False).to_csv(p1, index=False, encoding="utf-8-sig")
    _format_floats(_sector_last(df_swing), as_str=False).to_csv(p2, index=False, encoding="utf-8-sig")
    log.info(f"结果已留档：{p1.name} / {p2.name}")
    return {"ultra": str(p1), "swing": str(p2)}
