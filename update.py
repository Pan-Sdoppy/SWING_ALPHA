# -*- coding: utf-8 -*-
"""
SWING_ALPHA · 数据增量更新入口
--------------------------------------------------------------------
职责：把本地全市场主板日线数据增量更新到最新交易日；已最新则秒级跳过。
  · 历史日线：baostock（前复权，全局登录一次，连续失败自动重连）
  · 股票列表：akshare 新浪全市场快照，缓存 7 天，失败回退过期缓存
  · 每只股票独立 parquet：data/raw/{code}.parquet，单股失败互不影响
  · 元数据 data/raw/_meta.json 秒级判断增量，可从 parquet 重建
  · 单股重试3次(1s/2s) + 失败股补下载2轮；网络失败与节假日空表严格区分
  · 单线程下载（akshare V8 引擎不支持多线程；baostock 亦顺序调用最稳）

用法：
  python update.py                 # 全市场增量更新
  python update.py --limit 20      # 仅更新前20只（小样本验证）
  python update.py --codes 600000 000001
  python update.py --list-only     # 只刷新股票列表/行业，不下载日线
  python update.py --rebuild-meta  # 从已有 parquet 重建 _meta.json
"""
import argparse
import sys
import traceback

from swingalpha import storage
from swingalpha.data_pipeline import run_update, run_backfill
from swingalpha.utils import get_logger

log = get_logger()


def main():
    ap = argparse.ArgumentParser(description="SWING_ALPHA 数据增量更新")
    ap.add_argument("--limit", type=int, default=None, help="仅更新前 N 只（调试）")
    ap.add_argument("--codes", nargs="*", default=None, help="指定股票代码")
    ap.add_argument("--list-only", action="store_true", help="只刷新列表/行业")
    ap.add_argument("--rebuild-meta", action="store_true", help="从 parquet 重建元数据")
    ap.add_argument("--backfill", metavar="YYYY-MM-DD", default=None,
                    help="向更早日期回填历史（回测用），如 --backfill 2020-07-01")
    args = ap.parse_args()

    try:
        if args.rebuild_meta:
            meta = storage.rebuild_meta()
            log.info(f"重建完成，共 {len(meta)} 只")
            return
        if args.backfill:
            stats = run_backfill(args.backfill, limit=args.limit, codes=args.codes)
            log.info(f"回填统计：{ {k: v for k, v in stats.items() if k != 'failed_remaining'} }")
            if stats.get("failed_remaining"):
                log.warning(f"仍有 {len(stats['failed_remaining'])} 只失败，再次运行即可续传")
            return
        stats = run_update(limit=args.limit, codes=args.codes,
                           list_only=args.list_only)
        if not args.list_only:
            log.info(f"统计：{ {k: v for k, v in stats.items() if k != 'failed_remaining'} }")
            if stats.get("failed_remaining"):
                log.warning(f"仍有 {len(stats['failed_remaining'])} 只失败，"
                            f"再次运行本脚本即可续传，已下载数据不受影响")
    except Exception as e:  # noqa
        log.error(f"更新失败：{e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
