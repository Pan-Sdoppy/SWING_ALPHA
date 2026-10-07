# -*- coding: utf-8 -*-
"""
SWING_ALPHA · 每日选股预测入口
--------------------------------------------------------------------
职责：把“交易日当天可做超短(隔日套利) 或 波段(超跌拐头)”的股票全部筛出并打印。
本脚本【不做 update 的增量落库工作】：
  · 盘后/盘前/周末/节假日：直接基于 data 存量日线 + 因子缓存扫描；
  · 盘中(工作日 9:30~15:00)：拉取新浪实时快照，在内存追加“当日临时K线”
    （仅缓存到 data/intraday，不写入正式增量），重算后扫描，并对当日触板股
    拉分钟线判定封板时间/开板/红旗；盘后请先运行 update.py 落库当日数据。

输出两张表：【超短·隔日套利】与【波段·超跌拐头】，所属板块列固定在最右，
同时在 data/processed/predictions/ 留档 CSV。

用法：
  python predict.py                 # 自动判断盘中/盘后
  python predict.py --cpu           # 强制 CPU（不使用 GPU）
  python predict.py --no-minute     # 盘中不拉分钟线（更快）
  python predict.py --codes 600000 000001   # 小样本调试
  python predict.py --spot-cache    # 盘中使用最近一次快照缓存（离线调试）
"""
import argparse
import sys
import traceback

from swingalpha.engine import run_predict
from swingalpha.utils import get_logger

log = get_logger()


def main():
    ap = argparse.ArgumentParser(description="SWING_ALPHA 每日超短/波段选股")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 后端")
    ap.add_argument("--no-minute", action="store_true", help="盘中不拉候选股分钟线")
    ap.add_argument("--spot-cache", action="store_true", help="用最近盘中快照缓存")
    ap.add_argument("--no-save", action="store_true", help="只打印，不留档 CSV")
    ap.add_argument("--codes", nargs="*", default=None, help="指定股票（调试）")
    args = ap.parse_args()
    try:
        run_predict(force_cpu=args.cpu, spot_cache=args.spot_cache,
                    codes=args.codes, no_minute=args.no_minute,
                    save=not args.no_save)
    except RuntimeError as e:
        # 可预期的使用问题（如未先下载数据）：给清晰提示，不打 traceback
        log.error(str(e))
        sys.exit(2)
    except Exception as e:  # noqa
        log.error(f"预测失败：{e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
