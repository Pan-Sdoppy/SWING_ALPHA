# -*- coding: utf-8 -*-
"""
SWING_ALPHA · 历史回测入口
--------------------------------------------------------------------
回测 2017 年至最新交易日，分【超短·隔日套利】与【波段·A杀低吸】两套：
  · 严格无未来函数：信号只用截至当日 t 的数据；超短 t 收盘(涨停价)买、t+1 开盘卖；
    波段 t+1 开盘买，放量大阳线结束退出。
  · 写死最优参数：超短 top1，波段 bigbar/top3。
  · 输出年度(2017-2025)与 2026 逐月明细（胜率/平均收益/盈亏比/区间收益/最大回撤）；
    CSV 留档 data/processed/backtests/。

前置：先更新数据（数据从2016年起下载）
  python update.py

用法：
  python backtest.py                 # 全市场回测 2017-01 至今
  python backtest.py --cpu           # 强制 CPU
  python backtest.py --codes 600000 000001   # 小样本调试
  python backtest.py --start 2020-06-01      # 自定义起点
"""
import argparse
import sys
import traceback

from config.strategy_params import BACKTEST
from swingalpha.backtest_engine import run_backtest
from swingalpha.utils import get_logger

log = get_logger()


def main():
    ap = argparse.ArgumentParser(description="SWING_ALPHA 超短/波段历史回测")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 后端")
    ap.add_argument("--codes", nargs="*", default=None, help="指定股票（调试）")
    ap.add_argument("--start", default=None, help=f"回测起点 YYYY-MM-DD，默认 {BACKTEST['start']}")
    ap.add_argument("--bypass-season", action="store_true", help="实验：不区分大盘四季，每天都买（绕过冬天门控）")
    ap.add_argument("--first-day-loss-exit", action="store_true", help="实验：波段入场当天收盘亏损即平仓")
    args = ap.parse_args()
    try:
        run_backtest(force_cpu=args.cpu, codes=args.codes,
                     start=args.start or BACKTEST["start"],
                     bypass_season=args.bypass_season,
                     first_day_loss_exit=args.first_day_loss_exit)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(2)
    except Exception as e:  # noqa
        log.error(f"回测失败：{e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
