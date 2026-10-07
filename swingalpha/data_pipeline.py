# -*- coding: utf-8 -*-
"""
数据增量更新管线（update.py 调用）。
单线程（baostock 无 V8 依赖；akshare V8 引擎不支持多线程，本项目下载亦单线程）。
"""
from __future__ import annotations
import socket
import time
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
from tqdm import tqdm

from .config import BAOSTOCK, UNIVERSE, INDICES
from .utils import get_logger
from . import storage, universe
from .datasource.baostock_source import BaoStockSession
from .calendar_util import latest_trading_day

log = get_logger()


def _incremental_start(last_date: Optional[str]) -> str:
    if not last_date:
        return BAOSTOCK["start_date"]
    d = datetime.strptime(last_date, "%Y-%m-%d").date() + timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _login_with_retry(sess: "BaoStockSession", attempts: Optional[int] = None) -> bool:
    """初始登录带退避重试：瞬时 WinError 10054 / 接收数据异常时不整体中止。
    首次失败后按 1s/2s/4s/8s 递增重试，每次先 logout 清掉坏 socket。"""
    attempts = attempts or (UNIVERSE["retry_times"] + 1)
    backoffs = list(UNIVERSE["retry_backoff"]) + [4.0, 8.0]
    for k in range(attempts):
        if sess.login():
            return True
        if k < attempts - 1:
            wait = backoffs[min(k, len(backoffs) - 1)]
            log.warning(f"baostock 登录未成功，{wait:.0f}s 后重试（第 {k + 2}/{attempts} 次）…")
            try:
                sess.logout()
            except Exception:  # noqa
                pass
            time.sleep(wait)
    return False


def _download_one(sess: BaoStockSession, code: str, start: str, end: str):
    """单股带递增重试。返回 (raw_df, status): status in ok/network。
    超时/乱码(0xa9等)基本意味着当前连接已坏——首次失败即重连，再退避重试，
    避免在同一条坏连接上空等满 3 次。"""
    delays = list(UNIVERSE["retry_backoff"]) + [UNIVERSE["retry_backoff"][-1]]
    for attempt in range(UNIVERSE["retry_times"]):
        df, status = sess.query_daily(code, start, end,
                                     adjustflag=BAOSTOCK["adjustflag"])
        if status == "ok":
            return df, "ok"
        if attempt < UNIVERSE["retry_times"] - 1:
            sess.reconnect(cooldown=UNIVERSE["reconnect_cooldown"])
            time.sleep(delays[min(attempt, len(delays) - 1)])
    return None, "network"


def update_indices(sess: BaoStockSession, end_date: str):
    log.info("更新指数日线…")
    for idx, cname in INDICES.items():
        start = BAOSTOCK["start_date"]
        old = storage.load_index(idx)
        if old is not None and len(old):
            start = (datetime.strptime(str(old["date"].iloc[-1]), "%Y-%m-%d").date()
                     + timedelta(days=1)).strftime("%Y-%m-%d")
        if start > end_date:
            continue
        df, st = None, "network"
        for attempt in range(3):
            df, st = sess.query_index_daily(idx, start, end_date)
            if st == "ok":
                break
            time.sleep(1.0 + attempt)
        if st != "ok" or df is None:
            log.warning(f"指数 {cname} 更新失败：{st}")
            continue
        if len(df):
            for c in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if old is not None and len(old):
                df = pd.concat([old, df], ignore_index=True).drop_duplicates("date")
                df = df.sort_values("date")
            storage.save_index(idx, df)
    log.info("指数日线更新完成")


def run_backfill(start_date: str, limit: Optional[int] = None,
                 codes: Optional[List[str]] = None) -> dict:
    """向更早日期回填（为回测补历史）。对每只已落地个股，下载 [start_date, 本地最早日-1]
    并前置拼接。断点续传/单股失败隔离/补下载/元数据周期落盘与 run_update 一致。"""
    stock_df, src = storage.get_stock_list(
        cache_days=UNIVERSE["stock_list_cache_days"],
        timeout=UNIVERSE["stock_list_timeout"])
    if stock_df is None:
        raise RuntimeError("无法获取股票列表，且本地无可用缓存，终止回填")

    sess = BaoStockSession(timeout=30)
    if not _login_with_retry(sess):
        raise RuntimeError("baostock 多次登录仍失败，请稍后重跑；已下载数据不会丢失。")
    try:
        if codes:
            targets = [c for c in codes if c in set(stock_df["code"])]
        else:
            targets = list(stock_df["code"])
        if limit:
            targets = targets[:limit]

        meta = storage.load_meta()
        name_map = dict(zip(stock_df["code"], stock_df["name"]))
        failed: List[str] = []
        stats = dict(skipped=0, backfilled=0, empty=0, failed=0, missing=0)
        consec_fail = 0

        def process(code: str):
            nonlocal consec_fail
            old = storage.load_daily(code)
            if old is None or len(old) == 0:
                stats["missing"] += 1          # 本地无数据，交给正常 update 全量
                return
            first = str(old["date"].iloc[0])
            if start_date >= first:
                stats["skipped"] += 1          # 已覆盖该起点
                return
            back_end = (datetime.strptime(first, "%Y-%m-%d").date()
                        - timedelta(days=1)).strftime("%Y-%m-%d")
            raw, stt = _download_one(sess, code, start_date, back_end)
            if stt == "network":
                failed.append(code)
                stats["failed"] += 1
                consec_fail += 1
                slow = sess.last_latency >= UNIVERSE["slow_latency"]
                if consec_fail >= UNIVERSE["max_consecutive_fail"] or slow:
                    sess.reconnect(cooldown=UNIVERSE["reconnect_cooldown"])
                    consec_fail = 0
                return
            consec_fail = 0
            if raw is None or len(raw) == 0:
                stats["empty"] += 1
                return
            new_df = universe.normalize_daily(raw, code, BAOSTOCK["amount_scale"])
            if len(new_df) == 0:
                stats["empty"] += 1
                return
            merged = (pd.concat([new_df, old], ignore_index=True)
                      .drop_duplicates("date").sort_values("date")
                      .reset_index(drop=True))
            storage.save_daily(code, merged)
            storage.update_meta(meta, code, merged, name_map.get(code, ""))
            stats["backfilled"] += 1

        log.info(f"历史回填（起点 {start_date}，单线程，已覆盖秒级 skip）…")
        every = UNIVERSE["proactive_reconnect_every"]
        sess.set_timeout(UNIVERSE["data_timeout"])
        sess.reconnect(cooldown=UNIVERSE["reconnect_cooldown"])
        for i, code in enumerate(tqdm(targets, ncols=88, desc="历史回填")):
            process(code)
            if (i + 1) % 25 == 0:
                storage.save_meta(meta)
            if every and (i + 1) % every == 0:
                sess.reconnect(cooldown=1.0)
                consec_fail = 0

        for r in range(UNIVERSE["repair_rounds"]):
            if not failed:
                break
            log.warning(f"回填补下载第 {r+1} 轮，待补 {len(failed)} 只；先重连…")
            sess.reconnect(cooldown=UNIVERSE["reconnect_cooldown"])
            todo = list(failed)
            failed.clear()
            stats["failed"] = 0
            consec_fail = 0
            for j, code in enumerate(tqdm(todo, ncols=88, desc=f"回填补{r+1}")):
                process(code)
                if (j + 1) % 25 == 0:
                    storage.save_meta(meta)

        storage.save_meta(meta)
        stats["failed_remaining"] = sorted(set(failed))
        stats["meta_total"] = len(meta)
        log.info(
            "回填完成：补齐 {backfilled}，已覆盖跳过 {skipped}，空窗 {empty}，"
            "本地缺失 {missing}，仍失败 {fr}".format(
                backfilled=stats["backfilled"], skipped=stats["skipped"],
                empty=stats["empty"], missing=stats["missing"],
                fr=len(stats["failed_remaining"])))
        if stats["failed_remaining"]:
            log.warning("仍失败（不影响已回填数据，重跑续传）: "
                        + ",".join(stats["failed_remaining"][:50]))
        return stats
    finally:
        sess.logout()
        log.info("baostock 已登出")


def run_update(limit: Optional[int] = None, codes: Optional[List[str]] = None,
               list_only: bool = False) -> dict:
    """
    全量/增量更新。返回统计 dict。
    limit/codes: 调试用小样本。
    """
    # 1) 股票列表（在线，失败回退过期缓存；读入已强制主板非ST）
    stock_df, src = storage.get_stock_list(
        cache_days=UNIVERSE["stock_list_cache_days"],
        timeout=UNIVERSE["stock_list_timeout"])
    if stock_df is None:
        raise RuntimeError("无法获取股票列表，且本地无可用缓存，终止更新")

    # 2) 登录 baostock（全局复用）；瞬时连接重置(10054)自动退避重试，不一次失败就中止。
    #    日历/指数批量查询阶段用30秒，个股阶段恢复15秒
    sess = BaoStockSession(timeout=30)
    if not _login_with_retry(sess):
        raise RuntimeError(
            "baostock 多次登录仍失败（疑似网络抖动或服务临时限流，如 WinError 10054）。"
            "请过几分钟重跑 update.py：已下载个股与元数据不会丢失，重跑自动秒级 skip。")

    try:
        # 行业归属一次性获取并合并、回写列表缓存
        try:
            ind = sess.query_all_industry()
            stock_df = universe.merge_industry(stock_df, ind)
        except Exception as e:  # noqa
            log.warning(f"行业分类获取失败，所属板块将留空: {e}")
            stock_df["industry"] = stock_df.get("industry", "")
        storage.save_stock_list(stock_df)

        expected = latest_trading_day(sess)
        log.info(f"预期最新交易日 = {expected}（股票列表来源: {src}，主板非ST {len(stock_df)} 只）")

        # 指数为批量查询，放宽到30秒并带重试；个股阶段再恢复15秒快判
        socket.setdefaulttimeout(30)
        update_indices(sess, expected.strftime("%Y-%m-%d"))
        # baostock 的 socket 在 login 时即固定超时；个股阶段需用15s超时【重建新连接】，
        # 否则旧连接仍按30s干等，坏连接会把单股拖到数十秒。
        sess.set_timeout(UNIVERSE["data_timeout"])
        sess.reconnect(cooldown=UNIVERSE["reconnect_cooldown"])
        if list_only:
            return {"universe": len(stock_df), "source": src, "expected": str(expected)}

        if codes:
            targets = [c for c in codes if c in set(stock_df["code"])]
        else:
            targets = list(stock_df["code"])
        if limit:
            targets = targets[:limit]

        meta = storage.load_meta()
        name_map = dict(zip(stock_df["code"], stock_df["name"]))
        end_date = expected.strftime("%Y-%m-%d")

        failed: List[str] = []
        stats = dict(skipped=0, updated=0, empty=0, failed=0, new=0)
        consec_fail = 0

        def process(code: str, bar: bool = True):
            nonlocal consec_fail
            rec = meta.get(code, {})
            last = rec.get("last_date")
            start = _incremental_start(last)
            if last and last >= end_date:
                stats["skipped"] += 1
                return
            if not last:
                stats["new"] += 1
            raw, stt = _download_one(sess, code, start, end_date)
            if stt == "network":
                failed.append(code)
                stats["failed"] += 1
                consec_fail += 1
                # 连续失败，或单次查询异常缓慢（连接已老化/被限流），都立即重连
                slow = sess.last_latency >= UNIVERSE["slow_latency"]
                if consec_fail >= UNIVERSE["max_consecutive_fail"] or slow:
                    sess.reconnect(cooldown=UNIVERSE["reconnect_cooldown"])
                    consec_fail = 0
                return
            consec_fail = 0
            if raw is None or len(raw) == 0:
                # 空表=窗口内无数据（节假日/停牌/盘前未发布），不是网络失败
                # 工作日盘前空下载不更新元数据，盘后可重试
                stats["empty"] += 1
                return
            new_df = universe.normalize_daily(raw, code, BAOSTOCK["amount_scale"])
            if len(new_df) == 0:
                stats["empty"] += 1
                return
            old = storage.load_daily(code)
            if old is not None and len(old):
                new_df = (pd.concat([old, new_df], ignore_index=True)
                          .drop_duplicates("date").sort_values("date")
                          .reset_index(drop=True))
            storage.save_daily(code, new_df)
            storage.update_meta(meta, code, new_df, name_map.get(code, ""))
            stats["updated"] += 1

        # 快速路径：meta 全局最新日期已达预期 → 全部跳过，不逐只遍历
        # （个别停牌/空窗股 last_date 较早属正常，全局已最新说明当天无新数据可下）
        global_latest = max((rec.get("last_date", "") for rec in meta.values()), default="")
        if targets and global_latest >= end_date:
            stats["skipped"] = len(targets)
            log.info(f"meta 全局最新日期 = {global_latest}（预期 {end_date}），"
                     f"全部 {len(targets)} 只直接跳过，无需逐只遍历")
        else:
            log.info("第一轮下载（单线程，已有数据秒级 skip）…")
            every = UNIVERSE["proactive_reconnect_every"]
            for i, code in enumerate(tqdm(targets, ncols=88, desc="日线更新")):
                process(code)
                # 每25只落盘一次元数据：即使中途 Ctrl+C/崩溃，重跑也能秒级 skip 已下载部分
                if (i + 1) % 25 == 0:
                    storage.save_meta(meta)
                # 主动定期重连，避免长连接连续请求后被限流导致速度雪崩
                if every and (i + 1) % every == 0:
                    sess.reconnect(cooldown=1.0)
                    consec_fail = 0

            # 失败股票自动补下载 2 轮（重连重试）
            for r in range(UNIVERSE["repair_rounds"]):
                if not failed:
                    break
                log.warning(f"补下载第 {r+1} 轮，待补 {len(failed)} 只；先重连…")
                sess.reconnect(cooldown=UNIVERSE["reconnect_cooldown"])
                todo = list(failed)
                failed.clear()
                stats["failed"] = 0
                consec_fail = 0
                for j, code in enumerate(tqdm(todo, ncols=88, desc=f"补下载{r+1}")):
                    process(code)
                    if (j + 1) % 25 == 0:
                        storage.save_meta(meta)

        storage.save_meta(meta)
        stats["failed_remaining"] = sorted(set(failed))
        stats["meta_total"] = len(meta)
        stats["expected"] = str(expected)
        log.info(
            "更新完成：新增股 {new}，更新 {updated}，秒级跳过 {skipped}，"
            "空窗 {empty}，仍失败 {fr}".format(
                new=stats["new"], updated=stats["updated"],
                skipped=stats["skipped"], empty=stats["empty"],
                fr=len(stats["failed_remaining"])))
        if stats["failed_remaining"]:
            log.warning("仍失败股票（结尾汇总，不影响已下载数据）: "
                        + ",".join(stats["failed_remaining"][:50]))
        return stats
    finally:
        sess.logout()
        log.info("baostock 已登出")
