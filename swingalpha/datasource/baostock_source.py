# -*- coding: utf-8 -*-
"""
baostock 历史日线数据源（前复权）。
- 全局登录一次、下载完登出；长连接老化/限流时连续失败自动 logout/login 重连。
- 单股下载返回 (df_or_None, status)：
    status="ok"      正常拿到（可能为空 DataFrame，代表该窗口无数据/节假日）
    status="network" 网络/接口失败（None，调用方必须重试，绝不当作 skip）
- 访问限制守卫：
    ① 日请求计数（持久化 data/raw/_baostock_usage.json），官方上限5万次/日，
       达4.5万警告、达5万拒绝新查询（防黑名单）。
    ② 并发连接守卫：baostock 为全局单例会话，禁止多进程/多线程同时登录。
"""
from __future__ import annotations
import json
import os
import socket
import time
from datetime import date
from typing import Optional, Tuple

import pandas as pd

from ..utils import get_logger, code_to_bs
from ..config import BAOSTOCK
from ..storage import RAW_DIR

log = get_logger()

# 官方日请求上限
DAILY_LIMIT = 50000
SAFE_WARN = 45000   # 达此值警告
USAGE_FILE = RAW_DIR / "_baostock_usage.json"

# 并发连接守卫：类级别追踪已登录实例（baostock 全局单例，禁止并发）
_active_session = None


def _load_usage() -> dict:
    """读取今日请求计数；日期变更则重置。"""
    today = date.today().isoformat()
    try:
        if USAGE_FILE.exists():
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
    except Exception:
        pass
    return {"date": today, "count": 0}


def _save_usage(data: dict):
    try:
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _tick_request() -> int:
    """递增今日请求计数，返回当前计数；达上限时抛 RuntimeError。"""
    data = _load_usage()
    data["count"] = data.get("count", 0) + 1
    _save_usage(data)
    cnt = data["count"]
    if cnt >= DAILY_LIMIT:
        raise RuntimeError(
            f"baostock 今日请求已达 {cnt}/{DAILY_LIMIT} 上限，继续查询将被官方黑名单控制，"
            f"请明日再运行（已下载数据不受影响）")
    if cnt == SAFE_WARN:
        log.warning(f"baostock 今日请求已达 {cnt}/{DAILY_LIMIT}（90%），请注意控制用量")
    return cnt


class BaoStockSession:
    def __init__(self, timeout: int = 15):
        import baostock as bs  # 延迟导入
        self.bs = bs
        self.timeout = timeout
        self.logged_in = False
        self.login_count = 0
        self.last_latency = 0.0   # 最近一次查询耗时(秒)，供调用方判断连接是否老化

    def set_timeout(self, timeout: int):
        """调整 socket 超时；需在下次 login/reconnect 后才作用于新连接。"""
        self.timeout = timeout
        socket.setdefaulttimeout(timeout)

    def login(self) -> bool:
        global _active_session
        # 并发连接守卫：baostock 全局单例，禁止同时多个登录
        if _active_session is not None and _active_session is not self:
            log.error("检测到 baostock 并发连接！baostock 不支持多进程/多线程同时登录，"
                      "请确保单线程串行使用。当前已有活跃会话，拒绝重复登录。")
            return False
        socket.setdefaulttimeout(self.timeout)
        try:
            rs = self.bs.login()
            if rs.error_code != "0":
                log.error(f"baostock 登录失败: {rs.error_msg}")
                self.logged_in = False
                return False
            self.logged_in = True
            self.login_count += 1
            _active_session = self
            log.info(f"baostock 已登录（第 {self.login_count} 次）")
            return True
        except Exception as e:  # noqa
            log.error(f"baostock 登录异常: {e}")
            self.logged_in = False
            return False

    def logout(self):
        global _active_session
        try:
            self.bs.logout()
        except Exception:  # noqa
            pass
        self.logged_in = False
        if _active_session is self:
            _active_session = None

    def reconnect(self, cooldown: float = 2.0, retries: int = 2) -> bool:
        """logout 后重新登录；登录本身也可能遇到瞬时 10054，故带 retries 次重试。"""
        log.warning(f"连接老化/限流，冷却 {cooldown:.0f}s 后 baostock 重连…")
        self.logout()
        time.sleep(cooldown)
        for k in range(retries):
            if self.login():
                return True
            self.logout()
            if k < retries - 1:
                time.sleep(cooldown)
        log.error("baostock 重连登录失败，将在后续个股连续失败时再次触发重连")
        return False

    def query_daily(self, plain_code: str, start_date: str, end_date: str,
                    adjustflag: str = "2", fields: Optional[str] = None
                    ) -> Tuple[Optional[pd.DataFrame], str]:
        """下载单只股票（或指数）日线，返回 (df, status)。"""
        _tick_request()  # 日请求计数+上限守卫
        fields = fields or BAOSTOCK["fields"]
        bs_code = code_to_bs(plain_code)
        t0 = time.time()
        try:
            rs = self.bs.query_history_k_data_plus(
                bs_code, fields,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag=adjustflag,
            )
        except Exception as e:  # noqa
            self.last_latency = time.time() - t0
            return None, f"network:{e}"

        if rs is None or rs.error_code != "0":
            self.last_latency = time.time() - t0
            return None, f"network:{getattr(rs, 'error_msg', 'rs is None')}"

        rows = []
        try:
            while rs.next():
                rows.append(rs.get_row_data())
        except Exception as e:  # noqa
            # 含 UnicodeDecodeError(0xa9 错误页)、接收数据异常、timeout 等，一律按网络失败重试
            self.last_latency = time.time() - t0
            return None, f"network:{e}"
        self.last_latency = time.time() - t0

        if not rows:
            return pd.DataFrame(columns=fields.split(",")), "ok"
        return pd.DataFrame(rows, columns=fields.split(",")), "ok"

    def query_index_daily(self, index_code: str, start_date: str, end_date: str):
        """指数日线（不复权 adjustflag=3）。index_code 如 sh000852 / 000852。"""
        _tick_request()
        idx = index_code if index_code.startswith(("sh.", "sz.")) else \
            ("sh." + index_code.lstrip("shz"))
        fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
        try:
            rs = self.bs.query_history_k_data_plus(
                idx, fields, start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3",
            )
            if rs is None or rs.error_code != "0":
                return None, f"network:{getattr(rs,'error_msg','idx')}"
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame(columns=fields.split(",")), "ok"
            return pd.DataFrame(rows, columns=fields.split(",")), "ok"
        except Exception as e:  # noqa
            return None, f"network:{e}"

    def query_all_industry(self) -> pd.DataFrame:
        """一次性获取全市场行业分类（仅1次请求）。"""
        _tick_request()
        try:
            rs = self.bs.query_stock_industry()
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()
        except Exception as e:  # noqa
            log.warning(f"行业分类获取失败: {e}")
            return pd.DataFrame()

    def query_stock_industry(self, bs_code: str) -> Optional[pd.DataFrame]:
        """单只股票行业查询（带日请求计数）。bs_code 如 sh.600000。"""
        _tick_request()
        try:
            rs = self.bs.query_stock_industry(code=bs_code)
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()
        except Exception as e:  # noqa
            log.warning(f"单只行业查询失败 {bs_code}: {e}")
            return None
