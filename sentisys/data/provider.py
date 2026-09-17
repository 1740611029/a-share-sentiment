"""数据源封装：akshare（新浪/腾讯），带本地 parquet 缓存。

说明：东方财富接口在部分网络环境不可用，因此全部行情数据
（个股日线、指数日线）走新浪源；涨跌停/炸板/连板等指标
由日线数据按各板块涨跌停规则数学推导，不依赖第三方涨停池。
"""
from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd

from ..constants import INDEX_NAMES


class DataProvider:
    def __init__(self, cfg: dict):
        self.cache_dir = Path(cfg["data"]["cache_dir"])
        self.stock_dir = self.cache_dir / "stocks"
        self.index_dir = self.cache_dir / "index"
        for d in (self.stock_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.retry = int(cfg["data"].get("retry", 3))

    # ---------------- 股票列表 ----------------
    def get_stock_list(self, refresh: bool = False) -> pd.DataFrame:
        """返回 [code, name, list_date]。缓存 1 天。"""
        cache = self.cache_dir / "stock_list.parquet"
        if cache.exists() and not refresh:
            if time.time() - cache.stat().st_mtime < 86400:
                return pd.read_parquet(cache)
        frames = []
        for symbol in ("主板A股", "科创板"):  # 该接口默认仅主板，科创板需单独拉取
            sh = ak.stock_info_sh_name_code(symbol=symbol)
            sh = sh.rename(columns={"证券代码": "code", "证券简称": "name", "上市日期": "list_date"})
            frames.append(sh[["code", "name", "list_date"]])
        sz = ak.stock_info_sz_name_code()
        sz = sz.rename(columns={"A股代码": "code", "A股简称": "name", "A股上市日期": "list_date"})
        frames.append(sz[["code", "name", "list_date"]])
        df = pd.concat(frames, ignore_index=True)
        df["code"] = df["code"].astype(str).str.zfill(6)
        # 上交所格式 19991110，深交所格式 1991-11-10，混合解析
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce", format="mixed")
        df = df.dropna(subset=["list_date"]).drop_duplicates("code")
        df.to_parquet(cache)
        return df

    # ---------------- 个股日线（前复权） ----------------
    @staticmethod
    def sina_symbol(code: str) -> str | None:
        if code.startswith(("60", "68", "11", "51", "58")):
            return "sh" + code
        if code.startswith(("00", "30", "12", "15", "16")):
            return "sz" + code
        return None

    def stock_cache_path(self, code: str) -> Path:
        return self.stock_dir / f"{code}.parquet"

    def load_stock_daily(self, code: str) -> pd.DataFrame | None:
        p = self.stock_cache_path(code)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                return None
        return None

    def fetch_stock_daily(self, code: str, start: str, end: str,
                          source: str = "auto") -> pd.DataFrame | None:
        """拉取前复权日线：腾讯源（纯HTTP，线程安全）优先，新浪源兜底。

        注意：新浪源依赖 py_mini_racer，非线程安全，只能在单线程下使用。
        source: auto / tx / sina
        """
        symbol = self.sina_symbol(code)
        if symbol is None:
            return None
        sources = ("tx", "sina") if source == "auto" else (source,)
        for source in sources:
            for _ in range(self.retry):
                try:
                    if source == "tx":
                        df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start,
                                                   end_date=end, adjust="qfq")
                    else:
                        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start,
                                                 end_date=end, adjust="qfq")
                    if df is None or df.empty:
                        break
                    df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                    df["date"] = pd.to_datetime(df["date"])
                    for c in ("open", "high", "low", "close", "volume", "amount"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                except Exception:  # noqa: BLE001
                    time.sleep(0.5)
        return None

    # ---------------- 指数日线 ----------------
    def load_index_daily(self, symbol: str, refresh: bool = False,
                         start: str = "20150101") -> pd.DataFrame:
        p = self.index_dir / f"{symbol}.parquet"
        cached = None
        if p.exists() and not refresh:
            cached = pd.read_parquet(p)
        need_start = start
        if cached is not None and not cached.empty:
            last = cached["date"].max()
            if (pd.Timestamp.now().normalize() - last).days <= 3:
                return cached
            need_start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        last_err = None
        new = None
        for _ in range(self.retry):
            try:
                new = ak.stock_zh_index_daily(symbol=symbol)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.5)
        if new is None:
            if cached is not None:
                return cached
            raise RuntimeError(f"指数 {symbol}({INDEX_NAMES.get(symbol, '')}) 拉取失败: {last_err}")
        new["date"] = pd.to_datetime(new["date"])
        new = new[["date", "open", "high", "low", "close", "volume"]]
        if cached is not None and not cached.empty:
            new = pd.concat([cached, new]).drop_duplicates("date").sort_values("date")
        new = new[new["date"] >= pd.to_datetime(need_start if cached is None else "19000101")]
        new = new.reset_index(drop=True)
        new.to_parquet(p)
        return new
