"""六大情绪因子计算：基于个股日线 + 指数日线，输出市场-日 级因子表。

因子分组（与配置权重对应）：
  adv_decline   涨跌家数：up/down/flat
  distribution  涨跌幅分布：ge3..ge9 / le3..le10
  limit         涨停/跌停/炸板/封板率/连板/最高连板/首板/高位股
  volume        成交额：amount 及 5/10/20 日均值、量比
  breadth       市场宽度：站上 5/10/20/60 日线比例、创新高/新低
  index         指数因素：参考指数 5 日 / 20 日涨跌幅（加权）

所有统计仅使用当日及历史数据，无未来函数。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import Market

DIST_BINS_UP = (3, 5, 7, 9)
DIST_BINS_DOWN = (3, 5, 7, 9, 10)


def _stock_features(df: pd.DataFrame, limit_pct: float, tol: float,
                    hp_gain: float, hp_window: int, nh_window: int) -> pd.DataFrame:
    """单票特征：涨跌幅、涨跌停、炸板、连板、均线、新高新低、高位股。"""
    df = df.sort_values("date").reset_index(drop=True)
    close, high, low = df["close"], df["high"], df["low"]
    prev = close.shift(1)
    pct = (close / prev - 1.0) * 100.0
    intraday_up = (high / prev - 1.0) * 100.0

    limit_up = pct >= (limit_pct - tol)
    limit_dn = pct <= -(limit_pct - tol)
    broken = (intraday_up >= (limit_pct - tol)) & (~limit_up)

    # 连板计数：连续 limit_up 的天数
    grp = (~limit_up).cumsum()
    lianban = limit_up.groupby(grp).cumsum().where(limit_up, 0)

    out = pd.DataFrame({
        "date": df["date"], "pct": pct, "amount": df["amount"],
        "limit_up": limit_up, "limit_dn": limit_dn, "broken": broken,
        "lianban": lianban,
    })
    for n in (5, 10, 20, 60):
        ma = close.rolling(n).mean()
        out[f"above_ma{n}"] = (close > ma).where(ma.notna()).astype(float)
    rh = close.rolling(nh_window).max().shift(1)
    rl = close.rolling(nh_window).min().shift(1)
    out["new_high"] = (close > rh).where(rh.notna()).astype(float)
    out["new_low"] = (close < rl).where(rl.notna()).astype(float)
    hp_high = close.rolling(hp_window).max()
    hp_low = low.rolling(hp_window).min()
    out["high_pos"] = ((close >= hp_high) & (close / hp_low - 1.0 >= hp_gain)) \
        .where(hp_low.notna()).astype(float)
    return out


def compute_factors(provider, cfg: dict, market: str | Market,
                    codes_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """计算某市场全部历史每日因子，缓存到 parquet 并返回。"""
    market = Market(market)
    from ..data.universe import build_universe
    if codes_df is None:
        stock_list = provider.get_stock_list()
        codes_df = build_universe(stock_list, market, cfg)
    tol = float(cfg.get("limit_tolerance", 0.3))
    hp_gain = float(cfg["sentiment"].get("high_position_gain", 0.5))
    hp_window = int(cfg["sentiment"].get("high_position_window", 60))
    nh_window = int(cfg["sentiment"].get("new_highlow_window", 250))
    limit_map = dict(zip(codes_df["code"], codes_df["limit_pct"]))

    frames = []
    for code in codes_df["code"]:
        df = provider.load_stock_daily(code)
        if df is None or df.empty:
            continue
        feat = _stock_features(df, limit_map.get(code, 10.0), tol,
                               hp_gain, hp_window, nh_window)
        frames.append(feat)
    if not frames:
        raise RuntimeError(f"市场 {market.value} 无可用个股日线缓存，请先执行数据更新")
    big = pd.concat(frames, ignore_index=True)

    # ---------- 按交易日聚合 ----------
    def agg(g: pd.DataFrame) -> pd.Series:
        traded = g["pct"].notna().sum()
        pct = g["pct"]
        s = {
            "traded": traded,
            "up": (pct > 0).sum(),
            "down": (pct < 0).sum(),
            "flat": (pct == 0).sum(),
        }
        for b in DIST_BINS_UP:
            s[f"ge{b}"] = (pct >= b).sum()
        for b in DIST_BINS_DOWN:
            s[f"le{b}"] = (pct <= -b).sum()
        lu, ld, bk = g["limit_up"].sum(), g["limit_dn"].sum(), g["broken"].sum()
        s["limit_up"], s["limit_dn"], s["broken"] = lu, ld, bk
        s["seal_rate"] = lu / (lu + bk) if (lu + bk) > 0 else np.nan
        s["first_board"] = (g["lianban"] == 1).sum()
        s["lianban2p"] = (g["lianban"] >= 2).sum()
        s["max_lianban"] = int(g["lianban"].max()) if traded else 0
        s["high_pos"] = g["high_pos"].sum(skipna=True)
        s["amount"] = g["amount"].sum()
        for n in (5, 10, 20, 60):
            col = g[f"above_ma{n}"]
            s[f"above_ma{n}"] = col.sum(skipna=True) / col.notna().sum() if col.notna().any() else np.nan
        s["new_high"] = g["new_high"].sum(skipna=True)
        s["new_low"] = g["new_low"].sum(skipna=True)
        return pd.Series(s)

    factors = big.groupby("date").apply(agg, include_groups=False).sort_index()

    # ---------- 成交额衍生 ----------
    amt = factors["amount"]
    factors["amount_chg"] = amt.pct_change()
    for n in (5, 10, 20):
        factors[f"amount_ma{n}"] = amt.rolling(n).mean()
    factors["amount_ratio"] = amt / factors["amount_ma20"]

    # ---------- 指数因子 ----------
    refs = cfg["markets"][market.value]["index_refs"]
    idx_ret5, idx_ret20, wsum = 0.0, 0.0, 0.0
    for symbol, w in refs.items():
        idx = provider.load_index_daily(symbol).set_index("date").sort_index()["close"]
        idx = idx.reindex(factors.index).ffill()
        idx_ret5 = idx_ret5 + float(w) * (idx / idx.shift(5) - 1.0)
        idx_ret20 = idx_ret20 + float(w) * (idx / idx.shift(20) - 1.0)
        wsum += float(w)
    factors["idx_ret5"] = idx_ret5 / wsum
    factors["idx_ret20"] = idx_ret20 / wsum

    path = Path(cfg["data"]["factor_dir"]) / f"{market.value}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    factors.reset_index().to_parquet(path)
    return factors


def load_factors(cfg: dict, market: str | Market) -> pd.DataFrame:
    market = Market(market)
    path = Path(cfg["data"]["factor_dir"]) / f"{market.value}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"因子表不存在: {path}，请先执行 update")
    df = pd.read_parquet(path)
    return df.set_index("date").sort_index()
