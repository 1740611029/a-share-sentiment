"""业务流水线：数据更新 → 因子 → 情绪分数 → 快照落库"""
from __future__ import annotations

import json

import pandas as pd

from .constants import Market, STATUS_ABNORMAL, STATUS_OK
from .data import DataProvider, build_universe, update_stock_history
from .engine import add_trend_and_divergence, compute_score, compute_subscores
from .engine.score import add_score_metrics
from .factors import compute_factors


def update_market_data(cfg: dict, provider: DataProvider, market: str,
                       start: str | None = None, limit: int | None = None,
                       refresh_list: bool = False) -> dict:
    """增量更新某市场全部股票日线。返回更新统计。"""
    market = Market(market)
    stock_list = provider.get_stock_list(refresh=refresh_list)
    universe = build_universe(stock_list, market, cfg)
    codes = list(universe["code"])
    if limit:
        codes = codes[:limit]
    if start is None:
        years = int(cfg["data"]["history_years"])
        start = (pd.Timestamp.now() - pd.Timedelta(days=years * 365)).strftime("%Y%m%d")
    end = pd.Timestamp.now().strftime("%Y%m%d")
    workers = int(cfg["data"]["workers"])
    print(f"[{market.value}] 股票池 {len(codes)} 只，更新日线 {start}~{end}")
    return update_stock_history(provider, codes, start, end, workers=workers)


def compute_and_store(cfg: dict, db, provider: DataProvider, market: str) -> dict:
    """因子计算 → 情绪分数 → 快照写入。返回最新一行摘要。"""
    market = Market(market)
    factors = compute_factors(provider, cfg, market)

    # 数据异常检测：当日可统计数 < 历史中位数 × min_pool_ratio
    ratio = float(cfg["filters"]["min_pool_ratio"])
    median_traded = factors["traded"].median()
    abnormal = factors["traded"] < median_traded * ratio

    subscores = compute_subscores(factors, cfg)

    # 参数：优先使用激活模型版本，否则用配置默认值
    ver = db.latest_version(market.value)
    if ver:
        params = json.loads(ver["params_json"])
        weights = params.get("weights", cfg["sentiment"]["weights"])
        panic_t = float(params.get("extreme_panic", cfg["sentiment"]["extreme_panic"]))
        hot_t = float(params.get("extreme_hot", cfg["sentiment"]["extreme_hot"]))
        calc_version = f"{market.value}-{ver['version']}"
    else:
        weights = cfg["sentiment"]["weights"]
        panic_t = float(cfg["sentiment"]["extreme_panic"])
        hot_t = float(cfg["sentiment"]["extreme_hot"])
        calc_version = f"{market.value}-V1-default"

    score = compute_score(subscores, weights, cfg)
    metrics = add_score_metrics(score, cfg)
    primary = cfg["markets"][market.value]["primary_index"]
    idx_close = provider.load_index_daily(primary).set_index("date")["close"]
    metrics = add_trend_and_divergence(metrics, idx_close, cfg)
    metrics["is_extreme_panic"] = metrics["score"] <= panic_t
    metrics["is_extreme_hot"] = metrics["score"] >= hot_t
    # 参考指数当日涨跌幅（相对前一日收盘）
    idx_ret_1d = idx_close.reindex(metrics.index).ffill().pct_change() * 100

    for dt, row in metrics.iterrows():
        if pd.isna(row["score"]):
            continue
        db.save_snapshot({
            "date": dt.strftime("%Y-%m-%d"),
            "market_type": market.value,
            "sentiment_score": round(float(row["score"]), 2),
            "sentiment_level": row["level"],
            "daily_change": _r(row["daily_change"]),
            "change_3d": _r(row["change_3d"]),
            "change_5d": _r(row["change_5d"]),
            "change_10d": _r(row["change_10d"]),
            "percentile20": _r(row.get("percentile20")),
            "percentile60": _r(row.get("percentile60")),
            "percentile120": _r(row.get("percentile120")),
            "percentile250": _r(row.get("percentile250")),
            "is_extreme_panic": int(bool(row["is_extreme_panic"])),
            "is_extreme_hot": int(bool(row["is_extreme_hot"])),
            "trend": row["trend"],
            "pattern": row["pattern"],
            "divergence": row["divergence"],
            "data_status": STATUS_ABNORMAL if bool(abnormal.get(dt, False)) else STATUS_OK,
            "calculation_version": calc_version,
            "idx_ret_1d": None if pd.isna(idx_ret_1d.get(dt)) else round(float(idx_ret_1d[dt]), 2),
        })
    last = metrics.dropna(subset=["score"]).iloc[-1]
    return {
        "market": market.value,
        "date": metrics.dropna(subset=["score"]).index[-1].strftime("%Y-%m-%d"),
        "score": round(float(last["score"]), 2),
        "level": last["level"],
        "version": calc_version,
    }


def _r(x):
    return None if x is None or pd.isna(x) else round(float(x), 2)
