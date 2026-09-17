"""情绪趋势与指数-情绪背离识别"""
from __future__ import annotations

import pandas as pd


def add_trend_and_divergence(df: pd.DataFrame, index_close: pd.Series, cfg: dict) -> pd.DataFrame:
    """输入 add_score_metrics 的结果，追加：
    trend           ↑ 情绪回升 / → 情绪平稳 / ↓ 情绪下降
    pattern         恐慌修复 / 高位降温 / None
    divergence      底背离(指数下跌情绪改善) / 顶背离(指数上涨情绪恶化) / None
    idx_ret5        主参考指数 5 日涨跌幅
    """
    band = float(cfg["sentiment"]["trend_flat_band"])
    lookback = int(cfg["sentiment"]["repair_lookback"])
    out = df.copy()

    chg = out["daily_change"]
    out["trend"] = "→"
    out.loc[chg > band, "trend"] = "↑"
    out.loc[chg < -band, "trend"] = "↓"

    # 恐慌修复：连续 3 日上升，且 lookback 日内曾处于恐慌区(<25)
    up3 = (chg > 0) & (chg.shift(1) > 0) & (chg.shift(2) > 0)
    was_cold = out["score"].shift(2).rolling(lookback).min() < 25
    # 高位降温：连续 3 日下降，且 lookback 日内曾处于过热区(>75)
    dn3 = (chg < 0) & (chg.shift(1) < 0) & (chg.shift(2) < 0)
    was_hot = out["score"].shift(2).rolling(lookback).max() > 75
    out["pattern"] = None
    out.loc[up3 & was_cold, "pattern"] = "恐慌修复"
    out.loc[dn3 & was_hot, "pattern"] = "高位降温"

    idx = index_close.reindex(out.index).ffill()
    ret5 = idx / idx.shift(5) - 1.0
    out["idx_ret5"] = ret5
    chg5 = out["score"] - out["score"].shift(5)
    out["divergence"] = None
    out.loc[(ret5 < -0.01) & (chg5 > 3), "divergence"] = "底背离"
    out.loc[(ret5 > 0.01) & (chg5 < -3), "divergence"] = "顶背离"
    return out
