"""情绪分数引擎：因子 → 0~100 子分数 → 加权总分 → 状态/分位/变化。

子分数 = 因子原始值在过去 factor_rank_window 个交易日中的分位 × 100。
反向指标（跌停数、炸板数等）对负值取分位，保证方向一致：
分数越高情绪越热，越低越恐慌。只使用当日及历史数据，无未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHT_KEYS = ["adv_decline", "distribution", "limit", "volume", "breadth", "index"]


def rolling_rank(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """当前值在过去 window 个交易日（含当日）中的分位，0~100。"""
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        w = arr[lo:i + 1]
        w = w[~np.isnan(w)]
        if len(w) < min_periods or np.isnan(arr[i]):
            continue
        out[i] = (w[:-1] <= w[-1]).mean() * 100.0 if len(w) > 1 else 50.0
    return pd.Series(out, index=s.index)


def _rank(s: pd.Series, window: int, min_periods: int, positive: bool = True) -> pd.Series:
    return rolling_rank(s if positive else -s, window, min_periods)


def compute_subscores(factors: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """因子表 → 六个因子组的 0~100 子分数。"""
    w = int(cfg["sentiment"]["factor_rank_window"])
    mp = int(cfg["sentiment"]["factor_rank_min_periods"])
    f = factors
    traded = f["traded"].replace(0, np.nan)

    sub = pd.DataFrame(index=f.index)
    # 1. 涨跌家数：上涨占比
    sub["adv_decline"] = _rank(f["up"] / traded, w, mp)
    # 2. 涨跌幅分布：大涨-大跌 加权净值
    dist_raw = ((f["ge3"] - f["le3"]) + 0.7 * (f["ge5"] - f["le5"])
                + 0.5 * (f["ge7"] - f["le7"])) / traded
    sub["distribution"] = _rank(dist_raw, w, mp)
    # 3. 涨停/跌停：涨停、封板率、连板高度、高位股（正）；跌停、炸板（反）
    parts = [
        _rank(f["limit_up"], w, mp),
        _rank(f["seal_rate"].fillna(0), w, mp),
        _rank(f["max_lianban"].astype(float), w, mp),
        _rank(f["high_pos"], w, mp),
        _rank(f["limit_dn"], w, mp, positive=False),
        _rank(f["broken"], w, mp, positive=False),
    ]
    sub["limit"] = pd.concat(parts, axis=1).mean(axis=1)
    # 4. 成交量/成交额：量比（成交额/20日均额）
    sub["volume"] = _rank(f["amount_ratio"], w, mp)
    # 5. 市场宽度：20/60日线上方比例 + 新高新低差
    breadth_parts = [
        _rank(f["above_ma20"], w, mp),
        _rank(f["above_ma60"], w, mp),
        _rank((f["new_high"] - f["new_low"]) / traded, w, mp),
    ]
    sub["breadth"] = pd.concat(breadth_parts, axis=1).mean(axis=1)
    # 6. 指数因素：加权指数 5/20 日涨跌幅
    idx_parts = [_rank(f["idx_ret5"], w, mp), _rank(f["idx_ret20"], w, mp)]
    sub["index"] = pd.concat(idx_parts, axis=1).mean(axis=1)
    return sub


def compute_score(subscores: pd.DataFrame, weights: dict, cfg: dict | None = None) -> pd.Series:
    """子分数 × 权重 → 0~100 情绪总分。

    加权合成值再做一次自身滚动分位映射（factor_rank_window），
    使最终分数在分布上近似均匀：score<=10 即过去一年中最差的 10% 交易日，
    保证极端阈值具有明确的极端语义。任一子分数缺失则当日不评分。
    """
    total, wsum = 0.0, 0.0
    for k in DEFAULT_WEIGHT_KEYS:
        w = float(weights.get(k, 0.0))
        total = total + subscores[k] * w
        wsum += w
    composite = (total / wsum if wsum > 0 else total).clip(0, 100)
    if cfg is None:
        return composite
    w = int(cfg["sentiment"]["factor_rank_window"])
    mp = int(cfg["sentiment"]["factor_rank_min_periods"])
    return rolling_rank(composite, w, mp)


def level_of(score: float, levels: list) -> str:
    """按配置的状态分级阈值返回状态名。"""
    name = levels[-1][1]
    for bound, label in levels:
        if score >= bound:
            name = label
        else:
            break
    return name


def add_score_metrics(score: pd.Series, cfg: dict) -> pd.DataFrame:
    """总分 → 变化量 / 历史分位 / 状态。"""
    out = pd.DataFrame({"score": score})
    out["daily_change"] = score.diff(1)
    out["change_3d"] = score.diff(3)
    out["change_5d"] = score.diff(5)
    out["change_10d"] = score.diff(10)
    mp = int(cfg["sentiment"]["factor_rank_min_periods"])
    for n in cfg["sentiment"]["percentile_windows"]:
        out[f"percentile{n}"] = rolling_rank(score, int(n), min(mp, int(n) // 2))
    levels = cfg["sentiment"]["levels"]
    out["level"] = score.map(lambda x: level_of(x, levels) if pd.notna(x) else None)
    return out
