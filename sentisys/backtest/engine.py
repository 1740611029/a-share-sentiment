"""历史回测：极端情绪信号 vs 阶段顶部/底部。

阶段底部支持两种明确的数学定义（pivot_mode 配置）：

zone（默认，价格位置 + 确认反弹）——交易日 t 同时满足：
  1. 价格位置：t 收盘价 <= 此前 pivot_left 个交易日最低收盘价 × (1 + pos_tol)；
  2. 确认反弹：t 之后 confirm_window 个交易日内，
     最大收盘价相对 t 日涨幅 >= rebound_pct。

strict（精确局部极值）——交易日 t 同时满足：
  1. t 日收盘价是 [t-pivot_left, t+confirm_window] 窗口内最低收盘价；
  2. 同 zone 条件 2。

阶段顶部为镜像定义。顶/底判定是事后验证标签，不参与信号生成，无未来函数泄漏。

信号判定：情绪 <= 极端恐慌阈值（恐慌信号，预期底部）/
          情绪 >= 极端过热阈值（过热信号，预期顶部）。
若 T+0 / T+1 / T+2 任一交易日形成对应顶/底 → 成功信号。
信号去重：同一方向信号冷却期 signal_cooldown 个交易日内只保留首个。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import SIGNAL_HOT, SIGNAL_PANIC


def find_pivots(close: pd.Series, kind: str, pivot_left: int,
                confirm_window: int, rebound_pct: float,
                mode: str = "zone", pos_tol: float = 0.02) -> pd.DatetimeIndex:
    """找出全部阶段顶/底交易日。label 性质，允许使用确认窗口内的后续价格。"""
    arr = close.to_numpy(dtype=float)
    n = len(arr)
    hits = []
    for i in range(n):
        hi = min(n, i + confirm_window + 1)
        fwd = arr[i + 1:hi]
        if len(fwd) == 0 or np.isnan(arr[i]):
            continue
        if kind == "bottom":
            if mode == "strict":
                lo = max(0, i - pivot_left)
                in_position = arr[i] == np.nanmin(arr[lo:hi])
            else:
                lo = max(0, i - pivot_left)
                prev = arr[lo:i]
                in_position = len(prev) > 0 and arr[i] <= np.nanmin(prev) * (1.0 + pos_tol)
            if in_position and (np.nanmax(fwd) / arr[i] - 1.0) >= rebound_pct:
                hits.append(close.index[i])
        else:
            if mode == "strict":
                lo = max(0, i - pivot_left)
                in_position = arr[i] == np.nanmax(arr[lo:hi])
            else:
                lo = max(0, i - pivot_left)
                prev = arr[lo:i]
                in_position = len(prev) > 0 and arr[i] >= np.nanmax(prev) * (1.0 - pos_tol)
            if in_position and (np.nanmin(fwd) / arr[i] - 1.0) <= -rebound_pct:
                hits.append(close.index[i])
    return pd.DatetimeIndex(hits)


def _signal_days(score: pd.Series, signal_type: str, panic_t: float,
                 hot_t: float, cooldown: int) -> list:
    if signal_type == SIGNAL_PANIC:
        mask = score <= panic_t
    else:
        mask = score >= hot_t
    days = list(score.index[mask.fillna(False)])
    picked, last = [], -10 ** 9
    positions = {d: i for i, d in enumerate(score.index)}
    for d in days:
        if positions[d] - last > cooldown:
            picked.append(d)
            last = positions[d]
    return picked


def run_backtest(score: pd.Series, close: pd.Series, params: dict,
                 start: str | None = None, end: str | None = None) -> dict:
    """对单一市场、单组参数执行回测。

    params: extreme_panic, extreme_hot, rebound_pct, pivot_left,
            confirm_window, signal_cooldown, observe_lags
    返回 {"signals": [...], "stats": {...}}（stats 由 summarize 汇总）。
    """
    close = close.reindex(score.index).ffill()
    lags = [int(x) for x in params.get("observe_lags", (0, 1, 2))]
    pivots = params.get("_pivots")
    if pivots is None:
        mode = params.get("pivot_mode", "zone")
        pos_tol = float(params.get("pos_tol", 0.02))
        pivots = {
            SIGNAL_PANIC: find_pivots(close, "bottom", int(params["pivot_left"]),
                                      int(params["confirm_window"]), float(params["rebound_pct"]),
                                      mode, pos_tol),
            SIGNAL_HOT: find_pivots(close, "top", int(params["pivot_left"]),
                                    int(params["confirm_window"]), float(params["rebound_pct"]),
                                    mode, pos_tol),
        }
    n = len(score)
    cw = int(params["confirm_window"])
    signals: list[dict] = []
    for stype in (SIGNAL_PANIC, SIGNAL_HOT):
        days = _signal_days(score, stype, float(params["extreme_panic"]),
                            float(params["extreme_hot"]), int(params["signal_cooldown"]))
        for d in days:
            if start and d < pd.Timestamp(start):
                continue
            if end and d > pd.Timestamp(end):
                continue
            i = score.index.get_loc(d)
            rec = {"signal_type": stype, "signal_date": d.strftime("%Y-%m-%d"),
                   "sentiment": round(float(score.loc[d]), 2),
                   "pivot_date": None, "lag": None,
                   "hit_t0": 0, "hit_t1": 0, "hit_t2": 0, "success": None}
            fwd_close = close.iloc[i:min(n, i + cw + 1)]
            if len(fwd_close) > 1:
                if stype == SIGNAL_PANIC:
                    rec["max_rebound"] = round(float(fwd_close.max() / fwd_close.iloc[0] - 1.0), 4)
                    rec["max_drawdown"] = round(float(fwd_close.min() / fwd_close.iloc[0] - 1.0), 4)
                else:
                    rec["max_rebound"] = round(float(fwd_close.min() / fwd_close.iloc[0] - 1.0), 4)
                    rec["max_drawdown"] = round(float(fwd_close.max() / fwd_close.iloc[0] - 1.0), 4)
            # 数据末端：确认窗口不足 → 未确认，不计入命中率
            if i + cw >= n:
                signals.append(rec)
                continue
            rec["success"] = 0
            for k in lags:
                j = i + k
                if j < n and score.index[j] in pivots[stype]:
                    rec[f"hit_t{k}"] = 1
                    if rec["success"] == 0:
                        rec["success"] = 1
                        rec["lag"] = k
                        rec["pivot_date"] = score.index[j].strftime("%Y-%m-%d")
            signals.append(rec)
    return {"signals": signals, "stats": summarize(signals, len(score))}


def summarize(signals: list[dict], trading_days: int, min_samples: int = 5) -> dict:
    """汇总统计：T+0/1/2 命中率、累计命中率、覆盖率、最大连续失败。"""
    def block(stype: str) -> dict:
        arr = [s for s in signals if s["signal_type"] == stype]
        confirmed = [s for s in arr if s.get("success") is not None]
        total, succ = len(confirmed), sum(1 for s in confirmed if s["success"] == 1)
        fails = total - succ
        t_rates = {}
        for k in (0, 1, 2):
            t_rates[f"t{k}_hits"] = sum(1 for s in confirmed if s.get(f"hit_t{k}") == 1)
            t_rates[f"t{k}_rate"] = round(t_rates[f"t{k}_hits"] / total, 4) if total else None
        max_streak, streak = 0, 0
        for s in confirmed:
            if s["success"] == 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return {
            "signals": total, "unconfirmed": len(arr) - total,
            "success": succ, "fail": fails,
            "hit_rate": round(succ / total, 4) if total else None,
            **t_rates,
            "coverage": round(total / trading_days, 4) if trading_days else None,
            "max_consec_fail": max_streak,
            "sample_enough": total >= min_samples,
            "note": None if total >= min_samples else "样本不足，暂不能验证90%命中率",
        }

    return {"panic": block(SIGNAL_PANIC), "hot": block(SIGNAL_HOT),
            "trading_days": trading_days}
