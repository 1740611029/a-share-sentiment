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

import math
from math import comb

import numpy as np
import pandas as pd

from ..constants import SIGNAL_HOT, SIGNAL_PANIC


# ============================================================
# 统计显著性：命中率必须和"瞎猜"比，而且要看样本够不够下结论
# 纯 Python 实现，不引入 scipy 依赖。
# ============================================================
def _binom_sf(k: int, n: int, p: float) -> float | None:
    """P(X >= k)，X ~ Binomial(n, p)。用于检验"命中数是否显著多于瞎猜"。"""
    if n <= 0 or p is None:
        return None
    if p <= 0:
        return 0.0 if k > 0 else 1.0
    if p >= 1:
        return 1.0 if k <= n else 0.0
    k = max(0, min(k, n))
    return float(sum(comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k, n + 1)))


def _wilson_ci(succ: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """命中率的 Wilson 95% 置信区间。小样本下比正态近似可信得多。"""
    if n <= 0:
        return None
    ph = succ / n
    d = 1.0 + z * z / n
    center = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def _random_baseline(pivot_idx: pd.DatetimeIndex, index: pd.DatetimeIndex,
                     lags: list[int]) -> float | None:
    """瞎猜基线：随机挑一天，往后 lags 天内撞上该方向标签的概率。

    这是命中率的"零点"。不看它，命中率再高也没有意义。
    """
    n = len(index)
    if n == 0 or len(pivot_idx) == 0:
        return None
    mask = index.isin(pivot_idx)
    max_lag = max(lags)
    usable = n - max_lag
    if usable <= 0:
        return None
    hits = sum(1 for i in range(usable)
               if any(mask[i + k] for k in lags if i + k < n))
    return round(hits / usable, 4)


def _perm_p(observed: list[float], pool: list[float],
            trials: int = 20000, seed: int = 0) -> float | None:
    """置换检验：随机抽同样多的天数，其平均收益 >= 信号组平均收益的概率。

    比二项检验功效高得多 —— 用的是收益的幅度，而不是"中/没中"这一比特信息。
    p 越小，说明信号之后的走势越不像随机。
    """
    n = len(observed)
    if n == 0 or not pool:
        return None
    obs_mean = float(np.mean(observed))
    rng = np.random.default_rng(seed)
    pool_arr = np.asarray(pool, dtype=float)
    draws = rng.choice(pool_arr, size=(trials, n), replace=True)
    means = draws.mean(axis=1)
    return float((means >= obs_mean).mean())


def _fwd_block(sig_g: dict, all_g: dict, min_samples: int) -> dict | None:
    """前瞻收益汇总：信号后平均走了多少 vs 随机挑一天，并做置换检验。"""
    if not sig_g or not all_g:
        return None
    rows = {}
    for k in sorted(sig_g):
        obs, pool = sig_g[k], all_g.get(k) or []
        if not obs or not pool:
            rows[k] = None
            continue
        mean_obs = float(np.mean(obs))
        mean_base = float(np.mean(pool))
        p = _perm_p(obs, pool)
        n = len(obs)
        # _perm_p 是单边的（P(随机 >= 观测)）。要同时抓"明显差"这一侧，
        # 需转成双尾：p2 = 2 * min(p, 1-p)。只看单边会漏掉信号反向的情况。
        p2 = None if p is None else min(1.0, 2.0 * min(p, 1.0 - p))
        if n < min_samples:
            label = "样本太少"
        elif p2 is not None and p2 < 0.05 and mean_obs > mean_base:
            label = "明显好于平时"
        elif p2 is not None and p2 < 0.05 and mean_obs < mean_base:
            label = "明显差于平时"
        else:
            label = "与平时无差别"
        rows[k] = {
            "n": n,
            "mean": round(mean_obs, 4),
            "base": round(mean_base, 4),
            "diff": round(mean_obs - mean_base, 4),
            "p": round(p, 4) if p is not None else None,
            "p2": round(p2, 4) if p2 is not None else None,
            "label": label,
        }
    return rows


def _verdict(total: int, succ: int, baseline, p_value, min_samples: int) -> tuple[str, str]:
    """给这个方向的成绩下一个结论，并给一句人话说明。

    统一用双尾检验：我们并没有先验理由假定信号一定正向 —— 命中率显著低于基线
    同样是重要信息（说明信号是反向的），单边检验会漏掉这种情况。
    """
    if total < min_samples:
        return ("样本太少，无法判断",
                f"只有 {total} 次信号，最少要 {min_samples} 次才谈得上统计意义。这个百分比先别当真。")
    if baseline is None or p_value is None:
        return ("无法判断", "缺少对照基线，算不出是否有优势。")
    lift = (succ / total) - baseline
    p2 = min(1.0, 2.0 * min(p_value, 1.0 - p_value))
    if p2 < 0.05 and lift > 0:
        return ("优于瞎猜，可用",
                f"真实优势 {lift * 100:+.1f} 个百分点，统计显著（双尾 p={p2:.3f}）。")
    if p2 < 0.05 and lift <= 0:
        return ("明显不如瞎猜，别用",
                f"真实优势 {lift * 100:+.1f} 个百分点，显著为负（双尾 p={p2:.3f}）。这个信号是反的。")
    return ("与瞎猜差不多，还看不出优势",
            f"真实优势 {lift * 100:+.1f} 个百分点，但双尾 p={p2:.3f} 不够显著 —— "
            f"目前的数据区分不了「有效」和「运气好」。样本再多一些再看。")


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

    # ---- 前瞻收益（效果量）----
    # 二元的"命中/未命中"在样本少时统计功效极差：n=11 时几乎什么都测不出来。
    # 补一组连续指标：信号之后 1/3/5/10 个交易日市场平均走了多少，
    # 再和"随机挑一天"的分布做置换检验。同样的样本，结论可靠得多。
    horizons = [1, 3, 5, 10]
    arr = close.to_numpy(dtype=float)

    def _all_gains(sign: float) -> dict[int, list[float]]:
        """全样本的前瞻收益分布。sign=1 看涨方向（见底），sign=-1 看跌方向（见顶）。"""
        out: dict[int, list[float]] = {k: [] for k in horizons}
        for k in horizons:
            for i in range(n - k):
                a, b = arr[i], arr[i + k]
                if np.isnan(a) or np.isnan(b) or a == 0:
                    continue
                out[k].append(sign * (b / a - 1.0))
        return out

    all_gains = {SIGNAL_PANIC: _all_gains(1.0), SIGNAL_HOT: _all_gains(-1.0)}
    sig_gains = {SIGNAL_PANIC: {k: [] for k in horizons},
                 SIGNAL_HOT: {k: [] for k in horizons}}

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
            sign = 1.0 if stype == SIGNAL_PANIC else -1.0
            for k in horizons:
                if i + k < n and not np.isnan(arr[i]) and arr[i] != 0 \
                        and not np.isnan(arr[i + k]):
                    sig_gains[stype][k].append(sign * (arr[i + k] / arr[i] - 1.0))
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

    # 瞎猜基线：与信号同一套判卷标准，随机挑一天撞上标签的概率
    baselines = {
        SIGNAL_PANIC: _random_baseline(pivots[SIGNAL_PANIC], score.index, lags),
        SIGNAL_HOT: _random_baseline(pivots[SIGNAL_HOT], score.index, lags),
    }
    return {"signals": signals,
            "stats": summarize(signals, len(score), baselines=baselines,
                               sig_gains=sig_gains, all_gains=all_gains)}


def summarize(signals: list[dict], trading_days: int, min_samples: int = 5,
              baselines: dict | None = None,
              sig_gains: dict | None = None,
              all_gains: dict | None = None) -> dict:
    """汇总统计：T+0/1/2 命中率、累计命中率、覆盖率、最大连续失败。

    baselines: {信号类型: 瞎猜基线概率}。传入后会额外计算统计显著性
    （真实优势 / 二项检验 p 值 / Wilson 置信区间 / 一句话结论）。
    sig_gains / all_gains: 前瞻收益（信号组 / 全样本），用于置换检验 ——
    比二元命中率功效高得多，样本少时尤其重要。
    """
    baselines = baselines or {}
    sig_gains = sig_gains or {}
    all_gains = all_gains or {}

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
        hit_rate = round(succ / total, 4) if total else None

        # ---- 统计显著性：和瞎猜比，而不是只看绝对数字 ----
        base = baselines.get(stype)
        p_value = _binom_sf(succ, total, base) if base is not None else None
        ci = _wilson_ci(succ, total)
        label, explain = _verdict(total, succ, base, p_value, min_samples)

        return {
            "signals": total, "unconfirmed": len(arr) - total,
            "success": succ, "fail": fails,
            "hit_rate": hit_rate,
            **t_rates,
            "coverage": round(total / trading_days, 4) if trading_days else None,
            "max_consec_fail": max_streak,
            "sample_enough": total >= min_samples,
            "note": None if total >= min_samples else "样本不足，暂不能验证90%命中率",
            # 新增：对照与显著性
            "baseline": base,
            "lift": round(hit_rate - base, 4) if (hit_rate is not None and base is not None) else None,
            "p_value": round(p_value, 4) if p_value is not None else None,
            "p2": round(min(1.0, 2.0 * min(p_value, 1.0 - p_value)), 4) if p_value is not None else None,
            "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None,
            "verdict": label,
            "verdict_detail": explain,
            # 前瞻收益：信号之后市场平均怎么走（连续指标，功效高于二元命中率）
            "fwd": _fwd_block(sig_gains.get(stype), all_gains.get(stype), min_samples),
        }

    return {"panic": block(SIGNAL_PANIC), "hot": block(SIGNAL_HOT),
            "trading_days": trading_days}
