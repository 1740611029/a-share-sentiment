"""波段信号回测引擎：T+0 ~ T+10 交易日观察窗口评估 + 统计汇总。

阶段低点定义（底部信号，panic）：
  观察窗口 close[T .. T+W]（W=observe_window，默认10个交易日）内出现最低价，
  且最低价之后窗口内最大收盘价相对最低价反弹 >= confirm_pct → 命中。

阶段高点定义（顶部信号，hot）：完全镜像——窗口内最高价之后回撤 >= confirm_pct。

数据末端窗口不足 W+1 个交易日的信号记为"未确认"（hit=None），不计入命中率。
观察窗口只用于事后验证标签，不参与信号生成，无未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import _binom_sf, _verdict, _wilson_ci
from ..constants import SIGNAL_HOT, SIGNAL_PANIC
from .models import apply_cooldown, trigger_masks

LAG_BUCKETS = [("T+0", 0, 0), ("T+1", 1, 1), ("T+2", 2, 2), ("T+3", 3, 3),
               ("T+4~T+5", 4, 5), ("T+6~T+10", 6, 10)]


def _swing_baseline(close: pd.Series, window: int, confirm_pct: float,
                    stype: str) -> float | None:
    """瞎猜基线：随机挑一天，用与信号完全相同的判定标准，命中概率是多少。

    波段模型的"命中"定义比较宽松（窗口内出现极值 + 之后反向走够 confirm_pct），
    所以基线往往不低。不看基线的命中率毫无意义 —— 与情绪回测同一原则。
    """
    n = len(close)
    usable = hits = 0
    for i in range(n - window):
        if np.isnan(close.iloc[i]):
            continue
        usable += 1
        if evaluate_signal(close, i, window, confirm_pct, stype).get("hit") == 1:
            hits += 1
    return round(hits / usable, 4) if usable else None


def evaluate_signal(close: pd.Series, i: int, window: int,
                    confirm_pct: float, signal_type: str) -> dict:
    """评估单个信号在 T+0~T+W 窗口内是否形成阶段低点/高点。"""
    n = len(close)
    seg = close.iloc[i:min(n, i + window + 1)].to_numpy(dtype=float)
    rec: dict = {"observe_end": close.index[min(n - 1, i + window)].strftime("%Y-%m-%d"),
                 "t0_close": round(float(seg[0]), 3), "hit": None}
    if len(seg) < window + 1 or np.isnan(seg[0]):
        return rec  # 数据末端：观察窗口不足 → 未确认
    min_pos = int(np.nanargmin(seg))
    max_pos = int(np.nanargmax(seg))
    win_min, win_max = float(seg[min_pos]), float(seg[max_pos])
    idx = close.index
    rec.update({
        "win_min": round(win_min, 3),
        "win_min_date": idx[i + min_pos].strftime("%Y-%m-%d"),
        "win_min_lag": min_pos,
        "win_max": round(win_max, 3),
        "win_max_date": idx[i + max_pos].strftime("%Y-%m-%d"),
        "win_max_lag": max_pos,
        "drop_t0_to_min": round(win_min / seg[0] - 1.0, 4),
        "rise_t0_to_max": round(win_max / seg[0] - 1.0, 4),
        "rebound_after_min": round(float(np.nanmax(seg[min_pos:])) / win_min - 1.0, 4),
        "drawdown_after_max": round(float(np.nanmin(seg[max_pos:])) / win_max - 1.0, 4),
    })
    if signal_type == SIGNAL_PANIC:
        rec["hit"] = int(rec["rebound_after_min"] >= confirm_pct)
    else:
        rec["hit"] = int(rec["drawdown_after_max"] <= -confirm_pct)
    return rec


def run_model_backtest(model_code: str, ctx: dict, params: dict,
                       start: str | None = None,
                       end: str | None = None) -> dict:
    """对单个市场、单套波段模型执行历史回测。"""
    score, close = ctx["score"], ctx["close"]
    window = int(params["observe_window"])
    cooldown = int(params["signal_cooldown"])
    confirm = float(params["confirm_pct"])
    masks = trigger_masks(model_code, ctx)
    signals: list[dict] = []
    for stype in (SIGNAL_PANIC, SIGNAL_HOT):
        for d in apply_cooldown(masks[stype], cooldown):
            if start and d < pd.Timestamp(start):
                continue
            if end and d > pd.Timestamp(end):
                continue
            i = score.index.get_loc(d)
            rec = {"signal_type": stype, "signal_date": d.strftime("%Y-%m-%d"),
                   "sentiment": round(float(score.loc[d]), 2)}
            rec.update(evaluate_signal(close, i, window, confirm, stype))
            signals.append(rec)
    stats = summarize(signals, close, params)
    return {"signals": signals, "stats": stats}


def _regime_of(close: pd.Series) -> pd.Series:
    """市场状态划分：收盘价 vs 250日均线（±3% 为震荡）。"""
    ma = close.rolling(250, min_periods=60).mean()
    dev = close / ma - 1.0
    return pd.Series(np.where(dev > 0.03, "bull",
                              np.where(dev < -0.03, "bear", "sideways")),
                     index=close.index)


def _lag_distribution(sigs: list[dict], lag_key: str) -> dict:
    confirmed = [s for s in sigs if s.get("hit") is not None and s.get(lag_key) is not None]
    total = len(confirmed)
    out = {}
    for label, lo, hi in LAG_BUCKETS:
        cnt = sum(1 for s in confirmed if lo <= s[lag_key] <= hi)
        out[label] = {"count": cnt,
                      "pct": round(cnt / total, 4) if total else None}
    return out


def _breakdown(sigs: list[dict], key_fn) -> dict:
    groups: dict[str, list] = {}
    for s in sigs:
        if s.get("hit") is None:
            continue
        groups.setdefault(key_fn(s), []).append(s)
    return {k: {"signals": len(v), "success": sum(x["hit"] for x in v),
                "hit_rate": round(sum(x["hit"] for x in v) / len(v), 4)}
            for k, v in sorted(groups.items())}


def summarize(signals: list[dict], close: pd.Series, params: dict) -> dict:
    """汇总单套模型的统计：胜率/信号数/平均高低点时间/最大反弹回撤/
    连续失败/盈亏比/年度分布/牛熊震荡分布。"""
    min_samples = int(params.get("min_samples", 5))
    regime = _regime_of(close)
    window = int(params.get("observe_window", 10))
    confirm = float(params.get("confirm_pct", 0.03))
    baselines = {
        SIGNAL_PANIC: _swing_baseline(close, window, confirm, SIGNAL_PANIC),
        SIGNAL_HOT: _swing_baseline(close, window, confirm, SIGNAL_HOT),
    }

    def block(stype: str) -> dict:
        arr = [s for s in signals if s["signal_type"] == stype]
        confirmed = [s for s in arr if s.get("hit") is not None]
        total = len(confirmed)
        succ = sum(1 for s in confirmed if s["hit"] == 1)
        max_streak, streak = 0, 0
        for s in confirmed:
            if s["hit"] == 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        if stype == SIGNAL_PANIC:
            lag_key, amp_key, risk_key = "win_min_lag", "rebound_after_min", "drop_t0_to_min"
        else:
            lag_key, amp_key, risk_key = "win_max_lag", "drawdown_after_max", "rise_t0_to_max"

        def avg(key):
            vals = [s[key] for s in confirmed if s.get(key) is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        avg_amp, avg_risk = avg(amp_key), avg(risk_key)
        hit_rate = round(succ / total, 4) if total else None

        # ---- 对照与显著性：命中率必须和"瞎猜"比 ----
        base = baselines.get(stype)
        p_value = _binom_sf(succ, total, base) if base is not None else None
        ci = _wilson_ci(succ, total)
        label, explain = _verdict(total, succ, base, p_value, min_samples)

        return {
            "signals": total, "unconfirmed": len(arr) - total,
            "success": succ, "fail": total - succ,
            "hit_rate": hit_rate,
            "avg_pivot_lag": avg(lag_key),
            "avg_max_rebound": avg_amp,        # 底部：平均最大反弹；顶部：平均最大回撤（负值）
            "avg_max_risk": avg_risk,          # 底部：平均信号日到低点跌幅；顶部：平均信号日到高点涨幅
            "profit_loss_ratio": (round(abs(avg_amp / avg_risk), 2)
                                  if avg_amp is not None and avg_risk not in (None, 0) else None),
            "max_consec_fail": max_streak,
            "lag_distribution": _lag_distribution(confirmed, lag_key),
            "yearly": _breakdown(confirmed, lambda s: s["signal_date"][:4]),
            "regime": _breakdown(
                confirmed,
                lambda s: str(regime.get(pd.Timestamp(s["signal_date"]), "sideways"))),
            "sample_enough": total >= min_samples,
            "note": None if total >= min_samples else "样本不足，统计仅供参考",
            # 对照与显著性
            "baseline": base,
            "lift": round(hit_rate - base, 4) if (hit_rate is not None and base is not None) else None,
            "p_value": round(p_value, 4) if p_value is not None else None,
            "p2": round(min(1.0, 2.0 * min(p_value, 1.0 - p_value)), 4) if p_value is not None else None,
            "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None,
            "verdict": label,
            "verdict_detail": explain,
        }

    return {"panic": block(SIGNAL_PANIC), "hot": block(SIGNAL_HOT),
            "observe_window": int(params["observe_window"])}
