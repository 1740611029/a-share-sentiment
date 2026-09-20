"""波段信号模型调度：复用原情绪指标输出（不修改），逐模型回测并落库。

情绪分数的来源与原历史回测完全一致：因子表 → 子分数 → 激活模型版本
权重加权 → 0~100 总分。波段模型只读取该分数，永远不反向修改它。

极端阈值口径（重要）：触发阈值同样取激活模型版本的 extreme_panic / extreme_hot，
而不是 config.sentiment 里的默认值 —— 与「情绪概览」页展示的阈值保持一致，
避免同一个系统出现两套分数线。尚未 optimize 时退回 config 默认值。
"""
from __future__ import annotations

import json

import pandas as pd

from ..engine import compute_score, compute_subscores
from ..factors import load_factors
from .engine import run_model_backtest
from .models import MODELS, build_context


def _active_params(cfg: dict, db, market: str) -> dict:
    """取该市场当前生效的模型参数（权重 + 极端阈值）。

    阈值口径必须与情绪概览页一致：概览页展示的是激活模型版本的阈值，
    因此这里也不能用 config 里的默认值，否则同一个系统里会出现两套分数线。
    无激活版本（尚未 optimize）时退回 config 默认值。
    """
    fallback = {
        "weights": dict(cfg["sentiment"]["weights"]),
        "extreme_panic": float(cfg["sentiment"]["extreme_panic"]),
        "extreme_hot": float(cfg["sentiment"]["extreme_hot"]),
        "version": None,
    }
    ver = db.latest_version(market)
    if not ver:
        return fallback
    try:
        p = json.loads(ver["params_json"])
    except (ValueError, TypeError):
        return fallback
    return {
        "weights": p.get("weights", fallback["weights"]),
        "extreme_panic": float(p.get("extreme_panic", fallback["extreme_panic"])),
        "extreme_hot": float(p.get("extreme_hot", fallback["extreme_hot"])),
        "version": ver.get("version"),
    }


def _original_score(cfg, db, provider, market: str) -> tuple[pd.Series, pd.DataFrame]:
    """按原系统逻辑重算情绪分数（与原 backtest 命令同一口径）。"""
    weights = _active_params(cfg, db, market)["weights"]
    factors = load_factors(cfg, market)
    subscores = compute_subscores(factors, cfg)
    return compute_score(subscores, weights, cfg), factors


def run_swing(cfg, db, provider, market: str,
              sample: str = "full", models: list[str] | None = None) -> dict:
    """对单个市场运行全部（或指定）波段模型回测并落库。

    sample: full=全历史；oos=最近1年（样本外，与原回测口径一致）。
    返回 {model_code: stats}。
    """
    sp = cfg.get("swing", {})
    active = _active_params(cfg, db, market)
    panic_t, hot_t = active["extreme_panic"], active["extreme_hot"]
    params = {
        "observe_window": int(sp.get("observe_window", 10)),
        "signal_cooldown": int(sp.get("signal_cooldown", 5)),
        "confirm_pct": float(sp.get("confirm_pct", 0.03)),
        "reversal_lookback": int(sp.get("reversal_lookback", 5)),
        "reversal_delta": float(sp.get("reversal_delta", 5)),
        "exit_zone_low": float(sp.get("exit_zone_low", 40)),
        "exit_zone_high": float(sp.get("exit_zone_high", 60)),
        "index_confirm_ret": float(sp.get("index_confirm_ret", 0.0)),
        "composite_index_tol": float(sp.get("composite_index_tol", 0.01)),
        "min_samples": int(sp.get("min_samples", 5)),
        # 记录本轮实际使用的阈值与来源，落库后可在页面上核对
        "extreme_panic": panic_t,
        "extreme_hot": hot_t,
        "threshold_version": active["version"],
    }
    score, factors = _original_score(cfg, db, provider, market)
    score = score.dropna()
    primary = cfg["markets"][market]["primary_index"]
    close = provider.load_index_daily(primary).set_index("date")["close"]

    ctx = build_context(score, close, factors, panic_t, hot_t)
    ctx["params"] = params

    start = None
    if sample == "oos":
        exclude = int(cfg["backtest"]["train_exclude_recent_days"])
        start = (score.index[-1] - pd.Timedelta(days=exclude)).strftime("%Y-%m-%d")

    out = {}
    for code in (models or list(MODELS)):
        res = run_model_backtest(code, ctx, params, start=start)
        period = (start or score.index[0].strftime("%Y-%m-%d"),
                  score.index[-1].strftime("%Y-%m-%d"))
        db.save_swing_run(market, code, sample, period, params,
                          res["stats"], res["signals"])
        out[code] = res["stats"]
    return out
