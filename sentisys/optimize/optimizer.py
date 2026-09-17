"""参数自动优化 + walk-forward 验证 + 模型版本落库。

防过拟合约束：
  1. 最近 train_exclude_recent_days 天为样本外区间，优化过程完全不可见；
  2. 训练区间内按时间分 K 段做 walk-forward 稳定性检验；
  3. 目标函数 = min(恐慌方向得分, 过热方向得分)，
     方向得分 = 分段命中率均值 - stability_penalty × 分段标准差，
     样本数不足 min_samples 的方向直接记 0 分；
  4. 最终参数只在样本外区间评估一次并如实记录。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import find_pivots, run_backtest, summarize
from ..constants import SIGNAL_HOT, SIGNAL_PANIC
from ..engine.score import compute_score, DEFAULT_WEIGHT_KEYS


def _fold_bounds(index: pd.DatetimeIndex, k: int) -> list[tuple[str, str]]:
    """把训练区间按时间切成 k 段。"""
    n = len(index)
    bounds = []
    for i in range(k):
        lo = index[n * i // k]
        hi = index[min(n - 1, n * (i + 1) // k - 1)]
        bounds.append((lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")))
    return bounds


def _objective(score: pd.Series, close: pd.Series, params: dict,
               folds: list[tuple[str, str]], min_samples: int,
               stability_penalty: float) -> float:
    """walk-forward 分段评估，返回综合得分（越高越好）。"""
    dir_scores = []
    for stype in (SIGNAL_PANIC, SIGNAL_HOT):
        rates, ns = [], []
        for (s, e) in folds:
            res = run_backtest(score, close, params, start=s, end=e)
            stats = res["stats"][stype]
            ns.append(stats["signals"])
            if stats["signals"] > 0 and stats["hit_rate"] is not None:
                rates.append(stats["hit_rate"])
        total_n = sum(ns)
        if total_n < min_samples or not rates:
            dir_scores.append(0.0)
            continue
        mean_r = float(np.mean(rates))
        std_r = float(np.std(rates)) if len(rates) > 1 else 0.0
        dir_scores.append(mean_r - stability_penalty * std_r)
    return min(dir_scores)


def _eval(score: pd.Series, close: pd.Series, params: dict,
          start: str, end: str, min_samples: int) -> dict:
    res = run_backtest(score, close, params, start=start, end=end)
    return res


def optimize_market(cfg: dict, db, market: str, subscores: pd.DataFrame,
                    close: pd.Series, progress: bool = True) -> dict:
    """对单一市场执行参数优化，保存模型版本与回测记录，返回结果摘要。"""
    bt = cfg["backtest"]
    opt = cfg["optimize"]
    min_samples = int(bt["min_samples"])
    stab_pen = float(opt["stability_penalty"])
    k_folds = int(bt["walk_forward_folds"])

    close = close.reindex(subscores.index).ffill()
    valid = subscores.dropna().index
    if len(valid) < 300:
        raise RuntimeError(f"{market} 有效历史不足（{len(valid)} 个交易日），无法优化")
    oos_start = valid[-1] - pd.Timedelta(days=int(bt["train_exclude_recent_days"]))
    train_idx = valid[valid < oos_start]
    oos_idx = valid[valid >= oos_start]
    if len(oos_idx) < 30:
        raise RuntimeError("样本外区间过短")
    folds = _fold_bounds(train_idx, k_folds)
    train_period = (train_idx[0].strftime("%Y-%m-%d"), train_idx[-1].strftime("%Y-%m-%d"))
    oos_period = (oos_idx[0].strftime("%Y-%m-%d"), oos_idx[-1].strftime("%Y-%m-%d"))

    base_params = {
        "extreme_panic": cfg["sentiment"]["extreme_panic"],
        "extreme_hot": cfg["sentiment"]["extreme_hot"],
        "rebound_pct": bt["rebound_pct"],
        "pivot_left": bt["pivot_left"],
        "pivot_mode": bt.get("pivot_mode", "zone"),
        "pos_tol": bt.get("pos_tol", 0.02),
        "confirm_window": bt["confirm_window"],
        "signal_cooldown": bt["signal_cooldown"],
        "observe_lags": bt["observe_lags"],
    }
    base_weights = dict(cfg["sentiment"]["weights"])

    # 枢轴（顶/底标签）只与 (rebound_pct, pos_tol) 有关，预计算缓存避免重复
    pivot_cache: dict = {}

    def pivots_for(rb: float, pos_tol: float) -> dict:
        key = (rb, pos_tol)
        if key not in pivot_cache:
            pivot_cache[key] = {
                SIGNAL_PANIC: find_pivots(close, "bottom", base_params["pivot_left"],
                                          base_params["confirm_window"], rb,
                                          base_params["pivot_mode"], pos_tol),
                SIGNAL_HOT: find_pivots(close, "top", base_params["pivot_left"],
                                        base_params["confirm_window"], rb,
                                        base_params["pivot_mode"], pos_tol),
            }
        return pivot_cache[key]

    def log(msg):
        if progress:
            print(msg)

    # ---------- 阶段 1：默认权重，搜索阈值 × 反弹幅度 × 位置容差 ----------
    log(f"[{market}] 阶段1：阈值×反弹幅度×位置容差网格搜索（训练期 {train_period[0]} ~ {train_period[1]}）")
    best = (-1.0, None)
    score_default = compute_score(subscores, base_weights, cfg)
    for panic_t in opt["panic_thresholds"]:
        for hot_t in opt["hot_thresholds"]:
            for rb in opt["rebound_pcts"]:
                for pt in opt.get("pos_tols", [base_params["pos_tol"]]):
                    p = {**base_params, "extreme_panic": panic_t,
                         "extreme_hot": hot_t, "rebound_pct": rb, "pos_tol": pt,
                         "_pivots": pivots_for(rb, pt)}
                    val = _objective(score_default, close, p, folds, min_samples, stab_pen)
                    if val > best[0]:
                        best = (val, p)
    best_params = best[1]
    log(f"  最优: 恐慌<={best_params['extreme_panic']} 过热>={best_params['extreme_hot']} "
        f"反弹>={best_params['rebound_pct']:.1%} 位置容差={best_params['pos_tol']:.0%} 得分 {best[0]:.3f}")

    # ---------- 阶段 2：固定阈值，随机搜索因子权重 ----------
    log(f"[{market}] 阶段2：因子权重随机搜索（{opt['weight_random_trials']} 次）")
    rng = np.random.default_rng(42)
    best_w = (best[0], base_weights)
    best_params["_pivots"] = pivots_for(best_params["rebound_pct"], best_params["pos_tol"])
    trials = [base_weights]
    for _ in range(int(opt["weight_random_trials"])):
        w = rng.dirichlet(np.ones(len(DEFAULT_WEIGHT_KEYS)) * 2.0)
        w = np.maximum(w, 0.05)
        w = w / w.sum()
        trials.append(dict(zip(DEFAULT_WEIGHT_KEYS, w.round(4))))
    for weights in trials:
        score = compute_score(subscores, weights, cfg)
        val = _objective(score, close, best_params, folds, min_samples, stab_pen)
        if val > best_w[0]:
            best_w = (val, weights)
    best_weights = best_w[1]
    log(f"  最优权重: {best_weights} 得分 {best_w[0]:.3f}")

    # ---------- 阶段 3：最优权重下复调阈值 ----------
    log(f"[{market}] 阶段3：最优权重下复调阈值")
    score_best = compute_score(subscores, best_weights, cfg)
    for panic_t in opt["panic_thresholds"]:
        for hot_t in opt["hot_thresholds"]:
            p = {**best_params, "extreme_panic": panic_t, "extreme_hot": hot_t}
            val = _objective(score_best, close, p, folds, min_samples, stab_pen)
            if val > best[0]:
                best = (val, p)
                best_params = p
    log(f"  最终: 恐慌<={best_params['extreme_panic']} 过热>={best_params['extreme_hot']} "
        f"得分 {best[0]:.3f}")

    final_params = {k: v for k, v in
                    {**best_params, "weights": {k: round(float(v), 4)
                                                for k, v in best_weights.items()}}.items()
                    if not k.startswith("_")}

    # ---------- 训练期 / 样本外 各评估一次 ----------
    train_res = _eval(score_best, close, final_params, *train_period, min_samples)
    oos_res = _eval(score_best, close, final_params, *oos_period, min_samples)
    log(f"[{market}] 训练期: 恐慌命中率 {_fmt(train_res, 'panic')} 过热命中率 {_fmt(train_res, 'hot')}")
    log(f"[{market}] 样本外: 恐慌命中率 {_fmt(oos_res, 'panic')} 过热命中率 {_fmt(oos_res, 'hot')}")

    # ---------- 落库：模型版本 + 回测运行 ----------
    version = db.next_version(market)
    version_id = db.save_model_version(
        market, version, final_params,
        f"{train_period[0]}~{train_period[1]}",
        f"{oos_period[0]}~{oos_period[1]}",
        train_res["stats"], oos_res["stats"], is_active=True)
    db.save_backtest_run(market, version_id, "train", train_period,
                         final_params, train_res["stats"], train_res["signals"])
    db.save_backtest_run(market, version_id, "oos", oos_period,
                         final_params, oos_res["stats"], oos_res["signals"])
    log(f"[{market}] 已保存模型版本 {version}（active）")
    return {"version": version, "params": final_params,
            "train_stats": train_res["stats"], "oos_stats": oos_res["stats"]}


def _fmt(res: dict, key: str) -> str:
    s = res["stats"][key]
    if s["hit_rate"] is None:
        return f"无信号({s['note'] or ''})"
    return f"{s['hit_rate']:.1%} (n={s['signals']})"
