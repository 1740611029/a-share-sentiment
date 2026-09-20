"""波段信号模型定义：多套独立模型，分别读取原情绪分数，互不影响。

核心原则：原情绪指标只负责"描述市场情绪"且保持固定不变；
各波段模型只读取情绪分数（外加指数/市场宽度等公开数据），
研究"情绪进入极端区域之后，未来 T+0~T+10 个交易日内是否形成阶段性低点/高点"。

五套基准模型：
  A 纯情绪极端模型          情绪 <= 恐慌阈值 / >= 过热阈值 直接触发（基准模型）
  B 情绪极端 + 情绪反转     极端区域后情绪明显修复/退潮才触发
  C 情绪极端 + 指数确认     极端情绪 + 主参考指数止跌/回落
  D 情绪极端 + 市场宽度确认 极端情绪 + 涨跌家数/涨跌停等宽度改善/恶化
  E 综合波段模型            极端 + 反转 + 宽度 + 指数 综合确认

每个模型对 panic（预期阶段底部）/ hot（预期阶段顶部）两个方向分别给出
触发掩码；触发日为 T，观察 T+0~T+10 个交易日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import SIGNAL_HOT, SIGNAL_PANIC

MODELS: dict[str, dict] = {
    "A": {
        "name": "纯情绪极端模型",
        "desc": "情绪 <= 恐慌阈值（或 >= 过热阈值）直接触发，观察 T+0~T+10 是否形成阶段低点/高点。最基础的基准模型。",
    },
    "B": {
        "name": "情绪极端+反转模型",
        "desc": "近5日内曾进入极端区域，且情绪自极端值明显修复（底部回升5分以上且仍处低位）才触发；顶部镜像。",
    },
    "C": {
        "name": "情绪极端+指数确认模型",
        "desc": "情绪进入极端区域，且主参考指数当日止跌（底部：涨跌幅>=0；顶部：涨跌幅<=0）才触发。",
    },
    "D": {
        "name": "情绪极端+宽度确认模型",
        "desc": "情绪进入极端区域，且市场宽度开始改善（底部：上涨家数占比回升且跌停数不增；顶部镜像）才触发。",
    },
    "E": {
        "name": "综合波段模型",
        "desc": "近5日内曾极端恐慌，且情绪开始修复 + 市场宽度改善 + 指数基本止跌，三者同时满足才触发；顶部镜像。",
    },
}


def _extreme_masks(score: pd.Series, panic_t: float, hot_t: float):
    low = (score <= panic_t).fillna(False)
    high = (score >= hot_t).fillna(False)
    return low, high


def _trigger_model_A(ctx: dict) -> dict:
    return {SIGNAL_PANIC: ctx["extreme_low"], SIGNAL_HOT: ctx["extreme_high"]}


def _trigger_model_B(ctx: dict) -> dict:
    """情绪极端 + 情绪反转：极端区域出现后情绪明显修复/退潮才触发。"""
    score, p = ctx["score"], ctx["params"]
    lb = int(p["reversal_lookback"])
    delta = float(p["reversal_delta"])
    roll_min = score.rolling(lb, min_periods=1).min()
    roll_max = score.rolling(lb, min_periods=1).max()
    panic_t, hot_t = ctx["panic_t"], ctx["hot_t"]
    bottom = ((roll_min <= panic_t)
              & (score >= roll_min + delta)
              & (score <= float(p["exit_zone_low"]))).fillna(False)
    top = ((roll_max >= hot_t)
           & (score <= roll_max - delta)
           & (score >= float(p["exit_zone_high"]))).fillna(False)
    return {SIGNAL_PANIC: bottom, SIGNAL_HOT: top}


def _trigger_model_C(ctx: dict) -> dict:
    """情绪极端 + 指数确认：指数当日止跌（底部）/ 回落（顶部）。"""
    ret = float(ctx["params"]["index_confirm_ret"])
    idx_ret1 = ctx["idx_ret1"]
    bottom = (ctx["extreme_low"] & (idx_ret1 >= ret)).fillna(False)
    top = (ctx["extreme_high"] & (idx_ret1 <= -ret)).fillna(False)
    return {SIGNAL_PANIC: bottom, SIGNAL_HOT: top}


def _trigger_model_D(ctx: dict) -> dict:
    """情绪极端 + 市场宽度确认：宽度改善（底部）/ 恶化（顶部）。"""
    adv_ratio, limit_dn, limit_up = ctx["adv_ratio"], ctx["limit_dn"], ctx["limit_up"]
    breadth_up = (adv_ratio > adv_ratio.shift(1)) & (limit_dn <= limit_dn.shift(1))
    breadth_dn = (adv_ratio < adv_ratio.shift(1)) & (limit_up <= limit_up.shift(1))
    bottom = (ctx["extreme_low"] & breadth_up).fillna(False)
    top = (ctx["extreme_high"] & breadth_dn).fillna(False)
    return {SIGNAL_PANIC: bottom, SIGNAL_HOT: top}


def _trigger_model_E(ctx: dict) -> dict:
    """综合波段模型：极端 + 情绪修复 + 宽度改善 + 指数止跌 同时满足。"""
    score, p = ctx["score"], ctx["params"]
    lb = int(p["reversal_lookback"])
    tol = float(p["composite_index_tol"])
    roll_min = score.rolling(lb, min_periods=1).min()
    roll_max = score.rolling(lb, min_periods=1).max()
    panic_t, hot_t = ctx["panic_t"], ctx["hot_t"]
    adv_ratio, limit_dn, limit_up = ctx["adv_ratio"], ctx["limit_dn"], ctx["limit_up"]
    idx_ret1 = ctx["idx_ret1"]
    repairing = score > score.shift(1)
    fading = score < score.shift(1)
    breadth_up = (adv_ratio > adv_ratio.shift(1)) & (limit_dn <= limit_dn.shift(1))
    breadth_dn = (adv_ratio < adv_ratio.shift(1)) & (limit_up <= limit_up.shift(1))
    bottom = ((roll_min <= panic_t) & repairing & breadth_up
              & (idx_ret1 >= -tol)).fillna(False)
    top = ((roll_max >= hot_t) & fading & breadth_dn
           & (idx_ret1 <= tol)).fillna(False)
    return {SIGNAL_PANIC: bottom, SIGNAL_HOT: top}


TRIGGERS = {
    "A": _trigger_model_A,
    "B": _trigger_model_B,
    "C": _trigger_model_C,
    "D": _trigger_model_D,
    "E": _trigger_model_E,
}


def build_context(score: pd.Series, close: pd.Series,
                  factors: pd.DataFrame, panic_t: float, hot_t: float) -> dict:
    """组装各模型触发所需的公共数据（全部对齐到 score 的交易日索引）。

    只使用当日及历史数据（shift/rolling 均向后看），无未来函数。
    """
    close = close.reindex(score.index).ffill()
    f = factors.reindex(score.index)
    traded = f["traded"].replace(0, np.nan)
    extreme_low, extreme_high = _extreme_masks(score, panic_t, hot_t)
    return {
        "score": score,
        "close": close,
        "idx_ret1": close.pct_change(),
        "adv_ratio": (f["up"] - f["down"]) / traded,
        "limit_dn": f["limit_dn"].astype(float),
        "limit_up": f["limit_up"].astype(float),
        "extreme_low": extreme_low,
        "extreme_high": extreme_high,
        "panic_t": panic_t,
        "hot_t": hot_t,
        "params": None,  # 由调用方填入 swing 参数
    }


def trigger_masks(model_code: str, ctx: dict) -> dict:
    """返回 {signal_type: bool Series}，True 表示当日触发该模型的波段信号。"""
    return TRIGGERS[model_code](ctx)


def apply_cooldown(mask: pd.Series, cooldown: int) -> list:
    """信号去重：冷却期内同方向只保留首个触发日，返回交易日列表。"""
    days = list(mask.index[mask.fillna(False)])
    picked, last = [], -10 ** 9
    positions = {d: i for i, d in enumerate(mask.index)}
    for d in days:
        if positions[d] - last > cooldown:
            picked.append(d)
            last = positions[d]
    return picked
