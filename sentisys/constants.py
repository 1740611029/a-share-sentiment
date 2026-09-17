"""常量定义"""
from enum import Enum


class Market(str, Enum):
    A_SHARE = "A_SHARE"
    CHINEXT = "CHINEXT"
    STAR = "STAR"


class Board(str, Enum):
    SH_MAIN = "SH_MAIN"
    SZ_MAIN = "SZ_MAIN"
    CHINEXT = "CHINEXT"
    STAR = "STAR"
    BJ = "BJ"


MARKET_NAMES = {"A_SHARE": "A股全市场", "CHINEXT": "创业板", "STAR": "科创板"}

# 主要指数代码（新浪）
INDEX_NAMES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sh000300": "沪深300",
    "sh000852": "中证1000",
    "sz399006": "创业板指",
    "sh000688": "科创50",
}

# 数据状态
STATUS_OK = "ok"
STATUS_STALE = "stale"        # 数据未更新（数据源不可用，保留上一交易日）
STATUS_ABNORMAL = "abnormal"  # 数据异常（缺失/不足）

# 信号类型
SIGNAL_PANIC = "panic"  # 极端恐慌 → 预期阶段底部
SIGNAL_HOT = "hot"      # 极端过热 → 预期阶段顶部
