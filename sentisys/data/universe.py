"""股票池构建与过滤"""
from __future__ import annotations

import pandas as pd

from ..constants import Board, Market


def classify_board(code: str) -> Board | None:
    """按代码前缀划分板块。"""
    if code.startswith(("600", "601", "603", "605")):
        return Board.SH_MAIN
    if code.startswith("688"):
        return Board.STAR
    if code.startswith(("000", "001", "002", "003")):
        return Board.SZ_MAIN
    if code.startswith(("300", "301", "302")):
        return Board.CHINEXT
    if code.startswith(("8", "4", "920")):
        return Board.BJ
    return None


def build_universe(stock_list: pd.DataFrame, market: str | Market,
                   cfg: dict, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """按市场过滤股票池：板块 / ST / 退市 / 上市天数。

    返回 [code, name, board, list_date, limit_pct]
    """
    market = Market(market)
    f = cfg["filters"]
    boards = set(cfg["markets"][market.value]["boards"])
    limit_rules = cfg["limit_rules"]

    df = stock_list.copy()
    df["board"] = df["code"].map(classify_board)
    df = df[df["board"].notna()]
    df["board"] = df["board"].map(lambda b: b.value if isinstance(b, Board) else b)
    df = df[df["board"].isin(boards)]

    if f.get("exclude_st", True):
        df = df[~df["name"].str.upper().str.contains("ST", na=False)]
    if f.get("exclude_delisting", True):
        df = df[~df["name"].str.contains("退", na=False)]
    if as_of is not None and f.get("min_list_days", 0) > 0:
        cutoff = as_of - pd.Timedelta(days=int(f["min_list_days"]))
        df = df[df["list_date"] <= cutoff]

    df["board"] = df["board"].map(lambda b: b.value if isinstance(b, Board) else b)
    df["limit_pct"] = df["board"].map(lambda b: float(limit_rules.get(b, 10.0)))
    return df.reset_index(drop=True)
