"""SQLite 持久化：每日情绪快照 / 模型版本 / 回测运行与信号明细"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment_snapshot (
    date TEXT NOT NULL,
    market_type TEXT NOT NULL,
    sentiment_score REAL,
    sentiment_level TEXT,
    daily_change REAL,
    change_3d REAL,
    change_5d REAL,
    change_10d REAL,
    percentile20 REAL,
    percentile60 REAL,
    percentile120 REAL,
    percentile250 REAL,
    is_extreme_panic INTEGER DEFAULT 0,
    is_extreme_hot INTEGER DEFAULT 0,
    trend TEXT,
    pattern TEXT,
    divergence TEXT,
    data_status TEXT DEFAULT 'ok',
    calculation_version TEXT,
    idx_ret_1d REAL,
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    PRIMARY KEY (date, market_type)
);
CREATE TABLE IF NOT EXISTS model_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_type TEXT NOT NULL,
    version TEXT NOT NULL,
    params_json TEXT NOT NULL,
    train_period TEXT,
    oos_period TEXT,
    train_stats_json TEXT,
    oos_stats_json TEXT,
    is_active INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE (market_type, version)
);
CREATE TABLE IF NOT EXISTS backtest_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_type TEXT NOT NULL,
    version_id INTEGER,
    sample_tag TEXT,          -- train / oos / full
    period_start TEXT,
    period_end TEXT,
    params_json TEXT,
    stats_json TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS backtest_signal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    market_type TEXT NOT NULL,
    signal_type TEXT NOT NULL,   -- panic / hot
    signal_date TEXT NOT NULL,
    sentiment REAL,
    pivot_date TEXT,
    lag INTEGER,                 -- 实际命中偏差 0/1/2，未命中为 NULL
    max_rebound REAL,
    max_drawdown REAL,
    hit_t0 INTEGER DEFAULT 0,
    hit_t1 INTEGER DEFAULT 0,
    hit_t2 INTEGER DEFAULT 0,
    success INTEGER,             -- 1 成功 / 0 失败 / NULL 未确认
    FOREIGN KEY (run_id) REFERENCES backtest_run(id)
);
CREATE TABLE IF NOT EXISTS swing_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_type TEXT NOT NULL,
    model_code TEXT NOT NULL,    -- A / B / C / D / E ...
    sample_tag TEXT,             -- full / oos
    period_start TEXT,
    period_end TEXT,
    params_json TEXT,
    stats_json TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS swing_signal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    market_type TEXT NOT NULL,
    model_code TEXT NOT NULL,
    signal_type TEXT NOT NULL,   -- panic / hot
    signal_date TEXT NOT NULL,
    sentiment REAL,
    t0_close REAL,
    observe_end TEXT,            -- 观察窗口结束日（T+10 或数据末端）
    win_min REAL,                -- T+0~T+10 最低价
    win_min_date TEXT,           -- 最低点日期
    win_min_lag INTEGER,         -- 最低点距离信号日（交易日）
    win_max REAL,                -- T+0~T+10 最高价
    win_max_date TEXT,           -- 最高点日期
    win_max_lag INTEGER,         -- 最高点距离信号日（交易日）
    drop_t0_to_min REAL,         -- 信号日 → 最低点跌幅（负值）
    rebound_after_min REAL,      -- 最低点后最大反弹
    rise_t0_to_max REAL,         -- 信号日 → 最高点涨幅
    drawdown_after_max REAL,     -- 最高点后最大回撤（负值）
    hit INTEGER,                 -- 1 命中 / 0 未命中 / NULL 未确认
    FOREIGN KEY (run_id) REFERENCES swing_run(id)
);
"""


class Database:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        # 兼容旧库：sentiment_snapshot 缺 idx_ret_1d 列则补上
        cols = [r["name"] for r in self.conn.execute(
            "PRAGMA table_info(sentiment_snapshot)").fetchall()]
        if "idx_ret_1d" not in cols:
            self.conn.execute("ALTER TABLE sentiment_snapshot ADD COLUMN idx_ret_1d REAL")
            self.conn.commit()
        self.conn.commit()

    # ---------------- 情绪快照 ----------------
    def save_snapshot(self, row: dict):
        cols = ["date", "market_type", "sentiment_score", "sentiment_level",
                "daily_change", "change_3d", "change_5d", "change_10d",
                "percentile20", "percentile60", "percentile120", "percentile250",
                "is_extreme_panic", "is_extreme_hot", "trend", "pattern",
                "divergence", "data_status", "calculation_version", "idx_ret_1d"]
        vals = [row.get(c) for c in cols]
        self.conn.execute(
            f"INSERT OR REPLACE INTO sentiment_snapshot ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})", vals)
        self.conn.commit()

    def get_history(self, market_type: str, days: int | None = None,
                    start: str | None = None, end: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM sentiment_snapshot WHERE market_type=?"
        params: list = [market_type]
        if start:
            sql += " AND date>=?"
            params.append(start)
        if end:
            sql += " AND date<=?"
            params.append(end)
        sql += " ORDER BY date DESC"
        if days and not start:
            sql += f" LIMIT {int(days)}"
        return pd.read_sql(sql, self.conn, params=params)

    def latest_snapshot(self, market_type: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sentiment_snapshot WHERE market_type=? "
            "ORDER BY date DESC LIMIT 1", (market_type,)).fetchone()
        return dict(row) if row else None

    # ---------------- 模型版本 ----------------
    def save_model_version(self, market_type: str, version: str, params: dict,
                           train_period: str, oos_period: str,
                           train_stats: dict, oos_stats: dict,
                           is_active: bool = False) -> int:
        if is_active:
            self.conn.execute("UPDATE model_version SET is_active=0 WHERE market_type=?",
                              (market_type,))
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO model_version "
            "(market_type, version, params_json, train_period, oos_period, "
            " train_stats_json, oos_stats_json, is_active) VALUES (?,?,?,?,?,?,?,?)",
            (market_type, version, json.dumps(params, ensure_ascii=False),
             train_period, oos_period,
             json.dumps(train_stats, ensure_ascii=False),
             json.dumps(oos_stats, ensure_ascii=False), int(is_active)))
        self.conn.commit()
        return cur.lastrowid

    def next_version(self, market_type: str) -> str:
        row = self.conn.execute(
            "SELECT COUNT(*) c FROM model_version WHERE market_type=?",
            (market_type,)).fetchone()
        return f"V{row['c'] + 1}"

    def latest_version(self, market_type: str, active_only: bool = True) -> dict | None:
        sql = "SELECT * FROM model_version WHERE market_type=?"
        if active_only:
            sql += " AND is_active=1"
        sql += " ORDER BY id DESC LIMIT 1"
        row = self.conn.execute(sql, (market_type,)).fetchone()
        if row is None and active_only:
            row = self.conn.execute(
                "SELECT * FROM model_version WHERE market_type=? ORDER BY id DESC LIMIT 1",
                (market_type,)).fetchone()
        return dict(row) if row else None

    def list_versions(self, market_type: str) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM model_version WHERE market_type=? ORDER BY id DESC",
            self.conn, params=(market_type,))

    # ---------------- 回测 ----------------
    def save_backtest_run(self, market_type: str, version_id: int | None,
                          sample_tag: str, period: tuple[str, str],
                          params: dict, stats: dict, signals: list[dict]) -> int:
        cur = self.conn.execute(
            "INSERT INTO backtest_run (market_type, version_id, sample_tag, "
            "period_start, period_end, params_json, stats_json) VALUES (?,?,?,?,?,?,?)",
            (market_type, version_id, sample_tag, period[0], period[1],
             json.dumps(params, ensure_ascii=False),
             json.dumps(stats, ensure_ascii=False)))
        run_id = cur.lastrowid
        for s in signals:
            self.conn.execute(
                "INSERT INTO backtest_signal (run_id, market_type, signal_type, "
                "signal_date, sentiment, pivot_date, lag, max_rebound, "
                "max_drawdown, hit_t0, hit_t1, hit_t2, success) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, market_type, s["signal_type"], s["signal_date"],
                 s.get("sentiment"), s.get("pivot_date"), s.get("lag"),
                 s.get("max_rebound"), s.get("max_drawdown"),
                 int(s.get("hit_t0", 0)), int(s.get("hit_t1", 0)),
                 int(s.get("hit_t2", 0)),
                 None if s.get("success") is None else int(s["success"])))
        self.conn.commit()
        return run_id

    def list_backtest_runs(self, market_type: str) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM backtest_run WHERE market_type=? ORDER BY id DESC",
            self.conn, params=(market_type,))

    def get_run_signals(self, run_id: int) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM backtest_signal WHERE run_id=? ORDER BY signal_date",
            self.conn, params=(run_id,))

    def get_run(self, run_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM backtest_run WHERE id=?",
                                (run_id,)).fetchone()
        return dict(row) if row else None

    # ---------------- 波段信号模型 ----------------
    def save_swing_run(self, market_type: str, model_code: str,
                       sample_tag: str, period: tuple[str, str],
                       params: dict, stats: dict, signals: list[dict]) -> int:
        """保存一套波段模型的回测结果。

        同一 (市场, 模型, 样本) 只保留最新一次运行：重算结果是确定性的，
        旧 run 及其信号先删除，避免历史页标记重复。
        """
        old = self.conn.execute(
            "SELECT id FROM swing_run WHERE market_type=? AND model_code=? "
            "AND sample_tag=?", (market_type, model_code, sample_tag)).fetchall()
        for r in old:
            self.conn.execute("DELETE FROM swing_signal WHERE run_id=?", (r["id"],))
        self.conn.execute(
            "DELETE FROM swing_run WHERE market_type=? AND model_code=? "
            "AND sample_tag=?", (market_type, model_code, sample_tag))
        cur = self.conn.execute(
            "INSERT INTO swing_run (market_type, model_code, sample_tag, "
            "period_start, period_end, params_json, stats_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (market_type, model_code, sample_tag, period[0], period[1],
             json.dumps(params, ensure_ascii=False),
             json.dumps(stats, ensure_ascii=False)))
        run_id = cur.lastrowid
        for s in signals:
            self.conn.execute(
                "INSERT INTO swing_signal (run_id, market_type, model_code, "
                "signal_type, signal_date, sentiment, t0_close, observe_end, "
                "win_min, win_min_date, win_min_lag, win_max, win_max_date, "
                "win_max_lag, drop_t0_to_min, rebound_after_min, "
                "rise_t0_to_max, drawdown_after_max, hit) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, market_type, model_code, s["signal_type"],
                 s["signal_date"], s.get("sentiment"), s.get("t0_close"),
                 s.get("observe_end"), s.get("win_min"), s.get("win_min_date"),
                 s.get("win_min_lag"), s.get("win_max"), s.get("win_max_date"),
                 s.get("win_max_lag"), s.get("drop_t0_to_min"),
                 s.get("rebound_after_min"), s.get("rise_t0_to_max"),
                 s.get("drawdown_after_max"),
                 None if s.get("hit") is None else int(s["hit"])))
        self.conn.commit()
        return run_id

    def list_swing_runs(self, market_type: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM swing_run"
        params: list = []
        if market_type:
            sql += " WHERE market_type=?"
            params.append(market_type)
        sql += " ORDER BY market_type, model_code"
        return pd.read_sql(sql, self.conn, params=params)

    def get_swing_run(self, run_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM swing_run WHERE id=?",
                                (run_id,)).fetchone()
        return dict(row) if row else None

    def get_swing_run_signals(self, run_id: int) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM swing_signal WHERE run_id=? ORDER BY signal_date",
            self.conn, params=(run_id,))

    def get_swing_signals(self, market_type: str | None = None,
                          start: str | None = None,
                          end: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM swing_signal WHERE 1=1"
        params: list = []
        if market_type:
            sql += " AND market_type=?"
            params.append(market_type)
        if start:
            sql += " AND signal_date>=?"
            params.append(start)
        if end:
            sql += " AND signal_date<=?"
            params.append(end)
        sql += " ORDER BY signal_date"
        return pd.read_sql(sql, self.conn, params=params)

    def close(self):
        self.conn.close()
