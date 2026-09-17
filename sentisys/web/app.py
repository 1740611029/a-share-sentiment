"""Web 界面：实时首页 / 情绪历史 / 历史回测"""
from __future__ import annotations

import json

import pandas as pd
from flask import Flask, jsonify, render_template, request

from ..constants import MARKET_NAMES, Market
from ..storage import Database

LEVEL_COLORS = {
    "极度恐慌": "#1a7f37", "恐慌": "#2da44e", "偏冷": "#7bc96f",
    "中性": "#9e9e9e", "偏热": "#f0a35e", "过热": "#e8590c", "极度过热": "#d6336c",
}


def create_app(cfg: dict) -> Flask:
    app = Flask(__name__)
    db = Database(cfg["data"]["db_path"])

    # ---------------- 数据组装 ----------------
    def current_cards() -> list[dict]:
        cards = []
        for m in Market:
            snap = db.latest_snapshot(m.value)
            ver = db.latest_version(m.value)
            oos = None
            if ver and ver.get("oos_stats_json"):
                oos = json.loads(ver["oos_stats_json"])
            card = {"market": m.value, "name": MARKET_NAMES[m.value],
                    "version": ver["version"] if ver else "V1-default",
                    "oos": oos, "sentiment_score": None, "date": None,
                    "sentiment_level": None, "color": "#9e9e9e",
                    "daily_change": None, "change_3d": None, "change_5d": None,
                    "percentile20": None, "percentile60": None,
                    "percentile250": None, "trend": None, "pattern": None,
                    "divergence": None, "signal": "无"}
            if snap:
                card.update(snap)
                card["color"] = LEVEL_COLORS.get(snap["sentiment_level"], "#9e9e9e")
                if snap["is_extreme_panic"]:
                    card["signal"] = "极端恐慌"
                elif snap["is_extreme_hot"]:
                    card["signal"] = "极端过热"
                else:
                    card["signal"] = "无"
            cards.append(card)
        return cards

    def history_rows(market: str, days: int | None, start=None, end=None) -> pd.DataFrame:
        df = db.get_history(market, days=days, start=start, end=end)
        return df.sort_values("date") if not df.empty else df

    # ---------------- 页面 ----------------
    @app.route("/")
    def index():
        return render_template("index.html", cards=current_cards())

    @app.route("/history")
    def history():
        days = int(request.args.get("days", 60))
        series = {}
        for m in Market:
            df = history_rows(m.value, days)
            series[m.value] = df.to_dict("records")
        return render_template("history.html", days=days,
                               market_names=MARKET_NAMES, series=series)

    @app.route("/backtest")
    def backtest():
        market = request.args.get("market", Market.A_SHARE.value)
        versions = db.list_versions(market).to_dict("records")
        runs = db.list_backtest_runs(market).to_dict("records")
        run_id = request.args.get("run_id", type=int)
        if run_id is None and runs:
            run_id = runs[0]["id"]
        run, signals, stats = None, [], None
        if run_id:
            run = db.get_run(run_id)
            signals = db.get_run_signals(run_id).to_dict("records")
            stats = json.loads(run["stats_json"]) if run else None
            if run:
                run["params"] = json.loads(run["params_json"])
        return render_template("backtest.html", market=market,
                               market_names=MARKET_NAMES, versions=versions,
                               runs=runs, run=run, stats=stats, signals=signals)

    # ---------------- API ----------------
    @app.route("/api/current")
    def api_current():
        return jsonify(current_cards())

    @app.route("/api/history")
    def api_history():
        days = request.args.get("days", type=int)
        start, end = request.args.get("start"), request.args.get("end")
        out = {}
        for m in Market:
            out[m.value] = history_rows(m.value, days, start, end).to_dict("records")
        return jsonify(out)

    @app.route("/api/signal_detail")
    def api_signal_detail():
        """点击历史曲线极端信号：返回该信号在最近一次回测中的 T+0/1/2 表现。"""
        market = request.args.get("market")
        date = request.args.get("date")
        runs = db.list_backtest_runs(market)
        for _, r in runs.iterrows():
            sigs = db.get_run_signals(int(r["id"]))
            hit = sigs[sigs["signal_date"] == date]
            if not hit.empty:
                s = hit.iloc[0].to_dict()
                s["run_id"] = int(r["id"])
                s["sample_tag"] = r["sample_tag"]
                stats = json.loads(r["stats_json"])
                s["cum_hit_rate"] = stats.get(s["signal_type"], {}).get("hit_rate")
                return jsonify(s)
        return jsonify({"signal_date": date, "note": "该信号不在已有回测记录中"})

    return app
