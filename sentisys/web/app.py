"""Web 界面：情绪概览 / 情绪历史 / 波段信号 / 模型回测 / 模型对比 / 情绪回测 / 术语对照表"""
from __future__ import annotations

import json

import pandas as pd
from flask import Flask, jsonify, render_template, request
from markupsafe import Markup, escape

from ..constants import MARKET_NAMES, Market
from ..storage import Database
from ..swing.models import MODELS

LEVEL_COLORS = {
    "极度恐慌": "#34D399", "恐慌": "#4ADE80", "偏冷": "#86EFAC",
    "中性": "#94A3B8", "偏热": "#FBBF24", "过热": "#FB923C", "极度过热": "#F87171",
}
LEVEL_ORDER = ["极度恐慌", "恐慌", "偏冷", "中性", "偏热", "过热", "极度过热"]

# ============================================================
# 术语大白话词典
# 模板里写 {{ tip('命中率') }}，页面上就会出现带虚线下划线的词，
# 鼠标放上去（手机上点一下）弹出这句人话解释。
# 新增术语只要往这里加一条即可，无需改模板结构。
# ============================================================
GLOSSARY: dict[str, str] = {
    # ---------------- 情绪分怎么读 ----------------
    "情绪分": "给市场量体温的分数，0~100。0 分＝过去一年里最冷的一天，100 分＝最热的一天。",
    "分位": "「过去这么多个交易日里，有百分之多少的日子比现在更凉」。数字越大越热，越小越冷。",
    "20日分位": "最近一个月里，有多少比例的日子比现在冷。数字越大＝近期越热。",
    "60日分位": "最近三个月里，有多少比例的日子比现在冷。数字越大＝近期越热。",
    "250日分位": "最近一年里，有多少比例的日子比现在冷。数字越大＝当前越接近一年中最热的位置。",
    "滚动分位": "拿今天和过去固定一段时间的每一天逐一比较，看今天排第几。这样分数不会随时间推移而失真。",
    "日变化": "今天的情绪分比上一个交易日涨跌了几分。正数＝情绪在回暖。",
    "趋势": "日变化超过 ±2 分才显示箭头，小波动一律算平稳，避免把噪声误读成趋势。",
    "参考指数": "判断这个市场冷热时参照的大盘：A股看上证指数、创业板看创业板指、科创板看科创50。后面跟着的是它当天的涨跌幅。",
    "恐慌修复": "情绪连续 3 天回升，而且前几天还待在恐慌区（分数低于 25）。相当于「刚从坑里往上爬」。",
    "高位降温": "情绪连续 3 天回落，而且前几天还待在过热区（分数高于 75）。相当于「热劲正在退」。",
    "底背离": "指数在跌，情绪反而在改善。潜台词是「跌势没那么凶了」。",
    "顶背离": "指数在涨，情绪反而在恶化。潜台词是「涨得有点虚」。",

    # ---------------- 极端信号 ----------------
    "极端恐慌": "比「恐慌」更严格的一档：分数跌破这个市场自己的阈值才算，例如 A股 当前是 ≤3。",
    "极端过热": "比「过热」更严格的一档：分数涨破这个市场自己的阈值才算，例如 A股 当前是 ≥94。",
    "极端阈值": "每个市场单独调出来的分数线，分数越过它才算「极端」。概览页卡片下方的刻度线会标出当前生效的值。",
    "极端区域状态": "当前分数有没有越过极端阈值。越过了才会启动后面的观察窗口。",

    # ---------------- 顶底与回测 ----------------
    "阶段底部": "系统定义的「局部最低点」：某天是前后一段时间的低位，而且之后确实反弹了至少 3%。",
    "阶段顶部": "系统定义的「局部最高点」：某天是前后一段时间的高位，而且之后确实回落了至少 3%。",
    "阶段低/高点": "系统定义的局部最低点／最高点。判定标准：是前后一段时间的极值，且之后确实反向走了至少 3%。",
    "反弹确认幅度": "判定阶段底部要求的最低反弹幅度，目前固定 3%。这个数字写死、不参与调参——相当于考试的判卷标准，一旦允许它自动调，成绩就会注水。",
    "位置容差": "判定顶/底时允许的偏差范围，目前固定 2%。同样写死、不参与调参。",
    "确认窗口": "信号发出后盯多少个交易日来判断成败。默认 10 个交易日（约两周）。",
    "命中率": "发出信号后真的出现了阶段底部（或顶部）的比例。90% ＝ 十次里对九次。注意：这个数字必须和「瞎猜基线」比才有意义。",
    "瞎猜基线": "随便挑一天，往后几天内撞上「阶段底部」标签的概率。信号命中率减去它，才是信号的真实价值。",
    "真实优势": "信号命中率 − 瞎猜基线。大于 0 才说明信号带着有效信息；接近 0 或为负，就说明这个信号没用。",
    "p值": "「这个成绩是运气好」的概率。p<0.05 才算站得住 —— 意思是纯靠运气也能拿到这个成绩的可能性不到 5%。数字越大越不可信。",
    "95%置信区间": "真实命中率大概落在什么范围。比如 [44%, 84%] 表示「真实水平很可能在这个区间里」。区间很宽就说明样本还太少，说不准。",
    "统计显著性": "同时满足两点才算：① 命中率明显高于瞎猜基线；② 样本够多，p<0.05。只满足①不满足②，说明还区分不了「有效」和「运气好」。",
    "前瞻收益": "信号发出后 1/3/5/10 个交易日，市场平均涨跌了多少。比「中/没中」信息量大得多 —— 同样 11 次信号，「对了几次」只有两种取值，而「涨了百分之几」能区分出「小赚」和「大赚」。样本少时尤其有用。",
    "置换检验": "判断「信号后的走势是不是运气」的方法：从全部交易日里随机抽同样多的天数，重复两万次，看有多少次的平均涨幅能达到信号组的水平。达到的比例就是 p 值，越小越不像运气。",
    "双尾检验": "同时检验「显著好于平时」和「显著差于平时」两种可能。比单边检验保守，但不会漏掉「信号其实是反的」这种情况。",
    "样本外": "最近一年被封存起来、完全不参与调参的数据。拿这部分检验相当于「模拟考」，成绩才可信。",
    "训练期": "用来调参的那段历史。在它上面拿到的成绩相当于「练习册」上的成绩，天然好看。",
    "样本数 n": "一共发出过多少次信号。n 太小百分比就没意义——2 次里中 2 次也是 100%，不能当真。",
    "信号总数": "一共发出过多少次信号。样本太少时百分比会剧烈波动，请先看这个数。",
    "T+0": "信号发出的当天。",
    "T+1": "信号发出后的第 2 个交易日。",
    "T+2": "信号发出后的第 3 个交易日。",
    "T+0~T+2 累计": "这三天里任意一天命中就算成功，所以累计命中率一定大于等于单天命中率。",
    "偏差": "真正的最低点（或最高点）比信号晚几天才出现。偏差 1 ＝ 信号发出后第二天才见底。",
    "最大反弹": "信号发出后价格最多涨了多少。方向对了，也要看能赚出多少空间。",
    "最大回撤": "信号发出后价格最多跌了多少。用来看「就算方向对了，中间得扛多大波动」。",
    "信号覆盖率": "信号有多稀有 ＝ 信号次数 ÷ 总交易日数。覆盖率低说明信号很少出现。",
    "最大连续失败": "历史上最长连续失手几次。这个数字往往比命中率更值得看——命中率 90% 但连续失手 5 次，实际也很难扛。",
    "未确认": "信号发出后观察期还没走完，结果未知，所以不计入命中率。数据越新，未确认的信号越多。",
    "回测区间": "这段统计覆盖的起止日期。区间越长，结论越稳。",
    "运行编号": "每跑一次回测都会生成一个编号，方便对照不同参数下的结果。旧记录不会被覆盖。",

    # ---------------- 波段模型 ----------------
    "波段信号": "在「情绪进入极端区域」的基础上再叠加别的条件筛出来的可能转折点。它是历史统计工具，不给任何买卖建议。",
    "观察窗口": "信号发出后盯 10 个交易日，看这段时间里有没有形成阶段低点（或高点）。",
    "当前进度": "观察窗口走到第几天了，以及是否已经能判定命中。",
    "信号日": "触发模型的那一天，T+0 就从这天算起。",
    "窗口低点": "观察窗口这 10 天里的最低价，以及它出现在第几天。",
    "窗口高点": "观察窗口这 10 天里的最高价，以及它出现在第几天。",
    "平均低点时间": "平均在信号后第几个交易日见到最低点，写成 T+3.2 这样。数字越小＝信号来得越及时。",
    "盈亏比": "平均能涨的幅度 ÷ 平均要先承受的跌幅。大于 1 表示赚的空间比亏的空间大。",
    "市场状态": "按指数与 250 日均线的偏离度划分：高出 3% 以上算牛市、低 3% 以上算熊市、中间算震荡。用来看模型在不同行情下稳不稳。",
    "模型回测": "把某一套筛选条件单独拎出来，统计它历史上所有信号的表现。",
}


def _tip(term: str):
    """把术语包成带悬浮说明的 span；词典里没收录的原样返回，不会报错。"""
    text = GLOSSARY.get(term)
    if not text:
        return term
    return Markup(
        f'<span class="term" tabindex="0" data-tip="{escape(text)}">{escape(term)}</span>'
    )


def create_app(cfg: dict) -> Flask:
    app = Flask(__name__)
    db = Database(cfg["data"]["db_path"])
    # 模板里可直接调用 {{ tip('命中率') }}
    app.jinja_env.globals["tip"] = _tip

    # ---------------- 数据组装 ----------------
    def _enrich_oos(oos: dict | None, market: str) -> dict | None:
        """给模型版本里存的样本外统计补上"对照与显著性"字段。

        早期生成的版本（如本次口径修正前跑的）没有 baseline / p_value / verdict，
        这里从最新一条 sample_tag=oos 的回测记录里补，避免重跑耗时的 optimize。
        """
        if not oos:
            return oos
        keys = ("baseline", "lift", "p_value", "p2", "ci_low", "ci_high",
                "verdict", "verdict_detail", "fwd")
        need = any(oos.get(k, {}).get("verdict") is None
                   for k in ("panic", "hot") if k in oos)
        if not need:
            return oos
        for r in db.list_backtest_runs(market).to_dict("records"):
            if r.get("sample_tag") != "oos":
                continue
            try:
                st = json.loads(r["stats_json"])
            except (ValueError, TypeError):
                continue
            for k in ("panic", "hot"):
                if k in oos and k in st:
                    for f in keys:
                        oos[k].setdefault(f, st[k].get(f))
            break
        return oos

    def current_cards() -> list[dict]:
        cards = []
        for m in Market:
            snap = db.latest_snapshot(m.value)
            ver = db.latest_version(m.value)
            oos = None
            panic_t, hot_t = cfg["sentiment"]["extreme_panic"], cfg["sentiment"]["extreme_hot"]
            if ver:
                if ver.get("oos_stats_json"):
                    oos = _enrich_oos(json.loads(ver["oos_stats_json"]), m.value)
                try:
                    p = json.loads(ver["params_json"])
                    panic_t = p.get("extreme_panic", panic_t)
                    hot_t = p.get("extreme_hot", hot_t)
                except (ValueError, TypeError):
                    pass
            card = {"market": m.value, "name": MARKET_NAMES[m.value],
                    "version": ver["version"] if ver else "V1-default",
                    "oos": oos, "panic_t": panic_t, "hot_t": hot_t,
                    "sentiment_score": None, "date": None,
                    "sentiment_level": None, "color": "#94A3B8",
                    "daily_change": None, "change_3d": None, "change_5d": None,
                    "percentile20": None, "percentile60": None,
                    "percentile250": None, "trend": None, "pattern": None,
                    "divergence": None, "signal": "无"}
            if snap:
                card.update(snap)
                card["color"] = LEVEL_COLORS.get(snap["sentiment_level"], "#94A3B8")
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

    # ---------------- 波段信号数据组装 ----------------
    def swing_marks(market: str, start: str | None, end: str | None) -> list[dict]:
        """某市场日期区间内的波段信号，按 (日期,方向) 聚合触发模型。"""
        df = db.get_swing_signals(market, start=start, end=end)
        if df.empty:
            return []
        out = []
        for (d, st), g in df.groupby(["signal_date", "signal_type"]):
            out.append({
                "date": d, "signal_type": st,
                "models": sorted(g["model_code"].unique().tolist()),
                "hits": {r["model_code"]: (None if pd.isna(r["hit"]) else int(r["hit"]))
                         for _, r in g.iterrows()},
            })
        return sorted(out, key=lambda x: x["date"])

    def swing_live() -> list[dict]:
        """实时波段信号页：三市场当前情绪 + 各模型触发状态 + 观察中的信号。"""
        cards = []
        for m in Market:
            snap = db.latest_snapshot(m.value)
            latest_date = snap["date"] if snap else None
            # 极端阈值取激活模型版本，与概览页口径一致
            def _num(v, default):
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return default
                return int(f) if f.is_integer() else f

            panic_t = _num(cfg["sentiment"]["extreme_panic"], 10)
            hot_t = _num(cfg["sentiment"]["extreme_hot"], 90)
            ver = db.latest_version(m.value)
            if ver:
                try:
                    p = json.loads(ver["params_json"])
                    panic_t = _num(p.get("extreme_panic"), panic_t)
                    hot_t = _num(p.get("extreme_hot"), hot_t)
                except (ValueError, TypeError):
                    pass
            models = []
            for code, meta in MODELS.items():
                info = {"code": code, "name": meta["name"], "status": "未触发",
                        "signal": None}
                if latest_date:
                    df = db.get_swing_signals(m.value, start=latest_date)
                    df = df[df["model_code"] == code]
                    if not df.empty:
                        info["status"] = "已触发"
                        sig = df.iloc[0].to_dict()
                        info["signal"] = {k: (None if pd.isna(v) else v)
                                          for k, v in sig.items()}
                models.append(info)
            # 观察中 / 最近 30 个自然日内的信号（生命周期）
            recent = []
            if latest_date:
                since = (pd.Timestamp(latest_date)
                         - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
                df = db.get_swing_signals(m.value, start=since)
                if not df.empty:  # full/oos 两套样本并存时去重
                    df = df.drop_duplicates(
                        subset=["model_code", "signal_date", "signal_type"])
                    df = df.where(pd.notna(df), None)
                recent = df.to_dict("records")
            cards.append({"market": m.value, "name": MARKET_NAMES[m.value],
                          "snap": snap, "models": models, "recent": recent,
                          "panic_t": panic_t, "hot_t": hot_t,
                          "version": ver["version"] if ver else "V1-default",
                          "color": LEVEL_COLORS.get(
                              snap["sentiment_level"], "#94A3B8") if snap else "#94A3B8"})
        return cards

    def swing_run_view(run: dict | None) -> dict | None:
        if not run:
            return None
        run = dict(run)
        run["stats"] = json.loads(run["stats_json"])
        run["params"] = json.loads(run["params_json"])
        return run

    # ---------------- 顶栏状态注入（base.html 使用） ----------------
    @app.context_processor
    def inject_nav_meta():
        """提供各市场最新分数 + 数据日期与滞后天数，供顶栏状态条展示。"""
        today = pd.Timestamp.now().normalize()
        markets, latest = {}, None
        for m in Market:
            snap = db.latest_snapshot(m.value)
            if not snap:
                continue
            markets[m.value] = {
                "name": MARKET_NAMES[m.value],
                "date": snap["date"],
                "score": snap["sentiment_score"],
                "level": snap["sentiment_level"],
                "color": LEVEL_COLORS.get(snap["sentiment_level"], "#94A3B8"),
                "abnormal": snap.get("data_status") == "abnormal",
            }
            d = pd.Timestamp(snap["date"])
            if latest is None or d > latest:
                latest = d
        return {"nav_meta": {
            "today": today.strftime("%Y-%m-%d"),
            "latest": latest.strftime("%Y-%m-%d") if latest is not None else None,
            "lag": int((today - latest).days) if latest is not None else None,
            "markets": markets,
        }}

    # ---------------- 页面 ----------------
    @app.route("/")
    def index():
        return render_template("index.html", cards=current_cards())

    @app.route("/history")
    def history():
        days = int(request.args.get("days", 60))
        series, marks = {}, {}
        for m in Market:
            df = history_rows(m.value, days)
            series[m.value] = df.to_dict("records")
            start = df["date"].min() if not df.empty else None
            marks[m.value] = swing_marks(m.value, start, None)
        return render_template("history.html", days=days,
                               market_names=MARKET_NAMES, series=series,
                               swing_marks=marks)

    @app.route("/swing")
    def swing():
        return render_template("swing.html", cards=swing_live(), models=MODELS)

    @app.route("/swing/backtest")
    def swing_backtest():
        market = request.args.get("market", Market.A_SHARE.value)
        model = request.args.get("model", "A")
        sample = request.args.get("sample", "full")
        runs = {r["model_code"]: r
                for r in db.list_swing_runs(market).to_dict("records")
                if r["sample_tag"] == sample}
        if not runs and sample != "full":
            sample = "full"
            runs = {r["model_code"]: r
                    for r in db.list_swing_runs(market).to_dict("records")
                    if r["sample_tag"] == "full"}
        run = swing_run_view(runs.get(model))
        signals = []
        if run:
            df = db.get_swing_run_signals(run["id"])
            signals = df.where(pd.notna(df), None).to_dict("records")
        return render_template("swing_backtest.html", market=market,
                               market_names=MARKET_NAMES, models=MODELS,
                               model=model, sample=sample, run=run,
                               signals=signals)

    @app.route("/swing/compare")
    def swing_compare():
        rows = []
        seen = set()
        for r in sorted(db.list_swing_runs().to_dict("records"),
                        key=lambda x: 0 if x["sample_tag"] == "full" else 1):
            key = (r["market_type"], r["model_code"])
            if key in seen:      # 同一市场同一模型只展示一份（优先 full 样本）
                continue
            seen.add(key)
            stats = json.loads(r["stats_json"])
            rows.append({"market": r["market_type"],
                         "market_name": MARKET_NAMES.get(r["market_type"], r["market_type"]),
                         "model_code": r["model_code"],
                         "model_name": MODELS.get(r["model_code"], {}).get("name", r["model_code"]),
                         "period": f"{r['period_start']} ~ {r['period_end']}",
                         "panic": stats["panic"], "hot": stats["hot"]})
        order = {m.value: i for i, m in enumerate(Market)}
        rows.sort(key=lambda x: (order.get(x["market"], 9), x["model_code"]))
        return render_template("swing_compare.html", rows=rows)

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

    @app.route("/glossary")
    def glossary():
        return render_template("glossary.html", market_names=MARKET_NAMES)

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

    @app.route("/api/swing_detail")
    def api_swing_detail():
        """点击历史曲线波段信号标记：返回该日各模型的信号明细。"""
        market = request.args.get("market")
        date = request.args.get("date")
        df = db.get_swing_signals(market, start=date, end=date)
        if df.empty:
            return jsonify({"signal_date": date,
                            "note": "该日期没有波段信号记录（请先运行 python run.py swing）"})
        df = df.drop_duplicates(subset=["model_code", "signal_type"])
        df = df.where(pd.notna(df), None)
        runs = {}
        for r in sorted(db.list_swing_runs(market).to_dict("records"),
                        key=lambda x: 0 if x["sample_tag"] == "full" else 1):
            runs.setdefault(r["model_code"], r)
        signals = []
        for s in df.to_dict("records"):
            stats = json.loads(runs[s["model_code"]]["stats_json"]) \
                if s["model_code"] in runs else {}
            s["model_name"] = MODELS.get(s["model_code"], {}).get("name", s["model_code"])
            s["cum_hit_rate"] = stats.get(s["signal_type"], {}).get("hit_rate")
            signals.append(s)
        return jsonify({"signal_date": date, "signals": signals})

    return app
