"""A股三市场情绪指标系统 CLI

用法：
  python run.py update    [--market A_SHARE CHINEXT STAR] [--limit N] [--refresh-list]
  python run.py optimize  [--market ...]
  python run.py backtest  [--market ...] [--sample full|train|oos]
  python run.py swing     [--market ...] [--model A B C D E] [--sample full|oos]
  python run.py serve     [--port 5000]
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from sentisys.config import load_config
from sentisys.constants import MARKET_NAMES, Market
from sentisys.data import DataProvider
from sentisys.factors import load_factors
from sentisys.pipeline import compute_and_store, update_market_data
from sentisys.storage import Database

ALL_MARKETS = [m.value for m in Market]


def _markets(args) -> list[str]:
    return args.market or ALL_MARKETS


def cmd_update(args):
    cfg = load_config(args.config)
    provider = DataProvider(cfg)
    db = Database(cfg["data"]["db_path"])
    for m in _markets(args):
        stats = update_market_data(cfg, provider, m, limit=args.limit,
                                   refresh_list=args.refresh_list)
        print(f"[{m}] 日线更新完成: {stats}")
        summary = compute_and_store(cfg, db, provider, m)
        print(f"[{m}] {MARKET_NAMES[m]} {summary['date']} "
              f"情绪 {summary['score']}（{summary['level']}）[{summary['version']}]")
        _update_swing(cfg, db, provider, m)


def _update_swing(cfg, db, provider, market: str):
    """增量刷新波段信号（独立模块；失败不影响原情绪指标流程）。"""
    try:
        from sentisys.swing import run_swing
        out = run_swing(cfg, db, provider, market, sample="full")
        n = sum(out[c]["panic"]["signals"] + out[c]["hot"]["signals"]
                for c in out)
        print(f"[{market}] 波段信号已刷新（{len(out)} 套模型，{n} 个已确认信号）")
    except Exception as e:  # noqa: BLE001
        print(f"[{market}] 波段信号刷新失败（不影响情绪指标）: {e}")


def cmd_swing(args):
    from sentisys.swing import run_swing
    cfg = load_config(args.config)
    provider = DataProvider(cfg)
    db = Database(cfg["data"]["db_path"])
    for m in _markets(args):
        out = run_swing(cfg, db, provider, m, sample=args.sample,
                        models=args.model)
        for code, stats in out.items():
            p, h = stats["panic"], stats["hot"]
            fmt = lambda b: (f"{b['hit_rate'] * 100:.1f}%" if b["hit_rate"] is not None else "—")
            print(f"[{m}] 模型{code} 底部命中率 {fmt(p)} (n={p['signals']}) "
                  f"顶部命中率 {fmt(h)} (n={h['signals']})")


def cmd_optimize(args):
    from sentisys.optimize import optimize_market
    cfg = load_config(args.config)
    provider = DataProvider(cfg)
    db = Database(cfg["data"]["db_path"])
    from sentisys.engine import compute_subscores
    for m in _markets(args):
        factors = load_factors(cfg, m)
        subscores = compute_subscores(factors, cfg)
        primary = cfg["markets"][m]["primary_index"]
        close = provider.load_index_daily(primary).set_index("date")["close"]
        result = optimize_market(cfg, db, m, subscores, close)
        print(json.dumps({"market": m, "version": result["version"],
                          "params": result["params"]}, ensure_ascii=False, indent=2))


def cmd_backtest(args):
    from sentisys.backtest import run_backtest
    from sentisys.engine import compute_score, compute_subscores
    cfg = load_config(args.config)
    provider = DataProvider(cfg)
    db = Database(cfg["data"]["db_path"])
    bt = cfg["backtest"]
    for m in _markets(args):
        ver = db.latest_version(m)
        if ver:
            params = json.loads(ver["params_json"])
            version_id, vname = ver["id"], ver["version"]
        else:
            params = {**bt, "weights": cfg["sentiment"]["weights"],
                      "extreme_panic": cfg["sentiment"]["extreme_panic"],
                      "extreme_hot": cfg["sentiment"]["extreme_hot"]}
            version_id, vname = None, "V1-default"
        factors = load_factors(cfg, m)
        subscores = compute_subscores(factors, cfg)
        score = compute_score(subscores, params.get("weights", cfg["sentiment"]["weights"]), cfg)
        primary = cfg["markets"][m]["primary_index"]
        close = provider.load_index_daily(primary).set_index("date")["close"]

        start = end = None
        if args.sample in ("train", "oos"):
            valid = score.dropna().index
            oos_start = valid[-1] - pd.Timedelta(days=int(bt["train_exclude_recent_days"]))
            if args.sample == "train":
                end = (oos_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start = oos_start.strftime("%Y-%m-%d")
        res = run_backtest(score, close, params, start=start, end=end)
        period = (start or score.dropna().index[0].strftime("%Y-%m-%d"),
                  end or score.dropna().index[-1].strftime("%Y-%m-%d"))
        run_id = db.save_backtest_run(m, version_id, args.sample, period,
                                      params, res["stats"], res["signals"])
        print(f"[{m}] 回测完成 run_id={run_id} 版本={vname} 区间={period[0]}~{period[1]}")
        print(json.dumps(res["stats"], ensure_ascii=False, indent=2))


def cmd_serve(args):
    from sentisys.web.app import create_app
    cfg = load_config(args.config)
    app = create_app(cfg)
    app.run(host="127.0.0.1", port=args.port, debug=True, use_reloader=True)


def main():
    ap = argparse.ArgumentParser(description="A股三市场情绪指标系统")
    ap.add_argument("--config", default=None, help="用户配置文件（覆盖 default.yaml）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("update", help="更新数据并计算当日情绪快照")
    p.add_argument("--market", nargs="*", choices=ALL_MARKETS)
    p.add_argument("--limit", type=int, default=None, help="限制股票数（冒烟测试用）")
    p.add_argument("--refresh-list", action="store_true", help="强制刷新股票列表")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("optimize", help="参数自动优化（训练期隔离 + 样本外验证）")
    p.add_argument("--market", nargs="*", choices=ALL_MARKETS)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("backtest", help="使用当前激活版本参数执行历史回测")
    p.add_argument("--market", nargs="*", choices=ALL_MARKETS)
    p.add_argument("--sample", choices=["full", "train", "oos"], default="full")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("swing", help="波段信号模型回测（T+0~T+10 阶段低点/高点）")
    p.add_argument("--market", nargs="*", choices=ALL_MARKETS)
    p.add_argument("--model", nargs="*", choices=list("ABCDE"),
                   default=None, help="只运行指定模型，默认全部")
    p.add_argument("--sample", choices=["full", "oos"], default="full")
    p.set_defaults(func=cmd_swing)

    p = sub.add_parser("serve", help="启动 Web 界面")
    p.add_argument("--port", type=int, default=5000)
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
