"""分片下载个股日线（多进程并行用，每进程处理不重叠的一片）。

用法：
  python scripts/download_shard.py --shard 0 --n 6 --source sina
  python scripts/download_shard.py --shard 5 --n 6 --source tx --workers 8

新浪源(py_mini_racer)非线程安全：进程内单线程，靠多进程提速。
腾讯源纯HTTP：进程内可多线程。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentisys.config import load_config          # noqa: E402
from sentisys.constants import Market             # noqa: E402
from sentisys.data import DataProvider, build_universe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--source", choices=["sina", "tx"], default="sina")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    provider = DataProvider(cfg)
    stock_list = provider.get_stock_list()

    codes, seen = [], set()
    for m in Market:  # 三个市场并集，去重
        u = build_universe(stock_list, m, cfg)
        for c in u["code"]:
            if c not in seen:
                seen.add(c)
                codes.append(c)

    years = int(cfg["data"]["history_years"])
    import pandas as pd
    start = (pd.Timestamp.now() - pd.Timedelta(days=years * 365)).strftime("%Y%m%d")
    end = pd.Timestamp.now().strftime("%Y%m%d")

    # 只处理缓存缺失或不完整的股票
    todo = []
    for c in codes:
        cached = provider.load_stock_daily(c)
        if cached is None or cached.empty:
            todo.append((c, start))
        else:
            last = cached["date"].max()
            if (pd.Timestamp.now().normalize() - last).days > 3:
                todo.append((c, (last + pd.Timedelta(days=1)).strftime("%Y%m%d")))
    todo = sorted(set(todo))
    shard = [t for i, t in enumerate(todo) if i % args.n == args.shard]
    print(f"[shard {args.shard}/{args.n} source={args.source}] "
          f"待下载 {len(shard)} / 总缺口 {len(todo)}", flush=True)

    ok, fail = 0, 0
    t0 = time.time()

    def one(item):
        nonlocal ok, fail
        code, need_start = item
        df = provider.fetch_stock_daily(code, need_start, end, source=args.source)
        if df is None or df.empty:
            fail += 1
            return
        cached = provider.load_stock_daily(code)
        if cached is not None and not cached.empty:
            df = pd.concat([cached, df]).drop_duplicates("date").sort_values("date")
        df.reset_index(drop=True).to_parquet(provider.stock_cache_path(code))
        ok += 1

    if args.source == "tx" and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(one, shard))
    else:
        for item in shard:
            one(item)
            done = ok + fail
            if done % 50 == 0:
                print(f"[shard {args.shard}] {done}/{len(shard)} "
                      f"用时 {time.time() - t0:.0f}s", flush=True)
    print(f"[shard {args.shard}] 完成 ok={ok} fail={fail} "
          f"用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
