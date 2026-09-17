"""个股日线全量/增量更新（多线程，断点续传）"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .provider import DataProvider


def update_stock_history(provider: DataProvider, codes: list[str],
                         start: str, end: str, workers: int = 12,
                         progress: bool = True) -> dict:
    """对 codes 逐个做增量更新：已有缓存则只补缺失区间。

    返回 {"updated": n, "skipped": n, "failed": [code,...]}
    """
    updated, skipped, failed = 0, 0, []

    def job(code: str):
        cached = provider.load_stock_daily(code)
        need_start = start
        if cached is not None and not cached.empty:
            last = cached["date"].max()
            if (pd.Timestamp.now().normalize() - last).days <= 3:
                return code, "skip"
            need_start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        new = provider.fetch_stock_daily(code, need_start, end)
        if new is None or new.empty:
            if cached is not None:
                return code, "skip"
            return code, "fail"
        if cached is not None and not cached.empty:
            new = pd.concat([cached, new]).drop_duplicates("date").sort_values("date")
        new.reset_index(drop=True).to_parquet(provider.stock_cache_path(code))
        return code, "ok"

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, c): c for c in codes}
        for fut in as_completed(futs):
            code, status = fut.result()
            done += 1
            if status == "ok":
                updated += 1
            elif status == "skip":
                skipped += 1
            else:
                failed.append(code)
            if progress and done % 100 == 0:
                print(f"  进度 {done}/{len(codes)}  更新 {updated} 跳过 {skipped} "
                      f"失败 {len(failed)}  用时 {time.time() - t0:.0f}s")
    return {"updated": updated, "skipped": skipped, "failed": failed}
