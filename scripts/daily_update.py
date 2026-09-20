"""每日更新脚本：拉取当日行情 → 计算三市场情绪快照 → 刷新波段信号。

与直接跑 `python run.py update` 的区别：
  1. 自带重试（网络/数据源偶发失败时不会一整天漏数据）；
  2. 日志落盘到 data/logs/，定时任务下也能事后排查；
  3. 单实例锁，避免上一次没跑完又被触发一次；
  4. 可选跑完自动 git commit + push（把当日快照同步到 GitHub）。

用法：
  python scripts/daily_update.py
  python scripts/daily_update.py --market CHINEXT          # 只更新创业板
  python scripts/daily_update.py --retries 3 --git-push    # 失败重试3次并推送
  python scripts/daily_update.py --skip-weekend            # 周六周日直接跳过
  python scripts/daily_update.py --dry-run                 # 只打印将要执行的命令
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "data" / "logs"
LOCK_FILE = LOG_DIR / "daily_update.lock"

# Windows 控制台默认 GBK，打印中文会 UnicodeEncodeError，统一改成 utf-8
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def log(msg: str, fh=None) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def pid_alive(pid: int) -> bool:
    """判断进程是否还活着（Windows 没有 os.kill(pid, 0) 语义，用 tasklist 兜底）。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace").stdout
            return str(pid) in out
        except Exception:  # noqa: BLE001
            return True  # 探测失败时保守认为还活着
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(fh=None) -> bool:
    """单实例锁：持有锁的进程已死（被 Ctrl+C / 任务管理器杀掉）时自动接管。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
        except Exception:  # noqa: BLE001
            pid = -1
        if pid != os.getpid() and pid_alive(pid):
            log(f"已有更新任务在运行（pid={pid}），本次跳过", fh)
            return False
        log("发现陈旧锁文件（原进程已退出），自动清理并接管", fh)
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def run_cmd(cmd: list[str], timeout: int | None, fh=None) -> tuple[int, str]:
    """执行命令，实时把输出同时打到控制台和日志文件。"""
    log("$ " + " ".join(cmd), fh)
    # 子进程输出是管道 → 默认块缓冲，长任务会"看起来卡住"，强制行缓冲
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    buf: list[str] = []
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            buf.append(line)
            print("  " + line, flush=True)
            if fh:
                fh.write("  " + line + "\n")
                fh.flush()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        log(f"命令超时（>{timeout}s），已终止", fh)
        return 124, "\n".join(buf)
    return proc.returncode, "\n".join(buf)


def git_sync(fh=None) -> bool:
    """把当日快照提交并推送（仅提交数据，代码改动不动）。"""
    def g(*args):
        return subprocess.run(["git", *args], cwd=str(ROOT),
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace")

    if g("rev-parse", "--is-inside-work-tree").returncode != 0:
        log("不是 git 仓库，跳过推送", fh)
        return False
    if g("diff", "--quiet", "--", "data/sentiment.db").returncode == 0:
        log("情绪快照无变化，无需提交", fh)
        return True

    today = datetime.now().strftime("%Y-%m-%d")
    steps = [
        ["add", "data/sentiment.db"],
        ["commit", "-m", f"chore(data): 更新 {today} 情绪快照"],
        ["push"],
    ]
    for args in steps:
        r = g(*args)
        if r.returncode != 0:
            log(f"git {' '.join(args)} 失败: {(r.stderr or r.stdout).strip()[:300]}", fh)
            return False
    log("已提交并推送当日快照到 GitHub", fh)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="A股情绪指标 - 每日更新")
    ap.add_argument("--market", nargs="*", default=None,
                    help="只更新指定市场（A_SHARE / CHINEXT / STAR），默认全部")
    ap.add_argument("--limit", type=int, default=None, help="限制股票数（冒烟测试用）")
    ap.add_argument("--config", default=None, help="自定义配置文件路径")
    ap.add_argument("--retries", type=int, default=2, help="失败重试次数（默认 2）")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="单次 update 超时秒数（默认 90 分钟）")
    ap.add_argument("--skip-weekend", action="store_true", help="周六周日直接跳过")
    ap.add_argument("--refresh-list", action="store_true", help="强制刷新股票列表")
    ap.add_argument("--git-push", action="store_true", help="更新成功后提交并推送快照")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"daily_update_{stamp}.log"

    if args.skip_weekend and datetime.now().weekday() >= 5:
        print("周末休市，跳过（--skip-weekend）")
        return 0

    cmd = [sys.executable, "run.py", "update"]
    if args.market:
        cmd += ["--market", *args.market]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.config:
        cmd += ["--config", args.config]
    if args.refresh_list:
        cmd.append("--refresh-list")

    print(f"A股情绪指标 - 每日更新  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"日志: {log_path}")
    if args.dry_run:
        print("[dry-run] " + " ".join(cmd))
        return 0

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 60}\n{datetime.now():%Y-%m-%d %H:%M:%S} 开始每日更新\n")
        if not acquire_lock(fh):
            return 0

        t0 = time.time()
        rc = 1
        for attempt in range(1, args.retries + 2):
            if attempt > 1:
                wait = 30 * (attempt - 1)
                log(f"第 {attempt - 1} 次重试前等待 {wait}s ...", fh)
                time.sleep(wait)
            rc, _ = run_cmd(cmd, args.timeout, fh)
            if rc == 0:
                break
            log(f"更新失败（exit={rc}），剩余重试 {args.retries - attempt + 1} 次", fh)

        elapsed = time.time() - t0
        if rc == 0:
            log(f"每日更新完成，用时 {elapsed / 60:.1f} 分钟", fh)
            if args.git_push:
                git_sync(fh)
        else:
            log(f"每日更新失败（exit={rc}），用时 {elapsed / 60:.1f} 分钟，详见日志", fh)
        release_lock()
        return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
