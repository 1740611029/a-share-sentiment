#!/usr/bin/env bash
# ============================================================
#  A股情绪指标 - 每日更新（Linux / macOS / Git Bash）
#  用法：
#    ./scripts/daily_update.sh
#    ./scripts/daily_update.sh --git-push      # 更新后自动推送快照
#    ./scripts/daily_update.sh --skip-weekend  # 周末跳过（配合 cron 用）
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[ERROR] 未找到 Python，请先安装 Python 3.10+" >&2
    exit 1
fi

echo "项目目录: $ROOT"
exec "$PY" scripts/daily_update.py "$@"
