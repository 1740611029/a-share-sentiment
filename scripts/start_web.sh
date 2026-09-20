#!/usr/bin/env bash
# ============================================================
#  A股情绪指标 - 启动 Web 界面（前台运行，Ctrl+C 停止）
#  用法：
#    ./scripts/start_web.sh        # 端口 5000
#    ./scripts/start_web.sh 5001   # 指定端口
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

PORT="${1:-5000}"
echo "启动 Web 界面: http://127.0.0.1:${PORT}"
exec "$PY" run.py serve --port "$PORT"
