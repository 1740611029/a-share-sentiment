@echo off
chcp 65001 >nul
REM ============================================================
REM  A股情绪指标 - 启动 Web 界面（前台运行，关闭窗口或 Ctrl+C 停止）
REM  默认端口 5000，改端口：scripts\start_web.bat 5001
REM  首次使用请先跑一次 scripts\daily_update.bat
REM ============================================================
cd /d "%~dp0.."

set "PY_CMD="
where py >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD (
    where python >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+ 并勾选 Add to PATH
    pause
    exit /b 1
)

if "%1"=="" (set "PORT=5000") else (set "PORT=%1")
echo 启动 Web 界面: http://127.0.0.1:%PORT%
echo （关闭本窗口或按 Ctrl+C 停止服务）
%PY_CMD% run.py serve --port %PORT%
pause
