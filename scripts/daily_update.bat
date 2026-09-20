@echo off
chcp 65001 >nul
REM ============================================================
REM  A股情绪指标 - 每日更新（Windows 双击即可运行）
REM  等价于：python scripts/daily_update.py
REM  可追加参数，例如在快捷方式里写：
REM    scripts\daily_update.bat --git-push
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

echo 项目目录: %CD%
%PY_CMD% scripts\daily_update.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] 更新失败，请查看上方输出或 data\logs\ 下的日志
    pause
    exit /b 1
)
