# ============================================================
#  A股情绪指标 - 注册 Windows 计划任务（每个交易日自动更新）
#
#  用法（PowerShell，普通用户权限即可，无需管理员）：
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -Time 19:00
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -GitPush
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -Uninstall
#
#  注册后在「任务计划程序」里能看到名为 A股情绪指标-每日更新 的任务。
#  注意：A股收盘 15:00，数据源日线通常在 15:30~17:00 后可用，默认 18:30 跑。
# ============================================================
param(
    [string]$Time = "18:30",
    [switch]$GitPush,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$TaskName = "A股情绪指标-每日更新"
$Root = Split-Path -Parent $PSScriptRoot

if ($Uninstall) {
    schtasks /Delete /TN $TaskName /F
    Write-Host "已删除计划任务：$TaskName"
    exit 0
}

# 找 Python：优先 py 启动器（可避开 PATH 里残留的 Python 2）
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $py = (Get-Command py).Source
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $py = (Get-Command python).Source
}
if (-not $py) {
    Write-Host "[ERROR] 未找到 Python，请先安装 Python 3.10+" -ForegroundColor Red
    exit 1
}

$script = Join-Path $Root "scripts\daily_update.py"
$extra = ""
if ($GitPush) { $extra = " --git-push" }
$tr = "`"$py`" `"$script`"$extra"

schtasks /Create /TN $TaskName /TR $tr /SC DAILY /ST $Time /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 注册失败，请以当前用户重新运行" -ForegroundColor Red
    exit 1
}

Write-Host "已注册计划任务：$TaskName"
Write-Host "  触发时间：每天 $Time"
Write-Host "  执行命令：$tr"
Write-Host "  工作目录：$Root"
Write-Host "  日志目录：$Root\data\logs"
Write-Host ""
Write-Host "手动立即跑一次验证：schtasks /Run /TN `"$TaskName`""
Write-Host "查看结果：            schtasks /Query /TN `"$TaskName`" /V /FO LIST"
Write-Host "卸载任务：            powershell -File scripts\install_daily_task.ps1 -Uninstall"
