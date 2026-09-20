# ============================================================
#  A股情绪指标 - 注册 Windows 计划任务（每天自动更新）
#
#  用法（PowerShell，普通用户权限即可，无需管理员）：
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -Time 19:00
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -GitPush
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -SkipWeekend
#    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -Uninstall
#
#  注册后在「任务计划程序」里能看到名为 A股情绪指标-每日更新 的任务。
#  注意：A股收盘 15:00，数据源日线通常在 15:30~17:00 后可用，默认 18:30 跑。
# ============================================================
param(
    [string]$Time = "18:30",
    [switch]$GitPush,
    [switch]$SkipWeekend,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$TaskName = "A股情绪指标-每日更新"
$Root = Split-Path -Parent $PSScriptRoot

if ($Uninstall) {
    schtasks /Delete /TN $TaskName /F | Out-Null
    Write-Host "已删除计划任务：$TaskName"
    exit 0
}

# ---------- 找 Python ----------
# 不能只依赖 Get-Command：某些环境（如被安全 shim 接管 PATH 的终端）里它查不到
# 明明存在的 py.exe，因此逐级回退到常见安装位置。
function Invoke-Python([string]$exe, [string]$preArgs, [string]$code) {
    $a = @()
    if ($preArgs) { $a += $preArgs }
    $a += @("-c", $code)
    return & $exe $a 2>$null
}

function Test-Python([string]$exe, [string]$preArgs) {
    if (-not (Test-Path $exe)) { return $null }
    $out = Invoke-Python $exe $preArgs "import sys; sys.stdout.write(sys.executable)"
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    return @{ Exe = $exe; Pre = $preArgs; Real = "$out".Trim() }
}

$candidates = @()
$pyCmd = Get-Command py -ErrorAction SilentlyContinue
if ($pyCmd) { $candidates += @{ Exe = $pyCmd.Source; Pre = "-3" } }
if (Test-Path (Join-Path $env:SystemRoot "py.exe")) {
    $candidates += @{ Exe = (Join-Path $env:SystemRoot "py.exe"); Pre = "-3" }
}
foreach ($c in @("python", "python3")) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { $candidates += @{ Exe = $cmd.Source; Pre = "" } }
}
# 用户级安装目录（Python 官方安装器默认位置）
$localPy = Join-Path $env:LOCALAPPDATA "Programs\Python"
if (Test-Path $localPy) {
    Get-ChildItem $localPy -Filter "python.exe" -Recurse -Depth 1 -ErrorAction SilentlyContinue |
        ForEach-Object { $candidates += @{ Exe = $_.FullName; Pre = "" } }
}

$chosen = $null
foreach ($c in $candidates) {
    $r = Test-Python $c.Exe $c.Pre
    if ($r) { $chosen = $r; break }
}
if (-not $chosen) {
    Write-Host "[ERROR] 未找到可用的 Python 3，请先安装 Python 3.10+" -ForegroundColor Red
    exit 1
}

# 校验实际解析到的解释器版本 >= 3.10（避开 PATH 里残留的 Python 2.7）
$ver = Invoke-Python $chosen.Exe $chosen.Pre "import sys;print('%d.%d'%sys.version_info[:2])"
if ($ver) { $ver = "$ver".Trim() }
if ($ver -and ([version]$ver -lt [version]"3.10")) {
    Write-Host "[ERROR] 解析到的 Python 版本过低：$ver（解释器 $($chosen.Real)），请改用 Python 3.10+" -ForegroundColor Red
    exit 1
}

# ---------- 组装命令 ----------
$script = Join-Path $Root "scripts\daily_update.py"
$extra = ""
if ($GitPush) { $extra += " --git-push" }
if ($SkipWeekend) { $extra += " --skip-weekend" }
$pre = if ($chosen.Pre) { "$($chosen.Pre) " } else { "" }
$tr = "`"$($chosen.Exe)`" $pre`"$script`"$extra"

schtasks /Create /TN $TaskName /TR $tr /SC DAILY /ST $Time /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 注册失败，请以当前用户重新运行" -ForegroundColor Red
    exit 1
}

Write-Host "已注册计划任务：$TaskName"
Write-Host "  触发时间：每天 $Time"
Write-Host "  执行命令：$tr"
Write-Host "  实际解释器：$($chosen.Real)（Python $ver）"
Write-Host "  项目目录：$Root"
Write-Host "  日志目录：$Root\data\logs"
Write-Host ""
Write-Host "手动立即跑一次验证：schtasks /Run /TN `"$TaskName`""
Write-Host "查看结果：            schtasks /Query /TN `"$TaskName`" /V /FO LIST"
Write-Host "卸载任务：            powershell -File scripts\install_daily_task.ps1 -Uninstall"
