# train.ps1 — 福袋模型一键训练（除采集外的全部训练）
#
# 一条命令完成：
#   1) 检测模型  data\fudai\model\fudai_clf.npz
#      样本数 < -ColdStartBelow（默认 30）时走冷启动：用一张种子图 bootstrap
#      样本数 >= 阈值时用全量样本训练
#   2) 状态模型  data\fudai\model\fudai_state.npz  （≥2 类才允许训练）
#   3) 训练后打印模型产物（大小/时间）
#
# 用法：
#   .\train.ps1                          # 全量训练（样本不足自动冷启动）
#   .\train.ps1 -DryRun                  # 只评估，不写任何文件
#   .\train.ps1 -ColdStartBelow 20       # 自定义冷启动阈值
#   .\train.ps1 -Seed data\fudai\seed\seed_vgn.png  # 指定种子图
#   .\train.ps1 -SkipState               # 只训检测模型
#   .\train.ps1 -SkipDetector            # 只训状态模型
#   .\train.ps1 -Aug 16 -Epochs 800      # 状态模型增强倍数/轮数
#   .\train.ps1 -Predict data\fudai\icons # 用状态模型预测
#   .\train.ps1 -Watch -IntervalSec 300  # 样本数变化自动重训（常驻）
#
# 采集仍由 fudai_auto 包在运行时自动完成（data\fudai\icons + labels.csv）。
# 重训后需重启 main.py 才会加载新权重。

[CmdletBinding()]
param(
    [int]$Aug = 8,
    [int]$Epochs = 400,
    [switch]$DryRun,
    [string]$Predict = "",
    [string]$Labels = "",
    [string]$Model = "",
    [string[]]$Extra = @(),
    [ValidateSet('pending', 'joined', 'countdown')]
    [string]$ExtraLabel = "",
    [string]$Python = "",
    [int]$ColdStartBelow = 30,
    [string]$Seed = "",
    [switch]$SkipDetector,
    [switch]$SkipState,
    [switch]$Watch,
    [int]$IntervalSec = 300
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.UTF8Encoding]::new() } catch { }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stateModule = Join-Path $root 'train_fudai'
$detModule = Join-Path $root 'train_fudai_clf'
$bootModule = Join-Path $root 'bootstrap_fudai_clf'
$iconDir = Join-Path $root 'data\fudai\icons'
$modelDir = Join-Path $root 'data\fudai\model'
$detModel = Join-Path $modelDir 'fudai_clf.npz'
$stateModel = if ($Model) { $Model } else { Join-Path $modelDir 'fudai_state.npz' }

if (-not $SkipState -and -not (Test-Path -LiteralPath $stateModule)) {
    Write-Host "找不到 train_fudai 包: $stateModule" -ForegroundColor Red
    exit 1
}
if (-not $SkipDetector -and -not (Test-Path -LiteralPath $detModule)) {
    Write-Host "找不到 train_fudai_clf 包: $detModule" -ForegroundColor Red
    exit 1
}

# ---- 解析 python ----
if (-not $Python) {
    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPy) {
        $Python = $venvPy
    }
    else {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source }
        else {
            Write-Host "未找到 .venv\Scripts\python.exe，PATH 里也没有 python" -ForegroundColor Red
            exit 1
        }
    }
}
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "python 不存在: $Python" -ForegroundColor Red
    exit 1
}

function Invoke-Detector {
    param([switch]$Dry)
    $count = Get-SampleCount
    $seedPath = if ($Seed) { $Seed } else { Join-Path $root 'data\fudai\seed\seed_vgn.png' }

    if ($count -lt $ColdStartBelow) {
        Write-Host "`n=== [1] 冷启动（样本 $count < 阈值 $ColdStartBelow，用种子图 bootstrap） ===" -ForegroundColor Yellow
        if ($Dry) {
            Write-Host "  dry-run：跳过冷启动训练，不写模型" -ForegroundColor DarkGray
            return 0
        }
        if (-not (Test-Path -LiteralPath $seedPath)) {
            Write-Host "种子图不存在: $seedPath" -ForegroundColor Red
            return 1
        }
        if (-not (Test-Path -LiteralPath $bootModule)) {
            Write-Host "找不到 bootstrap_fudai_clf 包: $bootModule" -ForegroundColor Red
            return 1
        }
        Push-Location $root
        try { & $Python @('-m', 'bootstrap_fudai_clf', '--seed', $seedPath, '--install') | Out-Host }
        finally { Pop-Location }
        return $LASTEXITCODE
    }

    $a = @('-m', 'train_fudai_clf')
    if ($Dry) { $a += '--dry-run' }
    Write-Host "`n=== [1] 检测模型（样本 $count ≥ 阈值 $ColdStartBelow） $(if ($Dry) { '(dry-run)' } else { $detModel }) ===" -ForegroundColor Cyan
    Push-Location $root
    try { & $Python @a | Out-Host }
    finally { Pop-Location }
    return $LASTEXITCODE
}

function Invoke-State {
    param([switch]$Dry)
    $a = @('-m', 'train_fudai', '--aug', $Aug, '--epochs', $Epochs)
    if ($Dry) { $a += '--dry-run' }
    if ($Labels) { $a += @('--labels', $Labels) }
    if ($Model) { $a += @('--model', $Model) }
    if ($Extra.Count -gt 0 -and $Extra[0]) { $a += @('--extra') + $Extra }
    if ($ExtraLabel) { $a += @('--extra-label', $ExtraLabel) }
    Write-Host "`n=== [2] 状态模型 $(if ($Dry) { '(dry-run)' } else { $stateModel }) ===" -ForegroundColor Cyan
    Push-Location $root
    try { & $Python @a | Out-Host }
    finally { Pop-Location }
    return $LASTEXITCODE
}

function Get-SampleCount {
    if (-not (Test-Path -LiteralPath $iconDir)) { return 0 }
    return (Get-ChildItem -LiteralPath $iconDir -Filter *.png -File).Count
}

function Show-Result {
    if (-not $SkipDetector) {
        $f = Get-Item -LiteralPath $detModel -ErrorAction SilentlyContinue
        if ($f) { Write-Host ("  检测模型  {0}  {1:N0} B  {2:HH:mm:ss}" -f $f.Name, $f.Length, $f.LastWriteTime) -ForegroundColor Green }
    }
    if (-not $SkipState) {
        $f = Get-Item -LiteralPath $stateModel -ErrorAction SilentlyContinue
        if ($f) { Write-Host ("  状态模型  {0}  {1:N0} B  {2:HH:mm:ss}" -f $f.Name, $f.Length, $f.LastWriteTime) -ForegroundColor Green }
    }
}

function Invoke-All {
    param([switch]$Dry)
    if (-not $SkipDetector) {
        $code = Invoke-Detector -Dry:$Dry
        if ($code -ne 0) { Write-Host "检测模型训练失败（exit $code）" -ForegroundColor Yellow; return $code }
    }
    if (-not $SkipState) {
        $code = Invoke-State -Dry:$Dry
        if ($code -eq 2) {
            Write-Host "状态模型跳过：样本类别不足（冷启动阶段正常）" -ForegroundColor Yellow
            return 0
        }
        if ($code -ne 0) { Write-Host "状态模型训练失败（exit $code）" -ForegroundColor Yellow; return $code }
    }
    return 0
}

# ---- 预测模式 ----
if ($Predict) {
    $a = @('-m', 'train_fudai', '--predict', $Predict)
    if ($Model) { $a += @('--model', $Model) }
    Push-Location $root
    try { & $Python @a }
    finally { Pop-Location }
    exit $LASTEXITCODE
}

# ---- 常驻重训模式 ----
if ($Watch) {
    Write-Host "常驻模式：每 $IntervalSec 秒检查样本数，有变化就全量重训（Ctrl+C 退出）" -ForegroundColor Cyan
    $lastCount = -1
    while ($true) {
        $count = Get-SampleCount
        if ($count -ne $lastCount) {
            $t0 = Get-Date
            Write-Host "`n[$($t0.ToString('HH:mm:ss'))] 样本 $count 张，开始训练" -ForegroundColor Cyan
            $code = Invoke-All
            $cost = [int]((Get-Date) - $t0).TotalSeconds
            if ($code -eq 0) {
                Show-Result
                Write-Host "训练完成，耗时 ${cost}s（重启 main.py 后加载新模型）" -ForegroundColor Green
                $lastCount = $count
            }
            else {
                Write-Host "训练失败（exit $code），样本数变化后重试" -ForegroundColor Yellow
            }
        }
        Start-Sleep -Seconds $IntervalSec
    }
}

# ---- 单次全量训练 ----
$t0 = Get-Date
$code = Invoke-All -Dry:$DryRun
$cost = [int]((Get-Date) - $t0).TotalSeconds
if ($code -eq 0) {
    if (-not $DryRun) { Show-Result }
    Write-Host "完成，耗时 ${cost}s" -ForegroundColor Green
}
else {
    Write-Host "失败（exit $code）" -ForegroundColor Red
}
exit $code
