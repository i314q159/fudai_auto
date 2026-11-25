# train.ps1 — 福袋模型一键训练（除采集外的全部训练）
#
# 一条命令完成：
#   1) 检测模型  data\fudai\model\fudai_clf.npz    （fudai_auto.py 运行时用）
#   2) 状态模型  data\fudai\model\fudai_state.npz  （待参与/已参与/倒计时 3 分类，分析用）
#   3) 训练后打印两个模型的产物（大小/时间）
#
# 用法：
#   .\train.ps1                          # 全部训练
#   .\train.ps1 -DryRun                  # 两个模型都只评估，不写任何文件
#   .\train.ps1 -SkipState               # 只训检测模型
#   .\train.ps1 -SkipDetector            # 只训状态模型
#   .\train.ps1 -Aug 16 -Epochs 800      # 状态模型增强倍数/轮数
#   .\train.ps1 -Predict data\fudai\icons # 用状态模型预测
#   .\train.ps1 -Watch -IntervalSec 300  # 样本数变化自动全量重训（常驻）
#
# 采集仍由 fudai_auto.py 在运行时自动完成（data\fudai\icons + labels.csv）。
# 重训后需重启 fudai_auto.py 才会加载新权重。

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
    [switch]$SkipDetector,
    [switch]$SkipState,
    [switch]$Watch,
    [int]$IntervalSec = 300
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.UTF8Encoding]::new() } catch { }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stateScript = Join-Path $root 'train_fudai.py'
$detScript = Join-Path $root 'train_fudai_clf.py'
$iconDir = Join-Path $root 'data\fudai\icons'
$modelDir = Join-Path $root 'data\fudai\model'
$detModel = Join-Path $modelDir 'fudai_clf.npz'
$stateModel = if ($Model) { $Model } else { Join-Path $modelDir 'fudai_state.npz' }

if (-not $SkipState -and -not (Test-Path -LiteralPath $stateScript)) {
    Write-Host "找不到 train_fudai.py: $stateScript" -ForegroundColor Red
    exit 1
}
if (-not $SkipDetector -and -not (Test-Path -LiteralPath $detScript)) {
    Write-Host "找不到 train_fudai_clf.py: $detScript" -ForegroundColor Red
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
    $a = @($detScript)
    if ($Dry) { $a += '--dry-run' }
    Write-Host "`n=== [1] 检测模型 $(if ($Dry) { '(dry-run)' } else { $detModel }) ===" -ForegroundColor Cyan
    & $Python @a | Out-Host
    return $LASTEXITCODE
}

function Invoke-State {
    param([switch]$Dry)
    $a = @($stateScript, '--aug', $Aug, '--epochs', $Epochs)
    if ($Dry) { $a += '--dry-run' }
    if ($Labels) { $a += @('--labels', $Labels) }
    if ($Model) { $a += @('--model', $Model) }
    if ($Extra.Count -gt 0 -and $Extra[0]) { $a += @('--extra') + $Extra }
    if ($ExtraLabel) { $a += @('--extra-label', $ExtraLabel) }
    Write-Host "`n=== [2] 状态模型 $(if ($Dry) { '(dry-run)' } else { $stateModel }) ===" -ForegroundColor Cyan
    & $Python @a | Out-Host
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
        if ($code -ne 0) { Write-Host "状态模型训练失败（exit $code）" -ForegroundColor Yellow; return $code }
    }
    return 0
}

# ---- 预测模式 ----
if ($Predict) {
    $a = @($stateScript, '--predict', $Predict)
    if ($Model) { $a += @('--model', $Model) }
    & $Python @a
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
                Write-Host "训练完成，耗时 ${cost}s（重启 fudai_auto.py 后加载新模型）" -ForegroundColor Green
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
