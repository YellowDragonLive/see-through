<#
.SYNOPSIS
    see-through 批量推理：单进程、单线程、多张图串联，模型资源只加载一次。

.DESCRIPTION
    与「循环调 infer_psd.cmd」不同：本脚本只启动一个 python.exe，
    LayerDiff3D / Marigold 各加载一次，N 张图依次跑完。
    省掉每张图约 20 秒的进程启动 + import + 5.6GB NF4 权重读取，
    显存峰值与单张一致（两个大模型不会同时驻留）。

.EXAMPLE
    .\infer_batch.ps1 "D:\img\a.png" "D:\img\b.png" "D:\img\c.png"
.EXAMPLE
    .\infer_batch.ps1 -InputDir "D:\img" -OutputDir "D:\out" -Resolution 768 -Offline
#>
[CmdletBinding()]
param(
    [Parameter(Position=0, ValueFromRemainingArguments=$true)][string[]]$Inputs,
    [string]$InputDir,
    [string]$OutputDir,
    [int]$Resolution = 768,
    [int]$DepthResolution = 720,
    [int]$Steps = 30,
    [int]$Seed = 42,
    [switch]$CpuOffload,
    [switch]$GroupOffload,
    [switch]$SkipVramCheck,
    [switch]$Offline,
    [switch]$CleanOutput,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

function Fail([string]$msg) {
    Write-Output ''
    Write-Output ("错误：{0}" -f $msg)
    exit 1
}

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$EnvPrefix   = Join-Path $env:USERPROFILE '.conda\envs\see_through'
$PythonExe   = Join-Path $EnvPrefix 'python.exe'
$Driver      = Join-Path $PSScriptRoot 'py\infer_batch.py'
$InferScript = Join-Path $RepoRoot 'inference\scripts\inference_psd_quantized.py'

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { Fail "环境未安装，请先运行 scripts\setup_env.cmd" }
if (-not (Test-Path -LiteralPath $Driver -PathType Leaf)) { Fail "批量驱动缺失：$Driver" }
if (-not (Test-Path -LiteralPath $InferScript -PathType Leaf)) { Fail "官方推理脚本缺失：$InferScript" }
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'assets'))) { Fail "assets 缺失，请先运行 scripts\setup_env.cmd" }

# 收集输入：位置参数 + 可选的输入目录（-Recurse 关闭，只扫一层）
$paths = @()
if ($Inputs) { $paths += $Inputs }
if ($InputDir) {
    if (-not (Test-Path -LiteralPath $InputDir -PathType Container)) { Fail "输入目录不存在：$InputDir" }
    $paths += @(Get-ChildItem -LiteralPath $InputDir -File |
                Where-Object { $_.Extension.ToLower() -in @('.png','.jpg','.jpeg','.webp','.bmp') } |
                ForEach-Object { $_.FullName })
}
if ($paths.Count -eq 0) { Fail "没有输入图片。用法：infer_batch.cmd <图1> <图2> <图3> 或 -InputDir <目录>" }

$resolved = @()
foreach ($p in $paths) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Fail "输入图片不存在：$p" }
    $resolved += (Resolve-Path -LiteralPath $p).Path
}
if ($resolved.Count -gt 20) { Fail "一次最多 20 张，当前 $($resolved.Count) 张；请分批跑" }

if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot 'workspace\batch_out' }
$outFull = [IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path -LiteralPath $outFull)) { New-Item -ItemType Directory -Path $outFull -Force | Out-Null }

Set-Location -LiteralPath $RepoRoot

# 同 infer_psd.ps1：过滤 WorkBuddy 注入的 sitecustomize shim，避免批量写文件被误判。
$origPyPath = $env:PYTHONPATH
if ($origPyPath) {
    $keptPy = @($origPyPath.Split(';') | Where-Object {
        $_ -and -not ($_ -like '*WorkBuddy*' -and $_ -like '*shim*')
    })
    $env:PYTHONPATH = $keptPy -join ';'
}

# -Offline 前置校验缓存，避免 LayerDiff 跑完才发现深度模型没下。
# 不用 python -c：Windows PowerShell 5.1 会吃掉多行参数里的双引号。
if ($Offline) {
    $checkScript = Join-Path $PSScriptRoot 'py\check_cache.py'
    if (-not (Test-Path -LiteralPath $checkScript -PathType Leaf)) { Fail "缺少缓存校验脚本：$checkScript" }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $check = & $PythonExe $checkScript 2>&1
    $cacheExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($cacheExit -ne 0) {
        Fail "-Offline 要求两个 NF4 模型都已缓存；当前缺失：$($check -join ' ')。请先运行 scripts\prefetch_models.cmd"
    }
}

# 每次批量独立 run 目录；中间产物用短名 001/002/003，避开 Windows 长路径限制。
$tag     = 'batch_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$runDir  = Join-Path $RepoRoot "workspace\batch\$tag"
if (-not (Test-Path -LiteralPath $runDir)) { New-Item -ItemType Directory -Path $runDir -Force | Out-Null }
$manifestPath = Join-Path $runDir 'manifest.json'
$reportPath   = Join-Path $runDir 'report.json'

$items = @()
foreach ($p in $resolved) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($p)
    $items += [ordered]@{
        src  = $p
        dst  = (Join-Path $outFull ($stem + '.psd'))
        name = $stem
    }
}
$json = ([ordered]@{ items = $items }) | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText($manifestPath, $json, (New-Object System.Text.UTF8Encoding($false)))

if ($CheckOnly) {
    Write-Output ''
    Write-Output '--- 批量自检通过（未启动推理）---'
    Write-Output ("  模式      : 单进程 / 单线程 / 模型只加载一次")
    Write-Output ("  图片数量  : {0}" -f $items.Count)
    foreach ($it in $items) { Write-Output ("    - {0}" -f $it['src']) }
    Write-Output ("  输出目录  : {0}" -f $outFull)
    Write-Output ("  分辨率    : {0} / depth {1} / steps {2}" -f $Resolution, $DepthResolution, $Steps)
    Write-Output ("  离线缓存  : {0}" -f $(if ($Offline) { '已校验' } else { '未校验（未加 -Offline）' }))
    Write-Output ("  清单      : {0}" -f $manifestPath)
    exit 0
}

if (-not $SkipVramCheck) {
    Write-Output '--- 显存体检（只读，不会关闭任何进程）---'
    & (Join-Path $PSScriptRoot 'gpu_mem.ps1') | Select-Object -First 7
    Write-Output ''
}

$env:PYTORCH_CUDA_ALLOC_CONF = ''
if ($Offline) {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
}

$pyArgs = @(
    'scripts/py/infer_batch.py',
    '--manifest', $manifestPath,
    '--repo-root', $RepoRoot,
    '--work-dir', $runDir,
    '--report', $reportPath,
    '--resolution', "$Resolution",
    '--depth-resolution', "$DepthResolution",
    '--steps', "$Steps",
    '--seed', "$Seed"
)
if ($CpuOffload)   { $pyArgs += '--cpu-offload' }
if ($GroupOffload) { $pyArgs += '--group-offload' }

Write-Output '--- 开始批量推理（单进程 / 单线程 / 模型只加载一次）---'
Write-Output ("  图片      : {0} 张" -f $items.Count)
Write-Output ("  分辨率    : {0} / depth {1} / steps {2}" -f $Resolution, $DepthResolution, $Steps)
Write-Output ("  输出目录  : {0}" -f $outFull)
Write-Output ("  中间产物  : {0}" -f $runDir)
Write-Output ("  group offload / cpu offload: {0} / {1}" -f $(if ($GroupOffload) { '开' } else { '关' }), $(if ($CpuOffload) { '开' } else { '关' }))
Write-Output ''

$sw = [Diagnostics.Stopwatch]::StartNew()
& $PythonExe @pyArgs
$code = $LASTEXITCODE
$sw.Stop()

Write-Output ''
Write-Output ("--- 进程结束：退出码 {0}，墙钟耗时 {1:N1} 秒 ---" -f $code, $sw.Elapsed.TotalSeconds)
if (Test-Path -LiteralPath $reportPath) {
    try {
        $r = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Output ("  成功/总数  : {0}/{1}" -f $r.ok, $r.count)
        Write-Output ("  峰值显存   : {0:N2} GB" -f $r.peak_vram_gb)
        Write-Output ("  固定开销   : 进程+import {0:N1}s + LayerDiff 加载 {1:N1}s + Marigold 加载 {2:N1}s" -f $r.startup_s, $r.build_layerdiff_s, $r.build_marigold_s)
        Write-Output ("  驱动计时   : {0:N1} 秒" -f $r.total_s)
        foreach ($it in $r.items) {
            Write-Output ("    [{0}] {1}  LD {2}s / MG {3}s / PSD {4}s  -> {5}" -f $it.stem, $it.status, $it.layerdiff_s, $it.marigold_s, $it.psd_s, $it.dst)
            if ($it.error) { Write-Output ("         错误：{0}" -f $it.error) }
        }
        Write-Output ("  报告       : {0}" -f $reportPath)
    } catch { Write-Output ("  报告解析失败：{0}" -f $_) }
}

if ($CleanOutput -and $code -eq 0) {
    $batchRoot = Join-Path $RepoRoot 'workspace\batch'
    if ($runDir.StartsWith($batchRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $runDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Output '已清理本次中间产物。'
    }
}
if ($code -ne 0) { Fail "批量推理有失败项，详见上方逐张状态与 $reportPath" }
Write-Output '--- 完成 ---'
