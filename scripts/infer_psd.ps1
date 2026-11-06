<#
.SYNOPSIS
    see-through 单图推理：输入图片 -> 输出分层 PSD，纯命令行，不启动任何 webui。

.EXAMPLE
    .\infer_psd.ps1 "D:\img\a.png" "D:\out\a.psd"
.EXAMPLE
    .\infer_psd.ps1 -InputImage "D:\img\a.png" -OutputPsd "D:\out\a.psd" -CpuOffload
#>
[CmdletBinding()]
param(
    [Parameter(Position=0, Mandatory=$true)][string]$InputImage,
    [Parameter(Position=1, Mandatory=$true)][string]$OutputPsd,
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
$InferScript = Join-Path $RepoRoot 'inference\scripts\inference_psd_quantized.py'

if (-not (Test-Path -LiteralPath $InputImage -PathType Leaf)) { Fail "输入图片不存在：$InputImage" }
$InputImage = (Resolve-Path -LiteralPath $InputImage).Path
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { Fail "环境未安装，请先运行 scripts\setup_env.cmd" }
if (-not (Test-Path -LiteralPath $InferScript -PathType Leaf)) { Fail "官方推理脚本缺失：$InferScript" }
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'assets'))) { Fail "assets 缺失，请先运行 scripts\setup_env.cmd" }

if ([IO.Path]::GetExtension($OutputPsd).ToLower() -ne '.psd') {
    $OutputPsd = $OutputPsd.TrimEnd('\') + '.psd'
}
$outFull = [IO.Path]::GetFullPath($OutputPsd)
$outDir  = Split-Path -Parent $outFull
if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

Set-Location -LiteralPath $RepoRoot

# 同 setup_env.ps1：WorkBuddy 经 PYTHONPATH 注入的 sitecustomize 会把删除操作
# 计入批量删除保护。推理过程会写大量中间文件，避免跑到一半被误判中断。
# 只对本次派生的 python 子进程过滤，不改全局环境、不删用户文件。
$origPyPath = $env:PYTHONPATH
if ($origPyPath) {
    $keptPy = @($origPyPath.Split(';') | Where-Object {
        $_ -and -not ($_ -like '*WorkBuddy*' -and $_ -like '*shim*')
    })
    $env:PYTHONPATH = $keptPy -join ';'
}

$checkScript = Join-Path $PSScriptRoot 'py\check_cache.py'
# 不用 python -c 传多行代码：Windows PowerShell 5.1 会把参数里的双引号吃掉，
# 导致 Python 收到残缺代码（File "<string>", line 4）。统一改为独立 .py 文件。
if ($Offline) {
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

if ($CheckOnly) {
    Write-Output ''
    Write-Output '--- 自检通过（未启动推理）---'
    Write-Output ("  Python    : {0}" -f $PythonExe)
    Write-Output ("  输入图片  : {0}" -f $InputImage)
    Write-Output ("  离线缓存  : {0}" -f $(if ($Offline) { '已校验' } else { '未校验（未加 -Offline）' }))
    Write-Output ("  分辨率    : {0}" -f $Resolution)
    exit 0
}

if (-not $SkipVramCheck) {
    Write-Output '--- 显存体检（只读，不会关闭任何进程）---'
    & (Join-Path $PSScriptRoot 'gpu_mem.ps1') | Select-Object -First 7
    Write-Output ''
}

# 源码真实链路（inference_psd_quantized.py + common/utils/inference_utils.py）：
#   推理脚本 saved = save_dir / <图片名>，further_extr(saved) 内部再拼 'optimized'，
#   最终 psd_savep = save_dir / <图片名>.psd（在 save_dir 内部，不是父目录）。
# 每次用独立 run tag，避免多次推理互相覆盖。
$rawStem  = [IO.Path]::GetFileNameWithoutExtension($InputImage)
$safeStem = $rawStem -replace '[^A-Za-z0-9_\-]', '_'
if (-not $safeStem) { $safeStem = 'image' }
$tag       = '{0}_{1}' -f $safeStem, (Get-Date -Format 'yyyyMMdd_HHmmss')
$runDirRel = "workspace/runs/$tag"
$psdRel    = "workspace/runs/$tag/$rawStem.psd"
$runsRoot  = Join-Path $RepoRoot 'workspace\runs'
if (-not (Test-Path -LiteralPath $runsRoot)) { New-Item -ItemType Directory -Path $runsRoot -Force | Out-Null }

# Windows 不支持 expandable_segments，避免留下一个无效且会触发告警的配置。
$env:PYTORCH_CUDA_ALLOC_CONF = ''
# 8GB 卡实测：官方默认开启的 group_offload 会把 NF4 权重搬到 CPU，
# bitsandbytes 再取回时 cuBLAS 上下文崩溃（CUBLAS_STATUS_ALLOC_FAILED）。
# 关闭后 tag 编码一次通过，且峰值显存仍在 4GB 以内，因此默认关闭。
# -Offline 用于模型已缓存但代理抖动（HF 返回 502）时跳过联网校验。
if ($Offline) {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
}

$pyArgs = @(
    'inference/scripts/inference_psd_quantized.py',
    '--srcp', $InputImage,
    '--save_dir', $runDirRel,
    '--save_to_psd',
    '--resolution', "$Resolution",
    '--resolution_depth', "$DepthResolution",
    '--num_inference_steps', "$Steps",
    '--seed', "$Seed"
)
if ($CpuOffload) { $pyArgs += '--cpu_offload' }
if (-not $GroupOffload) { $pyArgs += '--no_group_offload' }

Write-Output '--- 开始推理（纯 CLI，无 webui）---'
Write-Output ("  输入      : {0}" -f $InputImage)
Write-Output ("  输出      : {0}" -f $outFull)
Write-Output ("  分辨率    : {0} / depth {1}" -f $Resolution, $DepthResolution)
Write-Output ("  CPU offload: {0}" -f $(if ($CpuOffload) { '开' } else { '关' }))
Write-Output ("  group offload: {0}" -f $(if ($GroupOffload) { '开' } else { '关（8GB 卡默认）' }))
Write-Output ("  离线模式  : {0}" -f $(if ($Offline) { '开' } else { '关' }))
Write-Output ''

$startTime = Get-Date
$sw = [Diagnostics.Stopwatch]::StartNew()
& $PythonExe @pyArgs
$code = $LASTEXITCODE
$sw.Stop()
if ($code -ne 0) {
    Fail "推理退出码 $code。先试 -Resolution 512；8GB 卡不要开 -GroupOffload / -CpuOffload。"
}

$produced = Join-Path $RepoRoot $psdRel
if (-not (Test-Path -LiteralPath $produced)) {
    $runFull = Join-Path $RepoRoot $runDirRel
    $cand = Get-ChildItem -LiteralPath $runFull -Filter '*.psd' -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notlike '*_depth.psd' } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($cand) { $produced = $cand.FullName }
}
if (-not (Test-Path -LiteralPath $produced)) { Fail "推理完成但找不到 PSD 产物，预期：$psdRel" }

Copy-Item -LiteralPath $produced -Destination $outFull -Force

$statsPath = Join-Path $RepoRoot "$runDirRel\$rawStem\stats.json"
$peak = 'N/A'
if (Test-Path -LiteralPath $statsPath) {
    try {
        $s = Get-Content -LiteralPath $statsPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $peak = '{0:N2}' -f $s.peak_vram_gb
    } catch { $peak = 'N/A' }
}

Write-Output ''
Write-Output '--- 完成 ---'
Write-Output ("  PSD       : {0}" -f $outFull)
Write-Output ("  大小      : {0:N1} MB" -f ((Get-Item -LiteralPath $outFull).Length / 1MB))
Write-Output ("  耗时      : {0:N1} 秒" -f $sw.Elapsed.TotalSeconds)
Write-Output ("  峰值显存  : {0} GB" -f $peak)
Write-Output ("  中间产物  : {0}" -f (Join-Path $RepoRoot $runDirRel))
Write-Output ''

if ($CleanOutput) {
    $runFull = Join-Path $RepoRoot $runDirRel
    if ($runFull.StartsWith((Join-Path $RepoRoot 'workspace\runs'), [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $runFull -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $produced -Force -ErrorAction SilentlyContinue
    Write-Output '已清理本次中间产物（分层 PNG / 深度图 / stats.json）。'
    Write-Output ''
}
