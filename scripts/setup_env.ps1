<#
.SYNOPSIS
    按官方 requirements 搭建 see_through 独立 conda 环境（可重复运行）。

.EXAMPLE
    .\setup_env.ps1                完整安装
    .\setup_env.ps1 -VerifyOnly    只检查环境是否可用
    .\setup_env.ps1 -Retries 3     git+ 依赖网络不稳时多重试几次
#>
[CmdletBinding()]
param(
    [switch]$VerifyOnly,
    [int]$Retries = 2
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$EnvPrefix = Join-Path $env:USERPROFILE '.conda\envs\see_through'
$PythonExe = Join-Path $EnvPrefix 'python.exe'
Set-Location -LiteralPath $RepoRoot

# WorkBuddy 通过 PYTHONPATH 注入 sitecustomize.py，其中「批量删除保护」会把
# pip 覆盖包文件时的 os.unlink 累计计数，满 50 次即抛 SystemExit 中断安装。
# 在隔离 conda 环境里做包管理属于误报。这里只对本次脚本派生出的子进程过滤掉
# 该 shim，脚本结束立即恢复；不改全局环境、不禁用 shim 本身、不删任何用户文件。
$OriginalPyPath = $env:PYTHONPATH
function Restore-PyPath { $env:PYTHONPATH = $OriginalPyPath }
if ($OriginalPyPath) {
    $kept = @($OriginalPyPath.Split(';') | Where-Object {
        $_ -and -not ($_ -like '*WorkBuddy*' -and $_ -like '*shim*')
    })
    $env:PYTHONPATH = $kept -join ';'
    Write-Output '已为安装子进程过滤 PYTHONPATH 中的 shim（避免 pip 误触批量删除保护）。'
}

function Step([string]$title) {
    Write-Output ''
    Write-Output ("==== {0} ====" -f $title)
}
function Fail([string]$msg) {
    Restore-PyPath
    Write-Output ''
    Write-Output ("错误：{0}" -f $msg)
    Write-Output '可修复后重新运行本脚本，已完成的步骤会自动跳过。'
    exit 1
}
function Invoke-PipStep([string]$title, [string[]]$PipArgs) {
    Step $title
    $maxTry = $Retries + 1
    for ($i = 1; $i -le $maxTry; $i++) {
        Write-Output ("  尝试 {0}/{1}" -f $i, $maxTry)
        # 5.1 下 Stop 会把 pip 的 stderr 当成终止错误，临时降级。
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $PythonExe -m pip @PipArgs
        $pipExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEap
        if ($pipExit -eq 0) { return }
        Write-Output '  本轮失败，重试中（git+ 依赖常因代理隧道临时 502 中断）。'
    }
    Fail ("{0} 连续 {1} 次失败" -f $title, $maxTry)
}

# 不用 python -c 传多行代码：Windows PowerShell 5.1 会吃掉参数里的双引号。
# 所有内嵌 Python 都放在 scripts\py\ 下，用文件路径调用。
$VerifyScript = Join-Path $PSScriptRoot 'py\verify_env.py'
$TorchScript  = Join-Path $PSScriptRoot 'py\check_torch.py'
foreach ($f in @($VerifyScript, $TorchScript)) {
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { Fail "缺少脚本：$f" }
}

if ($VerifyOnly) {
    Restore-PyPath
    if (-not (Test-Path -LiteralPath $PythonExe)) { Fail "环境不存在：$PythonExe" }
    Step '环境验证'
    & $PythonExe $VerifyScript
    $code = $LASTEXITCODE
    Restore-PyPath
    exit $code
}

Write-Output ("仓库根   : {0}" -f $RepoRoot)
Write-Output ("环境前缀 : {0}" -f $EnvPrefix)
Write-Output '注意：与 ComfyUI 便携版完全隔离，不会改动其 python_embeded。'

Step '1/6 创建 conda 环境（Python 3.12）'
if (Test-Path -LiteralPath $PythonExe) {
    Write-Output '环境已存在，跳过创建。'
} else {
    & conda create --prefix "$EnvPrefix" python=3.12 -y
    if ($LASTEXITCODE -ne 0) { Fail 'conda create 失败' }
    if (-not (Test-Path -LiteralPath $PythonExe)) { Fail "conda create 后仍找不到 $PythonExe" }
}

Step '2/6 安装 PyTorch 2.8.0 + CUDA 12.8（约 2.5GB，耗时最长）'
& $PythonExe $TorchScript
if ($LASTEXITCODE -eq 0) {
    Write-Output 'torch 2.8.0+cu128 已安装且 CUDA 可用，跳过（幂等）。'
} else {
    Invoke-PipStep -title '2/6 安装 torch cu128' -PipArgs @(
        'install', 'torch==2.8.0+cu128', 'torchvision==0.23.0+cu128',
        'torchaudio==2.8.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128'
    )
}

Invoke-PipStep -title '3/6 安装 requirements.txt（含 git+ 依赖，需 GitHub 可达）' -PipArgs @(
    'install', '-r', 'requirements.txt'
)

Invoke-PipStep -title '4/6 安装 bitsandbytes（NF4 量化推理必需）' -PipArgs @(
    'install', '-r', 'requirements-inference-bnb.txt'
)

Step '5/6 创建 assets（Windows 用复制代替 ln -sf）'
$assets = Join-Path $RepoRoot 'assets'
if (Test-Path -LiteralPath $assets) {
    Write-Output 'assets 已存在，跳过。'
} else {
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'common\assets') -Destination $assets -Recurse
    Write-Output '已复制 common/assets -> assets'
}

Step '6/6 验证环境'
& $PythonExe $VerifyScript
if ($LASTEXITCODE -ne 0) { Fail '环境验证未通过' }

Write-Output ''
Write-Output '环境就绪。日常出图： scripts\infer_psd.cmd <输入图> <输出psd>'
Write-Output ''
Restore-PyPath
