<#
.SYNOPSIS
    按官方 requirements 搭建 see_through 独立 conda 环境（可重复运行）。

.EXAMPLE
    .\setup_env.ps1              完整安装
.EXAMPLE
    .\setup_env.ps1 -VerifyOnly  只检查环境是否可用
#>
[CmdletBinding()]
param(
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$EnvPrefix = Join-Path $env:USERPROFILE '.conda\envs\see_through'
$PythonExe = Join-Path $EnvPrefix 'python.exe'
Set-Location -LiteralPath $RepoRoot

function Step([string]$title) {
    Write-Output ''
    Write-Output ("==== {0} ====" -f $title)
}
function Fail([string]$msg) {
    Write-Output ''
    Write-Output ("错误：{0}" -f $msg)
    Write-Output '可修复后重新运行本脚本，已完成的步骤会自动跳过。'
    exit 1
}

$verifyCode = @'
import sys
print('python', sys.version.split()[0])
import torch
print('torch', torch.__version__)
print('cuda_build', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
for name in ('transformers', 'diffusers', 'accelerate', 'psd_tools'):
    try:
        mod = __import__(name)
        print(name, getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print(name, 'IMPORT FAILED:', e)
try:
    import bitsandbytes as bnb
    print('bitsandbytes', bnb.__version__)
except Exception as e:
    print('bitsandbytes IMPORT FAILED:', e)
'@

if ($VerifyOnly) {
    if (-not (Test-Path -LiteralPath $PythonExe)) { Fail "环境不存在：$PythonExe" }
    Step '环境验证'
    & $PythonExe -c $verifyCode
    exit $LASTEXITCODE
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
& $PythonExe -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Fail 'torch cu128 安装失败，检查网络与代理' }

Step '3/6 安装 requirements.txt（含 git+ 依赖，需 GitHub 可达）'
& $PythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail 'requirements.txt 安装失败' }

Step '4/6 安装 bitsandbytes（NF4 量化推理必需）'
& $PythonExe -m pip install -r requirements-inference-bnb.txt
if ($LASTEXITCODE -ne 0) { Fail 'bitsandbytes 安装失败，这是 Windows 上最容易出问题的一步' }

Step '5/6 创建 assets（Windows 用复制代替 ln -sf）'
$assets = Join-Path $RepoRoot 'assets'
if (Test-Path -LiteralPath $assets) {
    Write-Output 'assets 已存在，跳过。'
} else {
    Copy-Item -LiteralPath (Join-Path $RepoRoot 'common\assets') -Destination $assets -Recurse
    Write-Output '已复制 common/assets -> assets'
}

Step '6/6 验证环境'
& $PythonExe -c $verifyCode
if ($LASTEXITCODE -ne 0) { Fail '环境验证未通过' }

Write-Output ''
Write-Output '环境就绪。日常出图： scripts\infer_psd.cmd <输入图> <输出psd>'
Write-Output ''
