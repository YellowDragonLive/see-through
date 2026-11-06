<#
.SYNOPSIS
    预下载 see-through NF4 推理所需的两个 HuggingFace 模型，支持镜像回退与断点续传。

.EXAMPLE
    .\prefetch_models.ps1
.EXAMPLE
    .\prefetch_models.ps1 -VerifyOnly
.EXAMPLE
    .\prefetch_models.ps1 -Retries 5 -Endpoint https://hf-mirror.com
#>
[CmdletBinding()]
param(
    [int]$Retries = 3,
    [string]$Endpoint = 'https://hf-mirror.com',
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$PythonExe = Join-Path $env:USERPROFILE '.conda\envs\see_through\python.exe'
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Output "错误：环境未安装，请先运行 scripts\setup_env.cmd"
    exit 1
}

# 同 setup_env.ps1：只对本次派生的 python 子进程过滤掉 WorkBuddy 注入的 shim。
$origPyPath = $env:PYTHONPATH
if ($origPyPath) {
    $keptPy = @($origPyPath.Split(';') | Where-Object {
        $_ -and -not ($_ -like '*WorkBuddy*' -and $_ -like '*shim*')
    })
    $env:PYTHONPATH = $keptPy -join ';'
}

# 预下载必须联网；离线变量会直接让 snapshot_download 抛 OfflineModeIsEnabled。
$env:HF_HUB_OFFLINE = ''
$env:TRANSFORMERS_OFFLINE = ''
$env:HF_ENDPOINT = $Endpoint
$env:HF_HUB_DISABLE_TELEMETRY = '1'

$verify = if ($VerifyOnly) { '1' } else { '0' }
$retries = $Retries

# 不用 python -c 传多行代码：Windows PowerShell 5.1 会吃掉参数里的双引号。
$pyScript = Join-Path $PSScriptRoot 'py\prefetch_models.py'
if (-not (Test-Path -LiteralPath $pyScript -PathType Leaf)) {
    Write-Output "错误：缺少预下载脚本：$pyScript"
    exit 1
}

$env:ST_VERIFY = $verify
$env:ST_RETRIES = "$retries"

Write-Output '--- 预下载 NF4 模型（约 5.6GB，可断点续传）---'
& $PythonExe $pyScript
$code = $LASTEXITCODE

if ($origPyPath) { $env:PYTHONPATH = $origPyPath }
Remove-Item Env:ST_VERIFY, Env:ST_RETRIES -ErrorAction SilentlyContinue

if ($code -ne 0) {
    Write-Output ''
    Write-Output "错误：模型未就绪。可换源重试： -Endpoint https://huggingface.co"
    Write-Output "或检查本机代理（FlClash）是否在线；下载期间不要关代理。"
    exit 1
}

Write-Output ''
if ($VerifyOnly) { Write-Output '两个模型缓存完整，可以直接跑推理。' }
else { Write-Output '模型已就绪，可以运行 scripts\infer_psd.cmd（可加 -Offline 跳过联网校验）。' }
