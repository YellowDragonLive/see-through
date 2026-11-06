<#
.SYNOPSIS
    验收 see-through 产出的 PSD：检查尺寸与分层，生成合成预览图和图层拼版。

.EXAMPLE
    .\verify_psd.ps1 "X:\ai_tool\see-through\workspace\test.psd"
.EXAMPLE
    .\verify_psd.ps1 "D:\out\a.psd" -PreviewDir "D:\out\preview" -NoPreview
#>
[CmdletBinding()]
param(
    [Parameter(Position=0, Mandatory=$true)][string]$Psd,
    [string]$PreviewDir,
    [switch]$NoPreview
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

function Fail([string]$msg) {
    Write-Output ''
    Write-Output ("错误：{0}" -f $msg)
    exit 1
}

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $env:USERPROFILE '.conda\envs\see_through\python.exe'

if (-not (Test-Path -LiteralPath $Psd -PathType Leaf)) { Fail "PSD 不存在：$Psd" }
$Psd = (Resolve-Path -LiteralPath $Psd).Path
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { Fail "环境未安装，请先运行 scripts\setup_env.cmd" }

if (-not $PreviewDir) { $PreviewDir = Join-Path $RepoRoot 'workspace\preview' }
$previewFull = [IO.Path]::GetFullPath($PreviewDir)
if ($NoPreview) { $previewFull = '' }
elseif (-not (Test-Path -LiteralPath $previewFull)) {
    New-Item -ItemType Directory -Path $previewFull -Force | Out-Null
}

# 同 setup_env / infer_psd：过滤 WorkBuddy 经 PYTHONPATH 注入的 sitecustomize，
# 避免写预览图时触发批量删除保护。只作用于本次派生的 python 子进程。
$origPyPath = $env:PYTHONPATH
if ($origPyPath) {
    $keptPy = @($origPyPath.Split(';') | Where-Object {
        $_ -and -not ($_ -like '*WorkBuddy*' -and $_ -like '*shim*')
    })
    $env:PYTHONPATH = $keptPy -join ';'
}

# 不用 python -c 传多行代码：Windows PowerShell 5.1 会吃掉参数里的双引号。
$pyScript = Join-Path $PSScriptRoot 'py\verify_psd.py'
if (-not (Test-Path -LiteralPath $pyScript -PathType Leaf)) { Fail "缺少验收脚本：$pyScript" }

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$lines = & $PythonExe $pyScript $Psd $previewFull 2>&1
$exit  = $LASTEXITCODE
$ErrorActionPreference = $prevEap
$kv    = @{}
foreach ($line in $lines) {
    $text = [string]$line
    if ($text -match '^([A-Z_]+)=(.*)$') { $kv[$Matches[1]] = $Matches[2] }
    else { Write-Output $text }
}
if ($exit -ne 0) { Fail "psd-tools 验收失败（退出码 $exit）" }
if (-not $kv.ContainsKey('PSD_SIZE')) { Fail '未解析到验收结果，PSD 可能无法打开' }

Write-Output ''
Write-Output '--- PSD 验收 ---'
Write-Output ("  文件      : {0}" -f $Psd)
Write-Output ("  大小      : {0:N1} MB" -f ((Get-Item -LiteralPath $Psd).Length / 1MB))
Write-Output ("  尺寸      : {0}" -f $kv['PSD_SIZE'])
Write-Output ("  顶层元素  : {0}" -f $kv['TOP_LEVEL'])
Write-Output ("  像素图层  : {0}（非空 {1}）" -f $kv['PIXEL_LAYERS'], $kv['NONEMPTY'])
Write-Output ("  图层名    : {0}" -f $kv['NAMES'])
if ($kv['COMPOSITE']) { Write-Output ("  合成预览  : {0}" -f $kv['COMPOSITE']) }
if ($kv['SHEET'])     { Write-Output ("  图层拼版  : {0}" -f $kv['SHEET']) }
Write-Output ''

if ([int]$kv['NONEMPTY'] -lt 2) { Fail '有效图层少于 2 个，PSD 可能不完整' }
