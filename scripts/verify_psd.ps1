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

$code = @'
import sys, os
from psd_tools import PSDImage
from PIL import Image, ImageDraw

psd_path = sys.argv[1]
out_dir  = sys.argv[2] if len(sys.argv) > 2 else ""
psd = PSDImage.open(psd_path)
if psd is None:
    raise SystemExit("cannot open PSD")

def collect(layers, depth=0):
    out = []
    for l in layers:
        if l.is_group():
            out.extend(collect(l, depth + 1))
        else:
            out.append((l, depth))
    return out

flat = collect(psd)
print("PSD_SIZE=%dx%d" % (psd.size[0], psd.size[1]))
print("TOP_LEVEL=%d" % len(psd))
print("PIXEL_LAYERS=%d" % len(flat))

def layer_image(l):
    for fn in ("topil", "composite"):
        f = getattr(l, fn, None)
        if f is None:
            continue
        try:
            im = f()
            if im is not None:
                return im.convert("RGBA")
        except Exception:
            pass
    return None

def on_white(im):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

imgs = []
nonempty = 0
for l, d in flat:
    im = layer_image(l)
    if im is not None and im.getchannel("A").getbbox() is not None:
        nonempty += 1
    imgs.append((l, im))

print("NONEMPTY=%d" % nonempty)
print("NAMES=" + " | ".join((l.name or "unnamed") for l, _ in flat))

if out_dir:
    stem = os.path.splitext(os.path.basename(psd_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    # psd.composite() 在本版本返回 RGB，透明区被填成黑色；
    # 这里按图层顺序手动叠到白底，更接近在 PS 里看到的效果。
    W, H = psd.size
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    for l, im in imgs:
        if im is None or not l.visible:
            continue
        patch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        patch.paste(im, l.offset)
        canvas = Image.alpha_composite(canvas, patch)
    p = os.path.join(out_dir, stem + "_composite.png")
    canvas.convert("RGB").save(p)
    print("COMPOSITE=" + p)

    cols, cw, ch, thumb = 4, 250, 270, 230
    rows = max(1, (len(imgs) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cw, rows * ch), (244, 245, 248))
    draw = ImageDraw.Draw(sheet)
    for i, (l, im) in enumerate(imgs):
        x, y = (i % cols) * cw, (i // cols) * ch
        draw.rectangle([x + 1, y + 1, x + cw - 2, y + ch - 2], outline=(214, 216, 222))
        if im is not None:
            t = on_white(im)
            t.thumbnail((thumb, thumb), RESAMPLE)
            sheet.paste(t, (x + (cw - t.width) // 2, y + 8 + (thumb - t.height) // 2))
        draw.text((x + 10, y + 244), (l.name or "unnamed")[:32], fill=(35, 38, 45))
    p = os.path.join(out_dir, stem + "_layers.png")
    sheet.save(p)
    print("SHEET=" + p)
'@

$lines = & $PythonExe -c $code $Psd $previewFull 2>&1
$exit  = $LASTEXITCODE
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
