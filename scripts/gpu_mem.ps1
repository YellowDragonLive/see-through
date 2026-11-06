<#
.SYNOPSIS
    see-through 推理前的 GPU 显存体检与清理工具。

.DESCRIPTION
    默认只读：列出每个进程占用的显存、判断哪些可以安全关闭，并给出建议。
    只有同时传入 -Clean -Yes 才会真正结束进程。
    -Deep 额外临时结束 explorer（桌面/任务栏会消失），用 -RestoreExplorer 恢复。

.EXAMPLE
    .\gpu_mem.ps1                          仅诊断，不做任何改动
.EXAMPLE
    .\gpu_mem.ps1 -Clean                   预览将被关闭的进程，仍不改动
.EXAMPLE
    .\gpu_mem.ps1 -Clean -Yes              真正关闭可安全关闭的进程
.EXAMPLE
    .\gpu_mem.ps1 -Clean -Yes -Deep        再临时结束 explorer
.EXAMPLE
    .\gpu_mem.ps1 -RestoreExplorer        恢复桌面/任务栏
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Yes,
    [switch]$Deep,
    [switch]$RestoreExplorer
)

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if ($RestoreExplorer) {
    $ex = Get-Process -Name explorer -ErrorAction SilentlyContinue
    if (-not $ex) {
        Start-Process explorer.exe
        Write-Output '已重新启动 explorer，桌面 / 任务栏应已恢复。'
    } else {
        Write-Output ("explorer 已在运行（PID {0}），无需恢复。" -f ($ex.Id -join ','))
    }
    Write-Output ''
    return
}

# 绝对不动：Windows 桌面合成内核、会话宿主、脚本自身
$ProtectNames = @(
    'dwm', 'csrss', 'svchost', 'fontdrvhost', 'sihost',
    'ShellHost', 'ShellExperienceHost', 'StartMenuExperienceHost',
    'SearchHost', 'TextInputHost', 'ApplicationFrameHost',
    'CrossDeviceResume', 'msedgewebview2', 'conhost',
    'WorkBuddy', 'powershell', 'pwsh', 'cmd', 'WindowsTerminal',
    'HipsDaemon', 'HipsTray', 'usysdiag', 'MsMpEng'
)

# 可临时结束、但会影响桌面交互，仅在 -Deep 时处理
$DeepNames = @('explorer')

# 明确可安全关闭的普通应用（脚本只精确匹配这张白名单）
$CloseableNames = @(
    'chrome', 'msedge', 'firefox', 'brave', 'opera', '360chrome',
    'Weixin', 'WeChat', 'WeChatAppEx', 'QQ', 'TIM',
    'fdm', 'SwiftList', 'SwiftList.App', 'SystemSettings', 'PhoneExperienceHost', 'YourPhone',
    'GameViewer', 'SunloginClient', 'ToDesk', 'WallpaperEngine',
    'steam', 'EpicGamesLauncher', 'Telegram', 'Discord', 'Spotify',
    'PhotosApp', 'Video.UI', 'Notion', 'Feishu', 'DingTalk'
)

# 谨慎：关掉会断网 / 可能正在跑任务，只提示不自动关
$HoldNames = @(
    'FlClash', 'Clash', 'ClashforWindows', 'v2ray', 'NekoRay', 'sing-box',
    'python', 'pythonw', 'ComfyUI', 'conda', 'jupyter'
)

function Get-Vram {
    $raw = & nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null
    if (-not $raw) { return $null }
    $parts = @(($raw | Select-Object -First 1) -split ',' | ForEach-Object { $_.Trim() })
    if ($parts.Count -lt 3) { return $null }
    try {
        return [pscustomobject]@{
            Total = [int]$parts[0]
            Used  = [int]$parts[1]
            Free  = [int]$parts[2]
        }
    } catch { return $null }
}

function Get-GpuProcessList {
    $samples = $null
    try {
        $counter = Get-Counter '\GPU Process Memory(*)\Local Usage' -ErrorAction Stop
        $samples = $counter.CounterSamples
    } catch {
        Write-Output '  [警告] 无法读取 GPU 进程计数器，将退回 nvidia-smi 进程列表。'
        $samples = $null
    }

    $mbByPid = @{}
    if ($samples) {
        foreach ($s in $samples) {
            # 实例名形如 pid_10316_luid_..._phys_0，必须容忍 pid_ 前缀。
            # 不能用 -notmatch + $Matches：PowerShell 的 -notmatch 不填充 $Matches。
            $hit = [regex]::Match($s.InstanceName, '^(?:pid_)?(\d+)_')
            if (-not $hit.Success) { continue }
            $procId = [int]$hit.Groups[1].Value
            $mb = [math]::Round(($s.CookedValue / 1MB), 1)
            if ($mb -le 0) { continue }
            if ($mbByPid.ContainsKey($procId)) { $mbByPid[$procId] += $mb }
            else { $mbByPid[$procId] = $mb }
        }
    } else {
        $raw = & nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null
        foreach ($line in $raw) {
            $hit = [regex]::Match([string]$line, '^\s*(\d+)')
            if ($hit.Success) { $mbByPid[[int]$hit.Groups[1].Value] = 0 }
        }
    }

    $rows = @()
    foreach ($procId in $mbByPid.Keys) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($p) { $name = $p.ProcessName; $known = $true }
        else     { $name = '<系统/无权限>'; $known = $false }

        if (-not $known -or $procId -le 4)        { $cat = '保留' }
        elseif ($DeepNames -contains $name)      { $cat = '深度' }
        elseif ($ProtectNames -contains $name)   { $cat = '保留' }
        elseif ($HoldNames -contains $name)      { $cat = '谨慎' }
        elseif ($CloseableNames -contains $name) { $cat = '可关' }
        else                                     { $cat = '未分类' }

        $rows += [pscustomobject]@{
            PID      = $procId
            Name     = $name
            MB       = [math]::Round($mbByPid[$procId], 1)
            Category = $cat
        }
    }
    return @($rows | Sort-Object MB -Descending)
}

$vram = Get-Vram
if (-not $vram) {
    Write-Output '错误：读不到 NVIDIA 显卡信息，请确认驱动正常、nvidia-smi 可用。'
    return
}

Write-Output ''
Write-Output '==================== GPU 显存体检 ===================='
Write-Output ("  总计 {0} MiB   已用 {1} MiB   空闲 {2} MiB" -f $vram.Total, $vram.Used, $vram.Free)
Write-Output ''

$list = Get-GpuProcessList
Write-Output ('{0,9}  {1,-6}  {2,-8}  {3}' -f '占用MB', '分类', 'PID', '进程')
Write-Output ('-' * 62)
foreach ($r in $list) {
    Write-Output ('{0,9}  {1,-6}  {2,-8}  {3}' -f $r.MB, $r.Category, $r.PID, $r.Name)
}
Write-Output ''

$sumCloseable = ($list | Where-Object { $_.Category -eq '可关' }   | Measure-Object MB -Sum).Sum
$sumDeep      = ($list | Where-Object { $_.Category -eq '深度' }   | Measure-Object MB -Sum).Sum
$sumProtect   = ($list | Where-Object { $_.Category -eq '保留' }   | Measure-Object MB -Sum).Sum
$sumHold      = ($list | Where-Object { $_.Category -eq '谨慎' }   | Measure-Object MB -Sum).Sum
$sumOther     = ($list | Where-Object { $_.Category -eq '未分类' } | Measure-Object MB -Sum).Sum
if (-not $sumCloseable) { $sumCloseable = 0 }
if (-not $sumDeep)      { $sumDeep      = 0 }
if (-not $sumProtect)   { $sumProtect   = 0 }
if (-not $sumHold)      { $sumHold      = 0 }
if (-not $sumOther)     { $sumOther     = 0 }

Write-Output '---------------- 可回收性判断 ----------------'
Write-Output ("  可安全关闭 : {0,7:N1} MB   (-Clean -Yes)" -f $sumCloseable)
Write-Output ("  深度清理   : {0,7:N1} MB   (explorer，需额外 -Deep)" -f $sumDeep)
Write-Output ("  系统/会话  : {0,7:N1} MB   (dwm、WorkBuddy 等，脚本不碰)" -f $sumProtect)
Write-Output ("  谨慎       : {0,7:N1} MB   (代理、正在运行的 Python，不自动关)" -f $sumHold)
Write-Output ("  未分类     : {0,7:N1} MB   (脚本一律不动)" -f $sumOther)
Write-Output ''
Write-Output '  说明：本机 CPU 为 i7-13700F（无核显），Windows 桌面只能由'
Write-Output '        RTX 3060 Ti 渲染。dwm 的桌面合成占用无法消除；'
Write-Output '        explorer 可临时结束（-Deep），但桌面/任务栏会消失。'
Write-Output '        WorkBuddy 请手动退出，脚本不代劳（会中断当前会话）。'
Write-Output ''
Write-Output '---------------- 更有效的省显存做法 ----------------'
Write-Output '  1. 推理前先关 Chrome / 微信 / 下载器等（-Clean -Yes）'
Write-Output '  2. 首次推理降到 --resolution 1024，必要时 --resolution_depth 720'
Write-Output '  3. 加 --cpu_offload（更省显存，但更慢）'
Write-Output '  4. 设环境变量 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
Write-Output '  5. -Clean -Yes -Deep 临时关 explorer，跑完 -RestoreExplorer 恢复'
Write-Output '  6. 远程用完后在设备管理器禁用 GameViewer Virtual Display Adapter'
Write-Output ''

if (-not $Clean) {
    Write-Output '默认只读，未关闭任何进程。'
    Write-Output '  预览可关进程 : .\gpu_mem.ps1 -Clean'
    Write-Output '  确认后执行   : .\gpu_mem.ps1 -Clean -Yes'
    Write-Output '  极简模式     : .\gpu_mem.ps1 -Clean -Yes -Deep'
    Write-Output '  恢复桌面     : .\gpu_mem.ps1 -RestoreExplorer'
    Write-Output ''
    return
}

$targets = @($list | Where-Object { $_.Category -eq '可关' -and $_.PID -gt 4 } | Sort-Object MB -Descending)
$deepTargets = @()
if ($Deep) {
    $deepTargets = @($list | Where-Object { $_.Category -eq '深度' } | Sort-Object MB -Descending)
}
if ($targets.Count -eq 0 -and $deepTargets.Count -eq 0) {
    Write-Output '没有命中白名单中的可关闭进程，无需操作。'
    Write-Output ''
    return
}

Write-Output '---------------- 将被关闭的进程 ----------------'
foreach ($t in $targets) {
    Write-Output ('  PID {0,-8} {1,-24} 约 {2} MB' -f $t.PID, $t.Name, $t.MB)
}
foreach ($t in $deepTargets) {
    Write-Output ('  PID {0,-8} {1,-24} 约 {2} MB   [深度: 桌面/任务栏会消失]' -f $t.PID, $t.Name, $t.MB)
}
Write-Output ''

if (-not $Yes) {
    Write-Output '【预览模式】以上进程不会被关闭。'
    Write-Output '注意：关闭浏览器会丢失未保存的标签页 / 表单内容，请先保存。'
    if ($Deep) { Write-Output '注意：-Deep 请在独立的 cmd 窗口运行，别在 WorkBuddy 会话内执行。' }
    $cmdHint = '确认无误后执行： .\gpu_mem.ps1 -Clean -Yes'
    if ($Deep) { $cmdHint += ' -Deep' }
    Write-Output $cmdHint
    Write-Output ''
    return
}

Write-Output '正在关闭（最多 10 个，仅白名单进程）...'
$closed = 0
foreach ($t in ($targets | Select-Object -First 10)) {
    $p = Get-Process -Id $t.PID -ErrorAction SilentlyContinue
    if (-not $p) { continue }
    if ($ProtectNames -contains $p.ProcessName) { continue }
    Stop-Process -Id $t.PID -Force -ErrorAction SilentlyContinue
    $closed++
}
$deepClosed = 0
if ($Deep) {
    foreach ($t in $deepTargets) {
        $p = Get-Process -Id $t.PID -ErrorAction SilentlyContinue
        if (-not $p) { continue }
        if ($DeepNames -notcontains $p.ProcessName) { continue }
        Stop-Process -Id $t.PID -Force -ErrorAction SilentlyContinue
        $deepClosed++
    }
}
Start-Sleep -Seconds 2
$after = Get-Vram
Write-Output ''
Write-Output ("已结束 {0} 个普通进程、{1} 个 explorer 进程。" -f $closed, $deepClosed)
if ($after) {
    $reclaimed = $vram.Used - $after.Used
    Write-Output ("显存：已用 {0} MiB -> {1} MiB，空闲 {2} MiB（回收 {3} MiB）" -f `
        $vram.Used, $after.Used, $after.Free, $reclaimed)
}
if ($deepClosed -gt 0) {
    Write-Output '桌面 / 任务栏已隐藏。恢复命令： .\gpu_mem.ps1 -RestoreExplorer'
}
Write-Output 'WorkBuddy 需手动退出，脚本不会自动关闭会话宿主。'
Write-Output ''
