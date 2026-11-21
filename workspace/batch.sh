#!/usr/bin/env bash
# ============================================================
#  see-through 批量分层 PSD —— 一键脚本（适配 v5.1 源头拆层）
# ============================================================
#
#  目录约定（都跟本脚本放在同一层）：
#    target/   放目标图片（png / jpg / jpeg / webp）
#    output/   放产物，每次运行开一个以时间命名的子目录
#    log/      放运行情况：ASCII 输入副本、逐张日志、采样数据、资源报表
#
#  用法：
#    bash batch.sh                      # 默认 1024px，直接走 v5.1 源头拆层与排序
#    bash batch.sh 1280                 # 指定分辨率（8GB 卡的天花板就是 1280）
#    bash batch.sh 1024 --no-preview    # 跳过预览图合成
#    bash batch.sh 1024 --fix           # 额外再跑一次外部 fix.py 后处理（默认关闭）
#    bash batch.sh 1024 --no-fix        # 关闭外部 fix.py（默认）
#    bash batch.sh 1024 --order depth   # 外部 fix.py 排序策略（需搭配 --fix）
#    bash batch.sh 1024 --depth-res 768 # 伪深度分辨率（默认 768，Marigold 原生训练分辨率）
#    bash batch.sh 1024 --steps 30      # 扩散推理步数（默认 30）
#
#  说明：
#    - 每张图独立进程，显存用完即释放，批量比连跑更稳
#    - 输入会复制成 ASCII 名 t01/t02...，避免中文进 PSD 图层名
#    - PSD 画布恒为 resolution × resolution（算法会居中补成正方形）
#    - v5.1 已在 common/utils/inference_utils.py 源头内置前后衣领伪深度三明治拆分、
#      拆臂/垂发、嘴部与睫毛清理、边缘去色晕与解剖排序，默认不再叠加外部 fix.py
# ============================================================
set -u

# ---------- 路径基准：以脚本所在目录为准 ----------
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

TARGET="${TARGET:-$HERE/target}"
OUTROOT="${OUTROOT:-$HERE/output}"
LOGROOT="${LOGROOT:-$HERE/log}"

RES=1024
DEPTH_RES=768
STEPS=30
SEED=42
PREVIEW=1
DO_FIX=0
ORDER=auto
FIX_PY="$REPO/fix.py"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
      exit 0 ;;
    --no-preview) PREVIEW=0 ;;
    --fix)        DO_FIX=1 ;;
    --no-fix)     DO_FIX=0 ;;
    --order)
      [ $# -ge 2 ] || { echo "[FAIL] --order 需要取值：auto|depth|raw|classic|arms_front|arms_back|hair_over_top"; exit 1; }
      ORDER="$2"; shift ;;
    --depth-res)
      [ $# -ge 2 ] || { echo "[FAIL] --depth-res 需要整数取值（如 768）"; exit 1; }
      DEPTH_RES="$2"; shift ;;
    --steps)
      [ $# -ge 2 ] || { echo "[FAIL] --steps 需要整数取值（如 30）"; exit 1; }
      STEPS="$2"; shift ;;
    [0-9]*)       RES="$1" ;;
    *)            echo "[FAIL] 不认识的参数：$1（-h 看用法）"; exit 1 ;;
  esac
  shift
done
[[ "$RES" =~ ^[0-9]+$ ]]       || { echo "[FAIL] 分辨率必须是数字：$RES"; exit 1; }
[[ "$DEPTH_RES" =~ ^[0-9]+$ ]] || { echo "[FAIL] 深度分辨率必须是数字：$DEPTH_RES"; exit 1; }
[[ "$STEPS" =~ ^[0-9]+$ ]]     || { echo "[FAIL] 推理步数必须是数字：$STEPS"; exit 1; }

PY="${PY:-$HOME/miniforge3/envs/seethrough/bin/python}"
TS=$(date +%Y%m%d_%H%M%S)
OUT_BATCH="$OUTROOT/$TS"
LOG_BATCH="$LOGROOT/$TS"

hr()   { printf '%s\n' "--------------------------------------------------------------------"; }
ok()   { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; exit 1; }

echo "===================================================================="
echo " see-through 批量分层   ${RES}px / 深度 ${DEPTH_RES}px / NF4 / ${STEPS} 步 / RTX 3060 Ti"
echo "===================================================================="

# ---------- 1. 环境检查 ----------
hr; echo "[1/5] 环境检查"
[ -x "$PY" ]  || fail "找不到 seethrough 的 python：$PY"
[ -d "$REPO" ] || fail "找不到项目目录：$REPO"
[ -d "$TARGET" ] || { mkdir -p "$TARGET"; fail "target 目录不存在，已创建：$TARGET
       把图片放进去再跑一次。"; }

nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/  /' \
  || fail "nvidia-smi 不可用"

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
# 实测锚点：1024px 峰值 6554 MiB，1280px 峰值 7803 MiB，按像素数内插/外推
NEED_MIB=$(awk -v r="$RES" 'BEGIN{
  a=1024*1024; b=1280*1280;
  v = 6554 + (7803-6554)*((r*r-a)/(b-a));
  if (v < 6554) v = 6554;
  printf "%d", v }')
echo "  显存预算：${RES}px 预计峰值 ${NEED_MIB} MiB / 当前空闲 ${FREE_MIB} MiB"

if pgrep -f "[m]ain.py --port 8189" >/dev/null 2>&1; then
  warn "ComfyUI/MOSS 常驻实例还在跑（约占 6.9 GB 显存）"
  echo "         先释放：bash ~/run_moss.sh --stop-server"
fi
if [ "$FREE_MIB" -lt "$NEED_MIB" ]; then
  warn "空闲显存不足，很可能 CUDA OOM"
  echo "         清显存：bash ~/cleanup_gpu.sh    （或直接重启）"
fi

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WIN_HUB="/media/eric/Windows/Users/13410/.cache/huggingface/hub"
if [ ! -d "$WIN_HUB" ] && [ -b /dev/disk/by-uuid/781046F51046BA40 ]; then
  echo "  [INFO] Windows 分区未挂载，自动挂载 ..."
  udisksctl mount -b /dev/disk/by-uuid/781046F51046BA40 2>&1 | sed 's/^/    /'
  sleep 1
fi
if [ -d "$WIN_HUB" ]; then
  export HUGGINGFACE_HUB_CACHE="$WIN_HUB"
  export HF_HUB_OFFLINE=1
  ok "模型缓存：Windows 分区离线命中（不下载）"
else
  warn "未找到 Windows 缓存，将在线下载约 5.7 GB"
fi

# ---------- 2. 收集输入 ----------
hr; echo "[2/5] 收集输入图片"
mkdir -p "$OUT_BATCH" "$LOG_BATCH/inputs"
FIXDIR="$OUT_BATCH/fixed"
FIXLOG="$LOG_BATCH/fix_status.tsv"
if [ "$DO_FIX" -eq 1 ]; then
  mkdir -p "$FIXDIR"
  : > "$FIXLOG"
fi
FIX_OK=0
FIX_NG=0

mapfile -t SRCS < <(find "$TARGET" -maxdepth 1 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | sort)
[ "${#SRCS[@]}" -gt 0 ] || fail "target 里没有图片：$TARGET"

MAP="$LOG_BATCH/inputs/map.tsv"
: > "$MAP"
i=0
for src in "${SRCS[@]}"; do
  i=$((i+1))
  ext="${src##*.}"; ext="${ext,,}"
  dst="$LOG_BATCH/inputs/$(printf 't%02d' "$i").${ext}"
  cp -f "$src" "$dst" || fail "复制失败：$src"
  printf '%s\t%s\t%s\n' "$(basename "$dst")" "$src" "$(du -h "$src" | cut -f1)" >> "$MAP"
done
ok "共 ${#SRCS[@]} 张 → ASCII 副本在 log/${TS}/inputs/"
sed 's/^/    /' "$MAP"

# ---------- 3. 采样 ----------
hr; echo "[3/5] 启动资源采样"
CSV="$LOG_BATCH/metrics.csv"
REPORT="$LOG_BATCH/report.md"
METRICS_QUIET=1 bash "$HOME/record_metrics.sh" 1 "$CSV" &
SAMPLER=$!
sleep 1
ok "采样中（每秒 1 点）"

stop_sampler() {
  kill -TERM "$SAMPLER" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SAMPLER" 2>/dev/null || break; sleep 0.2; done
  kill -9 "$SAMPLER" 2>/dev/null
  wait "$SAMPLER" 2>/dev/null
}

# ---------- 4. 逐张推理 ----------
hr
if [ "$DO_FIX" -eq 1 ]; then
  echo "[4/5] 逐张推理（v5.1 源头拆层）+ 外部 fix.py 后处理"
else
  echo "[4/5] 逐张推理（v5.1 源头拆层与排序，跳过外部 fix.py）"
fi
cd "$REPO" || fail "无法进入 $REPO"

TIMINGS="$LOG_BATCH/timings.tsv"
: > "$TIMINGS"
T_ALL0=$(date +%s)
FAILED=()
n=0

while IFS=$'\t' read -r name orig size; do
  n=$((n+1))
  stem="${name%.*}"
  LOG="$LOG_BATCH/${stem}.log"
  echo
  hr
  echo "[$n/${#SRCS[@]}] $stem  ←  $(basename "$orig")  ($size)"
  hr

  T0=$(date +%s)
  "$PY" inference/scripts/inference_psd_quantized.py \
    --srcp "$LOG_BATCH/inputs/$name" \
    --save_dir "$OUT_BATCH" \
    --save_to_psd \
    --resolution "$RES" \
    --resolution_depth "$DEPTH_RES" \
    --quant_mode nf4 \
    --no_group_offload \
    --num_inference_steps "$STEPS" \
    --seed "$SEED" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  dur=$(( $(date +%s) - T0 ))

  if [ $rc -ne 0 ]; then
    if grep -qiE "out of memory|OutOfMemory|CUDA error|illegal memory" "$LOG"; then
      why="显存错误"
    else
      why="非显存错误"
    fi
    echo "[FAIL] $stem 失败（rc=$rc / $why / ${dur}s）→ log/${TS}/${stem}.log"
    FAILED+=("$stem:$why")
    printf '%s\t%s\tFAIL(%s)\n' "$stem" "$dur" "$why" >> "$TIMINGS"
    if [ "$why" = "显存错误" ]; then
      echo "[ABORT] 显存问题，后面的不再跑（先清显存再重新执行本脚本）"
      break
    fi
    continue
  fi

  seg=$(grep -oE '"[a-z_]*time[a-z_]*": *[0-9.]+' "$OUT_BATCH/$stem.psd.json" 2>/dev/null | tr '\n' ' ')
  echo "[DONE] $stem  ${dur}s  ${seg}"
  printf '%s\t%s\tOK\t%s\n' "$stem" "$dur" "$seg" >> "$TIMINGS"

  # ---- 可选：外部 fix.py 后处理（仅在显式传 --fix 时启用） ----
  if [ "$DO_FIX" -eq 1 ]; then
    if [ -f "$FIX_PY" ]; then
      fx="$LOG_BATCH/fix_${stem}.log"
      if "$PY" "$FIX_PY" "$OUT_BATCH/$stem.psd" -o "$FIXDIR/${stem}_fixed.psd" \
           --preview --order "$ORDER" > "$fx" 2>&1; then
        nfx=$(grep -cE '^  \[' "$fx" 2>/dev/null || true)
        echo "[FIX ] $stem → fixed/${stem}_fixed.psd（${nfx:-0} 处修正）"
        FIX_OK=$((FIX_OK+1))
        printf '%s\tOK\t%s\n' "$stem" "${nfx:-0}" >> "$FIXLOG"
      else
        warn "$stem 后处理失败（log/${TS}/fix_${stem}.log），原始 PSD 未受影响"
        FIX_NG=$((FIX_NG+1))
        printf '%s\tFAIL\t-\n' "$stem" >> "$FIXLOG"
      fi
    else
      warn "找不到 fix.py：$FIX_PY，跳过拆层后处理"
      DO_FIX=0
    fi
  fi
done < "$MAP"

T_ALL=$(($(date +%s) - T_ALL0))
stop_sampler

# ---------- 5. 预览 + 报表 ----------
hr; echo "[5/5] 生成预览与报表"

if [ "$PREVIEW" -eq 1 ]; then
  "$PY" - "$OUT_BATCH" <<'PYEOF' 2>&1 | sed 's/^/  /'
import sys, glob, os
try:
    from psd_tools import PSDImage
except Exception as e:
    print(f"[WARN] psd_tools 不可用，跳过预览：{e}"); sys.exit(0)
d = sys.argv[1]
for p in [x for x in sorted(glob.glob(os.path.join(d, "*.psd"))) if not x.endswith("_depth.psd")]:
    try:
        img = PSDImage.open(p).composite()
        out = p[:-4] + "_preview.png"
        img.save(out)
        print(f"{os.path.basename(out):28s} {img.size[0]}x{img.size[1]}  {os.path.getsize(out)//1024} KB")
    except Exception as e:
        print(f"[WARN] {os.path.basename(p)} 合成失败：{e}")
PYEOF
fi

{
  echo "运行时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "分辨率：${RES}px   深度分辨率：${DEPTH_RES}px   图片数：${#SRCS[@]}"
  echo "总耗时：$((T_ALL/60)) 分 $((T_ALL%60)) 秒"
  echo
  printf '%-8s %-8s %-6s %s\n' "图片" "耗时" "状态" "分项"
  while IFS=$'\t' read -r a b c d; do
    printf '%-8s %-8s %-6s %s\n' "$a" "${b}s" "$c" "${d:-}"
  done < "$TIMINGS"
} | tee "$LOG_BATCH/summary.txt" | sed 's/^/  /'

if [ -s "$CSV" ]; then
  bash "$HOME/report_metrics.sh" "$CSV" "see-through 批量 ${RES}px × ${#SRCS[@]} 张" "$REPORT" 2>/dev/null \
    | sed -n '/## 一、总览/,$p'
fi

hr
echo "产物：$OUT_BATCH"
if [ "$DO_FIX" -eq 1 ]; then
  echo "       └ fixed/    外部 fix.py 后处理产物"
  echo "后处理：${FIX_OK} 张成功${FIX_NG:+, ${FIX_NG} 张失败}"
else
  echo "       （已由 v5.1 源头完成拆层、前后衣领分层与排序，可直接拖进 Anime2.5DRig）"
fi
echo "日志：$LOG_BATCH"
echo
echo "下次使用：把图片放进 target/，然后  bash batch.sh"

[ "${#FAILED[@]}" -eq 0 ] || { echo; warn "失败 ${#FAILED[@]} 张：${FAILED[*]}"; exit 1; }
exit 0