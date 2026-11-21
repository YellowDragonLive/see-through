#!/usr/bin/env bash
# fix_all.sh — 对 see-through 历史产物里所有「原始 PSD」批量重跑 fix.py 后处理
#
# 说明：
#   - 扫描 workspace 下所有 *.psd，排除 *_depth.psd / *_fixed.psd / old_psd/（归档区不参与）
#   - 每张输出到同目录 <stem>_fixed.psd（默认覆盖旧文件；--keep 保留已有）
#   - 有 <stem>_depth.psd 的走完整四阶段（拆臂/拆胸前垂发/排序/清嘴）；
#     没有 depth 的自动降级为纯规则排序（fix.py 内部处理）
#   - 日志：log/fixall_<时间戳>/<序号>_<stem>.log + status.tsv + run.log
#
# 用法：
#   bash fix_all.sh                 # 全量（并行 2）
#   bash fix_all.sh -j 3            # 并行 3
#   bash fix_all.sh --dry-run       # 只列清单
#   bash fix_all.sh --only 1280     # 只跑路径含 1280 的
#   bash fix_all.sh --keep          # 跳过已有 _fixed.psd
#   bash fix_all.sh --preview       # 额外输出 PNG 预览
#   bash fix_all.sh --order depth   # 指定排序策略（默认 auto）
#
# 注意：本机沙盒会静默杀掉长任务，务必用 systemd-run 起：
#   systemd-run --user --unit=seethrough-fixall --collect \
#     --property=StandardOutput=file:/path/run.log \
#     bash fix_all.sh
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PY="${PY:-$HOME/miniforge3/envs/seethrough/bin/python}"
FIX_PY="$REPO/fix.py"
SCAN_ROOT="${SCAN_ROOT:-$HERE}"

JOBS=2
ORDER=auto
PREVIEW=0
KEEP=0
DRY=0
ONLY=""
SCAN_OLD=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
      exit 0 ;;
    -j|--jobs) [ $# -ge 2 ] || { echo "[FAIL] -j 需要数字"; exit 1; }; JOBS="$2"; shift ;;
    --order)   [ $# -ge 2 ] || { echo "[FAIL] --order 需要取值"; exit 1; }; ORDER="$2"; shift ;;
    --only)    [ $# -ge 2 ] || { echo "[FAIL] --only 需要关键字"; exit 1; }; ONLY="$2"; shift ;;
    --scan-old) SCAN_OLD=1 ;;
    --preview) PREVIEW=1 ;;
    --keep)    KEEP=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "[FAIL] 不认识的参数：$1（-h 看用法）"; exit 1 ;;
  esac
  shift
done

[[ "$JOBS" =~ ^[0-9]+$ && "$JOBS" -ge 1 ]] || { echo "[FAIL] -j 必须是正整数"; exit 1; }
[ -f "$FIX_PY" ] || { echo "[FAIL] 找不到 fix.py: $FIX_PY"; exit 1; }
[ -x "$PY" ]     || { echo "[FAIL] 找不到 python: $PY"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG="$HERE/log/fixall_$TS"
mkdir -p "$LOG"

# ---------- 收集清单 ----------
LIST="$LOG/list.tsv"
: > "$LIST"
# 归档区 old_psd 默认跳过；--scan-old 时反过来只处理归档区
if [ "$SCAN_OLD" -eq 1 ]; then
  OLDFILTER=( -path "*/old_psd/*" )
else
  OLDFILTER=( ! -path "*/old_psd/*" )
fi
while IFS= read -r f; do
  base=$(basename "$f" .psd)
  [ -f "${f%.psd}_depth.psd" ] && has=1 || has=0
  if [ -n "$ONLY" ] && [[ "$f" != *"$ONLY"* ]]; then continue; fi
  if [ "$KEEP" -eq 1 ] && [ -f "${f%.psd}_fixed.psd" ]; then
    echo "SKIP(done)	$f" >> "$LIST"; continue
  fi
  echo "TODO	$has	$f" >> "$LIST"
done < <(find "$SCAN_ROOT" -type f -name "*.psd" \
           ! -name "*_depth.psd" ! -name "*_fixed.psd" \
           "${OLDFILTER[@]}" | sort)

TOTAL=$(grep -c '^TODO' "$LIST" || true)
WITH_D=$(awk -F'\t' '$1=="TODO" && $2==1' "$LIST" | wc -l)
NO_D=$((TOTAL - WITH_D))

echo "=============================================================="
echo " see-through 历史 PSD 批量 fix"
echo " 时间    : $TS"
echo " 扫描根  : $SCAN_ROOT"
echo " 待处理  : $TOTAL 张（有 depth $WITH_D / 无 depth $NO_D）"
echo " 并行    : $JOBS    排序: $ORDER    预览: $([ $PREVIEW -eq 1 ] && echo 是 || echo 否)"
echo " 日志    : $LOG"
echo "=============================================================="

if [ "$DRY" -eq 1 ]; then
  awk -F'\t' '$1=="TODO" { printf "  [%s] %s\n", ($2==1?"D":"-"), $3 }' "$LIST"
  echo "（--dry-run，未执行）"
  exit 0
fi
[ "$TOTAL" -eq 0 ] && { echo "没有待处理文件。"; exit 0; }

STATUS="$LOG/status.tsv"
: > "$STATUS"

# ---------- 单张处理（由 xargs 调用） ----------
one() {
  local f="$1" has="$2" log="$3" idx="$4" status="$5"
  local stem out lg t0 t1 rc
  stem=$(basename "$f" .psd)
  out="${f%.psd}_fixed.psd"
  lg="$log/${idx}_${stem}.log"
  t0=$(date +%s)
  local args=("$f" --order "$ORDER")
  [ "$PREVIEW" -eq 1 ] && args+=(--preview)
  "$PY" "$FIX_PY" "${args[@]}" > "$lg" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    printf '%s\tOK\t%ss\t%s\t%s\n' "$stem" "$((t1-t0))" "$(du -h "$out"|cut -f1)" "$has" >> "$status"
    echo "  [ OK ] ${stem}  $((t1-t0))s  $(du -h "$out"|cut -f1)"
  else
    printf '%s\tFAIL(rc=%s)\t-\t-\t%s\n' "$stem" "$rc" "$has" >> "$status"
    echo "  [FAIL] ${stem}  rc=$rc  → $lg"
  fi
}

export -f one
export PY FIX_PY ORDER PREVIEW

START=$(date +%s)
echo
echo "[1/2] 逐张 fix.py（并行 $JOBS）"
idx=0
while IFS=$'\t' read -r tag has f; do
  [ "$tag" = "TODO" ] || continue
  idx=$((idx+1))
  printf '%s\t%s\t%s\t%03d\n' "$f" "$has" "$LOG" "$idx"
done < "$LIST" | xargs -P "$JOBS" -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r f has log idx <<< "{}"
  one "$f" "$has" "$log" "$idx" "'"$STATUS"'"
' | tee "$LOG/progress.log"

END=$(date +%s)
OK=$(awk -F'\t' '$2=="OK"' "$STATUS" | wc -l)
NG=$(awk -F'\t' '$2 ~ /^FAIL/' "$STATUS" | wc -l)

echo
echo "[2/2] 汇总"
echo "  成功 $OK / 失败 $NG / 总 $TOTAL   总耗时 $(( (END-START)/60 ))分$(( (END-START)%60 ))秒"
echo "  明细: $STATUS"
if [ "$NG" -gt 0 ]; then
  echo "  失败清单:"
  awk -F"\t" '$2 ~ /^FAIL/ {print "    - "$1" "$2}' "$STATUS"
  exit 1
fi
echo "  全部完成。"
