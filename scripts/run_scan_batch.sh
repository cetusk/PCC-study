#!/usr/bin/env bash
# 走査モデルを手持ちの全 LAS/LAZ で測る。直列実行（タイル規模でピーク 26GB 使うため）。
#
#   bash scripts/run_scan_batch.sh            # 全部
#   bash scripts/run_scan_batch.sh 3          # 3 番目から
#   bash scripts/run_scan_batch.sh 1 5        # 1〜5 番目だけ
#
# 各実行のログは data/work/batch/ に残る。--trace と PCC_SCAN_DEBUG=1 を付けるので、
# 幾何の全候補（走査v1〜v4 を含む）と走査モデルの内訳が記録される。
# 1 件が失敗しても止まらない。安いものと一般性に効くものを先に置いてあるので、
# 途中で打ち切っても結果は残る。
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
export LD_LIBRARY_PATH="$HOME/tools/laszip-install/lib:${LD_LIBRARY_PATH:-}"
export PCC_SCAN_DEBUG=1
BIN=./cpp/build/pccnorm
OUT=data/work/batch
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "$BIN が無い。cpp をビルドすること" >&2; exit 1; }

# 名前|入力|追加引数
JOBS=(
  "small_autzen_trim|data/raw/small/autzen_trim.laz|"
  "small_plane|data/raw/small/plane.laz|"
  "small_fullwave|data/raw/small/fullwave.laz|"
  "small_simple1_4|data/raw/small/simple1_4.las|"
  "small_vegetation|data/raw/small/vegetation_1_3.las|"
  "red_rocks|data/raw/extrabytes/entwine_data_red-rocks.laz|"
  "workshop_TerraScan|data/raw/extrabytes/workshop_TM_551_101.laz|"
  "autzen2023_LasMonkey|data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz|"
  "ahn5_31HZ1_20|data/raw/ahn5/31HZ1_20.LAZ|"
  "ahn3_31HZ1_20|data/raw/ahn3/31HZ1_20.LAZ|"
  "ahn4_20M|data/raw/ahn4/31HZ1_20.LAZ|--max-points 20000000"
  "ahn4_tile_sample|data/raw/ahn4/31HZ1_20.LAZ|--sample-select 2000000"
  "ahn4_tile|data/raw/ahn4/31HZ1_20.LAZ|"
  "ahn4_tile21|data/raw/ahn4/31HZ1_21.LAZ|"
)

START=${1:-1}
END=${2:-${#JOBS[@]}}
echo "== バッチ開始 $(date '+%F %T')  全 ${#JOBS[@]} 件 / $START〜$END 番目 =="
i=0
for job in "${JOBS[@]}"; do
  i=$((i+1))
  [ "$i" -ge "$START" ] || continue
  [ "$i" -le "$END" ] || continue
  IFS='|' read -r name path extra <<< "$job"
  if [ ! -s "$path" ]; then echo "[$i/${#JOBS[@]}] skip（無い）: $path"; continue; fi
  echo "[$i/${#JOBS[@]}] $name  $(date '+%T')"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  $BIN pack "$path" "$OUT/$name.pcc2" --trace $extra \
      > "$OUT/$name.log" 2> "$OUT/$name.err" || echo "  !! 失敗（exit $?）" >&2
  t1=$(date +%s)
  sz=$(stat -c %s "$OUT/$name.pcc2" 2>/dev/null || echo 0)
  rm -f "$OUT/$name.pcc2"          # 数百 MB になるので残さない。数値はログにある
  printf "  %s 秒  出力 %.1f MB  %s\n" "$((t1-t0))" "$(echo "$sz/1000000" | bc -l)" \
         "$(grep -m1 '検証' "$OUT/$name.log" 2>/dev/null || echo '検証行なし')"
done

echo
echo "== 一覧 =="
printf "%-22s %12s %10s %10s %10s %10s %10s %10s %8s\n" \
       ファイル 点数 幾何v3 走査v1 走査v2 走査v3 走査v4 PCC2合計 検証
for job in "${JOBS[@]}"; do
  IFS='|' read -r name path extra <<< "$job"
  f="$OUT/$name.log"; [ -s "$f" ] || continue
  g()  { grep -oP "^\s+$1\s+\K[\d.]+" "$f" | head -1; }
  pts=$(grep -oP '^\s+\K[\d]+(?= 点 / 出所)' "$f" | head -1)
  tot=$(grep -oP '^PCC2\s+[\d.]+ MB\s+\K[\d.]+' "$f" | head -1)
  ver=$(grep -oP '全列一致 = \K\w+' "$f" | head -1)
  printf "%-22s %12s %10s %10s %10s %10s %10s %10s %8s\n" \
         "$name" "${pts:--}" "$(g 幾何v3)" "$(g 走査v1)" "$(g 走査v2)" \
         "$(g 走査v3)" "$(g 走査v4)" "${tot:--}" "${ver:--}"
done
echo "== バッチ終了 $(date '+%F %T') =="
