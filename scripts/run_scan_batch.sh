#!/usr/bin/env bash
# 走査モデルを手持ちの全 LAS/LAZ で測る。直列実行（タイル規模でピーク 26GB 使うため）。
#
#   bash scripts/run_scan_batch.sh            # 全部
#   bash scripts/run_scan_batch.sh 3          # 3 番目から
#   bash scripts/run_scan_batch.sh 1 5        # 1〜5 番目だけ
#
# 各実行のログは data/work/batch/ に残る。--trace と PCC_SCAN_DEBUG=1 を付けるので、
# 幾何の全候補と走査モデルの内訳が記録される。z の中央値予測と交差軸文脈の変種も
# 測るなら PCC_ALL_VARIANTS=1 を足す（既定は走査v1 と走査変換の 2 本）。
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
# red-rocks と simple1_4 は外してある。psid が定数なうえ gps_time も単一値で、
# 走査順そのものが作れないため走査モデルを試せない。
JOBS=(
  "small_autzen_trim|data/raw/small/autzen_trim.laz|"
  "small_plane|data/raw/small/plane.laz|"
  "small_fullwave|data/raw/small/fullwave.laz|"
  "small_vegetation|data/raw/small/vegetation_1_3.las|"
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
printf "%-22s %12s %9s %9s %9s %10s %9s %8s %6s\n" \
       ファイル 点数 幾何v3 走査v1 走査変換 採択 PCC2合計 中身 検証
for job in "${JOBS[@]}"; do
  IFS='|' read -r name path extra <<< "$job"
  f="$OUT/$name.log"; [ -s "$f" ] || continue
  g()  { grep -oP "^\s+$1\s+\K[\d.]+" "$f" | head -1; }
  pts=$(grep -oP '^\s+\K[\d]+(?= 点 / 出所)' "$f" | head -1)
  sel=$(grep -oP '^  X\+Y\+Z\s+\K\S+' "$f" | head -1)
  tot=$(grep -oP '^PCC2\s+[\d.]+ MB\s+\K[\d.]+' "$f" | head -1)
  emb=$(grep -q '^中身.*包んだ' "$f" && echo 包 || echo 自前)
  ver=$(grep -oP '全列一致 = \K\w+' "$f" | head -1)
  printf "%-22s %12s %9s %9s %9s %10s %9s %8s %6s\n" \
         "$name" "${pts:--}" "$(g 幾何v3)" "$(g 走査v1)" "$(g 走査変換)" \
         "${sel:--}" "${tot:--}" "$emb" "${ver:--}"
done
echo "== バッチ終了 $(date '+%F %T') =="
