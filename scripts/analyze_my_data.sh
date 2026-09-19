#!/usr/bin/env bash
# 手元の実データを一式解析する。
#   bash scripts/analyze_my_data.sh <file.laz|file.las|file.ply|file.bin> [点数上限]
#
# data/incoming/ にファイルを置いて、そのパスを渡すだけでよい。
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
F="${1:?使い方: bash scripts/analyze_my_data.sh <file> [点数上限]}"
N="${2:-2000000}"
STEM="$(basename "${F%.*}")"
mkdir -p results

echo "=================================================================="
echo " 1. ビット内訳レポート（どこにビットが使われているか）"
echo "=================================================================="
$PCCPY python/run_report.py "$F" --max-points "$N" --kmax 9 \
       --out "results/${STEM}_report.md"

echo
echo "=================================================================="
echo " 2. 表現の正規化レイヤー（可逆）"
echo "=================================================================="
$PCCPY python/bench_normalize.py "$F" --max-points "$N"

echo
echo "=================================================================="
echo " 3. 精度要件を動かした場合（誤差上限つき）"
echo "=================================================================="
for g in 2 3 4 5; do
  echo "--- 下位 ${g} ビットを落とす ---"
  $PCCPY python/bench_normalize.py "$F" --max-points "$N" --grid-bits "$g" \
    | grep -E '^B\. 合計|^削減|^検証'
done

echo
echo "=================================================================="
echo " 4. 点間隔以下の構造"
echo "=================================================================="
$PCCPY python/exp_subspacing.py "$F"

echo
echo "結果: results/${STEM}_report.md"
