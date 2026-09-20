#!/usr/bin/env bash
# C: 符号器が内部で並べ替える設定で、順序感度が変わるかを測る。
# G-PCC だけを回す（PCC2 は関係しない）。
set -u
cd "$(dirname "$0")/.."
run() {
  local tag="$1"; shift
  echo "=== $tag : $* ==="
  PCC_GPCC_ONLY=1 PCC_TMC13_ARGS="$*" \
    "$PCCPY" python/exp_order_matrix.py "data/work/order_sort_$tag.txt" \
    > "data/work/sort_$tag.log" 2>&1
  echo "  行 $(grep -cE 'ok$|NG$' "data/work/order_sort_$tag.txt")"
}
run azimuth "--sortInputByAzimuth=1"
run predgeom "--geomTreeType=1 --predGeomSort=1"
