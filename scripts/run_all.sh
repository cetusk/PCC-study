#!/usr/bin/env bash
# 全データセットに全計測を一斉にかける。
# 毎回結果を見て次を変えるのではなく、同じ条件で並べて比べるための実行。
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH="$HOME/tools/laszip-install/lib:${LD_LIBRARY_PATH:-}"
B=cpp/build/pccnorm
S=data/raw/stanford
K=data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data
OUT=results/matrix
MAXP=${MAXP:-400000}

# slug|name|path|kind
# slug はファイル名に使うので英数字に限る（環境をまたぐときに事故らないため）。
# name は表示用で、make_tables.py が同じ対応表を持つ。
DATASETS=(
  "AHN4_ALS|AHN4 航空LiDAR|data/raw/ahn4/31HZ1_20.LAZ|las"
  "autzen_PDAL|autzen (PDAL)|data/raw/small/autzen_trim.laz|las"
  "plane_PDAL|plane (PDAL)|data/raw/small/plane.laz|las"
  "fullwave_YellowScan|fullwave (YellowScan)|data/raw/small/fullwave.laz|las"
  "simple1_4_GlobalMapper|simple1_4 (GlobalMapper)|data/raw/small/simple1_4.las|las"
  "vegetation_RSSurvey|vegetation (RS Survey)|data/raw/small/vegetation_1_3.las|las"
  "KITTI_MLS|KITTI 車載LiDAR|$K/0000000000.bin|pts"
  "Bunny_rawscan|Bunny 生スキャン|$S/bunny/data/bun000.ply|pts"
  "Bunny_recon|Bunny 再構成|$S/bunny/reconstruction/bun_zipper.ply|pts"
  "Dragon_rawscan|Dragon 生スキャン|$S/dragon_stand/dragonStandRight_0.ply|pts"
  "Dragon_recon|Dragon 再構成|$S/dragon_recon/dragon_vrip_res2.ply|pts"
  "Armadillo_rawscan|Armadillo 生スキャン|$S/Armadillo_scans/ArmadilloBack_0.ply|pts"
  "Armadillo_recon|Armadillo 再構成|$S/Armadillo.ply|pts"
)

run() {   # run <出力名> <引数...>
  local tag="$1"; shift
  local f="$OUT/$tag.txt"
  local t0 t1
  t0=$(date +%s.%N)
  { echo "### $*"; $B "$@" 2>&1; } > "$f" 2>&1 || true
  t1=$(date +%s.%N)
  echo "__elapsed $(echo "$t1 - $t0" | bc) s" >> "$f"
}

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r slug name path kind <<< "$entry"
  [ -f "$path" ] || { echo "skip (無い): $path"; continue; }
  echo ">>> $name"
  run "${slug}__score"   score   "$path" --max-points $MAXP
  run "${slug}__octant"  octant  "$path" --max-points $MAXP
  run "${slug}__geom"    geom    "$path" --max-points $MAXP
  run "${slug}__persist" persist "$path" --max-points $MAXP
  if [ "$kind" = "las" ]; then
    run "${slug}__bench"    bench "$path" --max-points $MAXP
    run "${slug}__benchnd"  bench "$path" --max-points $MAXP --no-delta
  fi
done
echo "完了: $OUT"
