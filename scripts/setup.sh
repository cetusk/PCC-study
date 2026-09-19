#!/usr/bin/env bash
# 再起動後の環境復元。冪等（既にあるものはスキップ）。
#   bash scripts/setup.sh
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/home/agent/.venvs/pcc
TOOLS=/home/agent/tools

echo "== 1. ビルドツール =="
if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake ninja-build zstd xz-utils
fi
g++ --version | head -1; cmake --version | head -1

echo "== 2. Python 環境 =="
# 注意: プロジェクトは virtiofs マウント上でシンボリックリンクが作れないため
#       venv は必ず $HOME 側に作る（/c/... 上に作ると uv venv が壊れる）
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
  uv pip install -q --python "$VENV/bin/python" \
      numpy scipy pandas matplotlib "laspy[lazrs]" plyfile zstandard numba
fi
"$VENV/bin/python" -c "import numpy,laspy,zstandard,numba;print('py ok', numpy.__version__, laspy.__version__)"

echo "== 2b. C++ ビルドの依存 =="
# cpp/ をビルドするのに要るもの。これが無いと pccnorm が作れない。
need_pkg=0
for h in /usr/include/zstd.h /usr/include/eigen3/Eigen/Core /usr/include/mpfr.h /usr/include/CGAL/version.h; do
  [ -e "$h" ] || need_pkg=1
done
if [ "$need_pkg" = 1 ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      libzstd-dev libeigen3-dev libgmp-dev libmpfr-dev libboost-dev libcgal-dev
fi
# LASzip（LAS/LAZ の読み書き）。liblaszip_api は動的ロード用のシムなので
# liblaszip に直接リンクする必要がある（CPP_PORT.md の記録を参照）。
if [ ! -e "$TOOLS/laszip-install/include/laszip/laszip_api.h" ]; then
  mkdir -p "$TOOLS" && cd "$TOOLS"
  [ -d LASzip ] || git clone -q --depth 1 https://github.com/LASzip/LASzip.git
  cmake -S LASzip -B LASzip/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$TOOLS/laszip-install" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
  ninja -C LASzip/build >/dev/null && ninja -C LASzip/build install >/dev/null
  cd "$PROJ"
fi
# GUDHI（パーシステントホモロジー）。ヘッダのみ使う。
if [ ! -e "$TOOLS/ext/gudhi.3.10.1/include/gudhi/Alpha_complex.h" ]; then
  mkdir -p "$TOOLS/ext" && cd "$TOOLS/ext"
  curl -sL -o gudhi.tar.gz \
    https://github.com/GUDHI/gudhi-devel/releases/download/tags%2Fgudhi-release-3.10.1/gudhi.3.10.1.tar.gz
  tar xzf gudhi.tar.gz && rm -f gudhi.tar.gz
  cd "$PROJ"
fi
ls "$TOOLS/laszip-install/include/laszip/laszip_api.h" >/dev/null && echo "  laszip ok"
ls "$TOOLS/ext/gudhi.3.10.1/include/gudhi/Alpha_complex.h" >/dev/null && echo "  gudhi ok"

echo "== 3. TMC13 (G-PCC 参照ソフト) =="
if [ ! -x "$TOOLS/tmc13/build/tmc3/tmc3" ]; then
  mkdir -p "$TOOLS" && cd "$TOOLS"
  [ -d tmc13 ] || git clone -q --depth 1 https://github.com/MPEGGroup/mpeg-pcc-tmc13.git tmc13
  cd tmc13
  # cmake 4.x は古い cmake_minimum_required を拒否するため policy を明示
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
  ninja -C build -j"$(nproc)" >/dev/null
fi
"$TOOLS/tmc13/build/tmc3/tmc3" --help 2>&1 | head -1 || true

echo "== 4. 環境変数の永続化 =="
grep -q 'PCCPY' /etc/sandbox-persistent.sh 2>/dev/null || \
  echo "export PCCPY=$VENV/bin/python  # PCC-study" | sudo tee -a /etc/sandbox-persistent.sh >/dev/null
grep -q 'export TMC3=' /etc/sandbox-persistent.sh 2>/dev/null || \
  echo "export TMC3=$TOOLS/tmc13/build/tmc3/tmc3  # PCC-study" | sudo tee -a /etc/sandbox-persistent.sh >/dev/null

echo "== 5. データ =="
cd "$PROJ"

# すべて再取得できる。URL は 2026-09-19 に到達性とバイト数を実測して確認した。
#
# 切り詰められたダウンロードは部分読み込みでは成功したように見えるため気づきにくい
# （results/vendor_generality.md: ちょうど 10 MiB で切れた laz を 1 件掴んだ）。
# そこで取得のたびにバイト数を照合し、合わなければその場で落とす。
fetch() {   # fetch <url> <出力パス> <期待バイト数|0>
  local url="$1" out="$2" want="$3"
  if [ -s "$out" ]; then
    local have; have=$(stat -c %s "$out")
    if [ "$want" != "0" ] && [ "$have" != "$want" ]; then
      echo "  !! $out のサイズが違う（期待 $want / 実際 $have）。消して取り直す" >&2
      rm -f "$out"
    else
      return 0
    fi
  fi
  mkdir -p "$(dirname "$out")"
  echo "  取得中: $(basename "$out")"
  # 出力が端末でないとき（ログへのリダイレクト）は進捗メーターを出さない。
  # 進捗メーターは復帰文字で同じ行を上書きするので、ログに落とすと
  # 1 行が 2 万文字を超える塊になり、tail や grep で読めなくなる
  # （実測: 30 行 22,082 バイトのうち 21,060 バイトが 1 行の進捗メーターだった）。
  local q=""
  [ -t 1 ] || q="--no-progress-meter"
  # shellcheck disable=SC2086
  curl -L --retry 3 --fail $q -o "$out" "$url" || { echo "  !! 取得失敗: $url" >&2; return 1; }
  if [ "$want" != "0" ]; then
    local have; have=$(stat -c %s "$out")
    if [ "$have" != "$want" ]; then
      echo "  !! $out が期待のバイト数と違う（期待 $want / 実際 $have）。切り詰めを疑う" >&2
      return 1
    fi
  fi
}

# ---- 航空 LiDAR: AHN（オランダ全国。同一タイルの世代違いが揃う）
GEO=https://geotiles.citg.tudelft.nl
fetch "$GEO/AHN4_T/31HZ1_20.LAZ" data/raw/ahn4/31HZ1_20.LAZ 439742645
fetch "$GEO/AHN4_T/31HZ1_21.LAZ" data/raw/ahn4/31HZ1_21.LAZ 392220080
fetch "$GEO/AHN3_T/31HZ1_20.LAZ" data/raw/ahn3/31HZ1_20.LAZ 126062307
fetch "$GEO/AHN5_T/31HZ1_20.LAZ" data/raw/ahn5/31HZ1_20.LAZ 213066857

# ---- 小さい LAS/LAZ（開発の高速反復用。laspy のテストデータ）
LASPY=https://github.com/laspy/laspy/raw/master/tests/data
for f in autzen_trim.laz vegetation_1_3.las simple1_4.las plane.laz fullwave.laz; do
  fetch "$LASPY/$f" "data/raw/small/$f" 0
done

# ---- ExtraBytes を持つ他ベンダのファイル（results/vendor_generality.md）
fetch "$LASPY/extra.laz"      data/raw/extrabytes/extra.laz      29084
fetch "$LASPY/extrabytes.las" data/raw/extrabytes/extrabytes.las 66354
fetch "https://github.com/PDAL/data/raw/main/workshop/TM_551_101.laz" \
      data/raw/extrabytes/workshop_TM_551_101.laz 107643468
fetch "https://data.entwine.io/red-rocks.laz" \
      data/raw/extrabytes/entwine_data_red-rocks.laz 10188197
# 注意: 配信側が差し替えたらしく、計測に使った版は 184,509,941 バイトだった。
# サイズ照合は現行の版に合わせてある。数字を再現するときはこの差に注意すること。
fetch "https://s3.amazonaws.com/hobu-lidar/autzen-2023.copc.laz" \
      data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz 184518771

# ---- 車載 LiDAR: KITTI raw drive
if [ ! -d data/raw/kitti/2011_09_26 ]; then
  fetch "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip" \
        data/raw/kitti/2011_09_26_drive_0001_sync.zip 458643963
  (cd data/raw/kitti && unzip -q -o 2011_09_26_drive_0001_sync.zip)
fi

# ---- 三角測量スキャナ: Stanford 3D Scanning Repository
#      生スキャンと再構成を対にして使う（results/matrix_comparison.md ①）
STAN=http://graphics.stanford.edu/pub/3Dscanrep
fetch "$STAN/bunny.tar.gz"                   data/raw/stanford/bunny.tar.gz          4894286
fetch "$STAN/dragon/dragon_recon.tar.gz"     data/raw/stanford/dragon_recon.tar.gz  11197764
fetch "$STAN/dragon/dragon_stand.tar.gz"     data/raw/stanford/dragon_stand.tar.gz   6133186
fetch "$STAN/armadillo/Armadillo.ply.gz"     data/raw/stanford/Armadillo.ply.gz      3874291
fetch "$STAN/armadillo/Armadillo_scans.tar.gz" data/raw/stanford/Armadillo_scans.tar.gz 30365481
(cd data/raw/stanford
 [ -d bunny ]        || tar xzf bunny.tar.gz
 [ -d dragon_recon ] || tar xzf dragon_recon.tar.gz
 [ -d dragon_stand ] || tar xzf dragon_stand.tar.gz
 [ -d Armadillo_scans ] || tar xzf Armadillo_scans.tar.gz
 [ -s Armadillo.ply ] || gunzip -kf Armadillo.ply.gz)

# ---- 地上型レーザースキャナ: Würzburg の講堂（RIEGL VZ-400）
#      results/tls_polar_finding.md。450MB あるので最後に置く
fetch "http://kos.informatik.uni-osnabrueck.de/3Dscans/lecturehall.tar.xz" \
      data/raw/tls/lecturehall.tar.xz 451591544
(cd data/raw/tls && [ -d lecturehall ] || tar xJf lecturehall.tar.xz)

du -sh data/raw/*

echo
echo "== 完了。動作確認: =="
echo "  ./cpp/build/pccnorm pack data/raw/small/autzen_trim.laz /tmp/t.pcc2"
echo "  \$PCCPY python/run_report.py data/raw/small/autzen_trim.laz --kmax 4"
