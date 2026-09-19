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
mkdir -p data/raw/{ahn4,kitti,small}
if [ ! -s data/raw/ahn4/31HZ1_20.LAZ ]; then
  echo "  AHN4 タイル取得中 (440MB)..."
  curl -L --retry 3 -o data/raw/ahn4/31HZ1_20.LAZ \
    "https://geotiles.citg.tudelft.nl/AHN4_T/31HZ1_20.LAZ"
fi
if [ ! -s data/raw/kitti/2011_09_26_drive_0001_sync.zip ]; then
  echo "  KITTI raw drive 取得中 (459MB)..."
  curl -L --retry 3 -o data/raw/kitti/2011_09_26_drive_0001_sync.zip \
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"
  (cd data/raw/kitti && unzip -q -o 2011_09_26_drive_0001_sync.zip)
fi
for f in autzen_trim.laz vegetation_1_3.las simple1_4.las plane.laz fullwave.laz; do
  [ -s "data/raw/small/$f" ] || curl -sL -o "data/raw/small/$f" \
    "https://github.com/laspy/laspy/raw/master/tests/data/$f"
done
du -sh data/raw/*

echo
echo "== 完了。動作確認: =="
echo "  \$PCCPY python/run_report.py data/raw/small/autzen_trim.laz --kmax 4"
