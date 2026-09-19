# 再開手順 / 現在地

最終更新: 2026-09-19（走査モデル符号器の実装・往復検証まで）

## 1. 再開時にまずこれを打つ

```bash
cd <repo>
bash scripts/setup.sh          # 冪等。環境が生きていれば数秒で終わる
cd cpp && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
export LD_LIBRARY_PATH=$HOME/tools/laszip-install/lib:$LD_LIBRARY_PATH
```

`scripts/setup.sh` が復元するもの:

- build-essential / cmake / ninja
- **C++ のビルド依存**: LASzip（`$HOME/tools/laszip-install`）、zstd・CGAL・GUDHI・Eigen・GMP・MPFR
- Python venv `/home/agent/.venvs/pcc`（`$PCCPY`）
  - **注意**: venv はプロジェクト直下に作ってはいけない。virtiofs マウントは
    シンボリックリンクを作れず `uv venv` が壊れる。必ず `$HOME` 側に置く
- TMC13（G-PCC 参照ソフト、`$TMC3`）
  - cmake 4.x は古い `cmake_minimum_required` を拒否するので
    `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` が必須
- サンプルデータ（取得済みならスキップ）

## 2. 動作確認

```bash
./cpp/build/pccnorm pack data/raw/small/autzen_trim.laz /tmp/t.pcc2
# → 「検証  全列一致 = true」「決定性 バイト一致」が出れば正常
```

## 3. 現在地

| 項目 | 状態 |
|---|---|
| 正規化レイヤー（論文の構成） | 完了。タイル全体 47.908 bpp / 295.2 MB |
| PCC2 コンテナ（LAS/LAZ 非依存） | **完了・検証済み**。`notes/02_codec_container.md` |
| 幾何 v0〜v3 | 完了。最良 22.032 bpp（200万点） |
| 走査モデル符号器 | **実装・往復検証済み**。200万点で幾何 20.801 bpp |
| タイル全体での走査モデル | **未測定**（次の一手） |
| 可逆な多重解像度（mipmap / パッチ） | 未着手 |

データ:

- `data/raw/ahn4/31HZ1_20.LAZ` … 440MB / 49.3M点 / scale 1mm / PF8
- `data/raw/{ahn3,ahn5,kitti,stanford,tls,extrabytes,small}/` … 取得済
- いずれも git 管理外。`scripts/setup.sh` が取得する

## 4. 再開後に最初に流すコマンド

```bash
# タイル全体で走査モデルを測る（25〜40 分）
./cpp/build/pccnorm pack data/raw/ahn4/31HZ1_20.LAZ data/work/ahn4_full.pcc2
```

論文の構成 47.908 bpp を下回るかどうかが焦点。

## 5. 確定している前提

- 対象データ: ① float32 実スキャン (LAS/LAZ, ALS) ② 車載LiDAR (KITTI系) ③ 地上型 (TLS)
- 一次データは**可逆でなければならない**（再処理の要請から確定）
- ゴール: 技術検証・実装が主。CTC 準拠は後追いでよい
- 計算資源: CPU 16core / 30GB RAM / **GPU なし**
- 評価は常に 5 軸: bpp / enc時間 / dec時間 / ピークメモリ / 決定性

## 6. 次にやること

[`notes/04_next_steps.md`](notes/04_next_steps.md) に整理してある。
