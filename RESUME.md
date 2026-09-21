# 再開手順 / 現在地

最終更新: 2026-09-21（順序感度の三度目の測定と、評価プロトコルの論文まで）

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
| 正規化レイヤー（論文の構成） | 完了。31HZ1_20 で 47.908 bpp / 295.2 MB |
| PCC2 コンテナ（LAS/LAZ 非依存） | **完了・検証済み**。`notes/02_codec_container.md` |
| 幾何 v0〜v3 | 完了。最良は v3（軸をまたぐ文脈）。全件で安定して効く |
| 走査モデル符号器 | **完了**。v1 と走査変換の 2 本を既定に。v2/v3/v4 は既定オフ |
| 一括測定（10 ファイル・12 構成） | **完了**（2026-09-20 再実行、1 時間 42 分、全構成で検証通過） |
| 「決して悪化しない」保証 | **実装済み**（A-3）。符号器自体も 2026-09-21 に全 15 件で LASzip より短くなった（52 節、別の台） |
| 差分の値を表で指す符号器 | **完了**（52 節）。全列で LASzip より 15/15、中央値 −10.1% |
| 符号化の速度 | **2026-09-21 に 99.3 倍 → 約 15 倍**（対 LASzip、全列、四分位 [7.8, 31.4]）。サイズは不変。55〜58 節 |
| 速度の残り | 候補を減らす（1.7〜4.5% のサイズ悪化）か二値化を変える（サイズが一方向に動かない）かの取引。**指示待ち** |
| 報告値の往復検証 | **実装済み**。`--force-geom` で候補を固定、`--no-fallback` で退避路を切る |
| 順序感度の測定 | **完了**（30 節、3 回目で確定）。17 ファイル 41 ブロック 324 行すべて検証通過。車載・地上型も含む |
| 掃引の分割の改善 | **未着手**。勝敗を決めている量（B-1） |
| 可逆な多重解像度（mipmap / パッチ） | 未着手 |

測定の記録は [`results/scan_model_fitting.md`](results/scan_model_fitting.md) にある。

| 節 | 内容 |
|---|---|
| 19〜23 | 一括測定（10 ファイル・12 構成）。走査系の採択は 6 件 |
| **24・27・29** | **独立査読で見つかった誤りと撤回。先に読むこと** |
| 25・26 | G-PCC との比較（幾何のみ、10 ファイル） |
| 28 | 順序の予備試行（結論は 29 節で撤回） |
| 30 | 順序感度の本測定（41 ブロック × 8 条件 × 4 符号器、324 行） |
| 39〜42 | スライス上限・置換の実費用・内部ソート・鍵の掃引 |

**同じ型の誤りを繰り返している。** 24 回の自己点検と 5 回の独立査読で
9 つの型に整理した。**数字を出す前・主張する前・訂正した後に
[`notes/05_verification_checklist.md`](notes/05_verification_checklist.md) を通すこと。**
実例は記録の 24・27・29〜32 節にある。

データ:

- `data/raw/ahn4/31HZ1_20.LAZ` … 440MB / 49.3M点 / scale 1mm / PF8
- `data/raw/{ahn3,ahn5,kitti,stanford,tls,extrabytes,small}/` … 取得済
- いずれも git 管理外。`scripts/setup.sh` が取得する

## 4. 再開後に最初に流すコマンド

まず動作確認（上記 2）だけ通す。長時間の測定は用件が決まってから。
10 ファイル・12 構成を通しで測り直すなら:

```bash
bash scripts/run_scan_batch.sh          # 直列 12 ジョブ、約 1 時間 42 分
bash scripts/run_scan_batch.sh 5 9      # 5〜9 番目だけ
```

進行は `data/work/batch/batch.log`、各件の詳細は同じ場所の `<名前>.log` / `.err`。

## 5. 確定している前提

- 対象データ: ① float32 実スキャン (LAS/LAZ, ALS) ② 車載LiDAR (KITTI系) ③ 地上型 (TLS)
- 一次データは**可逆でなければならない**（再処理の要請から確定）
- ゴール: 技術検証・実装が主。CTC 準拠は後追いでよい
- 計算資源: CPU 16core / 30GB RAM / **GPU なし**
- 評価は常に 5 軸: bpp / enc時間 / dec時間 / ピークメモリ / 決定性

## 6. 次にやること

[`notes/04_next_steps.md`](notes/04_next_steps.md) に整理してある。
