# PCC-study — 3D 点群圧縮の技術検討と符号器の実装

実運用データ（航空 LiDAR / 車載 LiDAR / 地上型スキャナ）を対象に、
「どこにビットが使われているか」を実測し、そこから符号化技術を設計・実装する作業場。

**すべての数字は実測である。** 符号化は毎回その場で復号して全列の一致を確認し、
同じ入力を 2 回符号化してバイト一致（決定性）も確認する。通らなければ数字を出さない。

---

## 論文

| | 内容 |
|---|---|
| [点群圧縮における表現正規化レイヤー](paper/pcc.pdf) | 符号化の前段に置く可逆写像の族。恒等写像を含めて実測で選ぶことで「元より悪化しない」を構成的に保証する |
| [幾何から導く符号化順序による航空 LiDAR 属性の可逆符号化](paper/coding.pdf) | 容器が幾何を先に復号する性質を使い、座標だけで決まる順序と予測子を副情報なしで使う |

いずれも LASzip を内側のコーデックとして用いる構成である。
**本リポジトリの現在の主題は、その外側ではなく符号器そのものの開発**（下記）。

---

## 現在の到達点

### 自前符号器 PCC2 — LAS/LAZ に依存しないコンテナと符号器

| 構成 | AHN4 200万点 | タイル全体（4,929万点） | 依存 |
|---|---|---|---|
| 元の LAZ | 71.209 bpp | 71.349 bpp / 439.6 MB | — |
| LASzip + 正規化レイヤー（論文の構成） | 54.082 | **47.908 / 295.2 MB** | LASzip |
| PCC2（自前符号器のみ） | 50.736 | 48.904 / 301.3 MB | **なし** |
| PCC2 + 走査モデル | **49.940** | 未測定 | **なし** |

自前符号器は **LASzip から独立になった**が、タイル全体ではまだ論文の構成に 2.1% 負けている。
記録は [`results/pcc2_codec.md`](results/pcc2_codec.md)、設計は
[`notes/02_codec_container.md`](notes/02_codec_container.md)。

### 幾何の符号器（AHN4 200万点、全候補の実測）

| 符号器 | bit/点 |
|---|---|
| 1 次差分 + レンジ符号 | 25.211 |
| 直近 3 差分の中央値を予測子に | 23.921 |
| + 復帰番号ごとに予測器を分ける | 23.482（逆効果） |
| + 軸をまたぐ文脈 | 22.032 |
| **走査モデル経由** | **20.801** |
| （参考）LASzip | 22.219 |

効いたのは予測式ではなく**文脈の取り方**と**取得の物理モデル**である。

### ALS の走査モデル

航空 LiDAR の走査を点群だけから復元できることを実測で示した
（[`results/als_scan_structure.md`](results/als_scan_structure.md)）。

- **掃引内の走査角は時刻の 1 次関数**（幅 278 m の掃引を 4 パラメタで残差 1.4 cm）
- 掃引は gps_time の隙間で切れるので、順序と区切りに**副情報が要らない**
- 1 掃引には 2 台のスキャナが混ざる。EM で分けると面外 RMS が 32 m → 3.4 mm
- 点群は **0.24 × 0.265 m の 2 次元格子（高さ場）**になる
- 可逆・決定論のため、復号側は三角関数を呼ばない
  （3 回のせん断による整数回転 + 整数 CORDIC の正接、誤差は 1200 m 先で 0.051 mm）

---

## 主な計測結果

| 発見 | 規模 | 記録 |
|---|---|---|
| ALS の走査角は時刻から決まる | 幾何 −5.6%、0 次で −24% | [als_scan_structure.md](results/als_scan_structure.md) |
| 自前符号器が LAS/LAZ 非依存に | タイル全体 −31.5%、全列一致 | [pcc2_codec.md](results/pcc2_codec.md) |
| 面を送って点を引き直す符号化 | 同じ面忠実度でオクトツリーの 4.0 倍 | [lossy_surface_finding.md](results/lossy_surface_finding.md) |
| 歪み尺度の落とし穴 2 件 | Chamfer の下限は点密度だけで決まる | [lossy_surface_finding.md](results/lossy_surface_finding.md) |
| 推定を測定に替えるだけで戻った | 3.1 bit/点 | [lossless_refinements.md](results/lossless_refinements.md) |
| 車載 LiDAR の float32 は派生表現 | センサ精度内で −85.2% | [kitti_polar_finding.md](results/kitti_polar_finding.md) |
| 取得表現への復帰は地上型で逆効果 | +34〜57% | [tls_polar_finding.md](results/tls_polar_finding.md) |
| octree は点間隔以下でゼロ情報 | 13 データセットで H(c)=2.976〜3.000 | [subspacing_finding.md](results/subspacing_finding.md) |
| 反復測量の差分符号化 | 幾何ビットの 1.3% のみ | [repeat_survey_finding.md](results/repeat_survey_finding.md) |
| 位相は一度も律速にならない | センサ床が常に 10〜40 倍きつい | [matrix_comparison.md](results/matrix_comparison.md) |
| C++ 移植（結果一致・最大 86 倍） | Python とバイト一致 | [CPP_PORT.md](results/CPP_PORT.md) |
| 主要数値の再検証 42 項目 | PASS 42 / FAIL 0 | [VERIFICATION.md](results/VERIFICATION.md) |

---

## セットアップ

```bash
bash scripts/setup.sh      # 冪等。ビルド依存・Python 環境・データを復元する
```

復元されるもの: build-essential / cmake / ninja、LASzip・zstd・CGAL・GUDHI・Eigen、
Python venv（`$PCCPY`）、G-PCC 参照ソフト（`$TMC3`）、サンプルデータ。

```bash
cd cpp && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
export LD_LIBRARY_PATH=$HOME/tools/laszip-install/lib:$LD_LIBRARY_PATH
```

> virtiofs 上では ninja がヘッダの変更を検知しないことがある。
> ヘッダを編集したら `touch cpp/include/pcc/*.hpp cpp/src/*.cpp` してからビルドする。

---

## 使い方

### 自前符号器（PCC2）

```bash
# 符号化 → その場で復号して全列一致を確認 → 5 軸を出す
./cpp/build/pccnorm pack   <in.laz|in.bin|in.ply> <out.pcc2> [--max-points N] [--trace]
./cpp/build/pccnorm unpack <out.pcc2> [--las restored.laz]
```

主なオプション

| | 内容 |
|---|---|
| `--trace` | 候補ごとの実測 bpp を全部出す（採択の根拠） |
| `--sample-select N` | 符号器の選択だけ先頭 N 点で行う（選択のずれは小ファイルで 0.3%） |
| `--no-normalize` | 正規化（定数・複製・整数アフィンの除去）を行わない |
| `--no-spatial` | 属性の空間予測を候補に入れない |
| `--split-geom` | 幾何 3 軸を別々のストリームにする |

### 計測・診断

```bash
./cpp/build/pccnorm bench   <file.laz>      # 正規化レイヤーの E2E 評価＋復元検証
./cpp/build/pccnorm score   <file>          # 取得構造スコア（6 指標）
./cpp/build/pccnorm octant|geom|lfs|persist <file>
./cpp/build/pccnorm residue|extern <file.laz>   # 正規化後の内訳 / 外部化の損得
./cpp/build/pccnorm surf    <file>          # 面符号化と点符号化の率歪み比較
./cpp/build/pccnorm diff    <target> <ref>  # 反復測量の差分符号化
./cpp/build/pccnorm sphere  1.0 20000 0     # 正解つき形状で lfs を検証
```

### Python（対照実装と実験）

C++ はレンジコーダのバイト列まで Python と一致する。実験スクリプトもここにある。

```bash
$PCCPY python/run_report.py <file.laz> --max-points 2000000 --kmax 9 --out results/x.md
$PCCPY python/bench_normalize.py <file.laz> --max-points 2000000
$PCCPY python/exp_scan_full.py data/raw/ahn4/31HZ1_20.LAZ 3000000   # 走査モデルの 0 次評価
$PCCPY python/exp_scan_int.py                                        # 整数回転と固定小数 tan の検証
$PCCPY python/verify_all.py                                          # 主要数値の再検証
```

---

## ディレクトリ

```
cpp/        C++ 実装（本体）
  include/pcc/  frame・pcc2・scanmodel・rangecoder・las・normalize・geom ほか
  src/
python/     Python の対照実装と実験スクリプト
results/    計測記録（数字の出所）
notes/      設計メモと今後の予定
paper/      論文（PDF）
scripts/    環境復元・一括実行
data/       サンプルデータ（git 管理外。setup.sh が取得する）
```

---

## 次にやること

[`notes/04_next_steps.md`](notes/04_next_steps.md) に整理してある。要点だけ:

1. **タイル全体で走査モデルを測る** — 200 万点では幾何 −5.6%。論文の構成 47.908 bpp を下回るか
2. **「決して悪化しない」保証の回復** — LASzip を置き換えたので構成的な保証を失った
3. **可逆な多重解像度** — 走査モデルが与える 2 次元格子の上で mipmap / パッチ
4. **未論文化の成果の論文化** — 面符号化と歪み尺度、走査モデル、PCC2

作業規則（守ること）は `notes/04_next_steps.md` の末尾にある。
**部分集合で測った値を全体の性質として報告しない** — 2026-09-19 に実際にやらかした。

---

## 背景メモ

- [`notes/00_plan.md`](notes/00_plan.md) — 確定した前提と Phase 分け
- [`notes/01_normalization_layer.md`](notes/01_normalization_layer.md) — 正規化レイヤーの設計
- [`notes/02_codec_container.md`](notes/02_codec_container.md) — PCC2 コンテナの設計
- [`notes/03_scan_model_codec.md`](notes/03_scan_model_codec.md) — 走査モデル符号器の設計
- [`RESUME.md`](RESUME.md) — 中断・再開用の現在地

## 第三者のコード

`cpp/include/pcc/nanoflann.hpp` は [nanoflann](https://github.com/jlblancoc/nanoflann)（BSD-2-Clause）。
