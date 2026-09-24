# PCC-study — 3D 点群の可逆圧縮の研究と符号器の実装

実運用データ（航空 LiDAR・車載 LiDAR・地上型スキャナ・三角測量スキャナ）を対象に、
「どこにビットが使われているか」を実測し、そこから設計した可逆の点群符号器（PCC2）と、
その成果をまとめた論文 3 本を置いている。

指標の順は、可逆を前提に **サイズ → 速度 → メモリ**。

---

## 論文

姉妹論文で、依存は下から上への一方向。符号化論文は単独で読める。

| | 内容 |
|---|---|
| [Lossless Coding of Airborne LiDAR Attributes in a Geometry-Derived Order](paper/coding.pdf) | 容器が幾何を先に復号する性質を使い、座標だけで決まる順序と予測子を副情報なしで使う |
| [A Representation Normalization Layer for Point Cloud Compression](paper/pcc.pdf) | 符号化の前段に置く可逆写像の族。恒等写像を含めて実測で選ぶことで「元より悪化しない」を構成的に保証する。自前の器を単独で使った結果も載せる |
| [Input Order Is an Uncontrolled Variable in Lossless Point Cloud Compression Benchmarks](paper/order.pdf) | 点の並びが可逆符号器の比較を左右することを、8 通りの順序・4 符号器・17 ファイルの 41 ブロック、計 324 行で示す |

最初の 2 本の主な構成は LASzip を内側のコーデックとして用いる。自前の器（PCC2）を単独で使った結果は
`pcc.pdf` の「The container standing alone」にある。

---

## 主な結果（2026-09-24、各ファイルの中央 200 万点）

las 15 件の全列を、同じ列を詰め直した LASzip と比べた（`python/bench_all.py`）。
**点の次元・VLR・EVLR・ヘッダの欄まで全部保存したうえでの値**である。

| | 値 |
|---|---|
| LASzip より小さいファイル | **15/15** |
| 中央値 | **−16.3%** |
| 四分位 | [−32.9, −12.8] |
| 幅 | −38.2〜−9.8% |
| 幾何 / 属性 | −13.3% / −18.3%（どちらも 15/15） |

- **幾何だけ**（17 ファイル）: LAZ には 17/17 で勝つ（比の中央値 0.87）。G-PCC には公称値どうしで 11/17（0.91）。
  G-PCC は点の順序を捨てているので、そのままでは同じ仕事の比較ではない（`order.pdf`）
- **車載 LiDAR（KITTI、108 frame）**: float32 × 4（128 bit/点）に対して可逆で −83.3%。座標は 1 mm の整数倍で、
  器はそれを見つけてビット完全に整数化する
- **速度**（LASzip 比、中央値）: 符号化 47.1 倍、復号 9.9 倍（どちらも遅い）。候補を全部符号化して最短を採るため
- **メモリ**: 1 点あたり 928 byte（属性込み）。三者で最も重い

---

## 仕組みの要約

- **列ごとに、符号器の候補を実際に符号化して、いちばん短いものを採る。**恒等候補を必ず含むので、
  どの列も生のまま（64 bit）より長くならない
- **幾何を属性より先に復号する器。**属性の予測子は座標を副情報なしで使える。流れの順は
  点番号・gps_time・戻りの情報 → 幾何 → 属性。点の順序は並べ替えない
- **文脈の旗**: 桁長の記号化（記）、下位ビットの素通し（生）、文脈の束ね（束）、直前の残差の符号（符）、
  戻りの種類・同じパルスの続き（光）、z の曲面予測（面）、速い適応（速）、既に復号済みの列の類（類）
- **非可逆**（`--eps`、float の器だけ）: 極座標・直交の格子を実際に符号化して短いほうを選ぶ。属性はビット完全のまま
- **器の形**: 版 5（版 3・4 も読める）。タグつきの頭、流れの表、crc64

---

## 検証と再現性

- **pack の自己検証**: 選んだ流れを符号化し直してバイト一致（決定性）→ 復号して全列一致 →
  LAS なら `unpack --las` と同じ道で書き戻して元のファイルと欄ごとに照合。どれかに落ちたら器を残さない
- **検証一式** `python/verify_suite.py`（10 項目）: 直した不具合の回帰 63 件・15 件の往復と照合・
  スレッド数を変えた md5 一致・次元の被覆・`unpack --las` の生バイト比較・PLY・KITTI 108 frame・
  幾何の候補 58 本の強制・非可逆（極座標とデカルト格子）・命令セットの違う二値との一致
- **復号は libm の三角関数を呼ばない**（版 4 以降。非可逆の極座標は四則演算と floor だけの sin/cos、
  十進の格子は正確な 10^d の表）。二値の既定の命令セットは `x86-64`（SSE2 だけ）

---

## セットアップ

```bash
bash scripts/setup.sh      # 冪等。ビルド依存・Python 環境・G-PCC の参照ソフト・サンプルデータを用意する
```

置き場所は `PCC_VENV`・`PCC_TOOLS` で変えられる（既定は `$HOME/.venvs/pcc`・`$HOME/tools`）。
スクリプトは `PCCPY`（Python）と `TMC3`（G-PCC の参照ソフト）を環境変数で探すので、
`setup.sh` が最後に表示する `export` をシェルの設定に足しておく。

```bash
cd cpp && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
export LD_LIBRARY_PATH=$HOME/tools/laszip-install/lib:$LD_LIBRARY_PATH
# 検証一式の isa 項目用に、この機械の命令セットで建てた二値も作る（出力は既定の二値と同じ）
cmake -S . -B build_native -G Ninja -DCMAKE_BUILD_TYPE=Release -DPCC_MARCH=native && ninja -C build_native pccnorm
```

命令セットは `-DPCC_MARCH=<値>` で選ぶ（既定 `x86-64`）。既存のビルド用ディレクトリは値を
CMakeCache に覚えるので、既定を変えるときは `-DPCC_MARCH=x86-64` を渡して cmake を回し直す。

> 共有フォルダなど、ファイルの更新時刻が伝わりにくい場所では ninja がヘッダの変更を検知しないことがある。
> そのときは `touch cpp/include/pcc/*.hpp cpp/src/*.cpp` してからビルドする。

---

## 使い方

### 符号器（PCC2）

```bash
# 符号化 → 決定性・往復・LAS 書き戻しを確かめて確定 → 5 軸を出す
./cpp/build/pccnorm pack   <in.laz|in.bin|in.ply> <out.pcc2> [--max-points N] [--trace]
./cpp/build/pccnorm unpack <out.pcc2> [--las restored.laz] [--bin restored.bin]
```

主なオプション（pack）

| | 内容 |
|---|---|
| `--trace` | 候補ごとの実測 bpp を全部出す（採択の根拠） |
| `--eps <m>` | 非可逆。float の器（KITTI `.bin`・PLY）だけ。誤差上限を守る格子（極座標か直交）に量子化する。属性はビット完全のまま |
| `--no-verify` | 自己検証を全部飛ばす（ベンチで符号化の時間だけを測るとき） |
| `--no-fallback` | 自前の出力より、同じ点を書き直した基準の LAZ が短いときに包む退避路を切る（LAS 入力だけ。測定用） |
| `--force-geom <名前>` | 幾何の候補を 1 本に固定する（旗つきの名前も可）。選ばれなかった候補の往復に使う |
| `--fast-attr` | 属性の候補を 2 本に絞って速くする（サイズは伸びる） |
| `--sample-select N` | 符号器の選択を N 点の散らした標本で行う（既定より遅く、サイズも伸びうる） |
| `--no-normalize` | 正規化（定数・複製・整数アフィンの除去）を行わない |
| `--no-spatial` | 属性の空間予測を候補に入れない |
| `--split-geom` | 幾何 3 軸を別々のストリームにする |

`unpack --bin` は KITTI の `.bin` として書き戻す。PLY を書き出す口は無い。

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

### Python（対照実装・検証・ベンチ）

旧形式（PCC1）のレンジコーダは C++ と Python でバイト列まで一致する。

```bash
$PCCPY python/verify_suite.py [札]          # 検証一式（10 項目。測定と同時に回さない）
$PCCPY python/regress_fixes.py              # 直した不具合の回帰（63 件）
BENCH_N=2000000 $PCCPY python/bench_all.py  # サイズ・速度・メモリを LASzip と比べる
BENCH_N=2000000 $PCCPY python/bench_split.py  # 幾何と属性の内訳
BENCH_N=2000000 $PCCPY python/bench_three.py  # 幾何のみの三者比較（G-PCC・LAZ・PCC2）
$PCCPY python/bench_ab.py <旧 pccnorm> <新 pccnorm>   # 2 つの二値を交互に回して速度とメモリを比べる
```

測定のログは `data/work/grid/` に書かれる（`data/` は git の管理外）。

---

## ディレクトリ

```
cpp/        C++ 実装（本体）
  include/pcc/  frame・pcc2・scanmodel・rangecoder・las・normalize・geom ほか
  src/
  tests/fixtures/  回帰試験用の作り物のデータ
python/     Python の対照実装・検証・ベンチ・実験スクリプト
paper/      論文（PDF）
scripts/    環境の用意・一括実行
data/       サンプルデータ（git 管理外。setup.sh が取得する）
```

---

## ライセンス

[Apache License 2.0](LICENSE)。同梱・参照している第三者のコードとデータの出典は [NOTICE](NOTICE) にまとめてある。`cpp/include/pcc/nanoflann.hpp` は [nanoflann](https://github.com/jlblancoc/nanoflann)（BSD-2-Clause）。
