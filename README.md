# PCC-study — 3D 点群圧縮の技術検討と符号器の実装

実運用データ（航空 LiDAR / 車載 LiDAR / 地上型スキャナ）を対象に、
「どこにビットが使われているか」を実測し、そこから符号化技術を設計・実装する作業場。

符号化は毎回その場で復号して全列の一致を確認し、同じ入力を 2 回符号化してバイト一致（決定性）も確認済。

---

## 論文

姉妹論文で、依存は下から上への一方向。符号化論文は単独で読める。

| | 内容 |
|---|---|
| [Lossless Coding of Airborne LiDAR Attributes in a Geometry-Derived Order](paper/coding.pdf) | 容器が幾何を先に復号する性質を使い、座標だけで決まる順序と予測子を副情報なしで使う |
| [A Representation Normalization Layer for Point Cloud Compression](paper/pcc.pdf) | 符号化の前段に置く可逆写像の族。恒等写像を含めて実測で選ぶことで「元より悪化しない」を構成的に保証する。属性の符号化方式は上の論文に委ねる |

いずれも LASzip を内側のコーデックとして用いる構成である。
**本リポジトリの現在の主題は、その外側ではなく符号器そのものの開発**（下記）。

---

## 現在の到達点

### まとめ（2026-09-23 時点）

**現在の数値の正本は [`results/standing.md`](results/standing.md) §1 である。**ここには要点だけを
写す（写しが古くならないよう、測り直したら standing.md を先に直し、ここはそれに合わせる）。
文書の地図は [`results/INDEX.md`](results/INDEX.md)。

標本は各ファイルの**中央 200 万点**、las 15 件の全列を LASzip と比べた結果である
（`python/verify_suite.py` の size、ログ `data/work/grid/size_v10.log`）。
**点の次元・VLR・EVLR・ヘッダの欄まで全部保存したうえでの値**であり、`pack` は
`unpack --las` と同じ道で書き戻して元のファイルと照合している（`results/defects.md`）。

| | 中央 200 万点 |
|---|---|
| LASzip より小さい | **15/15** |
| 中央値 | **−14.3%** |
| 四分位 | [−32.2, −10.6] |
| 幅 | −36.5〜−8.5% |

幾何だけを取り出すと、**LAZ には 17/17 で勝つ**（比の中央値 0.90x）。
G-PCC には公称値どうしで 11/17（0.94x）。**G-PCC は点の順序を捨てている**ので
そのままでは同じ仕事の比較ではなく、順序の復元も含めて測ると **15/17 ＋ 2 件は決着せず**。
詳細は `results/standing.md` §2.1.1。

時間は LASzip（laspy 既定の lazrs 並列版）に対して次のとおり（中央 200 万点）。

| | 中央値 | 四分位 |
|---|---|---|
| 符号化 | **36.5x**（開始時 99.3x） | [25.2, 43.3] |
| **復号** | **9.5x** | [2.3, 12.0] |

メモリーは 1 点あたり中央値 **647 byte**（属性込み）。三者で最も重い（standing.md §6）。

**符号化が復号より掛かるのは、候補を全部符号化して最短を採るためである。**
復号は選ばれた 1 本しか通らない。

### 速度に効いた手（2026-09-21 までの記録。値はその時点の二値）

サイズを変えずに速くした手と、サイズのために速さを払った手の一覧である。

| | 効果 |
|---|---|
| 近傍の予測子表を P ごとにキャッシュ | 16.4 → 6.7 s |
| 属性の候補も並列に符号化 | 6.7 → 2.2 s |
| 最短を超えた候補を符号化の途中で打ち切る | 2.2 → 1.7 s |
| 近傍探索を 1 回にまとめる（木も 1 回） | 1.70 → 1.36 s |
| 表を幾何の列の裏で先読み | 1.36 → 1.27 s |
| 符号化器の模型を 2 byte に、ビット長を clz で | 400 万点で −16% |
| **二値算術符号化から分岐を消す** | **400 万点で −36〜41%** |
| 空間予測の作業領域を使い回す | 1.13 → 1.10 s |
| 当てはめの `tan` から引数の還元を外す | 当てはめ 0.09 → 0.05 s |
| 事前選別の標本の上限を 2.5 万点に | 1.06 → 1.00 s |
| **下位ビットを 1 記号で送る版を候補に加える** | 符号化は速いが候補が 2 倍。**サイズが −2.5 ポイント** |
| KD木の構築を並列に（復号の 34% が近傍表だった） | 復号 4.5x → 4.0x |
| **列の仕事と候補の仕事を同じ待ち行列に入れる** | **符号化 14.6x → 10.3x、最悪 39.1x → 16.8x** |
| （走査モデルの変種を全部出す — サイズのために速度を払う） | 符号化 10.9x → 12.5x、サイズは 3 件が縮み悪化ゼロ |
| 標本で不利になる候補を守り切る（65・66 節） | **事前選別の代償が 0 件 / 15 に** |
| 参照列を「差」と「空間予測後」の 2 軸で順位付ける（67 節） | **100 万点で 5 件が 0.28〜2.11% 縮んだ。絞りの代償は 0 件 / 15** |

（秒の行は AHN3 _20 100 万点・全 14 列の `enc`、倍率の行は対 LASzip の比。
太字でサイズに触れている行以外は、サイズを 1 bit も変えない）

**「LAZ に匹敵する（1〜2 倍）」には届いていない。** 機械の埋まり具合は
16 コアのうち 3.5 → 8.1 まで上げた（64 節）。残りは候補を減らす
（0.036〜4.5% のサイズ悪化を実測）か、下位ビットを確率モデルなしで書く
（候補として出しても 17 件で 1 度も採られない）かで、**どちらもサイズを
犠牲にする**。サイズを変えずに残っているのは、幾何の残差の算術復号を
波 0 と並行に先回りさせる案だけで、復号の 5% ほどである（63 節）。

### これまでの到達点の記録（09-19〜21。現在の値は `results/standing.md`）

以下の 4 つの表はそれぞれの時点の二値の値で、今の値ではない。経緯を追うために残す。

#### 自前符号器 PCC2 — LAS/LAZ に依存しないコンテナと符号器

| 構成 | AHN4 200万点 | 31HZ1_20（4,929万点） | 31HZ1_21（4,749万点） | 依存 |
|---|---|---|---|---|
| 元の LAZ | 71.209 bpp | 71.349 / 439.6 MB | 66.052 / 392.1 MB | — |
| LASzip + 正規化レイヤー（論文の構成） | 54.082 | **47.908 / 295.2 MB** | 未測定 | LASzip |
| PCC2（自前符号器） | **47.794** | 48.904 / 301.3 MB | **42.771 / 253.9 MB** | **なし** |

自前符号器は **LASzip から独立になった**。31HZ1_20 では論文の構成に 2.1% 負けているが、
隣接タイル 31HZ1_21 では走査モデルが採られて元の LAZ 比 −35.2% に届く。
**同じデータセットの隣り合うタイルで挙動が変わる**ことが 2026-09-19 の一括測定で分かった。
記録は [`results/pcc2_codec.md`](results/pcc2_codec.md)、設計は
[`notes/02_codec_container.md`](notes/02_codec_container.md)。

#### 10 ファイルでの LASzip との比較

2026-09-19 に 10 ファイル（小 4 件・ALS 6 件、計 2.4 億点）を 12 構成で一括測定した。
全件で往復検証の全列一致と決定性のバイト一致を通している
（[`results/scan_model_fitting.md`](results/scan_model_fitting.md) 16〜18 節）。

| | 件数 | 幅 |
|---|---|---|
| PCC2 が勝つ | 9 | −3.8% 〜 −35.5% |
| **符号器が負ける** | 3 | +1.1%（autzen_trim）/ +4.9%（AHN3）/ +8.2%（vegetation） |

負ける 3 件は、元の器を容器に包んで出す退避路（A-3）により、
**最終出力では +0.0〜0.3% に収まる**。ただしこれは運用上の保証であって、
符号器が追いついたわけではない。

> **2026-09-21 追記。** 差分の値が繰り返す列を表で指す符号器を足して、
> この 3 件はいずれも符号器自体で LASzip より短くなった
> （autzen_trim −9.5% / AHN3 _20 −5.4% / vegetation −11.0%）。
> 別の台での測定なのでこの表の数値とは直接比べられない
> （`python/bench_total.py`、las 15 件、中央から最大 20 万点。
> 上の表は 10 ファイル全点）。記録 52 節を参照。

#### 幾何の符号器（AHN4 200万点、全候補の実測）

| 符号器 | bit/点 |
|---|---|
| 1 次差分 + レンジ符号 | 25.211 |
| 直近 3 差分の中央値を予測子に | 23.921 |
| + 復帰番号ごとに予測器を分ける | 23.482（逆効果） |
| + 軸をまたぐ文脈 | 22.032 |
| **走査モデル経由** | **19.090** |
| （参考）LASzip | 22.219 |

効いたのは予測式ではなく**文脈の取り方**と**取得の物理モデル**である。
ただし走査モデルの優位は一般ではない。12 構成で測ると採られたのは 6 件で、
幅は −18.9% から +71% まで開く（下記）。軸をまたぐ文脈（22.032）は全件で安定して効く。

#### ALS の走査モデル

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
| ALS の走査角は時刻から決まる | 0 次で −24% | [als_scan_structure.md](results/als_scan_structure.md) |
| 走査モデル符号器と、当てはめの誤り | 幾何 −8.9%（200万点） | [scan_model_fitting.md](results/scan_model_fitting.md) |
| 走査モデルが勝つのは 12 構成中 6 件 | 勝敗は退避した線の面外ビット長で決まる | [scan_model_fitting.md](results/scan_model_fitting.md) |
| 標本選択は 10.2 倍速で +1.05% | 誤るのは 14 列中 X+Y+Z の 1 列だけ | [scan_model_fitting.md](results/scan_model_fitting.md) |
| 自前符号器が LAS/LAZ 非依存に | タイル全体 −31.5%、全列一致 | [pcc2_codec.md](results/pcc2_codec.md) |
| 面を送って点を引き直す符号化 | 同じ面忠実度でオクトツリーの 4.0 倍 | [lossy_surface_finding.md](results/lossy_surface_finding.md) |
| 歪み尺度の落とし穴 2 件 | Chamfer の下限は点密度だけで決まる | [lossy_surface_finding.md](results/lossy_surface_finding.md) |
| 推定を測定に替えるだけで戻った | 3.1 bit/点 | [lossless_refinements.md](results/lossless_refinements.md) |
| 車載 LiDAR の float32 は 1 mm 格子に乗る（可逆に整数化） | 無圧縮比 −83.2%、整数 LAS にした LASzip 比 −6.6% | [forward_operator.md](results/forward_operator.md) §4 |
| （撤回）車載 LiDAR の float32 は派生表現 | 前提の「格子は壊れている」が誤りだった | [kitti_polar_finding.md](results/kitti_polar_finding.md) |
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

現在の方向は [`results/approaches.md`](results/approaches.md)（試した手と残りの方向）にある。
以下は 09-20 時点の [`notes/04_next_steps.md`](notes/04_next_steps.md)（履歴）の要点:

1. **掃引の分割を良くする** — 勝敗を決めているのは当てはめではなく走査線の切り出し
2. **鍵の縮退の境界を詰める** — 順序感度の測定（30 節）で、走査モデルの順序不変性は鍵重複が 62% を超えるあたりで失われると分かった。統制した掃引（42 節）は原因が縮退であることを示すが、量子化が時刻の分解能も壊すため応答の形までは決められない。鍵重複 62〜82% のデータが要る
3. ~~**gps_time の符号器** — AHN3 が負ける主因~~ → **済**（52 節）。
   差分の値を直近 1024 個の表で指す符号器で、gps_time は 9 本の列で採択された
4. **可逆な多重解像度** — 走査モデルが与える 2 次元格子の上で mipmap / パッチ
5. **未論文化の成果の論文化** — 面符号化と歪み尺度、走査モデル、PCC2

作業規則は [`notes/04_next_steps.md`](notes/04_next_steps.md) の末尾、
点検の手順は [`notes/05_verification_checklist.md`](notes/05_verification_checklist.md) にある。
**部分集合で測った値を全体の性質として報告しない** — 2026-09-19 に実際にやらかした。

---

## 背景メモ

- [`notes/00_plan.md`](notes/00_plan.md) — 確定した前提と Phase 分け
- [`notes/01_normalization_layer.md`](notes/01_normalization_layer.md) — 正規化レイヤーの設計
- [`notes/02_codec_container.md`](notes/02_codec_container.md) — PCC2 コンテナの設計
- [`notes/03_scan_model_codec.md`](notes/03_scan_model_codec.md) — 走査モデル符号器の設計
- [`notes/05_verification_checklist.md`](notes/05_verification_checklist.md) — **点検の手引き。数字を出す前・主張する前・訂正した後に通す**
- [`RESUME.md`](RESUME.md) — 中断・再開用の現在地

## ライセンス

[Apache License 2.0](LICENSE)。同梱・参照している第三者のコードとデータの出典は [NOTICE](NOTICE) にまとめてある。`cpp/include/pcc/nanoflann.hpp` は [nanoflann](https://github.com/jlblancoc/nanoflann)（BSD-2-Clause）。
