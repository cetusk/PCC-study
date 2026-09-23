# PCC-study — 3D 点群圧縮の技術検討と符号器の実装

実運用データ（航空 LiDAR / 車載 LiDAR / 地上型スキャナ）を対象に、
「どこにビットが使われているか」を実測し、そこから符号化技術を設計・実装する作業場。

`pack` は書いた器をその場で確かめてから確定する。選んだ流れを新しい状態で符号化し直してバイト一致（決定性）、
復号して全列の一致、LAS なら元のファイルとの照合まで通らなければ器を残さない（元の LAZ を包んだときは
符号化し直しの代わりに器の書き直しの一致を見る）。

---

## 論文

姉妹論文で、依存は下から上への一方向。符号化論文は単独で読める。

| | 内容 |
|---|---|
| [Lossless Coding of Airborne LiDAR Attributes in a Geometry-Derived Order](paper/coding.pdf) | 容器が幾何を先に復号する性質を使い、座標だけで決まる順序と予測子を副情報なしで使う |
| [A Representation Normalization Layer for Point Cloud Compression](paper/pcc.pdf) | 符号化の前段に置く可逆写像の族。恒等写像を含めて実測で選ぶことで「元より悪化しない」を構成的に保証する。属性の符号化方式は上の論文に委ねる。自前の器を単独で使った結果（全列で LASzip より 15 件すべてで小さく、中央値 −16.3%）も載せる |
| [Input Order Is an Uncontrolled Variable in Lossless Point Cloud Compression Benchmarks](paper/order.pdf) | 点の並びが可逆符号器の比較を左右することを、8 通りの順序・4 符号器・17 ファイルの 41 ブロック、計 324 行で示す |

最初の 2 本の主な構成は LASzip を内側のコーデックとして用いる。
**本リポジトリの現在の主題は、その外側ではなく符号器そのものの開発**（下記）。

2026-09-23〜24 に 3 本とも数値と主張を見直した。主な訂正は、地上型スキャンで「極座標への復帰は
34〜57% 逆効果」としていた結果が、測る道具が重心を引いていたための誤りだったこと
（スキャナ中心なら 38〜43% 短い。[`results/tls_polar_finding.md`](results/tls_polar_finding.md)）と、
車載 LiDAR の可逆を汎用の符号器の値（46.183 bpp）で比べていたこと（器では座標だけで 16.901 bpp）。

---

## 現在の到達点

### まとめ（2026-09-23 時点）

**現在の数値の正本は [`results/standing.md`](results/standing.md) §1 である。**ここには要点だけを
写す（写しが古くならないよう、測り直したら standing.md を先に直し、ここはそれに合わせる）。
文書の地図は [`results/INDEX.md`](results/INDEX.md)。

標本は各ファイルの**中央 200 万点**、las 15 件の全列を LASzip と比べた結果である
（`python/bench_all.py`、ログ `data/work/grid/bench_2m_v21.log`）。
**点の次元・VLR・EVLR・ヘッダの欄まで全部保存したうえでの値**であり、`pack` は
`unpack --las` と同じ道で書き戻して元のファイルと照合している（`results/defects.md`）。

| | 中央 200 万点 |
|---|---|
| LASzip より小さい | **15/15** |
| 中央値 | **−16.3%**（red-rocks が中央を越えて −14.7% から跳ねた。ファイルごとの行は standing.md §1.5） |
| 四分位 | [−32.7, −12.1] |
| 幅 | −37.6〜−9.1% |

幾何だけを取り出すと、**LAZ には 17/17 で勝つ**（比の中央値 0.89x）。
G-PCC には公称値どうしで 11/17（0.91x）。**G-PCC は点の順序を捨てている**ので
そのままでは同じ仕事の比較ではなく、順序の復元も含めて測ると **15/17 ＋ 2 件は決着せず**。
詳細は `results/standing.md` §2.1.1。

時間は LASzip（laspy 既定の lazrs 並列版）に対して次のとおり（中央 200 万点）。

| | 中央値 | 四分位 |
|---|---|---|
| 符号化 | **43.9x**（開始時 99.3x） | [29.5, 56.9] |
| **復号** | **8.1x** | [2.3, 12.7] |

（復号の比は v19 の二値で 10.3x だった。下がったのは LASzip の復号の秒数が揺れたためで、PCC2 の復号の秒数の合計は −1.1%。LASzip との比は分母が揺れるので、費用は PCC2 の実時間で見る。）

メモリーは 1 点あたり中央値 **742 byte**（属性込み）。三者で最も重い（standing.md §6）。
2026-09-23 の改善ループでサイズを全件で縮めた代わりに、符号化が約 3 割遅くなった
（`results/losses.md` §27〜§33）。その後の構成の査読の直し（効かない候補を外す・命令セット）で
符号化は交互 3 回の比較で −5.9%（`ab_v19_v21.log`）。

**構成の全体像**（入力側・流れの順序・符号器と旗・選び方・器の形・検証）は
[`results/architecture.md`](results/architecture.md) にまとめてある。

### 検証と再現性

- **pack の自己検証**: 選んだ流れを符号化し直してバイト一致（決定性）→ 復号して全列一致 →
  LAS なら `unpack --las` と同じ道で書き戻して元のファイルと欄ごとに照合。どれかに落ちたら器を残さない
  （検証に落ちたら戻り値 2、復号そのものが失敗したら 1）
- **検証一式** `python/verify_suite.py`（10 項目、単独で回す）: 回帰 60 件・15 件の往復と照合・
  スレッド数を変えた md5 一致・次元の被覆・`unpack --las` の生バイト比較・PLY・KITTI 108 frame・
  幾何の候補 58 本の強制・非可逆（極座標とデカルト格子）・命令セットの違う二値との一致（81/81）
- **器の版は 4**（版 3 も読める）。**版 4 の器の復号は libm の三角関数を呼ばない**（非可逆の極座標は四則演算と
  floor だけの sin/cos、十進の格子は正確な 10^d の表）。版 3 の器は互換のため極座標だけ libm で読む。二値の既定の命令セットは `x86-64`（SSE2 だけ）

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
| LASzip + 正規化レイヤー（論文の構成） | 54.082 | **47.908 / 295.2 MB**（2026-09-23 に測り直して 47.881 / 295.0 MB） | 未測定 | LASzip |
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
| 標本選択は 10.2 倍速で +1.05%（当時。既定の経路を速くした後は `--sample-select` のほうが遅い） | 誤るのは 14 列中 X+Y+Z の 1 列だけ | [scan_model_fitting.md](results/scan_model_fitting.md) |
| 自前符号器が LAS/LAZ 非依存に | タイル全体 −31.5%、全列一致 | [pcc2_codec.md](results/pcc2_codec.md) |
| 面を送って点を引き直す符号化 | 同じ面忠実度でオクトツリーの 4.0 倍 | [lossy_surface_finding.md](results/lossy_surface_finding.md) |
| 歪み尺度の落とし穴 2 件 | Chamfer の下限は点密度だけで決まる | [lossy_surface_finding.md](results/lossy_surface_finding.md) |
| 推定を測定に替えるだけで戻った | 3.1 bit/点 | [lossless_refinements.md](results/lossless_refinements.md) |
| 車載 LiDAR の float32 は 1 mm 格子に乗る（可逆に整数化） | 無圧縮比 −83.3%（108 frame の中央値、強度込み。`kitti108_v19.log`）、座標だけなら走行全体で 16.901 bpp（−82.4%、`kitti_xyz_v22.log`）、整数 LAS にした LASzip 比 −6.6% | [forward_operator.md](results/forward_operator.md) §4（発見）・[standing.md](results/standing.md)（いまの値） |
| （撤回）車載 LiDAR の float32 は派生表現 | 前提の「格子は壊れている」が誤りだった | [kitti_polar_finding.md](results/kitti_polar_finding.md) |
| （訂正）取得表現への復帰は地上型でも効く。「逆効果」は極座標の中心を重心にして測った誤り | スキャナ中心で器が −38〜−43%（旧表は重心中心の見積りで +34〜57%） | [tls_polar_finding.md](results/tls_polar_finding.md) |
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
# 検証一式の isa 項目用に、この機械の命令セットで建てた二値も作る（出力は既定の二値と同じ）
cmake -S . -B build_native -G Ninja -DCMAKE_BUILD_TYPE=Release -DPCC_MARCH=native && ninja -C build_native pccnorm
```

命令セットは `-DPCC_MARCH=<値>` で選ぶ（既定 `x86-64`）。既存のビルド用ディレクトリは値を
CMakeCache に覚えるので、既定を変えても ninja を回し直すだけでは前の値のまま（`-DPCC_MARCH=x86-64` を
渡して cmake を回し直す）。`cpp/build` が native のままだと、検証一式の isa は「同じ命令セット」で落ちる。

> virtiofs 上では ninja がヘッダの変更を検知しないことがある。
> ヘッダを編集したら `touch cpp/include/pcc/*.hpp cpp/src/*.cpp` してからビルドする。

---

## 使い方

### 自前符号器（PCC2）

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
| `--sample-select N` | 符号器の選択を N 点の散らした標本で行う（いまは既定より遅く、サイズも伸びうる） |
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

### Python（対照実装と実験）

旧形式（PCC1）のレンジコーダは C++ と Python でバイト列まで一致する。実験スクリプトもここにある。

```bash
$PCCPY python/verify_suite.py [札]          # 検証一式（10 項目。測定と同時に回さない。札を省くと日付）
$PCCPY python/regress_fixes.py              # 直した不具合の回帰（60 件）
BENCH_N=2000000 $PCCPY python/bench_all.py  # サイズ・速度・メモリを LASzip と比べる
$PCCPY python/bench_ab.py <旧 pccnorm> <新 pccnorm>   # 2 つの二値を交互に回して速度とメモリを比べる
```

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

一覧は [`results/approaches.md`](results/approaches.md) §G（2026-09-24 に整理）にある。指標の順
（可逆を前提に、サイズ → 速度 → メモリ）で要点だけ写す。

- **サイズ**: 属性の同時符号化（属性は LASzip の bit の中央値 58.5%、未測定で伸びしろが最も大きい）／
  推定で切り落とす所（可逆: 幾何の予備選別・参照先の絞り、非可逆: 格子の選択）の損を測る／
  旗の重ね方を全組合せで測る／二次推定（SSE / APM）・KITTI のフレーム間予測（未測定）
- **速度**: 速 の選び直しで効かない版（恒等・素の幅・字母）を試さない／`--sample-select` を直すか外す
- **メモリ**: 走査v1 の 400 万点での超過
- **再現性**: 符号化側も libm に依らないようにする（可逆でも回転・走査モデルの係数が libm に依る）／
  タイル全体を版 4 で回す／x86 以外で確かめる
- **論文**: 地上型の極座標（正しい中心が要る）を成果として書く／要旨に地上型を入れるか／
  `order.tex` の先行技術（US10796457B2）・MPEG の試験系列・測定範囲の穴／対 G-PCC の未決着 2 件

進める順の提案は、タイルの確認と速度の小さな直し → 推定の損を測る → 属性の同時符号化 → 論文。

09-20 時点の計画のうち、鍵の縮退は論文の項目に含めた。走査の分割と可逆な多重解像度はいまの優先順では
後に回し、[`notes/04_next_steps.md`](notes/04_next_steps.md)（履歴）に残す。

作業規則は [`notes/04_next_steps.md`](notes/04_next_steps.md) の末尾、
点検の手順は [`notes/05_verification_checklist.md`](notes/05_verification_checklist.md) にある。
**部分集合で測った値を全体の性質として報告しない** — 2026-09-19 に実際にやらかした。

---

## 背景メモ

- [`results/architecture.md`](results/architecture.md) — **いまの構成の全体像**（コードから起こした一覧）
- [`notes/00_plan.md`](notes/00_plan.md) — 確定した前提と Phase 分け
- [`notes/01_normalization_layer.md`](notes/01_normalization_layer.md) — 正規化レイヤーの設計
- [`notes/02_codec_container.md`](notes/02_codec_container.md) — PCC2 コンテナの設計
- [`notes/03_scan_model_codec.md`](notes/03_scan_model_codec.md) — 走査モデル符号器の設計
- [`notes/05_verification_checklist.md`](notes/05_verification_checklist.md) — **点検の手引き。数字を出す前・主張する前・訂正した後に通す**
- [`RESUME.md`](RESUME.md) — 中断・再開用の現在地

## ライセンス

[Apache License 2.0](LICENSE)。同梱・参照している第三者のコードとデータの出典は [NOTICE](NOTICE) にまとめてある。`cpp/include/pcc/nanoflann.hpp` は [nanoflann](https://github.com/jlblancoc/nanoflann)（BSD-2-Clause）。
