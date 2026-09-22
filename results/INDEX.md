# 文書の索引

このリポジトリの文書は 50 本近くあり、書いた時期で前提も数値も違う。
**どれが今も正しく、どれが記録として残してあるだけか**をここにまとめる。

## 最初に読むもの

1. [`results/standing.md`](standing.md) — 既存技術（LASzip・G-PCC）と比べた現在地。**現在の数値の正本**。
2. [`results/architecture.md`](architecture.md) — 符号器の構成。コードから書き起こしたもの。
3. [`results/approaches.md`](approaches.md) — 試した手と、効いたもの・効かなかったもの・残った方向。
4. [`notes/07_status.md`](../notes/07_status.md) — 確定したこと・覆ったこと・未確定のこと。
5. [`notes/05_verification_checklist.md`](../notes/05_verification_checklist.md) — 誤りの型と点検の手順。

## 数値の正本は 1 か所

**現在の数値（サイズ・速度・メモリ・対 LASzip・対 G-PCC）は `results/standing.md` の §1 と
§2 以降の表だけに書く。**他の文書は standing.md を参照するか、書いた日付と出どころの
ログを添えて「その時点の値」として書く。

以前は README・RESUME・07_status・losses・scan_model_fitting の各所に「現在の値」が
あり、測り直すたびに一部だけが更新されて食い違った（2026-09-23 の棚卸しで 9 文書に
古い値が残っていた）。

測り直しの手順は `python/verify_suite.py`（検証一式）と `python/bench_all.py`・
`bench_three.py`・`bench_split.py`（数値）。ログは `data/work/grid/` に**日付か版の札を
付けて**残す（同じ名前に上書きしない）。

## 状態の札

| 札 | 意味 |
|---|---|
| **現行** | 今のシステムと今の数値を書いている。食い違ったら直す対象 |
| **参照** | 結論は今も有効な発見・設計。数値はその時点の実験のもの |
| **履歴** | 後の文書に置き換えられた記録。経緯を追うために残す |
| **撤回** | 撤回した主張を含む。冒頭か該当節に撤回の注記がある |

## 一覧

時期の記号: **P0** Python の試作（09-17）/ **P1** 正規化レイヤーと PCC1（09-17〜18）/
**P2** 自前符号器 PCC2（09-19）/ **SM** 走査モデル・査読・順序（09-19〜21）/
**P3** 全次元保存と負けの解消（09-22）/ **NF** 座標の外の構造・別分野（09-22〜23）

### 入口と手順

| 文書 | 札 | 時期 | 内容 |
|---|---|---|---|
| `README.md` | 現行 | P3 | 入口。論文・到達点・発見の一覧・使い方 |
| `RESUME.md` | 現行（手順）| P3 | 環境の復元と作業の再開手順 |
| `notes/05_verification_checklist.md` | 現行 | SM→ | 誤りの型と点検の手順（規則として有効） |
| `notes/07_status.md` | 現行 | P3/NF | 確定・覆った・未確定の整理 |

### 現在の構成と到達点

| 文書 | 札 | 時期 | 内容 |
|---|---|---|---|
| `results/standing.md` / `.html` | 現行（正本） | P3/NF | 既存技術と比べた現在地。html は同じ内容の図つき版 |
| `results/architecture.md` | 現行 | P3/NF | 符号器の構成（器・正規化・候補・後追い・復号の波） |
| `results/approaches.md` | 現行 | NF | 試した手の整理と残りの方向 |
| `results/losses.md` | 現行（改良の記録） | P3/NF | 負けを 1 つずつ潰した記録と査読 |
| `results/defects.md` | 現行 | 09-23 | コードの不具合を洗い出して直した記録（サイズは動かさず） |
| `results/structure.md` | 参照 | NF | 座標の外の構造（パルス・曲面予測・グラフ・Voronoi） |
| `results/newfields.md` | 参照 | NF | 別分野（音声・画像・汎用圧縮）の道具。符号つき文脈 |
| `results/forward_operator.md` | 参照 | P3 | 順演算子を逆に辿る 5 つの筋（KITTI の整数化・粗い格子） |
| `results/orderfree.md` | 参照 | NF | 点の順序を捨ててよいなら何が変わるか |
| `results/ml_concepts.md` | 参照 | NF | 学習の概念を決定的な算法として試す |
| `results/literature_2026.md` | 参照 | NF | 2026 年の文献調査 |

### 設計メモ

| 文書 | 札 | 時期 | 内容 |
|---|---|---|---|
| `notes/01_normalization_layer.md` | 参照 | P1 | 表現の正規化レイヤーの設計 |
| `notes/02_codec_container.md` | 参照 | P2 | PCC2 の器の設計 |
| `notes/03_scan_model_codec.md` | 参照 | SM | 走査モデル経由の幾何符号器の設計 |
| `notes/00_plan.md` | 履歴 | P0 | 最初の前提と段取り |
| `notes/04_next_steps.md` | 履歴 | SM | 09-20 時点の「今後」。現在の方向は approaches.md |
| `notes/06_timeline.md` | 履歴 | SM | 09-21 未明までの作業史 |

### 発見（今も有効な結論）

| 文書 | 札 | 時期 | 内容 |
|---|---|---|---|
| `results/als_scan_structure.md` | 参照 | SM | ALS の走査構造（走査角・掃引面・2 次元格子） |
| `results/acquisition_score.md` | 参照 | P1 | 取得構造の残り具合を測る 6 指標 |
| `results/attribute_spatial_finding.md` | 参照（一部訂正）| P1 | 幾何から決めた順序での属性予測 |
| `results/vendor_generality.md` | 参照 | P1 | 属性の冗長は 9 ベンダで一般的か |
| `results/ahn4_extrabytes_finding.md` | 参照（訂正済み）| P0/P1 | ExtraBytes 列の冗長 |
| `results/ahn4_31HZ1_20_4M.md` | 参照 | P0/P1 | AHN4 400 万点のビット内訳（09-18 生成） |
| `results/externalize_finding.md` | 参照 | P1 | 正規化の後に何が残るか |
| `results/repeat_survey_finding.md` | 参照 | P1 | 反復測量の差分は幾何の 1.3% しか効かない |
| `results/subspacing_finding.md` | 参照 | P1 | 点間隔より細かい構造は無い |
| `results/ect_finding.md` | 参照 | P1 | 位相（ECT）は圧縮にならない |
| `results/lfs_finding.md` | 参照 | P1 | 局所特徴サイズによる可変誤差予算 |
| `results/lossy_surface_finding.md` | 参照 | P1 | 面を送る非可逆符号化と歪み尺度の落とし穴 |
| `results/lossless_refinements.md` | 参照 | P1/P2 | 候補の拡張と選択の物差し |
| `results/stanford_finding.md` | 参照 | P1 | 密なオブジェクトスキャンでの検証 |
| `results/tls_polar_finding.md` | 参照 | P1 | 地上型では極座標への復帰が逆効果 |
| `results/matrix_comparison.md` | 参照 | P1 | 13 データ × 6 計測の一斉比較 |

### 履歴（後の文書に置き換えられた記録）

| 文書 | 札 | 時期 | 置き換えた文書・内容 |
|---|---|---|---|
| `results/scan_model_fitting.md` | 履歴 | SM→P3 | 74 節の作業記録。§74「最終」の値は古い → standing.md |
| `results/pcc2_codec.md` | 履歴 | P2 | PCC2 の初回実測 → standing.md |
| `results/COMPRESSION_STATUS.md` | 履歴 | P1 | 正規化レイヤー時代の「到達点」→ standing.md |
| `results/decision_table_measured.md` | 履歴 | P1 | 精度要件ごとの削減量（正規化レイヤー時代） |
| `results/fulltile_ahn4.md` | 履歴 | P1 | タイル全体での実測（正規化レイヤー時代） |
| `results/CPP_PORT.md` | 履歴 | P1 | Python から C++ への移植 |
| `results/VERIFICATION.md` | 履歴 | P1 | 09-18 の数値の再計算。規則は 05_verification_checklist へ |
| `results/autzen_trim.md` | 履歴 | P0 | autzen_trim のビット内訳 |
| `results/ahn4_31HZ1_20_4M.PARTIAL.md` | 履歴 | P0 | 上の完成版の途中経過 |

### 撤回を含む

| 文書 | 札 | 時期 | 撤回した主張 |
|---|---|---|---|
| `results/kitti_polar_finding.md` | 撤回 | P1 | 「float32 デカルト座標は派生表現」→ 実は 1 mm 格子（forward_operator.md §4） |
| `results/end_to_end_comparison.md` | 撤回（一部） | P1 | 3 方式比較の一部（56・81 行の訂正） |
| `results/combination_finding.md` | 撤回（見積り） | P1 | G-PCC 幾何との組み合わせの見積り |

### 論文

`.tex` は作業木にだけあり、git では PDF だけを追跡する（`.gitignore` の `/paper/*`）。

| 文書 | 札 | 内容 |
|---|---|---|
| `paper/pcc.pdf` | 現行（一部未再測定） | 正規化レイヤーの論文。§e2e（6.3% / 22.3%、295.2 MB）は測り直していない（`notes/07_status.md` §4.5） |
| `paper/coding.pdf` | 現行 | 幾何から導いた順序での属性の可逆符号化 |
| `paper/order.pdf` | 参照 | 入力順がベンチマークを左右すること |
