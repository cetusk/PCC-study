# 表現の正規化レイヤー（設計メモ）

## 何をする層か

点群を **センサが本来持っている量と分解能** に戻してから既存コーデックに渡す。
圧縮アルゴリズムそのものには手を入れない。

実運用データのビットの多くは「モデル化が足りない」のではなく
「表現が冗長」であることに使われている、という計測結果に基づく。

## 設計方針

1. **自動検出** — フィールド名をハードコードしない。関係はデータから見つける。
2. **完全検証** — 見つけた関係は全点で厳密に検証してからでないと採用しない。
3. **小さな仕様** — 復元に必要な情報は数百バイトの仕様に収め、ストリームに含める。
   ベンチマークでも必ず副情報を勘定に入れる。
4. **決定論的** — 外部符号化は整数のみのレンジコーダ。浮動小数を使わない
   （survey 攻め筋⑥: 可逆に学習モデルを使うなら決定性は後から足せない）。
5. **申告制** — 各操作は exact（ビット完全）か bounded（誤差上限つき）かを必ず申告し、
   bounded の上限は推定ではなく全点での実測値を書く。

## 実装した操作

| 操作 | 内容 | 可逆性 |
|---|---|---|
| `drop_constant` | 定数フィールドを送らない | exact |
| `drop_duplicate` | 他フィールドと完全に同一なので送らない | exact |
| `drop_affine` | `a*src + b` で厳密に表せるので送らない | exact |
| `residual_code` | 他フィールドとの差分を外部符号化 | exact |
| `palette` | 取りうる値が少ない場合に辞書+索引へ（float でもビット完全） | exact |
| `float_grid` | float が「整数×一定刻み」の場合に整数へ | exact（検証通過時のみ） |
| `polar_restore` | 回転式 LiDAR のデカルト座標を極座標格子へ | bounded |

## ファイル

```
python/normalize.py    操作の検出・適用・逆適用（依存グラフの反復解決つき）
python/rangecoder.py   決定論的レンジコーダ（LZMA 方式、整数のみ）
python/pccfile.py      PCC1 コンテナ（仕様＋ストリーム。副情報を必ず含める）
python/bench_normalize.py  LAS/LAZ 経路のエンドツーエンド評価＋復元検証
python/bench_polar.py      回転式 LiDAR 経路のエンドツーエンド評価＋誤差検証
```

## 使い方

```bash
# ALS (LAS/LAZ)
$PCCPY python/bench_normalize.py <file.laz> --max-points 2000000
$PCCPY python/bench_normalize.py <file.laz> --no-residual     # 重複削除のみ

# 回転式 LiDAR (.bin)
$PCCPY python/bench_polar.py '<dir>/*.bin' --r-step 0.002 --ang-deg 0.002
```

どちらも実行のたびに復元検証を行い、通らなければ数字を出さない。
