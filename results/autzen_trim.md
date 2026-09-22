# ビット内訳レポート  —  autzen_trim.laz

> **履歴**（2026-09-23 の索引 `results/INDEX.md` による）: Python 試作（09-17）のビット内訳。**現在の数値は `results/standing.md`**。

生成 2026-09-17 20:48   

## 0. データ概要
```
N = 110,000  src = data/raw/small/autzen_trim.laz
  int grid : scale=[0.01 0.01 0.01]  span(int)=[117746  56270  11425]  needed bits/axis=[17, 16, 14]
  attr intensity          dtype=uint16   unique=255 min=0 max=254
  attr return_number      dtype=uint8    unique=4 min=1 max=4
  attr number_of_returns  dtype=uint8    unique=4 min=1 max=4
  attr scan_direction_flag dtype=uint8    unique=2 min=0 max=1
  attr edge_of_flight_line dtype=uint8    unique=1 min=0 max=0
  attr classification     dtype=uint8    unique=2 min=1 max=2
  attr synthetic          dtype=uint8    unique=1 min=0 max=0
  attr key_point          dtype=uint8    unique=1 min=0 max=0
  attr withheld           dtype=uint8    unique=1 min=0 max=0
  attr scan_angle_rank    dtype=int8     unique=18 min=-18 max=-1
  attr user_data          dtype=uint8    unique=19 min=117 max=135
  attr point_source_id    dtype=uint16   unique=1 min=7326 max=7326
  attr gps_time           dtype=float64  unique=99331 min=245379.39843682514 max=245385.91112104454
  attr red                dtype=uint16   unique=192 min=40 max=236
  attr green              dtype=uint16   unique=172 min=55 max=228
  attr blue               dtype=uint16   unique=166 min=52 max=219
```

## 1. ベースライン（5軸同時計測）
```
codec                               size         bpp        enc        dec      peak
raw int32 xyz                  1.320 MB    96.000 bpp  enc    0.00s  dec    0.00s  peak       0MB  lossless  
LASzip                         0.591 MB    43.009 bpp  enc    0.01s  dec    0.02s  peak      70MB  lossless  
LASzip/geom-only               0.291 MB    21.159 bpp  enc    0.01s  dec    0.01s  peak      70MB  lossless  
G-PCC/TMC13 (geom, lossless)     0.289 MB    21.011 bpp  enc    0.22s  dec    0.19s  peak      73MB  lossless  
ref/original/zstd19/delta      0.311 MB    22.621 bpp  enc    0.08s  dec    0.00s  peak     122MB  lossless  
```

## 2. フィールド別ビット内訳（幾何 vs 属性）
```
N = 110,000

LASzip 実測  全体  43.009 bpp  /  幾何のみ  21.159 bpp  → 属性ぶん  21.850 bpp (50.8%)

field                 dtype       bits/pt  order0 H  raw bits  内訳
--------------------------------------------------------------------------------
intensity             uint16        6.949     7.681        16  ########################################
gps_time              float64       6.699    16.541        64  #######################################
red                   uint16        4.530     6.969        16  ##########################
green                 uint16        4.412     6.585        16  #########################
blue                  uint16        4.331     6.150        16  #########################
classification        uint8         0.918     0.791         8  #####
user_data             uint8         0.442     2.795         8  ###
number_of_returns     uint8         0.390     0.845         8  ##
return_number         uint8         0.366     0.529         8  ##
scan_angle_rank       int8          0.318     3.397         8  ##
scan_direction_flag   uint8         0.090     1.000         8  #
edge_of_flight_line   uint8         0.003    -0.000         8  
synthetic             uint8         0.003    -0.000         8  
key_point             uint8         0.003    -0.000         8  
withheld              uint8         0.003    -0.000         8  
point_source_id       uint16        0.003    -0.000        16  
--------------------------------------------------------------------------------
属性 孤立コスト合計                         29.461   (LASzip 実測の属性ぶん 21.850 bpp / 相関で回収済み +7.611)
幾何 孤立コスト                           22.586   (3軸合計, 参照パイプライン)
```

## 3. 点の順序が持つ冗長
```
理論上限 log2(N!)/N        =   15.305 bits/点  （集合として符号化すれば原理上不要な量）
参照PL: 取得順のまま         =   22.621 bpp
参照PL: Morton 並べ替え後    =   27.878 bpp   (★並べ替えで悪化 -5.257 bpp)
G-PCC (順序非依存・集合符号化) =   21.011 bpp
  → 取得順（走査線順）は Morton より予測しやすい。「順序は捨ててよい冗長」という素朴な前提はこのデータでは成立しない。
```

## 4. LSB スイープ（ノイズ床はどこか）
```

--- laszip_geom ---
  k   step[mm]       bpp   Δbpp/LSB      ノイズ度  判定
------------------------------------------------------------------------
  0      10.00    21.159        nan       nan  (基準)
  1      20.00    18.621      2.538      0.85  ▨ ほぼノイズ・弱い構造
  2      40.00    15.896      2.724      0.91  ▨ ほぼノイズ・弱い構造
  3      80.00    13.490      2.406      0.80  ▨ ほぼノイズ・弱い構造
  4     160.00    11.488      2.002      0.67  ▤ 構造あり
  5     320.00     9.931      1.557      0.52  ▤ 構造あり
  6     640.00     8.523      1.408      0.47  ▤ 構造あり

--- tmc13 ---
  k   step[mm]       bpp   Δbpp/LSB      ノイズ度  判定
------------------------------------------------------------------------
  0      10.00    21.011        nan       nan  (基準)
  1      20.00    20.153      0.859      0.29  □ 強い構造（モデル化余地）
  2      40.00    17.769      2.384      0.79  ▨ ほぼノイズ・弱い構造
  3      80.00    14.733      3.036      1.01  ■ ほぼ純ノイズ（圧縮不能）
  4     160.00    11.635      3.098      1.03  ■ ほぼ純ノイズ（圧縮不能）
  5     320.00     8.796      2.839      0.95  ■ ほぼ純ノイズ（圧縮不能）
  6     640.00     6.105      2.690      0.90  ▨ ほぼノイズ・弱い構造

```

_経過 5.0s_