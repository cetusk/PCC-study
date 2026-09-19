# 発見: LAS ExtraBytes の冗長（AHN4 実タイル）

## 訂正メモ

初出時に「float64 64bit で格納されている」と書いたが誤り。
ExtraBytes VLR を確認したところ、実際の格納は **uint16 / int16 + scale 0.01** で、
laspy が scale を適用して float64 に見せていただけだった。
bpp の実測値は本物の格納形式に対して測っていたので数字は変わらないが、
**無駄の正体は「容器の肥大」ではなく「完全な重複」**である。

```
ExtraBytes VLR:
  Amplitude    data_type=3 (uint16)  scale=0.01   'Echo signal amplitude [dB]'
  Reflectance  data_type=4 (int16)   scale=0.01   'Echo signal reflectance [dB]'
  Deviation    data_type=3 (uint16)  scale=None   'Pulse shape deviation'
```

## 事実

```
intensity 生: [1403 1400 1353 1376 1345 1383 1327 1380]
Amplitude 生: [1403 1400 1353 1376 1345 1383 1327 1380]
→ 200万点すべてでビット単位に一致（不一致 0 件）
```

**`Amplitude` は `intensity` の完全な重複。** 同じ uint16 を 2 箇所に書いている。

`Reflectance` は int16。intensity との関係:

```
H(Reflectance)            = 10.176 bit
H(Reflectance | intensity)=  5.937 bit
H(Reflectance - intensity)=  6.280 bit   ← 単純差分でほぼ下限に届く
```

物理的にも自然で、RIEGL の reflectance は amplitude から同距離の基準反射体の
値を引いたものなので、差は距離の滑らかな関数になる。

## LASzip 実測でのコスト（2M点）

```
現状                     71.252 bpp
Amplitude を除去          61.517 bpp   (− 9.735 bpp / −13.7%)
Reflectance を除去        61.480 bpp   (− 9.772 bpp / −13.7%)
両方を除去                51.745 bpp   (−19.507 bpp / −27.4%)
+ Deviation も除去        47.021 bpp   (−24.231 bpp / −34.0%)
```

## 正規化レイヤーでの実測（エンドツーエンド・検証済み）

`python/bench_normalize.py` が自動検出し、復元まで検証した結果:

```
A. 元のまま LAZ          71.240 bpp
B. 正規化後 合計          60.255 bpp   −15.4%（完全可逆）
```

検出内容:
- `Amplitude` ≡ `intensity` → 送らない
- `intensity` ← `Reflectance` との残差を外部符号化
- `red` ← `green` ← `blue` の残差連鎖

## 含意

- 幾何は全体の 31%（22.2 bpp）しかない。**幾何を何%削っても効きは 1/3 になる。**
- 学習型 PCC の論文は 8iVFB / MVUB / KITTI で評価する。これらには ExtraBytes が
  存在せず、属性も RGB か反射強度のみ。**実運用の LAS が抱えている冗長は、
  ベンチマークの構成上そもそも見えない。**

## 注意

AHN4（RIEGL 系センサ出力）に固有の可能性がある。一般化の前に他ベンダ・
他の実データで同じ分解をかけて確認が必要。ただし正規化レイヤーは
フィールド名をハードコードせず関係をデータから見つけるので、
別のパターンでも同じ枠組みで拾える。
