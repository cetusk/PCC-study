"""走査線をまたぐ近傍（画像符号化の「上の行」に当たるもの）に構造があるか。

いまの幾何の予測子は、どれも**格納順に沿って後ろを見る**（直前 W 点）。
画像の可逆符号器が使うのは「左」と**「上」**である。点群で「上」に当たるのは
**一つ前の走査線の対応する点**で、格納順では L 点前にある（L = 1 線の点数）。

L を探す方法: 遅れ L ごとに |P[i] − P[i−L]| の中央値を出し、谷を探す。
谷があれば走査線の周期が格納順に現れているということで、そこを予測に使える。
**副情報は要らない**（L は復号済みの点から同じ手順で探せる）。
"""
from __future__ import annotations
import sys, os
import numpy as np
import laspy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000


def run(lab, path):
    las = laspy.read(path)
    n = min(N, len(las.points))
    P = np.stack([np.asarray(las.X)[:n], np.asarray(las.Y)[:n],
                  np.asarray(las.Z)[:n]], 1).astype(np.int64)
    rn = np.asarray(las.return_number)[:n].astype(np.int64)
    # 1 回目の戻りだけを見る（多重戻りは別の話＝光線モデル）
    P1 = P[rn == 1]
    m = len(P1)
    if m < 20000:
        print("%-9s 1 回目の戻りが %d 点しかない" % (lab, m)); return
    lags = [1, 2, 3, 4] + list(range(8, 4096, 8))
    med = []
    for L in lags:
        if L >= m: break
        d = np.abs(P1[L:] - P1[:-L]).sum(1)
        med.append((float(np.median(d)), L))
    med.sort()
    base = [v for v, L in med if L == 1][0]
    best_far = [(v, L) for v, L in med if L >= 8][:3]
    print("%-9s 1 回目の戻り %d 点" % (lab, m))
    print("           遅れ 1 の |差| 中央値 %.0f（0.01 m 単位）" % base)
    print("           遅れ 8 以上で一番小さい 3 つ: " +
          "  ".join("L=%d %.0f (%.2f 倍)" % (L, v, v / base) for v, L in best_far))


for lab, path in [("USGS NY", "data/raw/usgs/NY_ClintonEssex_2014.laz"),
                  ("USGS AK", "data/raw/usgs/AK_Kenai_2008_000001.laz"),
                  ("AHN3", "data/raw/ahn3/31HZ1_20.LAZ"),
                  ("AHN5", "data/raw/ahn5/31HZ1_20.LAZ"),
                  ("workshop", "data/raw/extrabytes/workshop_TM_551_101.laz"),
                  ("red-rocks", "data/raw/extrabytes/entwine_data_red-rocks.laz")]:
    if os.path.exists(path):
        run(lab, path)
