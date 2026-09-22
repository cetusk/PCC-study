"""同一パルスの戻りは 1 本の直線に乗る、という物理を予測に使えるか。

いまの符号器は、2 回目の戻りを「直前の 2 回目の戻り」や「直近 W 点の最近傍」から
予測している。**同じパルスの 1 回目の戻り**は、物理的にはもっと近い相手である
（同一の光線上にあり、距離だけが違う）。

測るのは「残差がどれだけ短くなるか」であって、符号長そのものではない。
符号長は適応模型で別に測る（exp_raycode.py）。
"""
from __future__ import annotations
import sys, glob, os
import numpy as np
import laspy

N = int(sys.argv[2]) if len(sys.argv) > 2 else 500000


def load(p):
    las = laspy.read(p)
    n = min(N, len(las.points))
    X = np.asarray(las.X)[:n].astype(np.int64)
    Y = np.asarray(las.Y)[:n].astype(np.int64)
    Z = np.asarray(las.Z)[:n].astype(np.int64)
    rn = np.asarray(las.return_number)[:n].astype(np.int64)
    nr = np.asarray(las.number_of_returns)[:n].astype(np.int64)
    return X, Y, Z, rn, nr


def bits(a):
    """符号つき整数の並びを送るのに要る桁数の合計（1 値あたり）。"""
    a = np.abs(np.asarray(a, dtype=np.int64))
    return float(np.mean(np.where(a > 0, np.log2(a.astype(np.float64) + 1) + 1, 1)))


def run(lab, path):
    X, Y, Z, rn, nr = load(path)
    n = len(X)
    P = np.stack([X, Y, Z], 1)

    # 同じパルスの直前の戻り: 格納順で直前の点が「同じパルスの 1 つ前の戻り」か
    # （rn が 1 増えていれば同一パルス。LAS は 1 パルスの戻りを連続して書く）
    same = np.zeros(n, dtype=bool)
    same[1:] = (rn[1:] == rn[:-1] + 1) & (nr[1:] == nr[:-1])
    m = int(same.sum())
    if m < 100:
        print("%-9s 同一パルスの連続が %d 件しかない" % (lab, m)); return
    idx = np.nonzero(same)[0]

    # 1) いまの形に近い予測: 直前の点そのもの（無傾）
    d_prev = P[idx] - P[idx - 1]

    # 2) 光線モデル: 同じパルスの前の戻りから、**直前のパルスで測った戻り間ベクトル**
    #    だけ進んだ所。向きは復号済みのものだけから作る（副情報ゼロ）。
    #    直前に現れた「同じ戻り番号の組」の差分を覚えておく。
    pred = np.zeros_like(d_prev)
    last = {}
    for t, i in enumerate(idx):
        k = int(rn[i])
        v = last.get(k)
        if v is not None:
            pred[t] = v
        last[k] = P[i] - P[i - 1]
    d_ray = d_prev - pred

    # 3) 参考: 戻り間ベクトルの長さと向きのばらつき
    L = np.linalg.norm(d_prev.astype(np.float64), axis=1)
    u = d_prev.astype(np.float64) / np.maximum(L, 1)[:, None]
    cos = np.sum(u[1:] * u[:-1], 1)

    print("%-9s 点 %d  同一パルスの連続 %d (%.1f%%)" % (lab, n, m, 100 * m / n))
    print("           戻り間の距離 中央値 %.3f m  向きの連続性 cos 中央値 %.5f"
          % (np.median(L) * 0.01, np.median(cos)))
    print("           残差の桁数/値  直前の点から %.2f → 光線モデルで %.2f  (%+.1f%%)"
          % (bits(d_prev), bits(d_ray),
             100 * (bits(d_ray) / bits(d_prev) - 1)))


for lab, path in [("USGS NY", "data/raw/usgs/NY_ClintonEssex_2014.laz"),
                  ("USGS AK", "data/raw/usgs/AK_Kenai_2008_000001.laz"),
                  ("AHN3", "data/raw/ahn3/31HZ1_20.LAZ"),
                  ("AHN4", "data/raw/ahn4/31HZ1_20.LAZ"),
                  ("AHN5", "data/raw/ahn5/31HZ1_20.LAZ"),
                  ("vegetation", "data/raw/small/vegetation_1_3.las")]:
    if os.path.exists(path):
        run(lab, path)


def run2(lab, path):
    """比較相手を実際の勝者に近づける。

    USGS NY の勝者は 幾何v4W4（直近 4 点の最近傍＋傾き）。多重戻りの点では
    最近傍は「同じパルスの 1 つ前の戻り」になるはずなので、**素朴な直前の点**と
    比べるのは相手が弱すぎる。傾きを足した版と、戻り番号ごとに状態を分けた版
    （幾何v2 に相当）も並べる。
    """
    X, Y, Z, rn, nr = load(path)
    n = len(X)
    P = np.stack([X, Y, Z], 1)
    same = np.zeros(n, dtype=bool)
    same[1:] = (rn[1:] == rn[:-1] + 1) & (nr[1:] == nr[:-1])
    idx = np.nonzero(same)[0]
    if len(idx) < 100:
        return

    # 幾何v4 に相当: 直近 4 点のうち最も近い点 ＋ 直近 3 差分の中央値
    W = 4
    res_v4 = np.zeros((len(idx), 3), dtype=np.int64)
    res_ray = np.zeros((len(idx), 3), dtype=np.int64)
    res_v2 = np.zeros((len(idx), 3), dtype=np.int64)
    hist = []                      # 直近の差分（傾き用）
    lastk = {}                     # 戻り番号ごとの直前の点（幾何v2 相当）
    lastgap = {}                   # 戻り番号ごとの直前の戻り間ベクトル（光線）
    for t, i in enumerate(idx):
        lo = max(0, i - W)
        d = ((P[lo:i] - P[i]) ** 2).sum(1)
        j = lo + int(d.argmin())
        tr = np.median(np.array(hist[-3:]), axis=0).astype(np.int64) if len(hist) >= 3 \
             else np.zeros(3, dtype=np.int64)
        res_v4[t] = P[i] - (P[j] + tr)
        k = int(rn[i])
        pk = lastk.get(k)
        res_v2[t] = P[i] - (pk if pk is not None else P[i - 1])
        g = lastgap.get(k)
        res_ray[t] = P[i] - (P[i - 1] + (g if g is not None else 0))
        lastgap[k] = P[i] - P[i - 1]
        lastk[k] = P[i]
        hist.append(P[i] - P[i - 1])
    print("%-9s 同一パルスの点 %d 件での残差の桁数/値" % (lab, len(idx)))
    print("           幾何v4 相当 %.2f  幾何v2 相当 %.2f  **光線モデル %.2f**  (対 v4 %+.1f%%)"
          % (bits(res_v4), bits(res_v2), bits(res_ray),
             100 * (bits(res_ray) / bits(res_v4) - 1)))


if len(sys.argv) > 3 and sys.argv[3] == "strong":
    for lab, path in [("USGS NY", "data/raw/usgs/NY_ClintonEssex_2014.laz"),
                      ("USGS AK", "data/raw/usgs/AK_Kenai_2008_000001.laz"),
                      ("AHN3", "data/raw/ahn3/31HZ1_20.LAZ"),
                      ("AHN4", "data/raw/ahn4/31HZ1_20.LAZ"),
                      ("AHN5", "data/raw/ahn5/31HZ1_20.LAZ")]:
        if os.path.exists(path):
            run2(lab, path)
