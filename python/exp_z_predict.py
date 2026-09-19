"""走査モデルの z を、2 次元の近傍から予測できるか測る。

走査モデルが決まると点は (掃引 × 発射) の格子をなす。いまの符号器は
「左」（掃引内・同じ戻り番号の直前点）しか使っておらず、z の費用は
素の差分とほぼ同じ 9.6 bit で頭打ちになっている。

ここでは「上」（前の掃引の対応する点）を足したときの効果を測る。
上の対応付けは、本来は走査角 θ（gps_time と副情報から決まる）で行うが、
それは空間最近傍の近似でしかない。まず空間最近傍そのもの（オラクル）で
上限を測る。上限が「左」に勝たなければ、θ による対応も勝てない。

    $PCCPY python/exp_z_predict.py [点数]
"""
from __future__ import annotations
import sys
import numpy as np
import laspy
from scipy.spatial import cKDTree


def H0(v) -> float:
    v = np.asarray(v, dtype=np.int64)
    if len(v) < 2:
        return 0.0
    _, c = np.unique(v, return_counts=True)
    p = c / len(v)
    return float(-(p * np.log2(p)).sum())


def main(path='data/raw/ahn4/31HZ1_20.LAZ', npts=2_000_000, gap=200):
    with laspy.open(path) as f:
        p = f.read_points(npts)
    gb = np.array(p['gps_time'], dtype=np.float64).view(np.int64)
    sid = np.array(p['point_source_id'], dtype=np.int64)
    X = np.array(p['X'], dtype=np.int64)
    Y = np.array(p['Y'], dtype=np.int64)
    Z = np.array(p['Z'], dtype=np.int64)
    rn = np.array(p['return_number'], dtype=np.int64)
    n = len(gb)
    o = np.lexsort((np.arange(n), gb, sid))
    gb, sid, X, Y, Z, rn = gb[o], sid[o], X[o], Y[o], Z[o], rn[o]

    base = np.zeros(n, bool)
    base[0] = True
    base[1:] = (np.diff(gb) > gap) | (np.diff(sid) != 0)
    starts = np.flatnonzero(base)
    ends = np.r_[starts[1:], n]
    print(f"点 {n}  掃引 {len(starts)} 本  1 掃引あたり中央 {int(np.median(ends - starts))} 点")

    left = np.full(n, -1, np.int64)      # 掃引内・同じ戻り番号の直前点
    up = np.full(n, -1, np.int64)        # 前の掃引の空間最近傍（オラクル）
    prev_a, prev_b = -1, -1
    for a, b in zip(starts, ends):
        last = {}
        for t in range(a, b):
            r = int(rn[t])
            if r in last:
                left[t] = last[r]
            last[r] = t
        if prev_a >= 0 and b - a > 0 and prev_b - prev_a >= 1:
            tree = cKDTree(np.c_[X[prev_a:prev_b], Y[prev_a:prev_b]].astype(np.float64))
            _, j = tree.query(np.c_[X[a:b], Y[a:b]].astype(np.float64), k=1)
            up[a:b] = prev_a + j
        prev_a, prev_b = a, b

    ok = (left >= 0) & (up >= 0)
    upleft = np.full(n, -1, np.int64)
    m = ok & (up[np.maximum(left, 0)] >= 0)
    upleft[m] = up[left[m]]
    ok &= upleft >= 0
    print(f"左・上・左上がすべて取れた点 {ok.sum()} ({ok.mean():.3f})")

    z = Z[ok]
    a_ = Z[left[ok]]          # 左
    b_ = Z[up[ok]]            # 上
    c_ = Z[upleft[ok]]        # 左上

    med = np.where(c_ >= np.maximum(a_, b_), np.minimum(a_, b_),
          np.where(c_ <= np.minimum(a_, b_), np.maximum(a_, b_), a_ + b_ - c_))
    avg = (a_ + b_) // 2
    grad = a_ + b_ - c_       # 平面外挿

    print()
    print(f"{'予測子':<26}{'0次[bit]':>10}{'残差RMS[m]':>13}")
    for nm, pred in (("左（現行の実装）", a_),
                     ("上（オラクル上限）", b_),
                     ("左上", c_),
                     ("MED（JPEG-LS）", med),
                     ("平均（左,上）", avg),
                     ("平面外挿 左+上-左上", grad)):
        r = z - pred
        print(f"{nm:<26}{H0(r):>10.3f}{r.std()*0.001:>13.3f}")

    print()
    print("参考: 上の対応がどれだけ近いか")
    d = np.hypot((X[ok] - X[up[ok]]).astype(float), (Y[ok] - Y[up[ok]]).astype(float)) * 0.001
    dl = np.hypot((X[ok] - X[left[ok]]).astype(float), (Y[ok] - Y[left[ok]]).astype(float)) * 0.001
    print(f"  上までの距離 中央 {np.median(d):.3f} m / 左までの距離 中央 {np.median(dl):.3f} m")


if __name__ == '__main__':
    a = sys.argv[1:]
    main(npts=int(a[0]) if a else 2_000_000)
