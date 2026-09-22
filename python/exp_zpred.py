"""走査モデルの z を、どの予測子が一番短くできるかを実データで比べる。

z は全件で最大の成分なのに、走査モデルは「掃引内の同じ戻り番号の直前の点」
という素朴な予測のままである。候補を並べて符号長を見る。

  1. 直前の点（今）
  2. 直前 3 点の差の中央値（med3）
  3. 直前 2 点からの線形外挿
  4. 隣の走査線の最近傍点
  5. 隣の走査線＋直前の点（2 つの平均）
  6. 走査線内の位置からの 1 次（地面の傾き）

符号長は離散ラプラスの見積り。走査線ごとに見積もって点数で重み付ける。
"""
import sys, math
import numpy as np
from scipy.spatial import cKDTree


def bits(r):
    r = np.asarray(r, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    b = np.mean(np.abs(r - np.median(r)))
    return math.log2(2 * math.e * b) if b > 1e-9 else 0.0


def main(path, minn=32):
    a = np.loadtxt(path, skiprows=1, dtype=np.int64)
    line = a[:, 0]
    xy = a[:, 2:4].astype(np.float64)
    z = a[:, 4].astype(np.float64)
    n = len(a)
    brk = np.flatnonzero(np.diff(line) != 0) + 1
    segs = [s for s in np.split(np.arange(n), brk) if len(s) >= minn]
    if not segs:
        print('  走査線が短すぎる'); return
    npt = sum(len(s) for s in segs)

    names = ['直前の点（今）', 'med3', '線形外挿', '隣の線', '隣の線＋直前', '位置の1次']
    acc = np.zeros(len(names))
    prev = None
    nb_cov = 0
    for s in segs:
        zz = z[s]
        L = len(s)
        # 1. 直前の点
        p1 = np.concatenate([[zz[0]], zz[:-1]])
        # 2. med3
        d = np.diff(zz, prepend=zz[0])
        d3 = np.stack([np.roll(d, k) for k in (1, 2, 3)])
        d3[:, :3] = 0
        p2 = p1 + np.median(d3, axis=0)
        # 3. 線形外挿
        p3 = np.concatenate([[zz[0], zz[0]], 2 * zz[1:-1] - zz[:-2]])
        # 4/5. 隣の線
        p4 = p1.copy()
        if prev is not None:
            dd, nb = cKDTree(xy[prev]).query(xy[s], k=1)
            m = dd < 4000.0
            if m.sum() > 16:
                p4[m] = z[prev][nb[m]]
                nb_cov += int(m.sum())
        p5 = 0.5 * (p4 + p1)
        # 6. 走査線内の位置からの 1 次
        t = np.arange(L, dtype=np.float64)
        A = np.stack([t, np.ones(L)], 1)
        w, *_ = np.linalg.lstsq(A, zz, rcond=None)
        p6 = A @ w
        for i, p in enumerate((p1, p2, p3, p4, p5, p6)):
            acc[i] += L * bits(zz - p)
        prev = s
    acc /= npt
    base = acc[0]
    print(f'  走査線 {len(segs)} 本 / 点 {npt}  隣が取れた点 {100*nb_cov/npt:.1f}%')
    for nm, b in zip(names, acc):
        d = f'   （{100*(b-base)/base:+.1f}%）' if nm != names[0] else ''
        print(f'    {nm:<18}{b:7.3f} bit/点{d}')


if __name__ == '__main__':
    main(sys.argv[1])
