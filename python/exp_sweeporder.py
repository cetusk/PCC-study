"""掃引の中で、点が本当に発射順に並んでいるかを測る。

符号化器は (point_source_id, gps_time) で安定整列して発射順を復元する。
時刻に同値が多いと、同値の中は格納順のまま残る。それが発射順でなければ
掃引の中で点が飛び回り、走査線が「太く」見える。

連続比 = 隣り合う 2 点の距離の中央値 ÷ 点間隔。
1 に近ければ並びは発射順。大きければ飛んでいる。
"""
import sys
import numpy as np
import laspy
from scipy.spatial import cKDTree


def main(path, n=300000, minn=60):
    f = laspy.read(path)
    m = min(n, len(f.x))
    gi = np.asarray(f.gps_time[:m], dtype=np.float64).view(np.int64)
    src = np.asarray(f.point_source_id[:m], dtype=np.int64)
    P = np.column_stack([np.asarray(f.X[:m], dtype=np.float64),
                         np.asarray(f.Y[:m], dtype=np.float64)])
    o = np.lexsort((gi, src))
    gs, ss, Ps = gi[o], src[o], P[o]

    # 点間隔（最近傍距離の中央値）
    d, _ = cKDTree(P[:60000]).query(P[:60000], k=2)
    spacing = float(np.median(d[:, 1]))

    same = ss[1:] == ss[:-1]
    dg = gs[1:] - gs[:-1]
    pos = dg[same & (dg > 0)]
    thr = int(np.median(pos)) * 20
    tie = float(np.mean(dg[same] == 0))
    cuts = np.flatnonzero((~same) | (dg > thr)) + 1
    b = np.concatenate([[0], cuts, [m]])

    rat, rat_g = [], []
    for i in range(len(b) - 1):
        s, e = b[i], b[i + 1]
        if e - s < minn:
            continue
        step = np.linalg.norm(np.diff(Ps[s:e], axis=0), axis=1)
        rat.append(np.median(step))
        # 同値の中を「格納順」ではなく「最近傍をたどる順」に直したら縮むか
        # （ここでは上限の目安として、距離の小さいほうから並べ直した値を見る）
        rat_g.append(np.median(np.sort(step)[:len(step) // 2]))
    if not rat:
        print('  掃引が短すぎる'); return
    print(f'  点間隔 {spacing:.1f} 格子単位 / 時刻が同値の隣接点 {100*tie:.1f}%')
    print(f'  掃引内の隣り合う 2 点の距離 中央 {np.median(rat):.1f}'
          f'  → 連続比 {np.median(rat)/spacing:.2f}')
    print(f'  （並べ直しの目安: 下位半分の中央 {np.median(rat_g):.1f}'
          f'  → 連続比 {np.median(rat_g)/spacing:.2f}）')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300000)
