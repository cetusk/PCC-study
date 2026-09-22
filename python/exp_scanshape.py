"""掃引の形が「直線」か「回転（楕円）」かを測る。

走査モデルは掃引が鉛直面内の直線であることを前提にしている。
回転式（パーマー走査など）なら軌跡は楕円になり、面外のずれは
掃引の中の位置に対して正弦波になる。どちらかを見分ける。

  直線   … 面外のずれは小さく、位置と無関係
  回転   … 面外のずれが位置の正弦波で説明できる
"""
import sys
import numpy as np
import laspy


def sweeps(path, n):
    f = laspy.read(path)
    m = min(n, len(f.x))
    gi = np.asarray(f.gps_time[:m], dtype=np.float64).view(np.int64)
    src = np.asarray(f.point_source_id[:m], dtype=np.int64)
    X = np.asarray(f.X[:m], dtype=np.float64)
    Y = np.asarray(f.Y[:m], dtype=np.float64)
    o = np.lexsort((gi, src))
    gs, ss = gi[o], src[o]
    same = ss[1:] == ss[:-1]
    d = gs[1:] - gs[:-1]
    pos = d[same & (d > 0)]
    thr = int(np.median(pos)) * 20
    cuts = np.flatnonzero((~same) | (d > thr)) + 1
    return X[o], Y[o], np.concatenate([[0], cuts, [m]])


def offaxis(x, y):
    dx, dy = x - x.mean(), y - y.mean()
    w, v = np.linalg.eigh(np.cov(np.vstack([dx, dy])))
    a = v[:, np.argmax(w)]
    return dx * a[0] + dy * a[1], -dx * a[1] + dy * a[0]


def sine_fit(off, along, shuffle=False, seed=0):
    """面外を、主軸に沿った位置の正弦波で説明できるか。残り／元 を返す。

    37 個の周期から残差最小のものを選ぶので、**当てはまらなくても 1 より下に出る**。
    shuffle=True は帰無の基準: 面外の値を並べ替えて位置との関係を壊し、
    同じ手順を掛ける。この値と比べないと 0.485 が意味を持たない。
    """
    if shuffle:
        off = np.random.default_rng(seed).permutation(off)
    L = len(off)
    t = (along - along.min()) / max(np.ptp(along), 1e-9)
    best = 1.0
    for period in np.linspace(0.2, 2.0, 37):
        w = 2 * np.pi / period
        A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones(L)])
        c, *_ = np.linalg.lstsq(A, off, rcond=None)
        r = off - A @ c
        best = min(best, np.std(r) / max(np.std(off), 1e-9))
    return best


def main(path, n=300000, minn=60):
    X, Y, b = sweeps(path, n)
    rows = []
    for i in range(len(b) - 1):
        s, e = b[i], b[i + 1]
        if e - s < minn:
            continue
        along, off = offaxis(X[s:e], Y[s:e])
        rows.append((e - s, np.std(off), sine_fit(off, along),
                     np.std(off) / max(np.std(along), 1e-9),
                     sine_fit(off, along, shuffle=True, seed=i)))
        if len(rows) >= 400:
            break
    if not rows:
        print('  掃引が短すぎる'); return
    a = np.array(rows)
    print(f'  掃引 {len(rows)} 本（先頭から）  点 中央 {int(np.median(a[:,0]))}')
    print(f'  面外の標準偏差 中央 {np.median(a[:,1]):8.1f} 格子単位')
    print(f'  面外 / 主軸方向 の比 中央 {np.median(a[:,3]):.4f}')
    print(f'  正弦波を当てた後の 残り／元 中央 {np.median(a[:,2]):.3f}')
    print(f'  同じ手順を並べ替えた面外に掛けたとき（帰無）  中央 {np.median(a[:,4]):.3f}')
    print(f'  → 差 {np.median(a[:,4])-np.median(a[:,2]):+.3f}'
          f'（帰無より下がっていなければ、位置との関係は見えていない）')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300000)
