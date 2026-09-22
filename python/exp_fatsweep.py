"""太い掃引の中身が何なのかを測る。

記録は「2 本に分けても割に合わない」ことまでは確かめたが、
3 本以上なのか、そもそも直線に乗っていないのかは測っていない。

掃引ごとに主軸を取り、主軸に直交する向きのずれ（面外）の分布を見る。
  * 2 山なら 2 本の走査線が混ざっている
  * k 山なら k 本
  * 単峰で幅が広いなら、そもそも直線ではない（曲がっている）
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
    bounds = np.concatenate([[0], cuts, [m]])
    return X[o], Y[o], bounds


def offaxis(x, y):
    dx, dy = x - x.mean(), y - y.mean()
    w, v = np.linalg.eigh(np.cov(np.vstack([dx, dy])))
    a = v[:, np.argmax(w)]
    return -dx * a[1] + dy * a[0]


def modes(off, nbin=41):
    """面外の分布の山の数。山＝両隣より高く、全体の 10% 以上の高さがある点。"""
    h, e = np.histogram(off, bins=nbin)
    if h.max() == 0:
        return 0
    hh = h / h.max()
    k = 0
    for i in range(1, len(hh) - 1):
        if hh[i] >= hh[i - 1] and hh[i] > hh[i + 1] and hh[i] > 0.10:
            k += 1
    return max(k, 1)


def main(path, n=300000, minn=60):
    X, Y, b = sweeps(path, n)
    cnt = {}
    widths, curves = [], []
    tot = 0
    for i in range(len(b) - 1):
        s, e = b[i], b[i + 1]
        if e - s < minn:
            continue
        x, y = X[s:e], Y[s:e]
        off = offaxis(x, y)
        k = modes(off)
        cnt[k] = cnt.get(k, 0) + (e - s)
        widths.append(np.percentile(np.abs(off - np.median(off)), 90))
        # 曲がりの大きさ: 主軸に沿った位置の 2 次式で面外をどれだけ説明できるか
        t = np.linspace(-1, 1, len(off))
        V = np.vander(t, 3)
        w, *_ = np.linalg.lstsq(V, off, rcond=None)
        res = off - V @ w
        curves.append(np.std(res) / max(np.std(off), 1e-9))
        tot += e - s
    if not widths:
        print('  掃引が短すぎる'); return
    print(f'  掃引 {len(widths)} 本 / 点 {tot}')
    print(f'  面外の広がり（P90）中央 {np.median(widths):.1f} 格子単位')
    print(f'  2 次式を当てた後の残り / 元 の中央 {np.median(curves):.3f}'
          f'  （1 に近い＝曲がりでは説明できない）')
    print('  山の数ごとの点の割合: ' +
          '  '.join(f'{k} 山 {100*v/tot:.1f}%' for k, v in sorted(cnt.items())))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300000)
