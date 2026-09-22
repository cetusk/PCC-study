"""符号化器が実際に払っている残差の天井を測る。

PCC_RESID_DUMP が出す残差（面内・面外・z）を、3 方向から白色化してみる。
取れた分は「まだモデル化していない構造」の下限であり、
取れなかった分が床だという主張ではない。
"""
import sys, math
import numpy as np
from scipy.spatial import cKDTree


def bits(r):
    r = np.asarray(r, dtype=np.float64)
    b = np.mean(np.abs(r - np.median(r)))
    return math.log2(2 * math.e * b) if b > 1e-9 else 0.0


def whiten_ar(r, k):
    n = len(r)
    if n <= k + 8:
        return r
    A = np.stack([r[k - 1 - j: n - 1 - j] for j in range(k)], axis=1)
    y = r[k:]
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ w


def whiten_poly(r, deg):
    n = len(r)
    if n <= deg + 8:
        return r
    t = np.linspace(-1, 1, n)
    V = np.vander(t, deg + 1)
    w, *_ = np.linalg.lstsq(V, r, rcond=None)
    return r - V @ w


def main(path, minn=60):
    a = np.loadtxt(path, skiprows=1, dtype=np.int64)
    line, ok = a[:, 0], a[:, 1]
    xy = a[:, 2:4].astype(np.float64)
    R = a[:, 5:8].astype(np.float64)              # 面内 / 面外 / z
    names = ('面内', '面外', 'z   ')

    # 走査線ごとの区間（line は符号化順に単調とは限らないので境界で切る）
    brk = np.flatnonzero(np.diff(line) != 0) + 1
    seg = np.split(np.arange(len(line)), brk)
    seg = [s for s in seg if len(s) >= minn]
    if not seg:
        print('走査線が短すぎる'); return

    npt = sum(len(s) for s in seg)
    frac_ok = float(np.mean(ok[np.concatenate(seg)]))
    print(f'  走査線 {len(seg)} 本 / 点 {npt}  モデル採用率 {frac_ok:.3f}')
    print(f'  {"成分":<7}{"現在":>9}{"多項式+過去":>13}{"隣の線":>10}{"平均の大きさ":>14}')

    prev = None
    nb_used = 0
    out = []
    for c in range(3):
        b_now = b_wh = b_nb = 0.0
        prev_seg = None
        for s in seg:
            r = R[s, c]
            w = len(s) / npt
            b_now += w * bits(r)
            rp = whiten_poly(r, 5)
            b_wh += w * min(bits(whiten_ar(rp, k)) for k in (1, 2, 3, 4))
            bb = bits(r)
            if prev_seg is not None:
                d, nb = cKDTree(xy[prev_seg]).query(xy[s], k=1)
                m = d < 4000.0
                if m.sum() > 32:
                    A = np.column_stack([R[prev_seg, c][nb[m]], np.ones(int(m.sum()))])
                    ww, *_ = np.linalg.lstsq(A, r[m], rcond=None)
                    bb = bits(r[m] - A @ ww)
                    if c == 0:
                        nb_used += int(m.sum())
            b_nb += w * bb
            prev_seg = s
        mag = 2 ** b_now / (2 * math.e)
        out.append((b_now, b_wh, b_nb))
        print(f'  {names[c]:<7}{b_now:>9.3f}{b_wh:>13.3f}{b_nb:>10.3f}{mag:>13.1f}')
    tot = sum(o[0] for o in out)
    twh = sum(min(o) for o in out)
    print(f'  {"計":<7}{tot:>9.3f}{"":>13}{"":>10}')
    print(f'  → 3 成分それぞれの最良をとると {twh:.3f} bit/点（-{tot-twh:.3f}、'
          f'{100*(tot-twh)/tot:.1f}%）  ※隣が取れた点 {100*nb_used/npt:.1f}%')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)
