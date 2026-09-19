"""掃引を短く区切って走査モデルを局所的に当て、残差のエントロピーを測る。

走査の地上軌跡は 13 度程度の範囲でしか直線に見えない（results/als_scan_structure.md §9）。
そこで掃引をさらに cap 点ごとに区切り、区間ごとにモデルを当てる。
区間を短くするほどモデルは当たるが、副情報が増える。その釣り合いを測る。
"""
from __future__ import annotations
import sys
import numpy as np
import laspy
from scipy.optimize import least_squares

sys.path.insert(0, str(__file__).rsplit('/', 1)[0] if '/' in __file__ else '.')
from exp_scan_int import shear_forward, tan_fx, SH, TAN_SH   # noqa: E402


def H0(v) -> float:
    v = np.asarray(v, dtype=np.int64)
    if len(v) < 2:
        return 0.0
    _, c = np.unique(v, return_counts=True)
    p = c / len(v)
    return float(-(p * np.log2(p)).sum())


def group_diff(v: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """gid ごとに 1 次差分を取る（並びは保つ）。先頭は 0 にする。"""
    idx = np.argsort(gid, kind='stable')
    w = v[idx]
    d = np.diff(w, prepend=0)
    g = gid[idx]
    start = np.ones(len(w), bool)
    start[1:] = g[1:] != g[:-1]
    d[start] = 0
    out = np.empty_like(d)
    out[idx] = d
    return out


def segments(base: np.ndarray, cap: int) -> np.ndarray:
    """時刻の切れ目 base に加えて、cap 点ごとに区切った区間 id。"""
    k = np.cumsum(base) - 1
    n = len(base)
    first = np.zeros(k[-1] + 1, np.int64)
    first[k[base]] = np.flatnonzero(base)
    pos = np.arange(n, dtype=np.int64) - first[k]
    return k.astype(np.int64) * (n // max(cap, 1) + 2) + pos // cap


def fit_segment(Xs, Ys, Zs, gbs, sas):
    """1 区間に走査モデルを当て、(面内残差, 面外残差, 採用したか) を返す。"""
    n = len(Xs)
    xf = Xs.astype(float); yf = Ys.astype(float)
    u, _, _ = np.linalg.svd(np.vstack([xf - xf.mean(), yf - yf.mean()]),
                            full_matrices=False)
    dv = u[:, 0]
    th = np.arctan2(dv[1], dv[0])
    if th > np.pi / 2:
        th -= np.pi
    if th < -np.pi / 2:
        th += np.pi
    t2 = int(round(np.tan(-th / 2) * (1 << SH)))
    sn = int(round(np.sin(-th) * (1 << SH)))
    s_i, off_i = shear_forward(Xs, Ys, t2, sn)
    off_r = off_i - np.int64(np.median(off_i))
    shot = (gbs - gbs[0]).astype(float)
    zc = Zs.astype(float)
    sfl = s_i.astype(float)
    ang = np.radians(sas * 0.006)
    ta = np.tan(ang)
    c = np.polyfit(shot, ang, 1) if np.ptp(shot) > 0 else np.array([0.0, ang.mean()])
    Sz0 = zc.mean() + np.ptp(sfl) / max(np.ptp(ta), 1e-6)
    Sz0 = min(max(Sz0, zc.mean() + 1e5), zc.mean() + 3.9e6)

    def f(q):
        return sfl - (q[0] + (q[1] - zc) * np.tan(q[2] + q[3] * shot))

    try:
        r = least_squares(f, [sfl.mean(), Sz0, c[1], c[0]], method='trf',
                          bounds=([sfl.mean() - 2e6, zc.mean() + 5e4, -np.pi / 4 + 1e-6, -1e-2],
                                  [sfl.mean() + 2e6, zc.mean() + 4e6, np.pi / 4 - 1e-6, 1e-2]),
                          max_nfev=200)
        s0, Sz, th0, om = r.x
        tv = np.clip(np.round((th0 + om * shot) / (np.pi / 4) * (1 << 31)).astype(np.int64),
                     -(1 << 31), (1 << 31) - 1)
        pred = np.int64(round(s0)) + (((np.int64(round(Sz)) - Zs) * tan_fx(tv)) >> TAN_SH)
        res = s_i - pred
        # 退避: 掃引内の 1 次差分
        alt = np.diff(s_i, prepend=s_i[0])
        if np.abs(res).max() < np.abs(alt).max() * 4 and res.std() < alt.std():
            return res, off_r, True, n
    except Exception:
        pass
    return np.diff(s_i, prepend=s_i[0]), off_r, False, n


def main(path='data/raw/ahn4/31HZ1_20.LAZ', npts=2_000_000,
         caps=(100, 200, 400, 1000, 10**9)):
    with laspy.open(path) as f:
        p = f.read_points(npts)
    gb = np.array(p['gps_time'], dtype=np.float64).view(np.int64)
    sid = np.array(p['point_source_id'], dtype=np.int64)
    X = np.array(p['X'], dtype=np.int64)
    Y = np.array(p['Y'], dtype=np.int64)
    Z = np.array(p['Z'], dtype=np.int64)
    sa = np.array(p['scan_angle'], dtype=np.int64)
    rn = np.array(p['return_number'], dtype=np.int64)
    o = np.lexsort((np.arange(len(gb)), gb, sid))
    gb, sid, X, Y, Z, sa, rn = gb[o], sid[o], X[o], Y[o], Z[o], sa[o], rn[o]
    base = np.zeros(len(gb), bool)
    base[0] = True
    base[1:] = (np.diff(gb) > 200) | (np.diff(sid) != 0)
    k = np.cumsum(base) - 1
    Hz = H0(group_diff(Z, k * 8 + rn))
    print(f"点 {len(gb)}  掃引 {k[-1]+1} 本")
    print(f"z（掃引内・戻り番号別の 1 次差分） {Hz:.3f} bit")
    print(f"{'cap':>8} {'区間数':>8} {'採用率':>8} {'面内bit':>9} {'面外bit':>9} "
          f"{'面内RMS m':>10} {'合計':>8}")
    for cap in caps:
        seg = segments(base, cap)
        order = np.argsort(seg, kind='stable')
        sseg = seg[order]
        bounds = np.flatnonzero(np.r_[True, sseg[1:] != sseg[:-1], True])
        Rs = np.zeros(len(gb), np.int64)
        Ro = np.zeros(len(gb), np.int64)
        nok = 0
        nseg = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            ix = order[a:b]
            if len(ix) < 20:
                Rs[ix] = np.diff(X[ix], prepend=X[ix][0])
                Ro[ix] = np.diff(Y[ix], prepend=Y[ix][0])
                nseg += 1
                continue
            rs, ro, ok, n = fit_segment(X[ix], Y[ix], Z[ix], gb[ix], sa[ix])
            Rs[ix] = rs
            Ro[ix] = ro
            nok += n if ok else 0
            nseg += 1
        hs, ho = H0(Rs), H0(Ro)
        print(f"{cap:>8} {nseg:>8} {nok/len(gb):>8.3f} {hs:>9.3f} {ho:>9.3f} "
              f"{Rs.std()*0.001:>10.3f} {hs+ho+Hz:>8.3f}")


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'data/raw/ahn4/31HZ1_20.LAZ',
         int(a[1]) if len(a) > 1 else 2_000_000)
