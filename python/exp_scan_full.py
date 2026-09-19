"""走査モデル経由の幾何を、全点・整数演算で測る（決定版）。

results/als_scan_structure.md の追試でわかったこと:
  * 掃引は時刻の切れ目で切る（点の 99.6% が入る。副情報は不要）
  * 1 掃引には 2 台のスキャナの走査線が混ざっている。分けると面外 RMS が
    32 m から 3.4 mm になる（約 9,000 倍）
  * 分けかたは gps_time や走査角からは決まらないので、標識を送る必要がある
    （直前の標識を文脈にして 0.5 bit/点、全体では 0.08 bit/点）

ここで測るのは、分割まで含めた端から端までの 0 次エントロピーである。
"""
from __future__ import annotations
import sys
import numpy as np
import laspy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_scan_int import shear_forward, tan_fx, SH, TAN_SH   # noqa: E402
from scipy.optimize import least_squares                      # noqa: E402


def H0(v) -> float:
    v = np.asarray(v, dtype=np.int64)
    if len(v) < 2:
        return 0.0
    _, c = np.unique(v, return_counts=True)
    p = c / len(v)
    return float(-(p * np.log2(p)).sum())


def cond_H(lab: np.ndarray) -> float:
    """直前の値を文脈にした 2 値の条件付きエントロピー。"""
    if len(lab) < 2:
        return 0.0
    a, b = lab[:-1], lab[1:]
    h = 0.0
    for v in (0, 1):
        m = a == v
        if not m.any():
            continue
        q = min(max(float(b[m].mean()), 1e-9), 1 - 1e-9)
        h += m.mean() * (-(q * np.log2(q) + (1 - q) * np.log2(1 - q)))
    return h


def line_split(x: np.ndarray, y: np.ndarray, iters: int = 12) -> np.ndarray:
    """2 本の直線（鉛直な走査面）へ EM で分ける。初期値は順番の偶奇。"""
    lab = (np.arange(len(x)) % 2).astype(np.int8)
    for _ in range(iters):
        ps = []
        for g in (0, 1):
            m = lab == g
            if m.sum() < 5:
                return lab
            xm, ym = x[m].mean(), y[m].mean()
            u, _, _ = np.linalg.svd(np.vstack([x[m] - xm, y[m] - ym]), full_matrices=False)
            d = u[:, 0]
            ps.append((xm, ym, -d[1], d[0]))
        d0 = np.abs((x - ps[0][0]) * ps[0][2] + (y - ps[0][1]) * ps[0][3])
        d1 = np.abs((x - ps[1][0]) * ps[1][2] + (y - ps[1][1]) * ps[1][3])
        nl = (d1 < d0).astype(np.int8)
        if np.array_equal(nl, lab):
            break
        lab = nl
    return lab


def thinness(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5:
        return 1e18
    sv = np.linalg.svd(np.vstack([x - x.mean(), y - y.mean()]), compute_uv=False)
    return float(sv[1] / np.sqrt(len(x)))


def fit_line(Xs, Ys, Zs, gbs, sas):
    """走査線 1 本にモデルを当て、(面内残差, 面外残差, 採用可否) を返す。"""
    xf, yf = Xs.astype(float), Ys.astype(float)
    u, _, _ = np.linalg.svd(np.vstack([xf - xf.mean(), yf - yf.mean()]), full_matrices=False)
    dv = u[:, 0]
    th = np.arctan2(dv[1], dv[0])
    th -= np.pi * np.round(th / np.pi)
    t2 = int(round(np.tan(-th / 2) * (1 << SH)))
    sn = int(round(np.sin(-th) * (1 << SH)))
    s_i, off_i = shear_forward(Xs, Ys, t2, sn)
    off_r = off_i - np.int64(np.median(off_i))
    alt = np.diff(s_i, prepend=s_i[0])
    shot = (gbs - gbs[0]).astype(float)
    zc = Zs.astype(float)
    sfl = s_i.astype(float)
    ang = np.radians(sas * 0.006)
    ta = np.tan(ang)
    c = np.polyfit(shot, ang, 1) if np.ptp(shot) > 0 else np.array([0.0, float(ang.mean())])
    Sz0 = zc.mean() + np.ptp(sfl) / max(np.ptp(ta), 1e-6)
    Sz0 = min(max(Sz0, zc.mean() + 1e5), zc.mean() + 3.9e6)

    def f(q):
        return sfl - (q[0] + (q[1] - zc) * np.tan(q[2] + q[3] * shot))

    try:
        r = least_squares(f, [sfl.mean(), Sz0, c[1], c[0]], method='trf',
                          bounds=([sfl.mean() - 2e6, zc.mean() + 5e4, -np.pi / 4 + 1e-6, -1e-2],
                                  [sfl.mean() + 2e6, zc.mean() + 4e6, np.pi / 4 - 1e-6, 1e-2]),
                          max_nfev=150)
        s0, Sz, th0, om = r.x
        tv = np.clip(np.round((th0 + om * shot) / (np.pi / 4) * (1 << 31)).astype(np.int64),
                     -(1 << 31), (1 << 31) - 1)
        pred = np.int64(round(s0)) + (((np.int64(round(Sz)) - Zs) * tan_fx(tv)) >> TAN_SH)
        res = s_i - pred
        if res.std() < alt.std():
            return res, off_r, True
    except Exception:
        pass
    return alt, off_r, False


def main(path='data/raw/ahn4/31HZ1_20.LAZ', npts=600_000, split_deg=20.0):
    with laspy.open(path) as f:
        p = f.read_points(npts)
    gb = np.array(p['gps_time'], dtype=np.float64).view(np.int64)
    sid = np.array(p['point_source_id'], dtype=np.int64)
    X = np.array(p['X'], dtype=np.int64)
    Y = np.array(p['Y'], dtype=np.int64)
    Z = np.array(p['Z'], dtype=np.int64)
    sa = np.array(p['scan_angle'], dtype=np.int64)
    rn = np.array(p['return_number'], dtype=np.int64)
    n = len(gb)
    base0 = np.array([H0(np.diff(v)) for v in (X, Y, Z)])
    o = np.lexsort((np.arange(n), gb, sid))
    gb, sid, X, Y, Z, sa, rn = gb[o], sid[o], X[o], Y[o], Z[o], sa[o], rn[o]
    base = np.zeros(n, bool)
    base[0] = True
    base[1:] = (np.diff(gb) > 200) | (np.diff(sid) != 0)
    k = np.cumsum(base) - 1
    starts = np.flatnonzero(base)
    ends = np.r_[starts[1:], n]

    Rs = np.zeros(n, np.int64)
    Ro = np.zeros(n, np.int64)
    lab_all = np.zeros(n, np.int8)
    split_mask = np.zeros(n, bool)
    n_ok = n_line = n_split = 0
    for a, b in zip(starts, ends):
        if b - a < 20:
            Rs[a:b] = np.diff(X[a:b], prepend=X[a])
            Ro[a:b] = np.diff(Y[a:b], prepend=Y[a])
            continue
        xf, yf = X[a:b].astype(float), Y[a:b].astype(float)
        groups = [np.arange(a, b)]
        if np.ptp(sa[a:b]) * 0.006 >= split_deg:
            lab = line_split(xf, yf)
            r_before = thinness(xf, yf)
            g0 = np.flatnonzero(lab == 0) + a
            g1 = np.flatnonzero(lab == 1) + a
            if len(g0) >= 10 and len(g1) >= 10:
                w = (len(g0) * thinness(X[g0].astype(float), Y[g0].astype(float)) +
                     len(g1) * thinness(X[g1].astype(float), Y[g1].astype(float))) / (b - a)
                if w < r_before * 0.5:
                    groups = [g0, g1]
                    lab_all[a:b] = lab
                    split_mask[a:b] = True
                    n_split += 1
        for ix in groups:
            rs, ro, ok = fit_line(X[ix], Y[ix], Z[ix], gb[ix], sa[ix])
            Rs[ix] = rs
            Ro[ix] = ro
            n_line += 1
            n_ok += len(ix) if ok else 0

    # z は掃引内・戻り番号別の 1 次差分
    gid = k * 8 + rn
    idx = np.argsort(gid, kind='stable')
    w = Z[idx]
    dz = np.diff(w, prepend=0)
    g = gid[idx]
    st = np.ones(n, bool)
    st[1:] = g[1:] != g[:-1]
    dz[st] = 0

    # 標識の費用（分割した掃引だけ）
    hl = 0.0
    if split_mask.any():
        hl = cond_H(lab_all[split_mask]) * split_mask.mean()

    hs, ho, hz = H0(Rs), H0(Ro), H0(dz)
    print(f"点 {n}  掃引 {len(starts)} 本  走査線 {n_line} 本  分割した掃引 {n_split} 本")
    print(f"モデル採用 {n_ok/n:.3f}  分割された点 {split_mask.mean():.3f}")
    print(f"{'':16}{'bit/点':>9}{'RMS m':>10}")
    print(f"{'面内残差':16}{hs:9.3f}{Rs.std()*0.001:10.3f}")
    print(f"{'面外残差':16}{ho:9.3f}{Ro.std()*0.001:10.4f}")
    print(f"{'z':16}{hz:9.3f}{dz.std()*0.001:10.3f}")
    print(f"{'チャネル標識':14}{hl:9.3f}")
    print(f"{'合計':16}{hs+ho+hz+hl:9.3f}")
    print(f"{'現行（格納順1次差分）':12}{base0.sum():9.3f}"
          f"   X {base0[0]:.3f} / Y {base0[1]:.3f} / Z {base0[2]:.3f}")


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'data/raw/ahn4/31HZ1_20.LAZ',
         int(a[1]) if len(a) > 1 else 600_000)
