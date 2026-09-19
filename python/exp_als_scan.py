"""ALS の走査モデルを点群だけから当てはめる。

仮説: 1 掃引の点は、センサ位置を含む鉛直面上に乗り、
      天底からの角度は時刻の 1 次関数である。

これが成り立てば角度は時刻から決まるので、幾何は距離 r だけになる。
KITTI で −85% が出たのと同じ構造が ALS にもある、ということになる。

判定は残差で行う。測距精度（数 cm）に収まれば成立、桁が違えば不成立。
"""
from __future__ import annotations
import sys
import numpy as np
import laspy
from scipy.optimize import least_squares


def load(path: str, n: int, psid: int | None = None):
    with laspy.open(path) as f:
        p = f.read_points(n)
    d = dict(t=np.array(p['gps_time'], dtype=np.float64),
             sa=np.array(p['scan_angle'], dtype=np.float64) * 0.006,
             sid=np.array(p['point_source_id']),
             x=np.array(p.x, dtype=np.float64),
             y=np.array(p.y, dtype=np.float64),
             z=np.array(p.z, dtype=np.float64))
    if psid is not None:
        m = d['sid'] == psid
        d = {k: v[m] for k, v in d.items()}
    o = np.argsort(d['t'], kind='stable')
    return {k: v[o] for k, v in d.items()}


def sweep_index(t: np.ndarray, period: float) -> np.ndarray:
    return np.floor((t - t[0]) / period).astype(np.int64)


def fit_sweep(x, y, z, t, verbose=False):
    """1 掃引に走査モデルを当てはめ、残差を返す。

    手順は 2 段。まず (x,y) に直線を当てて鉛直な走査面を決め、
    面内の座標 s に落としてから s = s0 + (Sz - z) * tan(θ0 + ω(t - t0)) を解く。
    """
    xm, ym = x.mean(), y.mean()
    u, s_, _ = np.linalg.svd(np.vstack([x - xm, y - ym]), full_matrices=False)
    dirv = u[:, 0]                       # 面内の水平方向
    s = (x - xm) * dirv[0] + (y - ym) * dirv[1]
    off = -(x - xm) * dirv[1] + (y - ym) * dirv[0]     # 面からのずれ
    planarity = float(s_[1] / max(s_[0], 1e-12))
    tt = t - t.mean()

    def resid(p):
        s0, Sz, th0, om = p
        th = th0 + om * tt
        return s - (s0 + (Sz - z) * np.tan(th))

    # 初期値: センサは点の中央の上空 500 m、角度幅は ±20 度
    span = max(s.max() - s.min(), 1.0)
    p0 = [s.mean(), z.mean() + 500.0, 0.0, 0.0]
    if np.ptp(tt) > 0:
        p0[3] = (np.arctan2(span / 2, 500.0) * 2) / np.ptp(tt)
    r = least_squares(resid, p0, method='lm', max_nfev=4000)
    e = resid(r.x)
    return dict(n=len(x), planarity=planarity,
                off_rms=float(np.sqrt((off ** 2).mean())),
                res_rms=float(np.sqrt((e ** 2).mean())),
                res_p95=float(np.percentile(np.abs(e), 95)),
                s0=r.x[0], Sz=r.x[1], th0=np.degrees(r.x[2]),
                omega=np.degrees(r.x[3]), span=float(span))


def main(path, n=3_000_000, psid=80, period=4.2324e-3, nsweep=12):
    d = load(path, n, psid)
    k = sweep_index(d['t'], period)
    uk, cnt = np.unique(k, return_counts=True)
    good = uk[cnt >= 200]
    print(f"psid={psid}  {len(d['t'])} 点  掃引 {len(uk)} 本  "
          f"200点以上の掃引 {len(good)} 本  1掃引の点数 中央 {int(np.median(cnt))}")
    if len(good) == 0:
        print("十分な点数の掃引が無い。周期の推定が誤っている可能性がある。")
        return
    sel = good[np.linspace(0, len(good) - 1, min(nsweep, len(good))).astype(int)]
    print(f"{'掃引':>6} {'点数':>6} {'面の薄さ':>9} {'面外RMS':>9} "
          f"{'残差RMS':>9} {'残差P95':>9} {'高度':>8} {'角速度[度/s]':>12}")
    for kk in sel:
        m = k == kk
        r = fit_sweep(d['x'][m], d['y'][m], d['z'][m], d['t'][m])
        print(f"{kk:6d} {r['n']:6d} {r['planarity']:9.4f} {r['off_rms']:9.3f} "
              f"{r['res_rms']:9.3f} {r['res_p95']:9.3f} {r['Sz']:8.1f} {r['omega']:12.1f}")


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'data/raw/ahn4/31HZ1_20.LAZ',
         int(a[1]) if len(a) > 1 else 3_000_000,
         int(a[2]) if len(a) > 2 else 80,
         float(a[3]) if len(a) > 3 else 4.2324e-3)
