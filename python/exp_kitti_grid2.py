"""レーザ 1 本ぶんの z が一様な格子に乗っているかを、周波数で測る。

HDL-64E の順演算子は、レーザ l について
    z = (r + 距離の下駄) sin(仰角) + 原点の縦ずれ
である。r が 2mm の整数倍なら、**z はリングごとに刻み 2mm·|sin(仰角)| の
一様な格子に乗る**。下駄と縦ずれは格子の位相にしか効かないので、
刻みだけを探せばよい。

格子らしさは第 1 フーリエ係数の大きさ |mean exp(2πi z/s)| で測る。
位相に依らず、無相関なら 0、完全に格子上なら 1 に近づく。
"""
import sys, os
import numpy as np


def gridness(v, s):
    return float(np.abs(np.mean(np.exp(2j * np.pi * v / s))))


def best_step(v, s_lo, s_hi, coarse=4000, fine=400):
    """刻みを粗く掃いてから細かく詰める。"""
    ss = np.linspace(s_lo, s_hi, coarse)
    ph = 2j * np.pi * v[:, None] / ss[None, :]
    g = np.abs(np.mean(np.exp(ph), axis=0))
    i = int(np.argmax(g))
    lo = ss[max(i - 1, 0)]; hi = ss[min(i + 1, coarse - 1)]
    ss2 = np.linspace(lo, hi, fine)
    g2 = np.abs(np.mean(np.exp(2j * np.pi * v[:, None] / ss2[None, :]), axis=0))
    j = int(np.argmax(g2))
    return float(ss2[j]), float(g2[j])


def make_rings(xyz, k=64):
    r = np.linalg.norm(xyz, axis=1)
    el = np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1, 1))
    o = np.argsort(el)
    cut = np.sort(np.argsort(np.diff(el[o]))[-(k - 1):])
    lab = np.zeros(len(el), dtype=np.int64)
    lab[o] = np.searchsorted(cut, np.arange(len(el)))
    return r, el, lab


def probe(xyz, label, step=0.002, nmax=24, quiet=False):
    r, el, lab = make_rings(xyz)
    rows = []
    for l in np.unique(lab)[:nmax]:
        m = lab == l
        if m.sum() < 500:
            continue
        z = xyz[m, 2]
        e = float(np.median(el[m]))
        s_exp = abs(step * np.sin(e))
        if s_exp < 2e-5:
            continue
        s, g = best_step(z, s_exp * 0.5, s_exp * 1.6)
        rows.append((int(m.sum()), e, s_exp, s, g))
    if not rows:
        print(f'{label}: リングが取れない'); return
    gs = np.array([x[4] for x in rows])
    rel = np.array([x[3] / x[2] for x in rows])
    print(f'{label:<34} リング {len(rows)}  格子らしさ 中央 {np.median(gs):.3f} '
          f'最大 {gs.max():.3f}  刻み比 中央 {np.median(rel):.3f}')
    if not quiet:
        for n, e, se, s, g in rows[:6]:
            print(f'    仰角 {np.degrees(e):7.2f}°  n {n:6d}  '
                  f'予想刻み {se*1e6:7.1f} um  見つけた {s*1e6:7.1f} um  '
                  f'格子らしさ {g:.3f}')


def main(dirpath, nframe=1):
    files = sorted(f for f in os.listdir(dirpath) if f.endswith('.bin'))[:nframe]
    for fn in files:
        a = np.fromfile(os.path.join(dirpath, fn), dtype=np.float32).reshape(-1, 4)
        probe(a[:, :3].astype(np.float64), f'KITTI {fn}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
