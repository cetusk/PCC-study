"""KITTI の float32 xyz から、センサが測った距離の格子を取り戻せるかを試す。

順演算子は
    xyz = R(自己運動) · [ レーザ l の原点 + r · 方向(θ, l) ]
で、r は 2mm 刻みの整数のはず。ビット完全に再現するには、この順を逆に辿って
r が整数に戻ることが要る。戻らなければ、可逆のまま取得表現へ帰る道は無い。

段階を踏んで測る。
  0. 素の r が 2mm の倍数か（記録では倍数でない）
  1. レーザごとに距離の下駄 d を許すと倍数になるか
  2. レーザごとに原点のずれ（縦・横）も許すとどうか
「倍数らしさ」は r/2mm の小数部が 0 に寄っているかで測る。
無相関なら平均 0.25、完全に格子上なら 0。
"""
import sys, os, math
import numpy as np

STEP = 0.002


def frac_score(r, step=STEP):
    """格子からの外れ。無相関なら 0.25、格子上なら 0。"""
    f = np.abs(r / step - np.round(r / step))
    return float(np.mean(f))


def load_bin(p):
    a = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
    return a[:, :3].astype(np.float64), a[:, 3].astype(np.float64)


def rings(xyz, k=64):
    r = np.linalg.norm(xyz, axis=1)
    el = np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1, 1))
    # 64 本のレーザは仰角が離散。分位で切るのではなく、値の集まりで切る
    o = np.argsort(el)
    es = el[o]
    gaps = np.diff(es)
    cut = np.argsort(gaps)[-(k - 1):]
    cut = np.sort(cut)
    lab = np.zeros(len(el), dtype=np.int64)
    lab[o] = np.searchsorted(cut, np.arange(len(el)), side='left')
    return r, el, lab


def fit_offset(r, lo=-0.5, hi=0.5, n=4001):
    """距離の下駄 d を掃いて、格子に一番乗る値を探す。"""
    ds = np.linspace(lo, hi, n)
    best, bd = 1.0, 0.0
    for d in ds:
        s = frac_score(r + d)
        if s < best:
            best, bd = s, d
    return bd, best


def main(dirpath, nframe=3):
    files = sorted(f for f in os.listdir(dirpath) if f.endswith('.bin'))[:nframe]
    for fn in files:
        xyz, inten = load_bin(os.path.join(dirpath, fn))
        r, el, lab = rings(xyz)
        print(f'== {fn}  {len(xyz)} 点  リング {len(np.unique(lab))} 本 ==')
        print(f'  0. 素の r          外れ {frac_score(r):.4f}  （無相関 0.25 / 格子上 0）')

        # 1. レーザごとの距離の下駄
        sc, w = 0.0, 0
        for l in np.unique(lab):
            m = lab == l
            if m.sum() < 200:
                continue
            d, s = fit_offset(r[m])
            sc += s * m.sum(); w += m.sum()
        print(f'  1. レーザごとの下駄  外れ {sc/max(w,1):.4f}')

        # 2. 下駄に加えて、方位角に依る 1 次の伸び（自己運動の 1 次近似）
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        sc2, w2 = 0.0, 0
        for l in np.unique(lab):
            m = lab == l
            if m.sum() < 200:
                continue
            best = 1.0
            for k in np.linspace(-0.02, 0.02, 41):       # m/rad
                d, s = fit_offset(r[m] + k * az[m], -0.1, 0.1, 801)
                best = min(best, s)
            sc2 += best * m.sum(); w2 += m.sum()
        print(f'  2. ＋方位角の 1 次  外れ {sc2/max(w2,1):.4f}')
        print()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
