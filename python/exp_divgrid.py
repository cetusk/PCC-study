"""粗い格子で割ると符号長が縮むかを、実際の符号化器で測る。

見積り（log2(g) ビット浮く）は上限であって利得ではない。値の種類が少なければ
符号化器は既に取っている。**割った列を実際に符号化して比べる。**
"""
import os, re, subprocess, sys, tempfile
from math import gcd
import numpy as np
import laspy

PCC = 'cpp/build/pccnorm'


def col_bpp(path, extra=()):
    out = subprocess.run([PCC, 'pack', path, path + '.pcc2', *extra],
                         capture_output=True, text=True).stdout
    d = {}
    for ln in out.split('\n'):
        m = re.match(r'\s{2}(\S+)\s+\S+\s+([\d.]+) bpp', ln)
        if m:
            d[m.group(1)] = float(m.group(2))
    tot = None
    for ln in out.split('\n'):
        m = re.match(r'PCC2\s+\S+\s+\S+\s+([\d.]+) bpp', ln)
        if m:
            tot = float(m.group(1))
    ok = '全列一致 = true' in out
    return d, tot, ok


def grid_of(v):
    u = np.unique(v.astype(np.int64))
    if len(u) < 3:
        return 1
    d = np.diff(u); d = d[d > 0]
    g = 0
    for x in d:
        g = gcd(g, int(x))
        if g == 1:
            return 1
    return g


def main(path, n=300000, cols=None):
    f = laspy.read(path)
    tot = len(f.points)
    m = min(n, tot)
    st = max(0, tot // 2 - m // 2)
    tmp = tempfile.mkdtemp()
    base = os.path.join(tmp, 'base.las')

    h = laspy.LasHeader(version=f.header.version, point_format=f.header.point_format)
    h.scales, h.offsets = f.header.scales, f.header.offsets
    d0 = laspy.LasData(h); d0.points = f.points[st:st + m].copy()
    d0.write(base)
    b0, t0, ok0 = col_bpp(base)
    print(f'{os.path.basename(path)}  {m} 点   もとの合計 {t0:.3f} bpp（検証 {ok0}）')

    names = cols or [c for c in ('X', 'Y', 'Z', 'intensity', 'red', 'green', 'blue',
                                 'nir', 'gps_time', 'user_data', 'scan_angle',
                                 'point_source_id')
                     if hasattr(f, c)]
    # 格子を持つ列を先に洗い出す
    grids = {}
    for c in names:
        v = np.asarray(getattr(d0, c))
        vi = v.view(np.int64) if v.dtype.kind == 'f' else v.astype(np.int64)
        g = grid_of(vi)
        if g > 1:
            grids[c] = (g, v.dtype)
    if not grids:
        print('  粗い格子に乗っている列は無い'); return
    print('  格子: ' + ' '.join(f'{c}={g}' for c, (g, _) in grids.items()))

    # **全部まとめて割る。**1 列だけ割ると、列どうしの関係（参照列予測）が壊れる。
    # 座標は scale を掛け合わせて相殺するので、世界座標は 1 ミリも動かない。
    h2 = laspy.LasHeader(version=f.header.version, point_format=f.header.point_format)
    h2.scales = np.array([h.scales[i] * grids.get('XYZ'[i], (1, None))[0]
                          for i in range(3)])
    h2.offsets = h.offsets
    d1 = laspy.LasData(h2); d1.points = f.points[st:st + m].copy()
    for c, (g, dt) in grids.items():
        v = np.asarray(getattr(d1, c))
        vi = v.view(np.int64) if v.dtype.kind == 'f' else v.astype(np.int64)
        q = vi // g
        setattr(d1, c, q.astype(np.int64).view(np.float64) if dt.kind == 'f'
                else q.astype(dt))
    p1 = os.path.join(tmp, 'div.las'); d1.write(p1)
    # 世界座標が変わっていないことを確かめる（変わっていれば別のものを測っている）
    chk = laspy.read(p1)
    for i, c in enumerate('XYZ'):
        a = np.asarray(getattr(d0, c)).astype(np.float64) * h.scales[i] + h.offsets[i]
        b = np.asarray(getattr(chk, c)).astype(np.float64) * h2.scales[i] + h2.offsets[i]
        if not np.allclose(a, b, rtol=0, atol=h.scales[i] * 0.51):
            print(f'    ※ {c} の世界座標が動いた。この行は無効'); return
    b1, t1, ok1 = col_bpp(p1)
    for c in grids:
        a, bn = b0.get(c), b1.get(c)
        if a is None or bn is None:
            print(f'    {c:<16} 列が流れに出ない（参照列などに畳まれた）'); continue
        print(f'    {c:<16} {a:7.3f} → {bn:7.3f} bpp （{100*(bn-a)/max(a,1e-9):+.1f}%）')
    print(f'  合計 {t0:.3f} → {t1:.3f} bpp（{t1-t0:+.3f}、{100*(t1-t0)/t0:+.2f}%）'
          f'  検証 {ok1}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300000)
