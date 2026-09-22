"""LAS の整数列が、さらに粗い格子に乗っていないかを測る。

LAS は座標を int32 + scale で持つ。しかし宣言された scale が実際の分解能とは
限らない。値がすべて g の倍数なら、g で割ってから符号化すれば log2(g) ビット
ただで浮く。属性列（intensity、gps_time、色など）も同じである。

**これは可逆である。**g は差分の最大公約数として全点から求め、封筒に載せる。
"""
import sys
from math import gcd
from functools import reduce
import numpy as np
import laspy

COLS = ['X', 'Y', 'Z', 'intensity', 'gps_time', 'scan_angle', 'point_source_id',
        'user_data', 'classification', 'red', 'green', 'blue', 'nir']


def grid_of(v):
    """値の差の最大公約数。1 なら粗い格子には乗っていない。"""
    u = np.unique(v)
    if len(u) < 3:
        return 0
    d = np.diff(u)
    d = d[d > 0]
    if len(d) == 0:
        return 0
    # 全部の最大公約数。大きい順に効くので、間引かず全部たたむ
    g = 0
    for x in d.astype(np.int64):
        g = gcd(g, int(x))
        if g == 1:
            return 1
    return g


def main(path, n=2000000):
    f = laspy.read(path)
    tot = len(f.points)
    m = min(n, tot)
    st = max(0, tot // 2 - m // 2)
    rows = []
    for c in COLS:
        try:
            a = getattr(f, c)
        except Exception:
            continue
        v = np.asarray(a[st:st + m])
        if v.dtype.kind == 'f':
            v = v.view(np.int64)          # 実数列はビット列を整数として見る
        v = v.astype(np.int64)
        g = grid_of(v)
        if g <= 1:
            continue
        rows.append((c, g, float(np.log2(g)), len(np.unique(v))))
    print(f'{path.split("/")[-1]}  {m} 点')
    if not rows:
        print('  粗い格子に乗っている列は無い')
        return 0.0
    tot_bits = 0.0
    for c, g, b, nu in rows:
        print(f'    {c:<18} 格子 {g:<12} {b:5.2f} bit/点 浮く   異なる値 {nu}')
        tot_bits += b
    print(f'  合計 {tot_bits:.2f} bit/点')
    return tot_bits


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000000)
