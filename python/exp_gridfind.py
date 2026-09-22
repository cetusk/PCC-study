"""浮動小数で書かれた座標が、実は一様な格子の整数倍かどうかを見つける。

やり方: 値を並べ替えて隣どうしの差を取り、その中の最小級の差を種にして
候補の刻みを作る。各候補について |mean exp(2πi v/s)| を見て、1 に近ければ
その刻みの格子に乗っている。位相（原点のずれ）には依らない。

格子に乗っているなら、整数に直してから符号化できる。**これは可逆である。**
"""
import sys, os
import numpy as np


def ulp_at(v):
    """その値域での float32 の刻み。これより細かい「格子」は表現そのものであって、
    データの性質ではない（同語反復になる）。"""
    m = float(np.max(np.abs(np.asarray(v, dtype=np.float64))))
    if m <= 0:
        return 0.0
    return float(np.spacing(np.float32(m)))


def gridness(v, s):
    """刻み s の格子にどれだけ乗っているか。位相（原点のずれ）には依らない。

    **段の数を要求する。**s がデータの幅より大きいと全部が同じ段に落ち、
    格子らしさが 1 になってしまう（自明に成り立つだけで何も言っていない）。
    """
    if s <= 0 or not np.isfinite(s):
        return 0.0
    if float(np.ptp(v)) < 32.0 * s:          # 32 段は無いと格子とは言えない
        return 0.0
    return float(np.abs(np.mean(np.exp(2j * np.pi * (v / s)))))


def find_step(v, nsamp=200000, seed=0):
    """隣どうしの差の「およその最大公約数」を刻みとする。

    格子に乗っているなら、並べ替えた値の差はすべて刻みの整数倍になる。
    最小級の差を種にして、各差を種で割った丸め誤差が小さいかを見る。
    合っていれば d / round(d/種) の平均で種を磨く（周波数掃きより桁違いに正確）。
    """
    v = np.asarray(v, dtype=np.float64)
    if len(v) > nsamp:
        v = np.random.default_rng(seed).choice(v, nsamp, replace=False)
    u = np.unique(v)
    if len(u) < 100:
        return None, 0.0
    d = np.diff(u)
    d = d[d > 0]
    if len(d) == 0:
        return None, 0.0
    # 種は「float32 の刻みより粗い最小の差」。表現の刻みを拾うと同語反復になる。
    u4 = 4.0 * ulp_at(v)
    cand = d[d > u4]
    s = float(np.min(cand)) if len(cand) else float(np.min(d))
    for _ in range(6):
        q = d / s
        k = np.round(q)
        m = (k >= 1) & (k <= 4096) & (np.abs(q - k) < 0.25)
        if m.sum() < 8:
            break
        s_new = float(np.sum(d[m]) / np.sum(k[m]))     # 最小二乗の比
        if not np.isfinite(s_new) or s_new <= 0:
            break
        if abs(s_new - s) <= 1e-15 * s:
            s = s_new
            break
        s = s_new
    # 最後に値そのものから磨く。刻みの相対誤差は値域の端で積もるので、
    # 差分から出した刻みのままだと端で整数からずれる。
    for _ in range(3):
        k = np.round(v / s)
        m = k != 0
        if m.sum() < 8:
            break
        s2 = float(np.sum(v[m] * k[m]) / np.sum(k[m] * k[m]))
        if not np.isfinite(s2) or s2 <= 0 or abs(s2 - s) <= 1e-16 * s:
            s = s2 if np.isfinite(s2) and s2 > 0 else s
            break
        s = s2
    # 見つけた刻みが本当の刻みの約数（倍音）のことがある。整数倍に上げて、
    # まだ格子に乗っている最大のものを採る。
    g = gridness(v, s)
    if g > 0.98:
        for m in range(2, 33):
            gm = gridness(v, s * m)
            if gm > 0.98:
                s, g = s * m, gm
    # 十進で書かれた文字列から来た値は 10^-k（や 2.5·10^-k）の格子に乗る。
    # 差分から出した種がその約数でないと整数倍に上げても届かないので、
    # 十進の候補を直に当てる。粗いほうを優先して採る。
    if g <= 0.95:
        u4 = 4.0 * ulp_at(v)
        cands = []
        for k in range(-3, 12):
            for c in (1.0, 2.5, 5.0):
                t = c * 10.0 ** (-k)
                if t > u4:
                    cands.append(t)
        for t in sorted(cands, reverse=True):
            gt = gridness(v, t)
            if gt > 0.90:          # 十進の文字列は桁数が揃わないことがあり、
                                   # 一部の値が外れても格子は格子である
                s, g = t, gt
                break
    return s, g


def report(name, arrs, labels):
    print(f'== {name} ==')
    for a, lb in zip(arrs, labels):
        s, g = find_step(a)
        if s is None:
            print(f'  {lb:<6} 値が少なすぎる'); continue
        v = np.asarray(a, dtype=np.float64)
        q = v / s
        # 許容は刻みの相対誤差が値域の端で積もるぶんを見込む
        err = np.abs(q - np.round(q))
        # 値が float32 で持たれている場合、v 自身が刻みの整数倍から
        # 相対 6e-8 だけずれている。その積もりを許容に入れる。
        ok = 100 * np.mean(err < np.maximum(1e-3, np.abs(q) * 1.2e-7))
        u = ulp_at(v)
        real = s > 4 * u                      # 表現の刻みより粗いか
        if not real:
            tag = f'  ← float32 の刻み（{u:.6g}）。データの格子ではない'
        elif g > 0.99 and ok > 99.9:
            tag = '  ← 整数化できる'
        elif g > 0.90:
            tag = f'  ← ほぼ格子（外れる点がある）'
        else:
            tag = ''
        print(f'  {lb:<6} 刻み {s:.9g}  格子らしさ {g:.4f}  '
              f'整数倍の割合 {ok:6.2f}%{tag}')
    print()


def read_ply(path, nmax=400000):
    with open(path, 'rb') as f:
        hdr, fmt, n, props = [], None, 0, []
        while True:
            ln = f.readline()
            if not ln:
                return None
            t = ln.decode('ascii', 'replace').strip()
            hdr.append(t)
            if t.startswith('format'):
                fmt = t.split()[1]
            elif t.startswith('element vertex'):
                n = int(t.split()[2])
            elif t.startswith('property') and len(props) < 32 and 'list' not in t:
                props.append(t.split()[1:])
            elif t == 'end_header':
                break
        if fmt == 'ascii':
            names = [p[1] for p in props]
            if not all(c in names for c in 'xyz'):
                return None
            rows = []
            for _ in range(min(n, nmax)):
                ln = f.readline()
                if not ln:
                    break
                rows.append([float(x) for x in ln.split()[:len(props)]])
            a = np.array(rows, dtype=np.float64)
            return [a[:, names.index(c)] for c in 'xyz']
        if fmt != 'binary_little_endian':
            return None
        m = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
             'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4'}
        try:
            dt = np.dtype([(p[1], m[p[0]]) for p in props])
        except KeyError:
            return None
        a = np.fromfile(f, dtype=dt, count=min(n, nmax))
        names = [p[1] for p in props if p[1] in ('x', 'y', 'z')]
        if len(names) < 3:
            return None
        return [a[c].astype(np.float64) for c in ('x', 'y', 'z')]


def main(paths):
    for p in paths:
        if p.endswith('.bin'):
            a = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
            report(os.path.basename(p), [a[:, 0], a[:, 1], a[:, 2], a[:, 3]],
                   ['x', 'y', 'z', '強度'])
        elif p.endswith('.ply'):
            r = read_ply(p)
            if r is None:
                print(f'== {os.path.basename(p)} == 読めない形式\n'); continue
            report(os.path.basename(p), r, ['x', 'y', 'z'])


if __name__ == '__main__':
    main(sys.argv[1:])
