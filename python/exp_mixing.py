"""複数の文脈モデルを「選ぶ」のと「混ぜる」のを、実際の残差で比べる。

今の符号化器は候補を全部符号化して**最短の 1 本だけ**を採り、残りを捨てている。
情報理論では、複数モデルの予測を重み付きで混ぜた符号長は最良の単独モデル以下になる。
その差が何ビットあるかを、実装する前に測る。

符号長は静的な条件付きエントロピーで見積もる（適応符号化の細部は入れない）。
混合の重みは符号長を最小にするように解く（線形の対数なので凸で、素直に解ける）。
"""
import sys, math
import numpy as np
from scipy.optimize import minimize


def binarize(r, nbit=16):
    """残差を符号器と同じ形（ジグザグ→ビット列）に直す。"""
    z = np.where(r >= 0, 2 * r, -2 * r - 1).astype(np.int64)
    return np.clip(z, 0, (1 << nbit) - 1)


def bucket(v, edges):
    return np.searchsorted(edges, np.abs(v), side='right')


def model_bits(sym, ctx, nsym, smooth=0.5):
    """文脈ごとの分布で符号化したときのビット長／記号（静的）。"""
    nctx = int(ctx.max()) + 1
    cnt = np.zeros((nctx, nsym), dtype=np.float64)
    np.add.at(cnt, (ctx, sym), 1.0)
    cnt += smooth
    p = cnt / cnt.sum(1, keepdims=True)
    return p, float(-np.mean(np.log2(p[ctx, sym])))


def main(path, col=5, nbit=10, cap=300000):
    a = np.loadtxt(path, skiprows=1, dtype=np.int64, max_rows=cap)
    line = a[:, 0]
    r = a[:, col]
    sym = binarize(r, nbit)
    n = len(sym)
    top = sym >> max(nbit - 6, 0)                     # 上位 6 ビットを記号にする
    nsym = int(top.max()) + 1

    ed = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096])
    prev1 = np.concatenate([[0], sym[:-1]])
    prev2 = np.concatenate([[0, 0], sym[:-2]])
    off = a[:, 6]                                     # 面外
    zz = a[:, 7]                                      # z

    ctxs = {
        '直前 1 点': bucket(prev1, ed),
        '直前 2 点': bucket(prev1, ed) * (len(ed) + 1) + bucket(prev2, ed),
        '別成分（面外）': bucket(np.concatenate([[0], off[:-1]]), ed),
        '別成分（z）': bucket(np.concatenate([[0], zz[:-1]]), ed),
        '走査線内の位置': np.minimum(
            (np.arange(n) - np.concatenate([[0], np.maximum.accumulate(
                np.where(np.diff(line) != 0, np.arange(1, n), 0))])) // 32, 31),
    }

    print(f'  記号 {nsym} 種 / 点 {n}')
    Ps, bits = [], {}
    for name, c in ctxs.items():
        c = np.asarray(c, dtype=np.int64)
        c = c - c.min()
        p, b = model_bits(top, c, nsym)
        Ps.append(p[c])                               # 各点での予測分布
        bits[name] = b
        print(f'    {name:<16}{b:7.4f} bit/記号')
    best = min(bits.values())

    P = np.stack(Ps, axis=0)                          # (M, n, nsym)
    idx = (np.arange(n), top)
    Pi = P[:, idx[0], idx[1]]                         # (M, n) 実際に出た記号の確率

    M = len(Ps)

    def cost_global(w):
        w = np.exp(w); w = w / w.sum()
        return float(-np.mean(np.log2(np.maximum(w @ Pi, 1e-12))))

    res = minimize(cost_global, np.zeros(M), method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6})
    mix_g = res.fun
    wg = np.exp(res.x); wg = wg / wg.sum()

    # **重みを文脈で切り替える。**実際の混合器はこうしている。
    # 切り替えの文脈は直前の残差の大きさの階級。
    grp = bucket(prev1, ed)
    grp = grp - grp.min()
    ng = int(grp.max()) + 1
    mix_l = 0.0
    wl = np.zeros((ng, M))
    for g in range(ng):
        m = grp == g
        if m.sum() < 64:
            wl[g] = wg
            if m.sum():
                ww = wg
                mix_l += m.sum() * float(-np.mean(np.log2(np.maximum(ww @ Pi[:, m], 1e-12))))
            continue
        Pg = Pi[:, m]

        def c(w):
            w = np.exp(w); w = w / w.sum()
            return float(-np.mean(np.log2(np.maximum(w @ Pg, 1e-12))))
        r2 = minimize(c, np.zeros(M), method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        ww = np.exp(r2.x); wl[g] = ww / ww.sum()
        mix_l += m.sum() * r2.fun
    mix_l /= n

    print(f'    {"最良の単独":<16}{best:7.4f} bit/記号')
    print(f'    {"混ぜる（大域）":<16}{mix_g:7.4f} bit/記号   '
          f'（-{best-mix_g:.4f}、{100*(best-mix_g)/best:.1f}%）')
    print(f'    {"混ぜる（文脈別）":<16}{mix_l:7.4f} bit/記号   '
          f'（-{best-mix_l:.4f}、{100*(best-mix_l)/best:.1f}%）  重みの組 {ng}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
