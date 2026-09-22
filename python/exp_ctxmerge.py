"""文脈を束ねる（word2vec の概念）＋ 束ね方を要約して送る（蒸留の概念）。

ここまでの測定で繰り返し出たのは「模型が情報ではなく**薄まり**で律速している」
ことである。文脈を増やすと必ず悪くなり、粗くすると良くなる場面があった。

word2vec が「似た文脈に現れる語を近くに置く」なら、算法としての等価物は
**統計の似た文脈を束ねて統計を共有する**ことである。
蒸留が「大きな模型の知識を小さな模型に移す」なら、等価物は
**1 度目の走査で最適な束ね方を求め、その対応表だけを送る**ことである。

対応表は文脈の数 × log2(群の数) ビットしか要らない（24 文脈・8 群なら 9 byte）。
**学習済み模型は要らず、復号側は対応表を読むだけでよい。**
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "300000"))
SEL = sys.argv[1:] or ["vegetation", "plane", "extra", "USGS NY", "AHN3 _20", "KITTI"]
NCTX = 24
KMAX = 65


def bl(v: int) -> int:
    z = (v << 1) if v >= 0 else ((-v << 1) - 1)
    return int(z + 1).bit_length() - 1


def adaptive(sym, ctx, ngrp) -> float:
    f = np.ones((ngrp, KMAX)); tot = np.full(ngrp, float(KMAX))
    b = 0.0
    for s, c in zip(sym, ctx):
        b += -np.log2(f[c, s] / tot[c])
        f[c, s] += 32.0; tot[c] += 32.0
    return b


def merge_map(sym, ctx, nctx, G):
    """統計の似た文脈を貪欲に G 群へ束ねる（1 度目の走査でだけ使う）。"""
    cnt = np.ones((nctx, KMAX))
    for s, c in zip(sym, ctx):
        cnt[c, s] += 1.0
    p = cnt / cnt.sum(axis=1, keepdims=True)
    n_i = cnt.sum(axis=1)
    groups = [[i] for i in range(nctx)]
    prof = [p[i].copy() for i in range(nctx)]
    wt = [n_i[i] for i in range(nctx)]

    def cost_of(a, b):
        """2 つを束ねたときに増える符号長（重み付き KL）。"""
        m = (wt[a] * prof[a] + wt[b] * prof[b]) / (wt[a] + wt[b])
        return (wt[a] * np.sum(prof[a] * np.log2(prof[a] / m)) +
                wt[b] * np.sum(prof[b] * np.log2(prof[b] / m)))

    while len(groups) > G:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                c = cost_of(i, j)
                if best is None or c < best[0]:
                    best = (c, i, j)
        _, i, j = best
        w = wt[i] + wt[j]
        prof[i] = (wt[i] * prof[i] + wt[j] * prof[j]) / w
        wt[i] = w
        groups[i] = groups[i] + groups[j]
        del groups[j]; del prof[j]; del wt[j]
    m = np.zeros(nctx, dtype=np.int64)
    for g, mem in enumerate(groups):
        for i in mem:
            m[i] = g
    return m


print(f"標本 {N} 点。**桁数の符号長だけ** [bit/点]。対応表の費用も足してある。\n")
print(f"{'データ':<13}{'点':>8}{'いま24':>9}{'束ね8':>9}{'束ね4':>9}{'束ね2':>9}{'最良の減り':>11}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    d = np.diff(xyz, axis=0, prepend=xyz[:1])
    K = np.array([[bl(int(x)) for x in r] for r in d.tolist()])
    sym = []; ctx = []
    prev = [0, 0, 0]
    for i in range(len(K)):
        for c in range(3):
            sym.append(int(K[i, c])); ctx.append(c * NCTX + min(prev[c], NCTX - 1))
        prev = [int(K[i, 0]), int(K[i, 1]), int(K[i, 2])]
    sym = np.array(sym); ctx = np.array(ctx)
    base = adaptive(sym, ctx, 3 * NCTX) / n
    out = [base]
    for G in (8, 4, 2):
        mp = merge_map(sym, ctx, 3 * NCTX, 3 * G)
        side = 3 * NCTX * max(1, int(np.ceil(np.log2(3 * G))))
        out.append((adaptive(sym, mp[ctx], 3 * G) + side) / n)
    best = min(out)
    print(f"{lab:<13}{n:>8}" + "".join(f"{v:>9.3f}" for v in out)
          + f"{100*(best-base)/base:>10.1f}%")
