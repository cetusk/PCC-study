"""置換を「前に出した点の近傍の何番目か」で指したときの費用を測る。

復号側は集合を全部持っているので、次の点を「まだ出していない点のうち、
直前に出した点から近い順に数えて何番目か」で指せる。格納順が空間的に
連続していれば、この順位はほとんど 0〜数個に収まる。
添字の差を送るより桁違いに安い。**G-PCC に有利な側の見積りなので、
こちらの主張を検証するにはこの方式で測らなければならない。**

重複点があると対応は一意でないので、**最も安い対応を選ぶ**（同じく有利側）。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "2000000"))
KMAX = int(os.environ.get("PERM_K", "1024"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra"]


def adapt_bits(v: np.ndarray, n_far: int, n_tot: int, n: int) -> float:
    """**到達可能な**符号長。桁数を適応模型（初期 1、当たりに +32）で送り、
    下位ビットは生。経験分布をそのまま使う `code_bits` は学習の費用を数えて
    いないので到達できない。**こちらの bpp は実際の符号器の出力（学習費用込み）
    なので、置換側もこの標準で測らないと比較が不公平になる。**"""
    k = bitlen_of(v)
    lab = np.concatenate([k, np.full(n_far, 64, dtype=np.int64)])   # 64 = 近傍外
    msym = 65
    f = np.ones(msym, dtype=np.float64); tot = float(msym)
    bits = 0.0
    for kk in lab.tolist():
        bits += -np.log2(f[kk] / tot)
        f[kk] += 32.0; tot += 32.0
    bits += float(k.sum()) + n_far * np.log2(max(n, 2))
    return bits / n_tot


def bitlen_of(v: np.ndarray) -> np.ndarray:
    z = v.astype(np.int64)
    k = np.zeros(len(z), dtype=np.int64)
    t = (z + 1).astype(np.uint64)
    while True:
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1
        t[m] >>= np.uint64(1)
    return k


def code_bits(v: np.ndarray, n_far: int, n_tot: int, n: int) -> float:
    """桁数を経験分布で、下位ビットは生。近傍に入らなかったぶんは log2(n) を足す。
    **学習の費用を数えていないので到達できない**（G-PCC に有利な側の値）。"""
    z = v.astype(np.int64)
    k = np.zeros(len(z), dtype=np.int64)
    t = (z + 1).astype(np.uint64)
    while True:
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1
        t[m] >>= np.uint64(1)
    lab = np.concatenate([k, np.full(n_far, -1, dtype=np.int64)])
    _, c = np.unique(lab, return_counts=True)
    p = c / c.sum()
    h = float(-(p * np.log2(p)).sum())
    return (h * n_tot + float(k.sum()) + n_far * np.log2(max(n, 2))) / n_tot


def perm_cost(xyz: np.ndarray, kmax: int = KMAX) -> float:
    """近傍は塊に分けて引く。200 万点 × 1024 を一度に持つと 16 GB になる。"""
    n = len(xyz)
    pts = xyz.astype(np.float64)
    t = cKDTree(pts)
    k = min(kmax, n)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    same = np.ones(n, dtype=bool)
    same[1:] = ~np.all(xyz[order[1:]] == xyz[order[:-1]], axis=1)
    grp = np.empty(n, dtype=np.int64)
    grp[order] = np.cumsum(same) - 1
    ng = int(grp.max()) + 1
    left = np.bincount(grp, minlength=ng).astype(np.int64)
    ranks = np.empty(n - 1, dtype=np.int64)
    nr = 0
    far = 0
    left[grp[0]] -= 1
    B = 20000
    for s0 in range(0, n - 1, B):
        s1 = min(s0 + B, n - 1)
        _, nb = t.query(pts[s0:s1], k=k, workers=-1)
        nb = np.asarray(nb, dtype=np.int64)
        gnb = grp[nb]
        for q in range(s1 - s0):
            i = s0 + q
            row = gnb[q]
            tgt = grp[i + 1]
            alive = left[row] > 0
            # 同じ群は 1 回だけ数える（重複点はどれを指してもよい）
            first = np.ones(len(row), dtype=bool)
            first[1:] = row[1:] != row[:-1]
            ok = alive & first
            hit = np.flatnonzero(ok & (row == tgt))
            if hit.size:
                ranks[nr] = int(np.count_nonzero(ok[:hit[0]]))
                nr += 1
            else:
                far += 1
            left[tgt] -= 1
    return (code_bits(ranks[:nr], far, n - 1, n),
            adapt_bits(ranks[:nr], far, n - 1, n))


print(f"標本 {N} 点、近傍 {KMAX} 個。置換を近傍の順位で指したときの bit/点。\n")
print(f"{'データ':<13}{'点':>8}{'経験分布':>11}{'到達可能':>11}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    a, b = perm_cost(xyz)
    print(f"{lab:<13}{n:>8}{a:>11.3f}{b:>11.3f}")
