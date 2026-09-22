"""置換を「直前 m 点の近傍の和」の中の順位で指す。

復号側は集合も既出印も持っているので、候補を「直前 1 点の近傍」に限る必要はない。
**直前 m 点それぞれの近傍を合わせ、直前 m 点への最小距離で並べる**ほうが、
格納順が飛ぶ入力では安くなる。m は数ビットで送れる。

置換の費用は**下がる方向にしか動かない**。より良い指し方が見つかれば
「G-PCC＋順序」は小さくなる。つまりこの比較でこちらが勝つと言えるのは、
**余裕が十分大きいときだけ**である。
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
MS = [int(x) for x in os.environ.get("PERM_M", "1,2,4,9,16").split(",")]
SEL = sys.argv[1:] or ["plane", "extra", "vegetation"]


def adapt_sym(lab: np.ndarray, msym: int) -> float:
    f = np.ones(msym, dtype=np.float64); tot = float(msym)
    bits = 0.0
    for x in lab.tolist():
        bits += -np.log2(f[x] / tot)
        f[x] += 32.0; tot += 32.0
    return bits


def cost(ranks: np.ndarray, far: int, n: int) -> tuple[float, float]:
    """桁数方式と、順位をそのまま記号にする方式の両方。どちらも適応模型。"""
    nt = len(ranks) + far
    z = ranks.astype(np.int64)
    k = np.zeros(len(z), dtype=np.int64)
    t = (z + 1).astype(np.uint64)
    while True:
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1
        t[m] >>= np.uint64(1)
    lab = np.concatenate([k, np.full(far, 64, dtype=np.int64)])
    a = (adapt_sym(lab, 65) + float(k.sum()) + far * np.log2(max(n, 2))) / nt
    CAP = 512
    lab2 = np.concatenate([np.minimum(z, CAP), np.full(far, CAP, dtype=np.int64)])
    over = z[z >= CAP]
    b = (adapt_sym(lab2, CAP + 1) + far * np.log2(max(n, 2))
         + sum(float(int(v).bit_length()) * 2 for v in over.tolist())) / nt
    return a, b


def perm_m(xyz: np.ndarray, m: int, kmax: int) -> tuple[float, float]:
    n = len(xyz)
    pts = xyz.astype(np.float64)
    t = cKDTree(pts)
    k = min(kmax, n)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    same = np.ones(n, dtype=bool)
    same[1:] = ~np.all(xyz[order[1:]] == xyz[order[:-1]], axis=1)
    grp = np.empty(n, dtype=np.int64); grp[order] = np.cumsum(same) - 1
    left = np.bincount(grp, minlength=int(grp.max()) + 1).astype(np.int64)
    _, nb = t.query(pts, k=k, workers=-1)
    nb = np.asarray(nb, dtype=np.int64)
    ranks = np.empty(n - 1, dtype=np.int64); nr = 0; far = 0
    left[grp[0]] -= 1
    for i in range(n - 1):
        lo = max(0, i - m + 1)
        cand = np.unique(nb[lo:i + 1].ravel())
        d = np.min(np.abs(pts[cand][:, None, :] - pts[lo:i + 1][None, :, :]).sum(axis=2),
                   axis=1)
        o = np.argsort(d, kind="stable")
        row = grp[cand[o]]
        alive = left[row] > 0
        first = np.ones(len(row), dtype=bool)
        _, idx0 = np.unique(row, return_index=True)
        first[:] = False; first[idx0] = True
        ok = alive & first
        hit = np.flatnonzero(ok & (row == grp[i + 1]))
        if hit.size:
            ranks[nr] = int(np.count_nonzero(ok[:hit[0]])); nr += 1
        else:
            far += 1
        left[grp[i + 1]] -= 1
    return cost(ranks[:nr], far, n)


print(f"標本 {N} 点、近傍 {KMAX} 個。置換の費用（到達可能な適応模型）。\n")
print(f"{'データ':<13}{'m':>4}{'桁数方式':>11}{'順位を記号':>12}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    for m in MS:
        a, b = perm_m(xyz, m, KMAX)
        print(f"{lab:<13}{m:>4}{a:>11.3f}{b:>12.3f}")
