"""幾何の予測子を並べて、符号器と同じ数え方で比べる。

いまの幾何v3 系はどれも「直前の点 + 直近の差分から作った傾き」である。
リングや走査線に沿って並んだ入力では軌跡が滑らかな曲線になるので、
**2 次（曲率まで見る）の外挿**が当たる可能性がある。ここで当たりを付けてから
C++ に足す。見積りは適応模型で回す（標本内の条件付きエントロピーは使わない）。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
SEL = sys.argv[1:] or ["KITTI", "plane", "vegetation", "USGS NY", "autzen-2023", "extra"]
NCTX = 24


def med(a, b, c):
    return a + b + c - np.minimum(np.minimum(a, b), c) - np.maximum(np.maximum(a, b), c)


def preds(v: np.ndarray) -> dict[str, np.ndarray]:
    """v は 1 軸の整数列。各予測子の予測値（先頭は 0 埋め）を返す。"""
    n = len(v)
    p = {}
    s1 = np.concatenate(([0], v[:-1]))
    s2 = np.concatenate(([0, 0], v[:-2]))
    s3 = np.concatenate(([0, 0, 0], v[:-3]))
    d1 = s1 - s2
    d2 = s2 - s3
    p["直前"] = s1
    p["1次外挿"] = s1 + d1
    p["1次半"] = s1 + d1 // 2
    p["2次外挿"] = s1 + 2 * d1 - d2
    p["2次半"] = s1 + d1 + (d1 - d2) // 2
    d3 = s3 - np.concatenate(([0, 0, 0, 0], v[:-4]))
    p["中3"] = s1 + med(d1, d2, d3)
    p["中3半"] = s1 + med(d1, d2, d3) // 2
    p["平均4"] = s1 + (d1 + d2 + d3 + np.concatenate(([0]*4, v[:-4]) ) * 0) // 4
    for k in p:
        p[k][:4] = 0
    return p


def ctx_of(z: np.ndarray) -> np.ndarray:
    k = np.zeros(len(z), dtype=np.int64)
    t = (z.astype(np.uint64) + np.uint64(1))
    for _ in range(64):
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1
        t[m] >>= np.uint64(1)
    return np.minimum(k, NCTX - 1)


def bits3(res: list[np.ndarray]) -> float:
    """3 軸を符号器と同じ文脈（軸をまたぐ桁数）で数える。"""
    zz = [np.where(r >= 0, r * 2, -r * 2 - 1).astype(np.uint64) for r in res]
    kk = [ctx_of(z) for z in zz]
    n = len(res[0])
    cnt = np.ones((3 * NCTX, 65)); tot = cnt.sum(axis=1)
    bits = 0.0
    prev_kx = 0
    k0 = kk[0]; k1 = kk[1]
    cx0 = np.empty(n, dtype=np.int64)
    cx0[0] = 0; cx0[1:] = k0[:-1]
    cxs = [cx0, k0, np.minimum((k0 + k1) // 2, NCTX - 1)]
    for c in range(3):
        z = zz[c]; cx = cxs[c] + c * NCTX
        kb = ctx_of(z)
        for i in range(n):
            ci = int(cx[i]); ki = int(kb[i])
            bits += -np.log2(cnt[ci, ki] / tot[ci]) + ki
            cnt[ci, ki] += 32.0; tot[ci] += 32.0
    return bits / n


print(f"標本 {N} 点。3 軸合計 bit/点。適応模型で数えた見積り。\n")
hdr = None
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    cols = [np.asarray(xyz[:, i]).astype(np.int64) for i in range(3)]
    ps = [preds(c) for c in cols]
    names = list(ps[0].keys())
    if hdr is None:
        hdr = names
        print(f"{'データ':<13}" + "".join(f"{x:>10}" for x in names))
    row = []
    for nm in names:
        res = [cols[c] - ps[c][nm] for c in range(3)]
        row.append(bits3(res))
    b = min(row)
    print(f"{lab:<13}" + "".join(f"{v:>10.3f}" for v in row) + f"   最良 {names[row.index(b)]}")
