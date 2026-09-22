"""y を「同じ点の x の差」から予測できるかを測る。

1 点を符号化するとき x は y より先に送られる。走査線やリングに沿って並んだ
入力では進む向きが滑らかに変わるので、**その点の dx を知れば dy はほぼ決まる**。
いまの符号器は dx の桁数を文脈にするだけで、値としては使っていない。

予測は整数だけで作る（復号側と 1 ビットも違ってはいけない）。
向きは直近の (dx, dy) から取る。
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


def ctx_of(z):
    k = np.zeros(len(z), dtype=np.int64)
    t = z.astype(np.uint64) + np.uint64(1)
    for _ in range(64):
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1
        t[m] >>= np.uint64(1)
    return np.minimum(k, NCTX - 1)


def adaptive(res, cx, nctx):
    z = np.where(res >= 0, res * 2, -res * 2 - 1).astype(np.uint64)
    kb = ctx_of(z)
    cnt = np.ones((nctx, 65)); tot = cnt.sum(axis=1)
    bits = 0.0
    for i in range(len(res)):
        c = int(cx[i]); k = int(kb[i])
        bits += -np.log2(cnt[c, k] / tot[c]) + k
        cnt[c, k] += 32.0; tot[c] += 32.0
    return bits / len(res)


print(f"標本 {N} 点。y 軸だけ bit/点。\n")
print(f"{'データ':<13}{'いま(直前)':>12}{'向き追従':>10}{'差':>9}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    X = np.asarray(xyz[:, 0]).astype(np.int64)
    Y = np.asarray(xyz[:, 1]).astype(np.int64)
    dx = X - np.concatenate(([0], X[:-1]))
    dy = Y - np.concatenate(([0], Y[:-1]))
    pdx = np.concatenate(([0], dx[:-1]))
    pdy = np.concatenate(([0], dy[:-1]))
    # 向き追従: dy ≒ dx * (前の dy / 前の dx)。整数だけで作る。
    den = np.where(pdx == 0, 1, pdx)
    pred = np.where(pdx == 0, pdy, (dx * pdy) // den)
    r0 = dy
    r1 = dy - pred
    cx = ctx_of(np.where(dx >= 0, dx * 2, -dx * 2 - 1).astype(np.uint64))
    b0 = adaptive(r0, cx, NCTX)
    b1 = adaptive(r1, cx, NCTX)
    print(f"{lab:<13}{b0:>12.3f}{b1:>10.3f}{b1-b0:>9.3f}")
