"""区間ごとに符号器を選ぶ（蒸留の概念を選択原理に効かせる）。

いまは列ごとに符号器を 1 本選ぶ。だがファイルの途中で性質は変わる
（飛行線が変わる、地形が変わる、走査が切れる）。区間ごとに選び、
**その選択表だけを送る**（区間数 × log2(候補数) ビット）。
1 度目の走査で決めて表に畳むので、蒸留と同じ形である。

点ごとに選ぶのは「どれが勝ったか」を毎点送ることになって割に合わなかった
（`data/work/mix/mixpred.log`）。区間なら表は無視できる大きさになる。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "300000"))
SEG = [int(x) for x in os.environ.get("SEG", "1024,4096,16384,65536").split(",")]
SEL = sys.argv[1:] or ["USGS NY", "AHN3 _20", "AHN4 _20", "KITTI", "workshop", "red-rocks"]
NCTX = 24


def med3(a, b, c):
    return a + b + c - min(a, b, c) - max(a, b, c)


def preds(v):
    """予測子ごとの残差（点 × 予測子）。"""
    n = len(v)
    s1 = np.concatenate(([0], v[:-1])); s2 = np.concatenate(([0, 0], v[:-2]))
    s3 = np.concatenate(([0, 0, 0], v[:-3])); s4 = np.concatenate(([0]*4, v[:-4]))
    d1, d2, d3 = s1 - s2, s2 - s3, s3 - s4
    m3 = d1 + d2 + d3 - np.minimum(np.minimum(d1, d2), d3) - np.maximum(np.maximum(d1, d2), d3)
    P = np.stack([s1, s1 + d1, s1 + d1 // 2, s1 + m3, s1 + m3 // 2, s1 + (d1 + d2 + d3) // 3])
    P[:, :4] = v[:4]
    return v[None, :] - P


def bitlens(res):
    z = np.where(res >= 0, res * 2, -res * 2 - 1).astype(np.int64)
    k = np.zeros(len(z), dtype=np.int64)
    t = (z + 1).astype(np.uint64)
    while True:
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1; t[m] >>= np.uint64(1)
    return k


def run(K_all, choice):
    """**模型は通しで持つ。**予測子だけ choice[i] に従って切り替える。"""
    n = K_all.shape[1]
    f = np.ones((NCTX, 65)); tot = np.full(NCTX, 65.0)
    b = 0.0; c = 0
    for i in range(n):
        kk = int(K_all[choice[i], i])
        b += -np.log2(f[c, kk] / tot[c]) + kk
        f[c, kk] += 32.0; tot[c] += 32.0
        c = min(kk, NCTX - 1)
    return b


def pick(K_all, seg):
    """区間ごとに、その区間の桁数の合計が最小の予測子を選ぶ。"""
    n = K_all.shape[1]
    ch = np.zeros(n, dtype=np.int64)
    for a in range(0, n, seg):
        blk = K_all[:, a:a + seg]
        ch[a:a + seg] = int(blk.sum(axis=1).argmin())
    return ch


print(f"標本 {N} 点。幾何 3 軸合計 [bit/点]。選択表の費用も足してある。\n")
print(f"{'データ':<13}{'点':>8}{'列ごと1本':>11}" +
      "".join(f"{('区間'+str(s)):>10}" for s in SEG) + f"{'最良の減り':>11}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    glob = 0.0
    segs = {s: 0.0 for s in SEG}
    for c in range(3):
        R = preds(xyz[:, c])
        K = R.shape[0]
        KA = np.stack([bitlens(R[j]) for j in range(K)])
        glob += min(run(KA, np.full(n, j, dtype=np.int64)) for j in range(K))
        for s in SEG:
            ch = pick(KA, s)
            segs[s] += run(KA, ch) + np.ceil(n / s) * np.ceil(np.log2(K))
    base = glob / n
    row = [segs[s] / n for s in SEG]
    best = min(row)
    print(f"{lab:<13}{n:>8}{base:>11.3f}" + "".join(f"{v:>10.3f}" for v in row)
          + f"{100*(best-base)/base:>10.1f}%")
