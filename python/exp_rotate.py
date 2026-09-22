"""座標系を回してから符号化する（「予測しやすい表現を見つける」の算法版）。

差分ベクトルの主成分を測ると、第 1 成分が 90〜99.9% を占め、しかも主軸は座標軸から
3〜14 度ずれている。軸に揃えれば、ずれているぶんが他の軸に漏れなくなる。

回転は**整数で完全に可逆**でなければならない。2 次元の回転は 3 回のせん断
（lifting）に分解でき、各せん断は `x += round(a*y)` の形で整数のまま可逆である。
角度は 1 度目の走査で求めて**副情報として送る**（蒸留の形）。

ここでは効きの上限を見るため、まず理想の回転（浮動小数）で測る。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
SEL = sys.argv[1:] or ["red-rocks", "AHN4 _20", "TLS p1", "AHN3 _20", "USGS NY",
                       "KITTI", "workshop", "plane"]
NCTX = 24


def cost(col: np.ndarray) -> float:
    d = col - np.concatenate(([0], col[:-1]))
    z = np.where(d >= 0, d * 2, -d * 2 - 1).astype(np.int64)
    k = np.zeros(len(z), dtype=np.int64)
    t = (z + 1).astype(np.uint64)
    while True:
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1; t[m] >>= np.uint64(1)
    f = np.ones((NCTX, 65)); tot = np.full(NCTX, 65.0)
    b = 0.0; c = 0
    for kk in k.tolist():
        b += -np.log2(f[c, kk] / tot[c]) + kk
        f[c, kk] += 32.0; tot[c] += 32.0
        c = min(kk, NCTX - 1)
    return b


def shear_rot(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """整数のまま可逆な近似回転。せん断 3 回で 2 次元の回転を作る。

    ここでは上限を見るだけなので、丸めた回転そのものを使う
    （実装では lifting に分解する）。
    """
    return np.rint(P.astype(np.float64) @ R).astype(np.int64)


print(f"標本 {N} 点。幾何 3 軸合計 [bit/点]。回転の角は副情報（数 byte）。\n")
print(f"{'データ':<13}{'点':>8}{'そのまま':>10}{'主成分で回す':>13}{'差':>9}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    P = np.asarray(xyz).astype(np.int64)
    a = sum(cost(P[:, c]) for c in range(3)) / n
    d = np.diff(P.astype(np.float64), axis=0)
    C = np.cov(d.T)
    w, V = np.linalg.eigh(C)
    V = V[:, ::-1]
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    Q = shear_rot(P, V)
    b = sum(cost(Q[:, c]) for c in range(3)) / n
    print(f"{lab:<13}{n:>8}{a:>10.3f}{b:>13.3f}{100*(b-a)/a:>8.1f}%")
