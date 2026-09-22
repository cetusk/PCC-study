"""z を「既に復号した近傍の点」から予測できるかを測る。

いまの幾何符号器は、どの版も**格納順で前の点**からしか予測しない。
だが 1 点を符号化するとき x と y は z より先に送られるので、**その点の x,y を
知った状態で** z を予測してよい。復号側も同じものを持っているので副情報は要らない。
地形・建物のように z が (x,y) の関数になっている入力では、直前の点より
「同じ場所の既知の点」のほうが近い。

見積りは**適応模型で回す**。標本内の条件付きエントロピーは文脈を増やすほど
下がるので、符号長の見積りに使ってはいけない（notes/05 を参照）。
ここでは符号器と同じ「桁数を適応模型 + 下位ビットは生」で数える。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra", "USGS NY", "autzen-2023", "KITTI"]
NCTX = 24


def zigzag(v: int) -> int:
    return (v << 1) ^ (v >> 63) if v >= 0 else ((-v) << 1) - 1


def adaptive_bits(vals: np.ndarray, ctx: np.ndarray, nctx: int = NCTX) -> float:
    """桁数を文脈つき適応模型で、下位ビットを生で数えたときの総ビット数 / 点。"""
    cnt = np.ones((nctx, 65), dtype=np.float64)
    tot = cnt.sum(axis=1)
    bits = 0.0
    for v, c in zip(vals.tolist(), ctx.tolist()):
        z = zigzag(int(v))
        k = int(z + 1).bit_length() - 1
        bits += -np.log2(cnt[c, k] / tot[c]) + k
        cnt[c, k] += 32.0
        tot[c] += 32.0
    return bits / len(vals)


def ctx_of(v: np.ndarray) -> np.ndarray:
    z = np.where(v >= 0, v.astype(np.int64) * 2, -v.astype(np.int64) * 2 - 1)
    k = np.zeros(len(z), dtype=np.int64)
    t = (z + 1).astype(np.uint64)
    while np.any(t > 1):
        m = t > 1
        k[m] += 1
        t[m] >>= np.uint64(1)
    return np.minimum(k, NCTX - 1)


def grid_pred_z(X, Y, Z, cell):
    """(x,y) を格子に落とし、同じ／隣の升に入っている**既出**の点の z で予測。"""
    n = len(X)
    gx = (X // cell).astype(np.int64)
    gy = (Y // cell).astype(np.int64)
    last: dict[tuple[int, int], int] = {}
    pred = np.zeros(n, dtype=np.int64)
    prev = 0
    for i in range(n):
        a, b = int(gx[i]), int(gy[i])
        best = None
        for da in (0, -1, 1):
            for db in (0, -1, 1):
                v = last.get((a + da, b + db))
                if v is not None:
                    best = v if best is None else (best + v) // 2
        pred[i] = prev if best is None else best
        prev = int(Z[i])
        last[(a, b)] = prev
    return pred


print(f"標本 {N} 点。z 軸だけを比べる（bit/点）。適応模型で数えた見積り。\n")
print(f"{'データ':<13}{'点':>7}{'直前の点':>10}{'近傍の面':>10}{'差':>9}{'升':>8}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    X, Y, Z = (np.asarray(xyz[:, i]).astype(np.int64) for i in range(3))
    # 升の大きさ = 近傍間隔のめやす（範囲 / sqrt(点数)）
    span = max(int(X.max() - X.min()), int(Y.max() - Y.min()), 1)
    cell = max(1, int(span / max(np.sqrt(n), 1)))
    dz_prev = Z - np.concatenate(([0], Z[:-1]))
    dx = X - np.concatenate(([0], X[:-1]))
    dy = Y - np.concatenate(([0], Y[:-1]))
    c = np.minimum((ctx_of(dx) + ctx_of(dy)) // 2, NCTX - 1)
    b0 = adaptive_bits(dz_prev, c)
    dz_surf = Z - grid_pred_z(X, Y, Z, cell)
    b1 = adaptive_bits(dz_surf, c)
    print(f"{lab:<13}{n:>7}{b0:>10.3f}{b1:>10.3f}{b1-b0:>9.3f}{cell:>8}")
