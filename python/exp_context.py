"""残差の bit 長を、どの文脈で条件づけると短くなるかを符号化前に測る。

いまの 幾何v3 は「同じ点の前の軸の bit 長（X は直前の点の X の bit 長）」を
文脈にしている。文脈を増やすと 1 文脈あたりの標本が減るので、条件付き
エントロピーが下がっても実際には縮まないことがある。ここでは下がり幅の
上限だけを見る。

符号長 ≒ H(bit 長 | 文脈) + E[bit 長 − 1]（MSB より下は素通しのため）。
右の項は文脈に依らないので、H(k | 文脈) だけを比べればよい。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M

N = int(os.environ.get("BENCH_N", "200000"))
NCTX = 24
FILES = [("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz"),
         ("USGS NY",     "data/raw/usgs/NY_ClintonEssex_2014.laz"),
         ("workshop",    "data/raw/extrabytes/workshop_TM_551_101.laz"),
         ("KITTI",       "data/raw/kitti"),
         ("TLS p1",      "data/raw/tls/lecturehall/lecturehall1.pose1.object1.label.csv")]


def klen(z: np.ndarray) -> np.ndarray:
    k = np.where(z > 0, np.log2(np.maximum(z, 1)).astype(int) + 1, 0)
    return np.minimum(k, NCTX - 1)


def cond_H(k: np.ndarray, ctx: np.ndarray) -> float:
    """H(k | ctx) を標本から。単位は bit/値。文脈が多いほど下振れする。"""
    m = ctx.max() + 1
    key = ctx.astype(np.int64) * NCTX + k
    cnt = np.bincount(key, minlength=m * NCTX).reshape(m, NCTX).astype(float)
    tot = cnt.sum(1, keepdims=True)
    p = np.divide(cnt, np.where(tot > 0, tot, 1))
    h = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1)
    return float((tot[:, 0] * h).sum() / tot.sum())


def adaptive_bits(k: np.ndarray, ctx: np.ndarray) -> float:
    """逐次推定（KT）での符号長。実際の適応符号器と同じで、文脈を増やしすぎると
    必ず損をする。単位は bit/値。"""
    m = int(ctx.max()) + 1
    cnt = np.full((m, NCTX), 0.5)
    tot = np.full(m, 0.5 * NCTX)
    bits = 0.0
    kk = k.astype(np.int64)
    cc = ctx.astype(np.int64)
    for i in range(len(kk)):
        c = cc[i]
        v = kk[i]
        bits -= np.log2(cnt[c, v] / tot[c])
        cnt[c, v] += 1.0
        tot[c] += 1.0
    return bits / len(kk)


def med3(a, b, c):
    return a + b + c - np.maximum(np.maximum(a, b), c) - np.minimum(np.minimum(a, b), c)


def main():
    mode = os.environ.get("EXP_CTX", "adaptive")
    FN = adaptive_bits if mode == "adaptive" else cond_H
    globals()["FN"] = FN
    lab0 = "逐次推定の符号長" if mode == "adaptive" else "H(bit 長 | 文脈)"
    print(f"標本 {N} 点。{lab0} の 3 軸の和（bit/点）。予測子は med3。"
          f"同軸の bit 長は 8 段に丸めて文脈に足す。")
    heads = ["文脈なし", "前の軸(いま)", "+同軸1つ前", "+同軸2つ前", "両方"]
    print(f"{'データ':<13}" + "".join(f"{h:>13}" for h in heads) + f"{'いま比':>9}")
    for lab, path in FILES:
        kind = ("kitti" if "kitti" in path else ("tls" if path.endswith(".csv") else "las"))
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        x = M.read_block(kind, path, st, n)[0].astype(np.int64)
        d = np.diff(x, axis=0)
        z = np.zeros((1, 3), dtype=np.int64)
        pred = med3(np.concatenate([z, d[:-1]]), np.concatenate([z, z, d[:-2]]),
                    np.concatenate([z, z, z, d[:-3]]))
        r = d - pred
        k = klen(np.abs(r) * 2)                      # (n-1, 3)
        kp = np.concatenate([np.zeros((1, 3), dtype=k.dtype), k[:-1]])   # 同じ軸の 1 つ前
        kp2 = np.concatenate([np.zeros((2, 3), dtype=k.dtype), k[:-2]])  # 同じ軸の 2 つ前
        tot_h = [0.0] * 5
        for c in range(3):
            # いまの 幾何v3 と同じ文脈: X は直前の点の X、Y は同じ点の X、Z は X と Y の平均
            cur = kp[:, 0] if c == 0 else (k[:, 0] if c == 1 else (k[:, 0] + k[:, 1]) // 2)
            q = lambda a: np.minimum(a, 15) // 2          # 8 段に丸めた bit 長
            cands = [np.zeros(len(k), dtype=np.int64),
                     cur,
                     cur * 8 + q(kp[:, c]),
                     cur * 8 + q(kp2[:, c]),
                     (cur * 8 + q(kp[:, c])) * 8 + q(kp2[:, c])]
            for i, ct in enumerate(cands):
                tot_h[i] += FN(k[:, c], ct.astype(np.int64))
        print(f"{lab:<13}" + "".join(f"{v:>13.3f}" for v in tot_h)
              + f"{100*(min(tot_h)/tot_h[1]-1):>+8.1f}%")


if __name__ == "__main__":
    main()
