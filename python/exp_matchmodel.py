"""照合模型 — 「直前 K 個の並びが前に出たとき、次も同じか」を旗で送る。

高次の文脈をそのまま文脈にすると薄まって壊滅する（exp_hictx.py: 15〜18 bpp）。
**予測として使い、当たったかどうかだけを旗で送る**なら模型は数個で済む。
LZ の照合や PAQ の match model と同じ構造で、辞書は列から作るので学習済み模型は
要らない。復号側も同じ表を作れる。

ここでは桁数の三つ組を対象にする（仮数部は変わらないので除いてある）。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "300000"))
SEL = sys.argv[1:] or ["USGS NY", "AHN3 _20", "KITTI", "plane", "vegetation",
                       "AHN4 _20", "workshop", "red-rocks"]
NCTX = 24


def bl(v: int) -> int:
    z = (v << 1) if v >= 0 else ((-v << 1) - 1)
    return int(z + 1).bit_length() - 1


def base_cost(ks, n) -> float:
    """いまの形（軸ごと order-1）の桁数の符号長。"""
    f = np.ones((3 * NCTX, 65)); tot = np.full(3 * NCTX, 65.0)
    bits = 0.0; prev = (0, 0, 0)
    for t in ks:
        for c in range(3):
            ci = c * NCTX + min(prev[c], NCTX - 1)
            bits += -np.log2(f[ci, t[c]] / tot[ci])
            f[ci, t[c]] += 32.0; tot[ci] += 32.0
        prev = t
    return bits / n


def match_cost(ks, n, K, tbits=20) -> tuple[float, float]:
    """照合模型つき。戻りは (bit/点, 一致した割合)。"""
    T = 1 << tbits
    tbl = np.full(T, -1, dtype=np.int32)          # 畳んだ鍵 → 三つ組の番号
    vocab: list = []
    vid: dict = {}
    f = np.ones((3 * NCTX, 65)); tot = np.full(3 * NCTX, 65.0)
    # 旗の模型は「直前に当たったか」×「鍵が既知か」の 4 通りだけ
    fl = np.ones((4, 2))
    bits = 0.0; prev = (0, 0, 0); h = 0; last_hit = 0; nhit = 0; nseen = 0
    ring: list = []
    for t in ks:
        if t not in vid:
            vid[t] = len(vocab); vocab.append(t)
        tid = vid[t]
        # **鍵は「直前 K 個」から毎回作り直す。**転がし雑音を累積すると
        # 履歴全体の鍵になってしまい、二度と一致しない（一致率 61% → 1% になった）。
        if len(ring) >= K:
            hh = 0
            for u in ring[-K:]:
                vv = (u[0] * 31 + u[1]) * 31 + u[2]
                hh = ((hh * 2654435761) ^ (vv + 7)) & 0xFFFFFFFF
            key = hh & (T - 1)
        else:
            key = -1
        pred = tbl[key] if key >= 0 else -1
        known = 1 if pred >= 0 else 0
        fc = last_hit * 2 + known
        if known:
            nseen += 1
            ok = 1 if vocab[pred] == t else 0
            p = fl[fc, ok] / fl[fc].sum()
            bits += -np.log2(p)
            fl[fc, ok] += 1.0
            if fl[fc].sum() > 8192:
                fl[fc] *= 0.5
            last_hit = ok
            if ok:
                nhit += 1
        else:
            last_hit = 0
            ok = 0
        if not ok:
            for c in range(3):
                ci = c * NCTX + min(prev[c], NCTX - 1)
                bits += -np.log2(f[ci, t[c]] / tot[ci])
                f[ci, t[c]] += 32.0; tot[ci] += 32.0
        else:
            for c in range(3):
                ci = c * NCTX + min(prev[c], NCTX - 1)
                f[ci, t[c]] += 32.0; tot[ci] += 32.0
        if key >= 0:
            tbl[key] = tid
        ring.append(t)
        if len(ring) > K:
            ring.pop(0)
        prev = t
    return bits / n, (nhit / max(nseen, 1))


print(f"標本 {N} 点。**桁数の符号長だけ** [bit/点]。仮数部は共通。\n")
print(f"{'データ':<13}{'点':>8}{'いま':>8}{'照合K=2':>9}{'K=4':>8}{'K=8':>8}"
      f"{'K=16':>8}{'最良の減り':>11}{'一致率':>8}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    d = np.diff(xyz, axis=0, prepend=xyz[:1])
    ks = [tuple(bl(int(x)) for x in r) for r in d.tolist()]
    b0 = base_cost(ks, n)
    res = {K: match_cost(ks, n, K) for K in (2, 4, 8, 16)}
    best = min(v[0] for v in res.values())
    bk = [K for K in res if res[K][0] == best][0]
    print(f"{lab:<13}{n:>8}{b0:>8.3f}" + "".join(f"{res[K][0]:>{9 if K==2 else 8}.3f}" for K in (2, 4, 8, 16))
          + f"{100*(best-b0)/b0:>10.1f}%{100*res[bk][1]:>7.1f}%")
