"""桁数の文脈を高次にしたときの符号長（トークン化・埋め込みの概念の代入）。

いまの符号器は桁数の文脈に「直前 1 個の桁数」しか使っていない（order-1）。
だが桁数の三つ組の列には高次の繰り返しがある
（USGS NY は直前 8 個の並びが一致すると次も 61.4% 当たる）。

**直前 K 個の三つ組を畳んだ値を文脈にする。**辞書は列から作るので学習済み模型は
要らず、復号側も同じものを作れる。表は固定の大きさに畳む（衝突は許す）。

見積りは到達可能（適応模型、学習費用込み）。仮数部は生で数え、
**桁数の符号長だけを比べる**（そこしか変わらないため）。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "300000"))
SEL = sys.argv[1:] or ["USGS NY", "AHN3 _20", "plane", "vegetation", "KITTI"]
NCTX = 24


def bl(v: int) -> int:
    z = (v << 1) if v >= 0 else ((-v << 1) - 1)
    return int(z + 1).bit_length() - 1


def code(ks: list[tuple[int, int, int]], mode: str, tbits: int = 16) -> float:
    """桁数の列を符号化したときの総ビット。mode で文脈の作り方を変える。"""
    T = 1 << tbits
    f = np.ones((T, 65), dtype=np.float32)
    tot = np.full(T, 65.0, dtype=np.float32)
    bits = 0.0
    h2 = h4 = h8 = 0
    prev = (0, 0, 0)
    for t in ks:
        for c in range(3):
            k = t[c]
            if mode == "order1":          # いまの形: 直前の同じ軸の桁数
                ci = (c * NCTX + min(prev[c], NCTX - 1))
            elif mode == "cross":         # 軸をまたぐ（いまの幾何v3 相当）
                ci = (c * NCTX + min((prev[0] + prev[1]) // 2, NCTX - 1))
            elif mode == "h2":
                ci = (h2 * 3 + c) & (T - 1)
            elif mode == "h4":
                ci = (h4 * 3 + c) & (T - 1)
            else:
                ci = (h8 * 3 + c) & (T - 1)
            p = f[ci, k] / tot[ci]
            bits += -np.log2(p)
            f[ci, k] += 32.0; tot[ci] += 32.0
        v = (t[0] * 31 + t[1]) * 31 + t[2]
        h2 = ((h2 * 1315423911) ^ v) & 0xFFFFFFFF
        h4 = ((h4 * 2654435761) ^ (v + 7)) & 0xFFFFFFFF
        h8 = ((h8 * 40503) ^ (v + 13)) & 0xFFFFFFFF
        prev = t
    return bits


def code_mix(ks, tbits=16) -> float:
    """order1 と高次を**混ぜる**（どちらが当たるかは場所で変わる）。"""
    T = 1 << tbits
    f1 = np.ones((3 * NCTX, 65), dtype=np.float32); t1 = np.full(3 * NCTX, 65.0, dtype=np.float32)
    f2 = np.ones((T, 65), dtype=np.float32); t2 = np.full(T, 65.0, dtype=np.float32)
    w = np.array([0.5, 0.5]); bits = 0.0; h = 0; prev = (0, 0, 0)
    for t in ks:
        for c in range(3):
            k = t[c]
            a = c * NCTX + min(prev[c], NCTX - 1)
            b = (h * 3 + c) & (T - 1)
            p1 = f1[a, k] / t1[a]; p2 = f2[b, k] / t2[b]
            p = w[0] * p1 + w[1] * p2
            bits += -np.log2(max(p, 1e-12))
            po = np.array([w[0] * p1, w[1] * p2]); sm = po.sum()
            if sm > 0:
                w = 0.98 * w + 0.02 * (po / sm); w /= w.sum()
            f1[a, k] += 32.0; t1[a] += 32.0
            f2[b, k] += 32.0; t2[b] += 32.0
        v = (t[0] * 31 + t[1]) * 31 + t[2]
        h = ((h * 2654435761) ^ (v + 7)) & 0xFFFFFFFF
        prev = t
    return bits


print(f"標本 {N} 点。**桁数の符号長だけ** [bit/点]。仮数部は共通なので除いてある。\n")
print(f"{'データ':<13}{'点':>8}{'order1':>9}{'軸また':>9}{'高次2':>9}"
      f"{'高次4':>9}{'高次8':>9}{'混合':>9}{'最良の減り':>11}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    d = np.diff(xyz, axis=0, prepend=xyz[:1])
    ks = [tuple(bl(int(x)) for x in r) for r in d.tolist()]
    r = {m: code(ks, m) / n for m in ("order1", "cross", "h2", "h4", "h8")}
    r["mix"] = code_mix(ks) / n
    best = min(r.values())
    print(f"{lab:<13}{n:>8}{r['order1']:>9.3f}{r['cross']:>9.3f}{r['h2']:>9.3f}"
          f"{r['h4']:>9.3f}{r['h8']:>9.3f}{r['mix']:>9.3f}"
          f"{100*(best-r['order1'])/r['order1']:>10.1f}%")
