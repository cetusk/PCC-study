"""残差列の繰り返しを辞書として使えるか（トークン化・照合模型の見積り）。

いまの符号器は**繰り返しを一切見ていない**。各点を予測子と文脈で符号化するだけで、
「この並びは前にも出た」という情報を使っていない。テキスト圧縮の BPE や
LZ の照合模型にあたるものが無い。

測るもの: 幾何の差分の三つ組 (dx,dy,dz) の列で、
  直前 K 個の並びが過去に出ていたら、その次に来た三つ組は今回も同じか。
一致率が高ければ、照合模型（「前と同じ」の旗＋外れたときだけ通常符号化）が効く。

**辞書は列そのものから作るので、学習済み模型は要らない。**復号側も同じ列を
持っているので副情報も生じない。
"""
from __future__ import annotations
import os, sys, collections
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "500000"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra", "USGS NY", "KITTI", "AHN3 _20"]


def bl(v):
    z = (v << 1) if v >= 0 else ((-v << 1) - 1)
    return int(z + 1).bit_length() - 1


def stats(xyz: np.ndarray, kind: str):
    d = np.diff(xyz, axis=0, prepend=xyz[:1])
    if kind == "raw":
        trip = [tuple(r) for r in d.tolist()]
    elif kind == "bitlen":
        trip = [tuple(bl(x) for x in r) for r in d.tolist()]
    elif kind == "sign":
        trip = [tuple((1 if x > 0 else (-1 if x < 0 else 0)) for x in r)
                for r in d.tolist()]
    else:                       # 粗い差分（下位 2 bit を落とす）
        trip = [tuple(int(x) >> 2 for x in r) for r in d.tolist()]
    n = len(trip)
    out = {}
    for K in (1, 2, 4, 8):
        tbl: dict = {}
        hit = 0; seen = 0
        for i in range(K, n):
            key = tuple(trip[i - K:i])
            prev = tbl.get(key)
            if prev is not None:
                seen += 1
                if prev == trip[i]:
                    hit += 1
            tbl[key] = trip[i]
        out[K] = (hit, seen, n - K)
    same = sum(1 for i in range(1, n) if trip[i] == trip[i - 1])
    return out, len(set(trip)), same, n


print(f"標本 {N} 点。幾何の差分の三つ組の繰り返し。\n")
print(f"{'データ':<13}{'粒度':>7}{'相異なり':>9}{'直前と同':>9}"
      f"{'K=1':>8}{'K=2':>8}{'K=4':>8}{'K=8':>8}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    for kd in ("raw", "bitlen", "sign", "q2"):
        o, nd, same, nn = stats(xyz, kd)
        row = "".join(f"{100*o[K][0]/max(o[K][1],1):>7.1f}%" for K in (1, 2, 4, 8))
        print(f"{(lab if kd=='raw' else ''):<13}{kd:>7}{nd:>9}{100*same/nn:>8.1f}%{row}")
print()
print("  K=k の列は「直前 k 個の並びが過去に出たとき、次も同じだった割合」。")
print("  高いほど照合模型が効く。0% に近ければ繰り返しは無い。")
