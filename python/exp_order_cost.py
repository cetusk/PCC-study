"""順序そのものに要る情報量を測る。

系列の符号長 = 集合の符号長 + 順序の符号長 である。G-PCC は集合しか送らないので、
こちらが G-PCC を下回れるかは「集合を G-PCC より安く送れるか」ではなく
「集合 + 順序を G-PCC の集合より安く送れるか」で決まる。
順序のぶんを下から見積もっておかないと、届かない目標を追うことになる。

測り方: 復号側が既に集合を持っているとして、次に来る点を「現在の点からの
距離順で、まだ出ていない点の中の何番目か」で指す。その順位の経験エントロピーが
順序の符号長の見積りになる。**これは下限ではなく、この指し方での値**である。
より賢い指し方があればもっと安くなりうるが、0 にはならない。
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "50000"))
K = int(os.environ.get("ORDER_K", "128"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra", "USGS NY", "autzen-2023", "KITTI"]


def order_bits(xyz: np.ndarray, k: int = K) -> tuple[float, float]:
    n = len(xyz)
    t = cKDTree(xyz.astype(np.float64))
    _, nb = t.query(xyz.astype(np.float64), k=min(k, n))
    seen = np.zeros(n, dtype=bool)
    ranks = []
    far = 0
    for i in range(n - 1):
        seen[i] = True
        row = nb[i]
        r = 0
        hit = -1
        for j in row:
            if j == i or seen[j]:
                continue
            if j == i + 1:
                hit = r
                break
            r += 1
        if hit < 0:
            far += 1
        else:
            ranks.append(hit)
    a = np.asarray(ranks)
    if a.size:
        _, c = np.unique(a, return_counts=True)
        p = c / (len(ranks) + far)
        h = float(-(p * np.log2(p)).sum())
    else:
        h = 0.0
    # 近傍 K 個に入らなかったぶんは、少なくとも log2(n) bit 掛かるとみなす
    pf = far / (n - 1)
    if pf > 0:
        h += -pf * np.log2(pf) + pf * np.log2(max(n, 2))
    return h, pf


print(f"標本 {N} 点、近傍 {K} 個。順序を「次の点は近傍の何番目か」で指したときの費用。\n")
print(f"{'データ':<13}{'点':>7}{'順序 bit/点':>12}{'近傍外':>8}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    h, pf = order_bits(np.asarray(xyz))
    print(f"{lab:<13}{n:>7}{h:>12.3f}{100*pf:>7.1f}%")
