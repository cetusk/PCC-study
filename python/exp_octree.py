"""自前の占有率符号器（八分木）を組んで、G-PCC の八分木に届くかを測る。

**点の順序を捨てた世界の話である。**集合だけを送る。

八分木を幅優先でたどり、各節点の 8 bit の占有を 1 bit ずつ適応二値模型で送る。
文脈は G-PCC と同じ考え方で「親の面隣接 6 方向が埋まっているか」と
「この byte の何ビット目か」と「ここまでに送ったビット」。
前の段は全部確定しているので、親の隣接は復号側も引ける。

葉に達したら、その升に何点あるか（重なりの数）を送る。

見積りは到達可能な符号長（適応模型、学習費用込み）で数える。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS
from runpeak import run
from bench_pcc2 import TMC3, ENV, GFLAGS, write_ply
import tempfile

N = int(os.environ.get("BENCH_N", "200000"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra"]


class Bin:
    """適応二値模型の集まり。"""

    def __init__(self, n):
        self.p = np.full(n, 0.5)
        self.c = np.zeros((n, 2), dtype=np.float64) + 1.0

    def cost(self, i, b):
        t = self.c[i]
        p = t[b] / (t[0] + t[1])
        self.c[i, b] += 1.0
        if self.c[i].sum() > 4096:
            self.c[i] *= 0.5
        return -np.log2(p)


def octree_bits(P: np.ndarray, mode: int = 0, idcm: bool = True) -> tuple[float, int]:
    """(1 点あたりの bit, 深さ)。

    idcm=True で「節点の点が少なければ分割をやめて残りのビットを直接送る」を入れる。
    G-PCC が持っている仕掛けで、疎で深い入力では分割を続けると 1 段あたり 8 bit を
    1 点のために払うことになる（simple1_4 は深さ 30）。
    直接送るかどうかは 1 bit の旗で送る（復号側は旗を読んでから分岐する）。
    """
    n = len(P)
    q = P - P.min(axis=0)
    ext = int(q.max()) + 1
    D = max(1, int(ext - 1).bit_length())
    cur = {(0, 0, 0): np.arange(n)}
    occ_prev: set = {(0, 0, 0)}
    NC = 64 * 256 * 4
    mdl = Bin(NC)
    mul = Bin(64)
    dir_f = Bin(64)
    bits = 0.0
    N26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
           if (dx, dy, dz) != (0, 0, 0)]
    N6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for lv in range(D):
        sh = D - lv - 1
        nxt = {}
        occ_now = set()
        for key in sorted(cur):
            idx = cur[key]
            bx, by, bz = key
            if idcm and lv > 0:
                # 直接送るか。旗の文脈は「この節点の点数が 1 か」と段の深さ
                one = 1 if len(idx) <= 2 else 0
                fc = one * 32 + min(lv, 31)
                take = one == 1
                bits += dir_f.cost(fc, 1 if take else 0)
                if take:
                    # 点数（1 か 2）と、各点の残り sh+1 bit × 3 軸を生で
                    bits += dir_f.cost(63, 1 if len(idx) == 2 else 0)
                    bits += 3.0 * (sh + 1) * len(idx)
                    continue
            sub = {}
            for i in idx.tolist():
                cx = (int(q[i, 0]) >> sh) & 1
                cy = (int(q[i, 1]) >> sh) & 1
                cz = (int(q[i, 2]) >> sh) & 1
                sub.setdefault((cx << 2) | (cy << 1) | cz, []).append(i)
            byte = 0
            for s2 in sub:
                byte |= 1 << s2
            if mode == 0:
                nb = 0
                for t, (dx, dy, dz) in enumerate(N6):
                    if (bx + dx, by + dy, bz + dz) in occ_prev:
                        nb |= 1 << t
                lvb = 0
            else:
                c26 = sum(1 for (dx, dy, dz) in N26
                          if (bx + dx, by + dy, bz + dz) in occ_prev)
                nb = min(c26, 26) * 2 + (1 if len(idx) > 4 else 0)
                lvb = 0 if mode == 1 else (0 if lv < D // 3 else (1 if lv < 2 * D // 3 else 2))
            partial = 1
            for i in range(8):
                b = (byte >> i) & 1
                ci = ((nb % 64) * 256 + partial) * 4 + lvb
                bits += mdl.cost(ci % NC, b)
                partial = (partial << 1) | b
                if partial >= 256:
                    partial = 255
            for s2, lst in sub.items():
                cx, cy, cz = (s2 >> 2) & 1, (s2 >> 1) & 1, s2 & 1
                k = (bx * 2 + cx, by * 2 + cy, bz * 2 + cz)
                nxt[k] = np.asarray(lst)
                occ_now.add(k)
        cur = nxt
        occ_prev = occ_now
    for key, idx in cur.items():
        m = min(len(idx) - 1, 63)
        for i in range(m):
            bits += mul.cost(min(i, 63), 1)
        bits += mul.cost(min(m, 63), 0)
    return bits / n, D


def gpcc_bpp(xyz):
    """**同じ標本で** tmc3 を走らせる。2M の値と 200k の値は比べられない。"""
    d = Path(tempfile.mkdtemp())
    try:
        write_ply(d / "in.ply", xyz)
        run([TMC3, "--mode=0", f"--uncompressedDataPath={d/'in.ply'}",
             f"--compressedStreamPath={d/'g.bin'}"] + GFLAGS, ENV)
        return (d / "g.bin").stat().st_size * 8.0 / len(xyz)
    finally:
        for q2 in d.glob("*"):
            q2.unlink()
        d.rmdir()
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    bs = [octree_bits(xyz, m, True)[0] for m in (0, 1, 2)]
    D = octree_bits(xyz, 0, False)[1]
    b = min(bs)
    g = gpcc_bpp(xyz)
    print(f"{lab:<13}{n:>8}{D:>5}" + "".join(f"{v:>9.3f}" for v in bs)
          + f"{g:>9.3f}{100*(b-g)/g:>8.1f}%")
