"""八分木の文脈を強くする — 子の段の既に決まった隣接を使う。

前の版は「親の隣接」しか見ていなかった。G-PCC が強いのは、符号化している
その升自身の隣接（同じ段で既に決まっているもの）を文脈に使うからである。
幅優先で走査順を固定すれば、どの隣接が既知かは両側で一致する。
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import TMC3, ENV, GFLAGS, write_ply
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "50000"))
SEL = sys.argv[1:] or ["red-rocks", "workshop", "plane", "vegetation"]
N6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
N26 = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
       if (a, b, c) != (0, 0, 0)]


class Bin:
    def __init__(self, n):
        self.c = np.ones((n, 2))

    def cost(self, i, b):
        t = self.c[i]
        p = t[b] / (t[0] + t[1])
        t[b] += 1.0
        if t[0] + t[1] > 8192:
            self.c[i] *= 0.5
        return -np.log2(p)


def run_octree(P: np.ndarray, use_child_nb: bool, idcm: bool,
               aniso: bool = False) -> float:
    """aniso=True で、軸ごとに必要な段数だけ分割する（四分木・二分木に落ちる）。

    幅が軸で大きく違う入力（red-rocks は X 67031 対 Y 3350）を立方体で分割すると、
    短い軸に無駄な段を払う。G-PCC が持っている仕掛け。
    """
    n = len(P)
    q = P - P.min(axis=0)
    Da = [max(1, int(q[:, a].max()).bit_length()) for a in range(3)]
    D = max(Da)
    if not aniso:
        Da = [D, D, D]
    cur = {(0, 0, 0): np.arange(n)}
    occ_prev = {(0, 0, 0)}
    mdl = Bin(64 * 8 * 256)
    mul = Bin(64)
    dirf = Bin(64)
    bits = 0.0
    for lv in range(D):
        act = [a for a in range(3) if lv >= D - Da[a]]      # この段で分割する軸
        sh = [Da[a] - 1 - (lv - (D - Da[a])) for a in range(3)]
        nxt = {}
        occ_now = set()
        K = len(act)
        for key in sorted(cur):
            idx = cur[key]
            bx, by, bz = key
            rem = sum(sh[a] + 1 for a in act)
            if idcm and lv > 0 and len(idx) == 1 and rem <= 18:
                bits += dirf.cost(min(lv, 31), 1)
                bits += float(rem)
                continue
            if idcm and lv > 0 and rem <= 18:
                bits += dirf.cost(min(lv, 31), 0)
            sub = {}
            for i in idx.tolist():
                s2 = 0
                for t, a in enumerate(act):
                    s2 |= ((int(q[i, a]) >> sh[a]) & 1) << t
                sub.setdefault(s2, []).append(i)
            c26 = sum(1 for (dx, dy, dz) in N26
                      if (bx + dx, by + dy, bz + dz) in occ_prev)
            base = min(c26, 26) * 2 + (1 if len(idx) > 4 else 0)
            partial = 1
            for s2 in range(1 << K):
                b = 1 if s2 in sub else 0
                ch = [bx, by, bz]
                for t, a in enumerate(act):
                    ch[a] = ch[a] * 2 + ((s2 >> t) & 1)
                for a in range(3):
                    if a not in act:
                        ch[a] = [bx, by, bz][a]
                if use_child_nb:
                    cn = sum(1 for (dx, dy, dz) in N6
                             if (ch[0] + dx, ch[1] + dy, ch[2] + dz) in occ_now)
                else:
                    cn = 0
                ci = ((base % 64) * 8 + min(cn, 7)) * 256 + partial
                bits += mdl.cost(ci % (64 * 8 * 256), b)
                partial = ((partial << 1) | b) & 255
                if b:
                    occ_now.add((ch[0], ch[1], ch[2]))
            for s2, lst in sub.items():
                ch = [bx, by, bz]
                for t, a in enumerate(act):
                    ch[a] = ch[a] * 2 + ((s2 >> t) & 1)
                nxt[(ch[0], ch[1], ch[2])] = np.asarray(lst)
        cur = nxt
        occ_prev = occ_now
    for key, idx in cur.items():
        m = min(len(idx) - 1, 63)
        for i in range(m):
            bits += mul.cost(min(i, 63), 1)
        bits += mul.cost(min(m, 63), 0)
    return bits / n


def gpcc(xyz):
    d = Path(tempfile.mkdtemp())
    try:
        write_ply(d / "in.ply", xyz)
        run([TMC3, "--mode=0", f"--uncompressedDataPath={d/'in.ply'}",
             f"--compressedStreamPath={d/'g.bin'}"] + GFLAGS, ENV)
        return (d / "g.bin").stat().st_size * 8.0 / len(xyz)
    finally:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()


print(f"標本 {N} 点。集合だけ [bit/点]。\n")
print(f"{'データ':<13}{'点':>7}{'立方体':>9}{'軸ごと':>10}{'+直接':>9}{'G-PCC':>9}{'最良差':>9}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    a = run_octree(xyz, True, False, False)
    b = run_octree(xyz, True, False, True)
    c = run_octree(xyz, True, True, True)
    g = gpcc(xyz)
    best = min(a, b, c)
    print(f"{lab:<13}{n:>7}{a:>9.3f}{b:>10.3f}{c:>9.3f}{g:>9.3f}{100*(best-g)/g:>8.1f}%")
