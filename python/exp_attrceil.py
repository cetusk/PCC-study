"""属性の列に、まだ縮む余地があるかを適応模型で測る。

**標本内の条件付きエントロピーは使わない**（losses.md に書いた罠）。
符号器と同じやり方——桁長を適応度数表で送り、下位は 1 bit ずつ適応模型——を
Python で回して、**実際に払う符号長**を出す。相手は実測の bpp である。
"""
from __future__ import annotations
import sys, math
import numpy as np
import laspy

N = int(sys.argv[2]) if len(sys.argv) > 2 else 200000


class Freq:
    """符号器の FreqModel と同じ更新則（+32、合計 32768 で半減）。"""
    __slots__ = ("f", "n")
    def __init__(self, n):
        self.n = n
        self.f = np.ones(n, dtype=np.int64)
    def cost(self, s):
        t = self.f.sum()
        return -math.log2(self.f[s] / t)
    def bump(self, s):
        self.f[s] += 32
        if self.f.sum() >= (1 << 15):
            self.f = (self.f + 1) >> 1


class Bit:
    __slots__ = ("p",)
    def __init__(self): self.p = 2048
    def cost(self, b):
        q = self.p if b == 0 else 4096 - self.p
        return -math.log2(q / 4096)
    def bump(self, b):
        if b == 0: self.p += (4096 - self.p) >> 5
        else:      self.p -= self.p >> 5


def code_len(res, ctx, nctx):
    """残差の並びを、文脈つきで符号化したときの長さ（bit）。"""
    km = [Freq(66) for _ in range(nctx)]
    sm = [[Bit() for _ in range(64)] for _ in range(nctx)]
    tot = 0.0
    for i in range(len(res)):
        v = int(res[i]) if res[i] >= 0 else 0
        z = (v << 1) ^ (v >> 63) if False else int(res[i])
        z = (z << 1) ^ (z >> 63) if z < 0 else (z << 1)
        c = int(ctx[i])
        t = z + 1
        k = t.bit_length() - 1
        rem = t - (1 << k)
        tot += km[c].cost(k); km[c].bump(k)
        for j in range(k - 1, -1, -1):
            b = (rem >> j) & 1
            tot += sm[c][j].cost(b); sm[c][j].bump(b)
    return tot


def nn_prev(xyz, i, w=16):
    """直前 w 点のうち最も近いもの（幾何v4 と同じ形）。"""
    lo = max(0, i - w)
    d = ((xyz[lo:i] - xyz[i]) ** 2).sum(1)
    return lo + int(d.argmin())


def run(path):
    las = laspy.read(path)
    n = min(N, len(las.points))
    inten = np.asarray(las.intensity)[:n].astype(np.int64)
    xyz = np.stack([np.asarray(las.X)[:n], np.asarray(las.Y)[:n],
                    np.asarray(las.Z)[:n]], 1).astype(np.float64)
    rn = np.asarray(las.return_number)[:n].astype(np.int64)
    print("%s  点 %d  intensity 異なる値 %d" % (path, n, len(np.unique(inten))))

    # 近傍（直前 16 点の最近傍）を一度だけ作る
    nb = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        nb[i] = nn_prev(xyz, i)

    def rep(name, pred, ctx, nctx):
        res = inten - pred
        b = code_len(res, ctx, nctx)
        print("  %-34s %8.3f bpp" % (name, b / n))
        return b / n

    zero = np.zeros(n, dtype=np.int64)
    pv = np.concatenate([[0], inten[:-1]])
    pn = inten[nb]; pn[0] = 0

    # 文脈: 直前の残差の桁数（0-15 に丸める）
    def kctx(res):
        a = np.abs(res)
        k = np.zeros(len(a), dtype=np.int64)
        k[1:] = np.minimum(15, np.where(a[:-1] > 0,
                                        np.log2(np.maximum(a[:-1], 1)).astype(np.int64) + 1, 0))
        return k

    rep("直前（文脈なし）", pv, zero, 1)
    rep("直前（直前の桁数）", pv, kctx(inten - pv), 16)
    rep("空間1（文脈なし）", pn, zero, 1)
    rep("空間1（直前の桁数）", pn, kctx(inten - pn), 16)
    rep("空間1（戻り番号）", pn, np.minimum(rn, 7), 8)
    k1 = kctx(inten - pn)
    rep("空間1（桁数×戻り番号）", pn, k1 * 8 + np.minimum(rn, 7), 128)
    # 直前と空間の平均
    pm = (pv + pn) // 2
    rep("直前と空間1の平均（桁数）", pm, kctx(inten - pm), 16)


for p in sys.argv[1].split(","):
    run(p)


def run_color(path):
    """色 3 列に、可逆変換＋空間予測で縮む余地があるか。"""
    las = laspy.read(path)
    n = min(N, len(las.points))
    if not hasattr(las, "red"): print("%s 色なし" % path); return
    r = np.asarray(las.red)[:n].astype(np.int64)
    g = np.asarray(las.green)[:n].astype(np.int64)
    b = np.asarray(las.blue)[:n].astype(np.int64)
    xyz = np.stack([np.asarray(las.X)[:n], np.asarray(las.Y)[:n],
                    np.asarray(las.Z)[:n]], 1).astype(np.float64)
    nb = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        nb[i] = nn_prev(xyz, i)
    # 可逆色変換（YCoCg-R）
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    zero = np.zeros(n, dtype=np.int64)
    def kctx(res):
        a = np.abs(res); k = np.zeros(len(a), dtype=np.int64)
        k[1:] = np.minimum(15, np.where(a[:-1] > 0,
                    np.log2(np.maximum(a[:-1], 1)).astype(np.int64) + 1, 0))
        return k
    tot = 0.0
    print("%s  点 %d" % (path, n))
    for name, v in (("Y", y), ("Co", co), ("Cg", cg)):
        pn = v[nb]; pn[0] = 0
        res = v - pn
        c = code_len(res, kctx(res), 16) / n
        print("  %-8s 空間1（桁数） %8.3f bpp" % (name, c))
        tot += c
    print("  合計 %8.3f bpp" % tot)


if len(sys.argv) > 3 and sys.argv[3] == "color":
    for p in sys.argv[1].split(","):
        run_color(p)
