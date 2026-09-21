"""固定の遅れ L で予測した場合の残差を、L を掃引して測る。

格子状に並んだ入力（走査線を行として並べたもの）では、L 点前が空間的に真上に
あたることがある。L は流れの先頭に 1 度だけ書けばよいので、1 点あたりの費用は
無視できる。いまの 幾何v4 は直近 W=4/16 点しか見ないので、L が大きいと届かない。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
LMAX = int(os.environ.get("EXP_LMAX", "1024"))


def zb(d: np.ndarray) -> float:
    z = np.abs(d) * 2
    return float(np.mean(np.sum(np.where(z > 0, np.log2(z + 1).astype(int) + 1, 0), axis=1)))


def med3(a, b, c):
    return a + b + c - np.maximum(np.maximum(a, b), c) - np.minimum(np.minimum(a, b), c)


def main():
    print(f"標本 {N} 点。zigzag の bit 長の平均（3 軸の和）。L は 1〜{LMAX}。")
    print(f"{'データ':<13}{'点':>8}{'med3':>8}{'L=1':>8}{'最良 L':>8}{'その値':>8}{'med3比':>8}")
    for lab, path, kind in INPUTS:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        x = M.read_block(kind, path, st, n)[0].astype(np.int64)
        d = np.diff(x, axis=0)
        z = np.zeros((1, 3), dtype=np.int64)
        m3 = zb(d - med3(np.concatenate([z, d[:-1]]), np.concatenate([z, z, d[:-2]]),
                         np.concatenate([z, z, z, d[:-3]])))
        best, bl = None, 0
        for L in range(1, min(LMAX, len(x) - 1) + 1):
            v = zb(x[L:] - x[:-L])
            if best is None or v < best:
                best, bl = v, L
        one = zb(x[1:] - x[:-1])
        print(f"{lab:<13}{len(x):>8}{m3:>8.2f}{one:>8.2f}{bl:>8}{best:>8.2f}"
              f"{100*(best/m3-1):>+7.1f}%")


if __name__ == "__main__":
    main()
