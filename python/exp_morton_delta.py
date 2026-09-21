"""格納順のままモートン鍵を組み、1 階差分が軸別の差分より短いかを測る。

八分木順に並んだ入力（COPC など）なら鍵の差分が小さくなるはず、という見立てを
符号化の前に確かめる台。文脈なしの粗い見積りとして zigzag の bit 長の平均を使う。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M

N = 200000
FILES = [("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz"),
         ("USGS NY",     "data/raw/usgs/NY_ClintonEssex_2014.laz"),
         ("AHN4 _20",    "data/raw/ahn4/31HZ1_20.LAZ"),
         ("plane",       "data/raw/small/plane.laz")]


def morton(xyz: np.ndarray):
    x = (xyz - xyz.min(0)).astype(np.int64)
    b = max(1, int(np.ceil(np.log2(max(2, int(x.max()) + 1)))))
    k = np.zeros(len(x), dtype=object)
    for i in range(b):
        for c in range(3):
            k |= ((x[:, c].astype(object) >> i) & 1) << (3 * i + c)
    return k, b


def mean_bits(v) -> float:
    """zigzag した値の bit 長の平均。0 は 0 bit。"""
    return float(np.mean([int(abs(int(t)) * 2).bit_length() for t in v]))


def main():
    print(f"標本 {N} 点（ファイル中央）。格納順のまま。bit は zigzag の bit 長の平均。")
    print(f"{'データ':<13}{'bit深':>6}{'鍵長':>5}{'モートンΔ':>10}{'負':>7}"
          f"{'X Δ':>7}{'Y Δ':>7}{'Z Δ':>7}{'3 軸の和':>9}")
    for lab, path in FILES:
        tot = M.total_points("las", path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz = M.read_block("las", path, st, n)[0]
        k, b = morton(xyz)
        dk = np.diff(k)
        d = np.diff(xyz.astype(np.int64), axis=0)
        ax = [mean_bits(d[:, c]) for c in range(3)]
        print(f"{lab:<13}{b:>6}{3*b:>5}{mean_bits(dk):>10.2f}"
              f"{100*np.mean(np.array([int(t) for t in dk]) < 0):>6.1f}%"
              + "".join(f"{v:>7.2f}" for v in ax) + f"{sum(ax):>9.2f}")


if __name__ == "__main__":
    main()
