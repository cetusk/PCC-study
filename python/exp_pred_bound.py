"""窓を広げる方向の上限を、符号化する前に見積もる。

幾何v4 は直近 W 点の最近傍から予測し、何点前かを送っている。W を全域に
広げれば残差はどこまで縮むのか、その代わり指す費用はどれだけ掛かるのかを、
zigzag の bit 長と後退量の 0 次エントロピーで見る。

あわせて「塊ごとに原点（最小角）を送り、塊の中は相対座標で送る」案も測る。
原点は前の塊との差分で送る。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M

N = int(os.environ.get("BENCH_N", "50000"))
FILES = [("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz"),
         ("USGS NY",     "data/raw/usgs/NY_ClintonEssex_2014.laz"),
         ("AHN4 _20",    "data/raw/ahn4/31HZ1_20.LAZ"),
         ("KITTI",       "data/raw/kitti")]


def zlen(d: np.ndarray) -> np.ndarray:
    z = np.abs(d) * 2
    return np.where(z > 0, np.log2(z + 1).astype(int) + 1, 0).sum(1)


def H(v: np.ndarray) -> float:
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def read(lab: str, path: str) -> np.ndarray:
    kind = "kitti" if "kitti" in path else "las"
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    return M.read_block(kind, path, st, n)[0].astype(np.int64)


def global_nn(x: np.ndarray):
    """これまでに符号化した全点の中の最近傍と、その後退量。"""
    n = len(x)
    d = np.empty((n - 1, 3), dtype=np.int64)
    lag = np.empty(n - 1, dtype=np.int64)
    tree, base = None, 0
    for i in range(1, n):
        if tree is None or i - base >= 4096:
            base = i
            tree = cKDTree(x[:base])
        j = int(tree.query(x[i], k=1)[1])
        if i > base:                      # 木を建てた後の点は線形に見る
            k = base + int(np.argmin(np.abs(x[base:i] - x[i]).sum(1)))
            if np.abs(x[k] - x[i]).sum() < np.abs(x[j] - x[i]).sum():
                j = k
        d[i - 1] = x[i] - x[j]
        lag[i - 1] = i - j
    return d, lag


def main():
    print(f"標本 {N} 点。bit/点。全域の最近傍を使う場合の内訳。")
    print(f"{'データ':<13}{'直前の残差':>11}{'全域の残差':>11}{'残差の得':>9}"
          f"{'後退量の H':>11}{'差引':>8}")
    xs = {}
    for lab, path in FILES:
        x = read(lab, path)
        xs[lab] = x
        d, lag = global_nn(x)
        a = float(np.mean(zlen(np.diff(x, axis=0))))
        b = float(np.mean(zlen(d)))
        h = H(lag)
        print(f"{lab:<13}{a:>11.2f}{b:>11.2f}{a-b:>9.2f}{h:>11.2f}{a-b-h:>8.2f}")

    print(f"\n塊ごとに原点を送る場合。bit/点。")
    B = (16, 64, 256, 1024)
    print(f"{'データ':<13}{'直前':>8}" + "".join(f"{'塊'+str(b):>9}" for b in B))
    for lab, _ in FILES:
        x = xs[lab]
        row = [float(np.mean(zlen(np.diff(x, axis=0))))]
        for b in B:
            m = (len(x) // b) * b
            g = x[:m].reshape(-1, b, 3)
            o = g.min(1)
            res = float(np.mean(zlen((g - o[:, None, :]).reshape(-1, 3))))
            row.append(res + float(np.sum(zlen(np.diff(o, axis=0)))) / m)
        print(f"{lab:<13}{row[0]:>8.2f}" + "".join(f"{v:>9.2f}" for v in row[1:]))


if __name__ == "__main__":
    main()
