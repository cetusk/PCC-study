"""副情報を使わない予測子を並べて、残差の大きさを符号化前に比べる。

いまの 幾何v3 は軸ごとに prev + med3(直近 3 つの差分) を使っている。
同じ枠（両側が復号済みの点だけから計算できる）で他の予測子がどれだけ違うかを、
zigzag の bit 長の平均と、bit 長の 0 次エントロピーで見る。

添字を送る予測子（幾何v4）はここには入れない。送る費用を別に測る必要がある。
"""
from __future__ import annotations
import concurrent.futures as cf
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "50000"))


def zlen(d: np.ndarray) -> np.ndarray:
    z = np.abs(d) * 2
    return np.where(z > 0, np.log2(z + 1).astype(int) + 1, 0)


def med3(a, b, c):
    return a + b + c - np.maximum(np.maximum(a, b), c) - np.minimum(np.minimum(a, b), c)


def preds(x: np.ndarray) -> dict:
    """x は 1 軸ぶんの整数列。予測子ごとに残差を返す。"""
    d = np.diff(x)                       # d[i] = x[i+1] - x[i]
    z = np.zeros(1, dtype=np.int64)
    d1 = np.concatenate([z, d[:-1]])     # 1 つ前の差分
    d2 = np.concatenate([z, z, d[:-2]])
    d3 = np.concatenate([z, z, z, d[:-3]])
    d4 = np.concatenate([z, z, z, z, d[:-4]])
    out = {
        "直前": d,
        "線形": d - d1,
        "med3": d - med3(d1, d2, d3),
        "平均3": d - (d1 + d2 + d3) // 3,
        "平均4": d - (d1 + d2 + d3 + d4) // 4,
        "重み": d - (2 * d1 + d2) // 3,
        "med3の半": d - med3(d1, d2, d3) // 2,
    }
    return out


def one(item):
    lab, path, kind = item
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz = M.read_block(kind, path, st, n)[0].astype(np.int64)
    acc = {}
    for c in range(3):
        for k, r in preds(xyz[:, c]).items():
            acc[k] = acc.get(k, 0) + zlen(r)
    return {"lab": lab, "n": len(xyz),
            **{k: float(np.mean(v)) for k, v in acc.items()}}


def med3n(a, b, c):
    return med3(a, b, c)


def window_trend(x: np.ndarray, W: int = 4) -> dict:
    """幾何v4（直近 W 点の最近傍）に局所の傾きを足した場合の残差。"""
    m = len(x)
    d = np.diff(x, axis=0)
    base = np.empty((m - 1, 3), dtype=np.int64)
    tr = np.zeros((m - 1, 3), dtype=np.int64)
    for i in range(1, m):
        a = max(0, i - W)
        c = x[a:i]
        j = int(np.argmin(np.abs(c - x[i]).sum(1)))
        base[i - 1] = x[i] - c[j]
        if i >= 4:
            tr[i - 1] = med3(d[i - 2], d[i - 3], d[i - 4])
    return {"x_j のみ": base, "+med3": base - tr, "+med3の半": base - tr // 2}


def one_w(item):
    lab, path, kind = item
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz = M.read_block(kind, path, st, n)[0].astype(np.int64)
    return {"lab": lab, **{k: float(np.mean(zlen(v).sum(1) if v.ndim == 2 else zlen(v)))
                           for k, v in window_trend(xyz).items()}}


def main():
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, INPUTS))
    keys = ["直前", "線形", "med3", "平均3", "平均4", "重み", "med3の半"]
    res.sort(key=lambda r: r["lab"])
    print(f"標本 {N} 点。zigzag の bit 長の平均（3 軸の和）。小さいほど良い。"
          f"いまの 幾何v3 は med3。")
    print(f"{'データ':<13}" + "".join(f"{k:>10}" for k in keys) + f"{'最良':>10}{'med3比':>8}")
    win = {}
    for r in res:
        b = min(keys, key=lambda k: r[k])
        win[b] = win.get(b, 0) + 1
        print(f"{r['lab']:<13}" + "".join(f"{r[k]:>10.2f}" for k in keys)
              + f"{b:>10}{100*(r[b]/r['med3']-1):>+7.1f}%")
    print("\n  最良になった回数: " + "  ".join(f"{k} {v}" for k, v in sorted(win.items(), key=lambda t: -t[1])))

    # 幾何v4（添字を送る枠）に傾き補正を足した場合。添字の費用は両者で同じ。
    sel = [i for i in INPUTS if i[0] in
           ("AHN5 _20", "workshop", "USGS NY", "autzen-2023", "AHN4 _21")]
    with cf.ProcessPoolExecutor(max_workers=5) as ex:
        rw = list(ex.map(one_w, sel))
    kw = ["x_j のみ", "+med3", "+med3の半"]
    print(f"\n幾何v4（W=4）に局所の傾きを足した場合。bit/点。")
    print(f"{'データ':<13}" + "".join(f"{k:>11}" for k in kw))
    for r in rw:
        print(f"{r['lab']:<13}" + "".join(f"{r[k]:>11.2f}" for k in kw))


if __name__ == "__main__":
    main()
