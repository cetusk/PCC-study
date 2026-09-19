"""点群から位相構造（パーシステントホモロジー）を計算する。

狙い:
  量子化ステップをどこまで粗くしてよいかを、これまでは
  「センサのノイズ床」から決めてきた。これは計測側の都合である。
  位相は**物体側**の都合を与える。穴や連結成分が壊れない限界まで、
  という基準が立つなら、二つの基準の厳しい方を採ればよい。

測るもの:
  アルファ複体のフィルトレーションで β0（連結成分）, β1（ループ）,
  β2（空洞）が「正しい値で安定している尺度の帯」を求める。
  帯の下端は標本密度、上端は最小構造で決まる。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gudhi
from pcio import load


def persistence(pts: np.ndarray, max_alpha_sq: float = float("inf")):
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree(max_alpha_square=max_alpha_sq)
    st.compute_persistence(homology_coeff_field=2)
    out = {}
    for d in (0, 1, 2):
        iv = st.persistence_intervals_in_dimension(d)
        # アルファ複体の値は半径の二乗。半径に直す
        out[d] = np.sqrt(np.clip(np.asarray(iv, float), 0, None)) if len(iv) else np.zeros((0, 2))
    return out


def betti_at(iv_by_dim, r: float) -> tuple[int, int, int]:
    b = []
    for d in (0, 1, 2):
        iv = iv_by_dim[d]
        if len(iv) == 0:
            b.append(0); continue
        alive = (iv[:, 0] <= r) & ((iv[:, 1] > r) | ~np.isfinite(iv[:, 1]))
        b.append(int(alive.sum()))
    return tuple(b)


def significant(iv, min_life_ratio=3.0):
    """生存区間が十分長い特徴だけ数える（標本ノイズ由来の短命な穴を除く）。"""
    if len(iv) == 0:
        return 0
    fin = iv[np.isfinite(iv[:, 1])]
    inf = len(iv) - len(fin)
    if len(fin) == 0:
        return inf
    born = np.maximum(fin[:, 0], 1e-12)
    return int((fin[:, 1] / born >= min_life_ratio).sum()) + inf


def run(path, label, n_sample=6000, seed=0, quant_steps=None):
    pc = load(path)
    w = (pc.xyz_float if pc.xyz_float is not None
         else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(n_sample, len(w)), replace=False)
    pts = w[idx]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=2)
    sp = float(np.median(d[:, 1]))
    diam = float(np.linalg.norm(w.max(0) - w.min(0)))

    print(f"## {label}   標本 {len(pts):,} 点 / 全体 {len(w):,} 点")
    print(f"   標本の点間隔 {sp:.5g}   物体の対角 {diam:.5g}   比 1:{diam/sp:.0f}")
    iv = persistence(pts)
    print(f"   {'半径 r':>12}{'r/点間隔':>10}{'β0':>6}{'β1':>6}{'β2':>6}")
    for m in (0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        r = sp * m
        b0, b1, b2 = betti_at(iv, r)
        print(f"   {r:>12.5g}{m:>10.1f}{b0:>6}{b1:>6}{b2:>6}")
    print(f"   長寿命の特徴のみ: β1 = {significant(iv[1])}, β2 = {significant(iv[2])}")
    return iv, sp, w


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("label", nargs="?", default="")
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    run(a.path, a.label or Path(a.path).stem, a.n)


def topology_vs_quantization(path, label, n_sample=6000, seed=0,
                             ratios=(1/16, 1/8, 1/4, 1/2, 1, 2, 4)):
    """量子化ステップをどこまで粗くすると位相が壊れるかを測る。

    基準（量子化なし）のパーシステンス図とのボトルネック距離で比べる。
    距離が点間隔に比べて十分小さいうちは、物体の構造は保たれている。
    """
    from scipy.spatial import cKDTree
    pc = load(path)
    w = (pc.xyz_float if pc.xyz_float is not None
         else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
    rng = np.random.default_rng(seed)
    pts = w[rng.choice(len(w), min(n_sample, len(w)), replace=False)]
    d, _ = cKDTree(pts).query(pts, k=2)
    sp = float(np.median(d[:, 1]))

    base = persistence(pts)
    print(f"## {label}   標本 {len(pts):,} 点   点間隔 {sp:.5g}")
    print(f"   {'格子 v':>12}{'v/点間隔':>10}{'残った点':>10}"
          f"{'β1':>5}{'β2':>5}{'ボトルネック距離':>16}{'/点間隔':>9}  判定")
    print("   " + "-" * 82)
    print(f"   {'(基準)':>12}{'0':>10}{len(pts):>10}"
          f"{significant(base[1]):>5}{significant(base[2]):>5}{'—':>16}{'—':>9}")
    for m in ratios:
        v = sp * m
        q = np.unique(np.rint(pts / v).astype(np.int64), axis=0) * v
        if len(q) < 50:
            continue
        iv = persistence(q)
        bd = max(gudhi.bottleneck_distance(base[k], iv[k]) for k in (1, 2))
        rel = bd / sp
        judge = ("構造は保たれている" if rel < 0.5 else
                 "変化が出始めた" if rel < 1.5 else "位相が壊れた")
        print(f"   {v:>12.5g}{m:>10.3g}{len(q):>10,}"
              f"{significant(iv[1]):>5}{significant(iv[2]):>5}{bd:>16.5g}{rel:>9.2f}  {judge}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[-1] == "--quant":
    pass
