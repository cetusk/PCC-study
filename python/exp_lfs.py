"""局所特徴サイズ（local feature size）от Voronoi 極。

09 節で位相に求めて得られなかった「物体側の誤差上限」は、実はこちらにある。
中軸（medial axis）までの距離 = 局所特徴サイズ lfs を使うと、
  摂動 < lfs / 2 なら位相は保たれる
という古典的な結果がある（Amenta–Bern の pole、Niyogi–Smale–Weinberger）。

しかも lfs は点ごとに定まるので、**場所によって違う誤差予算**を出せる。
細い部分は細かく、平らな部分は粗く、という配分が原理的に正当化される。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import Voronoi, cKDTree


def local_feature_size(pts: np.ndarray) -> np.ndarray:
    """各点について、そのボロノイ胞の最も遠い頂点までの距離（極）を返す。

    面から十分密に標本されていれば、この距離は中軸までの距離に収束する。
    """
    vor = Voronoi(pts)
    lfs = np.zeros(len(pts))
    vv = vor.vertices
    for i, region_idx in enumerate(vor.point_region):
        verts = vor.regions[region_idx]
        verts = [v for v in verts if v >= 0]
        if not verts:
            lfs[i] = np.inf
            continue
        d = np.linalg.norm(vv[verts] - pts[i], axis=1)
        lfs[i] = d.max()
    return lfs


def report(w: np.ndarray, label: str, n_sample: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    pts = np.ascontiguousarray(w[rng.choice(len(w), min(n_sample, len(w)), replace=False)])
    d, _ = cKDTree(pts).query(pts, k=2)
    sp = float(np.median(d[:, 1]))
    lfs = local_feature_size(pts)
    fin = lfs[np.isfinite(lfs)]
    q = np.percentile(fin, [1, 5, 25, 50, 95])
    print(f"## {label}   標本 {len(pts):,}   点間隔 {sp:.5g}")
    print(f"   lfs 分位点  P1 {q[0]:.5g}  P5 {q[1]:.5g}  P25 {q[2]:.5g}  "
          f"中央 {q[3]:.5g}  P95 {q[4]:.5g}")
    print(f"   lfs / 点間隔  P1 {q[0]/sp:.2f}  P5 {q[1]/sp:.2f}  中央 {q[3]/sp:.2f}")
    print(f"   位相を保証する誤差上限  最悪点基準 ε = P1/2 = {q[0]/2:.5g}"
          f"   / 95%の点で ε = P5/2 = {q[1]/2:.5g}")
    return dict(spacing=sp, lfs=fin, eps_p1=float(q[0] / 2), eps_p5=float(q[1] / 2))


if __name__ == "__main__":
    from pcio import load
    S = "data/raw/stanford"
    for lbl, p, mx in [("Stanford Bunny", f"{S}/bunny/reconstruction/bun_zipper.ply", None),
                       ("Stanford Armadillo", f"{S}/Armadillo.ply", None),
                       ("AHN4 航空LiDAR", "data/raw/ahn4/31HZ1_20.LAZ", 400_000)]:
        pc = load(p, max_points=mx)
        w = (pc.xyz_float if pc.xyz_float is not None
             else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
        report(np.ascontiguousarray(w - w.mean(0)), lbl)
        print()
