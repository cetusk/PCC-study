"""4 つのデータセットを「格子/点間隔」比で揃えて比較する。

測る量（いずれも偏りのない推定）:
  H(子オクタント)                        … 一様なら 3.000
  H(子オクタント | 粗レベルの平面予測)    … 文脈は 8 通りだけなので小標本バイアスが無い
  被覆                                   … 予測に使える近傍セルがある点の割合
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pcio import load
from exp_regime import point_spacing
from exp_octant_predict import octant_experiment


def analyse(path, label, ratios=(1/64, 1/16, 1/4, 1/2, 1, 2),
            max_points=None, sample=120_000):
    pc = load(path, max_points=max_points)
    w = (pc.xyz_float if pc.xyz_float is not None
         else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
    w = w - w.min(0)
    sp = point_spacing(w)
    print(f"\n## {label}   N={len(w):,}   点間隔={sp:.6g}")
    print(f"  {'格子/点間隔':>12}{'被覆':>8}{'H(子)':>9}{'H(子|予測)':>12}"
          f"{'情報量':>9}{'的中率':>9}")
    print("  " + "-" * 60)
    for r in ratios:
        res = octant_experiment(w, sp * r, sample=sample)
        if res["n"] == 0:
            print(f"  {r:>12.4f}{0:>7.0f}%{'—':>9}{'—':>12}{'—':>9}{'—':>9}")
            continue
        mi = res["H"] - res["H_cond"]
        print(f"  {r:>12.4f}{res['coverage']*100:>7.0f}%{res['H']:>9.3f}"
              f"{res['H_cond']:>12.3f}{mi:>9.3f}{res['acc']*100:>8.1f}%")


if __name__ == "__main__":
    S = "data/raw/stanford"
    for path, label, mx in [
        (f"{S}/bunny/reconstruction/bun_zipper.ply", "Stanford Bunny (再構成)", None),
        (f"{S}/dragon_recon/dragon_vrip.ply", "Stanford Dragon (再構成)", None),
        (f"{S}/Armadillo.ply", "Stanford Armadillo", None),
        ("data/raw/ahn4/31HZ1_20.LAZ", "AHN4 (航空 LiDAR)", 600_000),
        ("data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/"
         "velodyne_points/data/0000000000.bin", "KITTI (車載 LiDAR)", None),
    ]:
        try:
            analyse(path, label)
        except Exception as e:
            print(f"\n## {label}: 失敗 {type(e).__name__}: {e}")
    print("\n基準: H(子)=3.000 が一様（＝情報ゼロ）。的中率の基準は 1/8=12.5%。")
