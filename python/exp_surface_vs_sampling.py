"""ビットを「物体の情報」と「サンプリング位置の情報」に分ける。

問い:
  octree の子オクタントに 3 bit 掛かるのは事実だが、それは
  「物体について 3 bit 分かる」という意味ではない。
  点がセル内のどこに落ちたかは、物体ではなく**標本の取り方**で決まる。

そこで総ビットを次の二つに分ける。
  表面の情報        点間隔の格子での占有を送るのに必要なビット
  サンプリングの情報 そこから格納精度まで位置を詰めるのに必要なビット

前者は物体の形そのもの、後者は「どこを標本したか」である。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pcio import load
from baselines import tmc13_bits
from exp_regime import point_spacing


def decompose(w: np.ndarray, label: str, fine_ratio: float = 1/256) -> dict:
    w = w - w.min(0)
    s = point_spacing(w)
    # 表面: 格子 = 点間隔
    q_surf = np.rint(w / s).astype(np.int64)
    b_surf = tmc13_bits(q_surf).bpp
    # 実質フル精度: 格子 = 点間隔 / 256
    q_fine = np.rint(w / (s * fine_ratio)).astype(np.int64)
    r = tmc13_bits(q_fine)
    b_fine = r.bpp
    return dict(label=label, n=len(w), spacing=s,
                b_surf=b_surf, b_fine=b_fine,
                b_sample=b_fine - b_surf,
                frac_surf=b_surf / b_fine, lossless=r.lossless)


def main():
    S = "data/raw/stanford"
    targets = [
        ("Stanford Bunny", f"{S}/bunny/reconstruction/bun_zipper.ply", None),
        ("Stanford Dragon", f"{S}/dragon_recon/dragon_vrip.ply", None),
        ("Stanford Armadillo", f"{S}/Armadillo.ply", None),
        ("AHN4 航空LiDAR", "data/raw/ahn4/31HZ1_20.LAZ", 1_500_000),
        ("KITTI 車載LiDAR", "data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/"
                            "velodyne_points/data/0000000000.bin", None),
    ]
    rows = []
    for label, path, mx in targets:
        pc = load(path, max_points=mx)
        w = (pc.xyz_float if pc.xyz_float is not None
             else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
        rows.append(decompose(w, label))

    print(f"{'データ':<22}{'N':>10}{'点間隔':>11}"
          f"{'表面':>9}{'サンプリング':>13}{'合計':>9}{'表面の割合':>11}")
    print("-" * 86)
    for r in rows:
        print(f"{r['label']:<22}{r['n']:>10,}{r['spacing']:>11.5g}"
              f"{r['b_surf']:>9.3f}{r['b_sample']:>13.3f}{r['b_fine']:>9.3f}"
              f"{r['frac_surf']*100:>10.1f}%")
    print("-" * 86)
    print("\n単位 bpp。「表面」= 点間隔の格子での占有。"
          "「サンプリング」= そこから点間隔/256 まで位置を詰める分。")
    print("可逆性:", {r['label']: r['lossless'] for r in rows})


if __name__ == "__main__":
    main()
