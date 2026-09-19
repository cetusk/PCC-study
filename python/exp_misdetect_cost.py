"""走査幾何の判定を誤ったときの損失を測る。

方針:
  判定を「分類」ではなく「選択」にすれば、外したときの損失は原理的に 0 にできる。
  符号化器は候補表現を実際に符号化して短い方を採り、選んだ結果を仕様に書く。
  復号器は仕様を読むだけなので推測しない。

ここでは誤判定の損失そのものを定量化する。
同じ誤差上限のもとで、直交格子量子化 と 極座標量子化 の符号長を比べる。
"""
from __future__ import annotations
import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import zstandard as zstd
from pcio import load
from baselines import byte_split
from exp_regime import point_spacing
import normalize as nz


def _bits(streams, n) -> float:
    tot = 0
    for v in streams:
        d = np.diff(np.asarray(v, np.int64), prepend=np.int64(0))
        z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
        tot += len(zstd.ZstdCompressor(level=19).compress(byte_split(z)))
    return tot * 8 / n


def grid_option(w, eps):
    v = 2 * eps / np.sqrt(3)                       # 最大誤差 = (√3/2)·v = eps
    q = np.rint(w / v).astype(np.int64)
    err = np.linalg.norm(q * v - w, axis=1)
    return _bits([q[:, 0], q[:, 1], q[:, 2]], len(w)), float(err.max())


def polar_option(w, eps, origin=None):
    o = np.zeros(3) if origin is None else origin
    p = w - o
    r = np.linalg.norm(p, axis=1)
    r99 = float(np.percentile(r, 99)) or 1.0
    dr = eps / np.sqrt(3)
    da = eps / (np.sqrt(3) * max(r99, 1e-9))       # 接線方向の誤差を同程度に割り当てる
    qr, qa, qe = nz.polar_forward(p, dr, da)
    rec = nz.polar_inverse(qr, qa, qe, dr, da) + o
    err = np.linalg.norm(rec - w, axis=1)
    return _bits([qr, qa, qe], len(w)), float(err.max())


def run():
    S = "data/raw/stanford"
    T = [("Stanford Bunny", f"{S}/bunny/reconstruction/bun_zipper.ply", None),
         ("Stanford Armadillo", f"{S}/Armadillo.ply", None),
         ("Bunny 生スキャン", f"{S}/bunny/data/bun000.ply", None),
         ("AHN4 航空LiDAR", "data/raw/ahn4/31HZ1_20.LAZ", 800_000),
         ("KITTI 車載LiDAR", "data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/"
                             "velodyne_points/data/0000000000.bin", None)]
    print(f"{'データ':<20}{'単調性':>8}{'判定':>8}"
          f"{'直交格子':>10}{'極座標':>10}{'極/直交':>9}{'誤差上限':>11}{'試行時間':>9}")
    print("-" * 88)
    for lbl, path, mx in T:
        pc = load(path, max_points=mx)
        w = (pc.xyz_float if pc.xyz_float is not None
             else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
        det = nz.detect_spinning_lidar(w)
        # 極座標の原点: 回転式と判定されれば原点、そうでなければ重心を仮に使う
        origin = None if det["is_spinning"] else w.mean(0)
        w0 = w - w.mean(0) if not det["is_spinning"] else w
        sp = point_spacing(w0)
        eps = sp / 16
        t0 = time.perf_counter()
        bg, eg = grid_option(w0, eps)
        bp, ep = polar_option(w0, eps, None if det["is_spinning"] else np.zeros(3))
        dt = time.perf_counter() - t0
        print(f"{lbl:<20}{det['monotonic_frac']*100:>7.0f}%"
              f"{('回転式' if det['is_spinning'] else 'それ以外'):>9}"
              f"{bg:>9.2f}b{bp:>9.2f}b{bp/bg*100:>8.0f}%"
              f"{max(eg,ep)*1000:>10.3f}mm{dt:>8.1f}s")
    print("-" * 88)
    print("\n『極/直交』が 100% を超えるほど、極座標を選ぶと損をする。"
          "誤差上限は両案で揃えてある。")


if __name__ == "__main__":
    run()
