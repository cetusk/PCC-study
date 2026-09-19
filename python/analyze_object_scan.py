"""密なオブジェクトスキャン（Stanford 等）の解析。

確かめたいこと:
  1. 表現の無駄は ALS / 車載 LiDAR 以外にもあるか
  2. 「点間隔以下では octree の情報がゼロ」という折れ点は、
     データの種類ではなく 格子/点間隔 比だけで決まるのか
"""
from __future__ import annotations
import sys, zlib, lzma
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import zstandard as zstd
from scipy.spatial import cKDTree
from pcio import load
from baselines import tmc13_bits, byte_split
from exp_regime import point_spacing, regime_curve, print_curve


def axis_grid_report(w: np.ndarray) -> str:
    """各軸が固定刻みの格子に乗っているかを調べる。"""
    L = [f"  {'軸':<4}{'unique':>10}{'必要bit':>9}{'推定刻み':>14}{'格子上':>9}"]
    L.append("  " + "-" * 48)
    for i, nm in enumerate("xyz"):
        u = np.unique(w[:, i])
        if len(u) < 3:
            L.append(f"  {nm:<4}{len(u):>10,}{1:>9}{'-':>14}{'-':>9}")
            continue
        d = np.diff(u)
        d = d[d > 0]
        step = float(np.median(d))
        k = w[:, i] / step
        on = float((np.abs(k - np.rint(k)) < 0.01).mean())
        bits = int(np.ceil(np.log2(len(u)))) if len(u) > 1 else 1
        L.append(f"  {nm:<4}{len(u):>10,}{bits:>9}{step*1000:>13.6f}mm{on*100:>8.1f}%")
    return "\n".join(L)


def float_bits_used(w: np.ndarray) -> str:
    a = np.ascontiguousarray(w.astype(np.float32))
    tot = []
    for i in range(3):
        u = a[:, i].copy().view(np.uint32)
        h = 0.0
        for b in range(32):
            p = float(((u >> np.uint32(b)) & 1).mean())
            if 0 < p < 1:
                h += -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        tot.append(h)
    return (f"  float32 のビット毎エントロピー合計: "
            f"x={tot[0]:.1f} y={tot[1]:.1f} z={tot[2]:.1f}  "
            f"計 {sum(tot):.1f} bit/点（格納は 96 bit/点）")


def baselines(w: np.ndarray, path: Path, sp: float) -> str:
    n = len(w)
    a32 = np.ascontiguousarray(w.astype(np.float32))
    raw = n * 12
    u = a32.view(np.uint32)
    d = np.diff(u.astype(np.int64), axis=0, prepend=u[:1].astype(np.int64))
    z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
    zs = len(zstd.ZstdCompressor(level=19).compress(byte_split(z)))
    # 点間隔に合わせてボクセル化してから G-PCC（8iVFB 等と同じ土俵）
    q = np.rint((w - w.min(0)) / sp).astype(np.int64)
    r = tmc13_bits(q)
    L = [f"  {'元ファイル（配布形式そのまま）':<38}{path.stat().st_size*8/n:>9.3f} bpp",
         f"  {'float32 xyz 無圧縮':<38}{raw*8/n:>9.3f} bpp",
         f"  {'float32 xyz 可逆圧縮（差分+zstd）':<38}{zs*8/n:>9.3f} bpp",
         f"  {'点間隔にボクセル化 + G-PCC 可逆':<38}{r.bpp:>9.3f} bpp"
         f"   （{len(np.unique(q,axis=0)):,} 個の異なるボクセル）"]
    return "\n".join(L)


def run(path, max_points=None, levels=13):
    p = Path(path)
    pc = load(p, max_points=max_points)
    w = (pc.xyz_float if pc.xyz_float is not None
         else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
    sp = point_spacing(w)
    ext = w.max(0) - w.min(0)
    print(f"# {p.name}   N={len(w):,}")
    print(f"  範囲 {ext[0]*1000:.1f} x {ext[1]*1000:.1f} x {ext[2]*1000:.1f} mm"
          f"   点間隔 中央値 {sp*1000:.4f} mm")
    if pc.attrs:
        print(f"  属性: {', '.join(f'{k}({v.dtype})' for k, v in pc.attrs.items())}")
    print()
    print("[1] 各軸は格子に乗っているか")
    print(axis_grid_report(w))
    print()
    print(float_bits_used(w))
    print()
    print("[2] ベースライン")
    print(baselines(w, p, sp))
    print()
    print("[3] レジーム曲線（1 レベルあたり何ビット掛かるか）")
    print_curve(regime_curve(w, p.stem, n_levels=levels))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--levels", type=int, default=13)
    a = ap.parse_args()
    run(a.path, a.max_points, a.levels)
