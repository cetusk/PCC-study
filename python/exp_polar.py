"""実験: 回転式 LiDAR の float32 デカルト座標を、センサ本来の極座標に戻す。

仮説（survey 攻め筋⑦「表現そのものを疑う」の実スキャン向け具体形）:
  デカルト float32 xyz (96 bit/点) は派生表現である。センサが本来出しているのは
  (距離, 方位角, 仰角) であり、それぞれ固有の分解能を持つ。
  デカルト空間で圧縮している限り、この構造は文脈モデルからは見えない。

測るもの: 量子化ステップを変えながら bpp と「デカルト空間での再構成誤差」を同時に出す。
        誤差をセンサの測距精度より十分小さく保ったまま、どこまで bpp が落ちるか。
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baselines import CODECS, byte_split


def to_polar(xyz: np.ndarray):
    p = xyz.astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    az = np.arctan2(p[:, 1], p[:, 0])
    el = np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-12), -1, 1))
    return r, az, el


def from_polar(r, az, el):
    c = np.cos(el)
    return np.stack([r * c * np.cos(az), r * c * np.sin(az), r * np.sin(el)], 1)


def delta_bits(q: np.ndarray, codec="zstd19") -> float:
    d = np.diff(q, prepend=q[:1]).astype(np.int64)
    z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
    return len(CODECS[codec](byte_split(z))[0]) * 8 / len(q)


def run(path, r_steps=(0.002, 0.001, 0.0005), ang_deg=(0.02, 0.01, 0.005, 0.002)):
    a = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    xyz = a[:, :3]
    n = len(a)
    r, az, el = to_polar(xyz)

    # 参考: デカルトのまま可逆で圧縮した場合
    u = np.ascontiguousarray(xyz).view(np.uint32)
    d = np.diff(u.astype(np.int64), axis=0, prepend=u[:1].astype(np.int64))
    dz = ((d << 1) ^ (d >> 63)).astype(np.uint64)
    cart = len(CODECS["zstd19"](byte_split(dz))[0]) * 8 / n

    print(f"# 極座標表現の実験  {Path(path).name}  N={n:,}")
    print(f"\n基準: float32 デカルト 格納 96.000 bpp")
    print(f"      同 可逆圧縮（取得順差分+zstd） {cart:.3f} bpp  "
          f"= raw の {cart/96*100:.1f}%\n")
    print(f"{'距離刻み':>9}{'角度刻み':>10}{'bpp':>9}{'vs raw':>8}{'vs 可逆':>9}"
          f"{'誤差RMS':>10}{'誤差P99':>10}{'誤差max':>10}")
    print("-" * 76)
    out = []
    for rs in r_steps:
        for ad in ang_deg:
            aq = np.radians(ad)
            qr = np.rint(r / rs).astype(np.int64)
            qa = np.rint(az / aq).astype(np.int64)
            qe = np.rint(el / aq).astype(np.int64)
            bits = delta_bits(qr) + delta_bits(qa) + delta_bits(qe)
            rec = from_polar(qr * rs, qa * aq, qe * aq)
            e = np.linalg.norm(rec - xyz.astype(np.float64), axis=1)
            print(f"{rs*1000:8.1f}mm{ad:9.3f}°{bits:9.3f}{bits/96*100:7.1f}%"
                  f"{bits/cart*100:8.1f}%{e.std()*1000:9.2f}mm{np.percentile(e,99)*1000:9.2f}mm"
                  f"{e.max()*1000:9.2f}mm")
            out.append((rs, ad, bits, e))
    print("-" * 76)
    print("\n注: 誤差はデカルト空間での再構成誤差。角度刻みの誤差は距離に比例して効く")
    print("    （遠方ほど大きい）ため、P99 と max の差が大きい。")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else
        "data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin")
