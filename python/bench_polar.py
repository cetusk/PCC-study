"""正規化レイヤーのエンドツーエンド評価（回転式 LiDAR 経路）。

A. 元の .bin をそのまま / 汎用圧縮 / デカルト可逆圧縮
B. 正規化（極座標復元 + パレット）して PCC1 コンテナに書いた場合

B は幾何が bounded（誤差上限つき）なので、毎回
  ・宣言した誤差上限を実際に超えていないか
  ・属性はビット完全に戻るか
を検証する。
"""
from __future__ import annotations
import sys, glob, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import zstandard as zstd
import normalize as nz
import pccfile
from baselines import byte_split


def cartesian_lossless_bytes(xyz: np.ndarray) -> int:
    u = np.ascontiguousarray(xyz).view(np.uint32)
    d = np.diff(u.astype(np.int64), axis=0, prepend=u[:1].astype(np.int64))
    z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
    return len(zstd.ZstdCompressor(level=19).compress(byte_split(z)))


def run_frame(path, r_step=0.002, ang_deg=0.002, tmp: Path | None = None):
    a = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    xyz, inten = a[:, :3], a[:, 3]
    n = len(a)

    # ---- 正規化
    plan = nz.Plan()
    geo = nz.make_polar_op(xyz, r_step, ang_deg)
    plan.ops.append(geo)
    pal = nz.detect_palette("intensity", inten)
    if pal:
        plan.ops.append(pal)

    ang = np.radians(ang_deg)
    qr, qa, qe = nz.polar_forward(xyz, r_step, ang)
    streams = {"qr": qr, "qaz": qa, "qel": qe}
    if pal:
        streams["intensity"] = nz.palette_forward(inten, pal.params["values"])
    else:
        streams["intensity"] = inten.view(np.int32).astype(np.int64)

    spec = {"n": n, "ops": [nz_op_to_dict(o) for o in plan.ops]}
    p = (tmp or Path(tempfile.mkdtemp())) / (Path(path).stem + ".pcc")
    B = pccfile.write(p, spec, streams)

    # ---- 検証
    spec2, back = pccfile.read(p, n)
    rec = nz.polar_inverse(back["qr"], back["qaz"], back["qel"], r_step, ang)
    err = np.linalg.norm(rec - xyz.astype(np.float64), axis=1)
    if pal:
        ri = nz.palette_inverse(back["intensity"], pal.params["values"], np.float32)
        int_ok = np.array_equal(ri.view(np.uint32), inten.view(np.uint32))
    else:
        int_ok = True
    declared = geo.params["max_err_m"]
    within = bool(err.max() <= declared + 1e-12)

    return dict(n=n, raw=n * 16, cart=cartesian_lossless_bytes(xyz), B=B,
                err_max=float(err.max()), err_rms=float(err.std()),
                declared=declared, within=within, int_ok=int_ok)


def nz_op_to_dict(o):
    from dataclasses import asdict
    return asdict(o)


def run(pattern, limit=None, r_step=0.002, ang_deg=0.002):
    fs = sorted(glob.glob(pattern))[:limit]
    tmp = Path(tempfile.mkdtemp())
    R = [run_frame(f, r_step, ang_deg, tmp) for f in fs]
    n = sum(r["n"] for r in R)
    raw = sum(r["raw"] for r in R); cart = sum(r["cart"] for r in R); B = sum(r["B"] for r in R)
    print(f"# 正規化レイヤー（回転式 LiDAR）  {len(R)} フレーム  {n:,} 点")
    print(f"  設定: 距離 {r_step*1000:.1f}mm 刻み / 角度 {ang_deg:.3f}° 刻み\n")
    print(f"{'':<40}{'bytes':>14}{'bpp':>10}{'vs raw':>9}")
    print("-" * 74)
    print(f"{'A1. 元の .bin（xyz+intensity）':<40}{raw:>14,}{raw*8/n:>10.3f}{100.0:>8.1f}%")
    print(f"{'A2. デカルト float32 可逆圧縮（xyz のみ）':<40}{cart:>14,}{cart*8/n:>10.3f}"
          f"{cart/raw*100:>8.1f}%")
    print(f"{'B.  正規化 + PCC1（xyz+intensity, 副情報込）':<40}{B:>14,}{B*8/n:>10.3f}"
          f"{B/raw*100:>8.1f}%")
    print("-" * 74)
    print(f"  A2 比 {(1-B/cart)*100:.1f}% 削減   （B は intensity も含むのに対し A2 は幾何のみ）")
    print()
    em = max(r["err_max"] for r in R)
    print(f"検証: 全フレームで誤差上限を守った = {all(r['within'] for r in R)}"
          f"   intensity ビット完全 = {all(r['int_ok'] for r in R)}")
    print(f"      最大再構成誤差 {em*1000:.2f} mm   RMS {np.mean([r['err_rms'] for r in R])*1000:.2f} mm")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--r-step", type=float, default=0.002)
    ap.add_argument("--ang-deg", type=float, default=0.002)
    a = ap.parse_args()
    run(a.pattern, a.limit, a.r_step, a.ang_deg)
