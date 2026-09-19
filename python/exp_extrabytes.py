"""実験: LAS の ExtraBytes に float64 で入っている「実は小さな整数」を正す。

AHN4 で判明したこと:
  Amplitude   ≡ intensity / 100          （完全な重複。真の情報量 0 bit）
  Reflectance = 整数 / 100, 範囲 [-1592, 3259]  （真の情報量 12 bit）
  どちらも float64 = 64 bit/点 で格納されている。

LASzip の ExtraBytes 圧縮はこれらを不透明なバイト列として扱うため、
この冗長を一切回収できない。ここを正すと何 bpp 浮くかを実測する。
"""
from __future__ import annotations
import sys, numpy as np, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import laspy
from pcio import load


def write_laz(hdr, n, xyz_int, attrs, path):
    las = laspy.LasData(hdr)
    las.X, las.Y, las.Z = xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2]
    names = set(las.point_format.dimension_names)
    for k, v in attrs.items():
        if k in names:
            las[k] = v
    t = time.perf_counter(); las.write(str(path)); return time.perf_counter() - t


def run(path, max_points=2_000_000):
    pc = load(path, max_points=max_points)
    n = pc.n
    src = pc.meta["las_header"]
    d = Path(tempfile.mkdtemp())
    out = []

    def case(name, pf_builder, attrs):
        hdr = pf_builder()
        hdr.scales, hdr.offsets = pc.scale, pc.offset
        p = d / (name.replace("/", "_") + ".laz")
        enc = write_laz(hdr, n, pc.xyz_int, attrs, p)
        b = p.stat().st_size
        out.append((name, b * 8 / n, enc))
        return b * 8 / n

    def base_hdr():
        h = laspy.LasHeader(version=src.version, point_format=src.point_format)
        for v in src.vlrs:
            h.vlrs.append(v)
        return h

    # (A) 現状再現
    a = case("A. 現状（元ファイル相当）", base_hdr, pc.attrs)

    # (B) Amplitude を落とす（intensity から完全に復元できる）
    def hdr_b():
        h = laspy.LasHeader(version=src.version, point_format=src.point_format)
        for v in src.vlrs:
            h.vlrs.append(v)
        return h
    at_b = {k: v for k, v in pc.attrs.items() if k != "Amplitude"}
    at_b["Amplitude"] = np.zeros(n)          # 定数で埋める＝実質削除
    b = case("B. Amplitude を除去", hdr_b, at_b)

    # (C) さらに Reflectance を int16(×100) 相当に量子化して float64 の無駄を消す
    at_c = dict(at_b)
    at_c["Reflectance"] = np.rint(pc.attrs["Reflectance"] * 100) / 100.0
    c = case("C. B + Reflectance を格子に載せ直す", hdr_b, at_c)

    # (D) 理論値: 両フィールドを本来のビット幅で送った場合
    #     Amplitude 0 bit, Reflectance 12 bit（さらにエントロピー符号化すれば減る）
    refl = np.rint(pc.attrs["Reflectance"] * 100).astype(np.int64)
    _, cnt = np.unique(refl, return_counts=True)
    p_ = cnt / cnt.sum()
    H_refl = float(-(p_ * np.log2(p_)).sum())

    print(f"# ExtraBytes 実験  {Path(path).name}  N={n:,}\n")
    print(f"{'ケース':<38}{'bpp':>10}{'削減':>10}")
    print("-" * 60)
    for name, bpp, enc in out:
        print(f"{name:<38}{bpp:10.3f}{(a-bpp)/a*100 if bpp!=a else 0:9.1f}%")
    print("-" * 60)
    print(f"\n参考: Reflectance の真のエントロピー（順序0） = {H_refl:.3f} bit/点")
    print(f"      Amplitude の真の情報量                = 0 bit/点（intensity から復元可）")
    print(f"      現状この 2 フィールドが占める格納幅   = 128 bit/点")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/ahn4/31HZ1_20.LAZ")
