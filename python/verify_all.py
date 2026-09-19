"""これまでに報告した数値を、独立に計算し直して突き合わせる。

報告のたびに訂正が出たので、どの主張がどの程度確かめられているかを
一覧にする。ここで PASS しなかったものは信用してはいけない。
"""
from __future__ import annotations
import sys, time, subprocess, tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

RESULTS = []


def check(name: str, got, expected, tol_rel=0.02, note=""):
    if expected is None:
        ok = None
    elif isinstance(expected, str) or isinstance(got, str):
        ok = str(got) == str(expected)
    elif isinstance(expected, bool) or isinstance(got, (bool, np.bool_)):
        ok = bool(got) == bool(expected)
    else:
        ok = abs(float(got) - float(expected)) <= tol_rel * max(abs(float(expected)), 1e-12)
    RESULTS.append((name, got, expected, ok, note))
    mark = "PASS" if ok else ("FAIL" if ok is False else "----")
    g = f"{got:.4f}" if isinstance(got, (int, float, np.floating)) else str(got)
    e = f"{expected:.4f}" if isinstance(expected, (int, float, np.floating)) else str(expected)
    print(f"  [{mark}] {name:<52} 実測 {g:>12}  報告 {e:>12}  {note}")
    return ok


def summary():
    p = sum(1 for r in RESULTS if r[3] is True)
    f = sum(1 for r in RESULTS if r[3] is False)
    u = sum(1 for r in RESULTS if r[3] is None)
    print(f"\n{'='*100}\n  PASS {p}  /  FAIL {f}  /  未照合 {u}   計 {len(RESULTS)}")
    if f:
        print("\n  一致しなかった項目:")
        for n, g, e, ok, note in RESULTS:
            if ok is False:
                print(f"    - {n}: 実測 {g} / 報告 {e}")


def verify_ahn4_baseline():
    print("\n[A] AHN4 のベースラインと属性の内訳")
    from pcio import load
    from baselines import laszip_bits
    pc = load("data/raw/ahn4/31HZ1_20.LAZ", max_points=2_000_000)
    tot = laszip_bits(pc, keep_attrs=True)
    geo = laszip_bits(pc, keep_attrs=False)
    check("AHN4 LASzip 全体 bpp", tot.bpp, 71.252)
    check("AHN4 LASzip 幾何のみ bpp", geo.bpp, 22.208)
    check("AHN4 属性ぶん bpp", tot.bpp - geo.bpp, 49.044)
    check("AHN4 幾何の割合", geo.bpp / tot.bpp, 0.3117, tol_rel=0.03)
    # 元ファイルとの整合
    import laspy
    with laspy.open("data/raw/ahn4/31HZ1_20.LAZ") as f:
        n_all = f.header.point_count
    file_bpp = Path("data/raw/ahn4/31HZ1_20.LAZ").stat().st_size * 8 / n_all
    check("元ファイル実測 bpp（再現の妥当性）", file_bpp, 71.369, tol_rel=0.01,
          note="書き直した LAZ と元ファイルが一致するか")
    # Amplitude 重複
    eq = bool(np.array_equal(pc.raw["Amplitude"], pc.raw["intensity"]))
    check("Amplitude ≡ intensity（ビット単位）", eq, True)
    check("Amplitude の dtype が uint16", str(pc.raw["Amplitude"].dtype), "uint16")
    return pc


def verify_entropy_claims(pc):
    print("\n[B] エントロピーに関する主張")
    def H(a):
        _, c = np.unique(a, return_counts=True)
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum())
    def Hc(a, b):
        ab = np.stack([a.astype(np.int64), b.astype(np.int64)], 1)
        _, c = np.unique(ab, axis=0, return_counts=True)
        p = c / c.sum()
        return float(-(p * np.log2(p)).sum()) - H(b)
    I = pc.raw["intensity"].astype(np.int64)
    R = pc.raw["Reflectance"].astype(np.int64)
    check("H(Reflectance)", H(R), 10.176)
    check("H(Reflectance | intensity)", Hc(R, I), 5.937)
    check("H(Reflectance − intensity)", H(R - I), 6.280)


def verify_normalization():
    print("\n[C] 正規化レイヤー（復元検証つき）")
    from bench_normalize import run
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run("data/raw/ahn4/31HZ1_20.LAZ", 2_000_000, enable_residual=True)
    out = buf.getvalue()
    check("復元検証が通ったか", r["ok"], True)
    check("正規化後の合計 bpp", r["B"] * 8 / 2_000_000, 60.255)
    check("削減率", (r["A"] - r["B"]) / r["A"], 0.154, tol_rel=0.05)


def verify_lsb_sweep():
    print("\n[D] LSB スイープ（Δ = 3.04 = 圧縮不能）")
    from pcio import load
    from lsbsweep import sweep
    pc = load("data/raw/ahn4/31HZ1_20.LAZ", max_points=4_000_000)
    rows = sweep(pc, ks=range(0, 5), codecs=("tmc13",), verbose=False)
    for i in range(1, 5):
        d = rows[i-1]["tmc13"] - rows[i]["tmc13"]
        check(f"G-PCC Δbpp/レベル (k={i-1}→{i})", d, [3.040, 3.041, 3.042, 3.037][i-1],
              tol_rel=0.01)
    check("k=0 の G-PCC bpp", rows[0]["tmc13"], 24.512)
    check("全段で可逆", all(r.get("tmc13_lossless") for r in rows), True)


def verify_octant():
    print("\n[E] オクタント予測（H=3.000, 情報量 0.05〜0.16）")
    from pcio import load
    from exp_regime import point_spacing
    from exp_octant_predict import octant_experiment
    for lbl, path, mx, expH, expMI in [
        ("AHN4", "data/raw/ahn4/31HZ1_20.LAZ", 600_000, 2.999, 0.144),
        ("KITTI", "data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/"
                  "velodyne_points/data/0000000000.bin", None, 3.000, 0.054),
    ]:
        pc = load(path, max_points=mx)
        w = (pc.xyz_float if pc.xyz_float is not None
             else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
        w = w - w.min(0)
        r = octant_experiment(w, point_spacing(w))
        check(f"{lbl} H(オクタント)", r["H"], expH, tol_rel=0.005)
        check(f"{lbl} 情報量 I(c;ĉ)", r["H"] - r["H_cond"], expMI, tol_rel=0.25,
              note="標本依存で揺れる")


def verify_surface_sampling():
    print("\n[F] 表面とサンプリングの分解（表面は 11〜22%）")
    from pcio import load
    from exp_surface_vs_sampling import decompose
    for lbl, path, mx, exp in [
        ("AHN4", "data/raw/ahn4/31HZ1_20.LAZ", 1_500_000, 0.141),
        ("Bunny", "data/raw/stanford/bunny/reconstruction/bun_zipper.ply", None, 0.107),
    ]:
        pc = load(path, max_points=mx)
        w = (pc.xyz_float if pc.xyz_float is not None
             else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
        r = decompose(w, lbl)
        check(f"{lbl} 表面の割合", r["frac_surf"], exp, tol_rel=0.08)


def verify_kitti_decomposition():
    print("\n[G] KITTI の内訳（精度 −65% / 座標系 −13%）")
    import glob, zstandard as zstd
    from baselines import byte_split
    import normalize as nz
    def bits(streams, n):
        t = 0
        for v in streams:
            d = np.diff(np.asarray(v, np.int64), prepend=np.int64(0))
            z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
            t += len(zstd.ZstdCompressor(level=19).compress(byte_split(z)))
        return t * 8 / n
    fs = sorted(glob.glob("data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/"
                          "velodyne_points/data/*.bin"))[:30]
    C, G, P = [], [], []
    for f in fs:
        a = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        xyz = a[:, :3].astype(np.float64); n = len(a)
        u = np.ascontiguousarray(a[:, :3]).view(np.uint32).astype(np.int64)
        d = np.diff(u, axis=0, prepend=u[:1]); z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
        C.append(len(zstd.ZstdCompressor(level=19).compress(byte_split(z))) * 8 / n)
        v = 0.00244; q = np.rint(xyz / v).astype(np.int64)
        G.append(bits([q[:, 0], q[:, 1], q[:, 2]], n))
        ang = np.radians(0.002); qr, qa, qe = nz.polar_forward(xyz, 0.002, ang)
        P.append(bits([qr, qa, qe], n))
    c, g, p = np.mean(C), np.mean(G), np.mean(P)
    check("float32 可逆 bpp", c, 46.39)
    check("直交格子 2.44mm bpp", g, 16.33)
    check("極座標 2mm/0.002° bpp", p, 14.20)
    check("精度要件の寄与", (c - g) / c, 0.648, tol_rel=0.03)
    check("座標系の寄与", (g - p) / g, 0.130, tol_rel=0.10)


def verify_morton():
    print("\n[H] Morton の得失")
    from pcio import load
    from baselines import ref_geometry_bits
    exp = {"Stanford Armadillo": -21.37, "AHN4": 4.575, "Bunny 生": 13.687}
    S = "data/raw/stanford"
    for lbl, path, mx in [("Stanford Armadillo", f"{S}/Armadillo.ply", None),
                          ("AHN4", "data/raw/ahn4/31HZ1_20.LAZ", 1_500_000),
                          ("Bunny 生", f"{S}/bunny/data/bun000.ply", None)]:
        pc = load(path, max_points=mx)
        if pc.xyz_int is not None:
            q = pc.xyz_int
        else:
            w = pc.xyz_float.astype(np.float64); w = w - w.min(0)
            q = np.rint(w / (w.max() / 200000.)).astype(np.int64)
        a = ref_geometry_bits(q, "original").bpp
        b = ref_geometry_bits(q, "morton").bpp
        check(f"{lbl} Morton 差 bpp", b - a, exp[lbl], tol_rel=0.05)


def verify_rangecoder():
    print("\n[I] 決定論的レンジコーダ")
    from rangecoder import encode_ints, decode_ints
    rng = np.random.default_rng(0)
    allok = True
    for name, v in [("小さい値", rng.integers(-5, 6, 20000)),
                    ("広い値", rng.integers(-3000, 3000, 20000)),
                    ("偏り", rng.normal(0, 3, 20000).astype(np.int64)),
                    ("全部同じ", np.zeros(20000, np.int64)),
                    ("極端", rng.integers(-10**9, 10**9, 5000))]:
        b = encode_ints(v)
        allok &= bool(np.array_equal(decode_ints(b, len(v)), v))
    check("全ケースで可逆", allok, True)
    v = rng.integers(-5, 6, 20000)
    _, c = np.unique(v, return_counts=True); p = c / c.sum()
    H = float(-(p * np.log2(p)).sum())
    r = len(encode_ints(v)) * 8 / len(v)
    check("順序0エントロピー比の超過", r / H - 1, 0.017, tol_rel=0.5)


def verify_acquisition():
    print("\n[J] 取得構造スコアの推奨が実測と一致するか")
    from pcio import load
    from acquisition import acquisition_score
    S = "data/raw/stanford"
    truth = {  # (格納順維持, Morton候補, 極座標候補) — 実測から確定した正解
        "Bunny 生": (True, False, False),
        "KITTI": (True, False, True),
        "AHN4": (True, False, False),
        "Armadillo": (False, True, False),
    }
    for lbl, path, mx in [("Bunny 生", f"{S}/bunny/data/bun000.ply", None),
                          ("KITTI", "data/raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/"
                                    "velodyne_points/data/0000000000.bin", None),
                          ("AHN4", "data/raw/ahn4/31HZ1_20.LAZ", 300_000),
                          ("Armadillo", f"{S}/Armadillo.ply", None)]:
        pc = load(path, max_points=mx)
        w = (pc.xyz_float if pc.xyz_float is not None
             else pc.xyz_int * pc.scale + pc.offset).astype(np.float64)
        r = acquisition_score(w)["recommend"]
        got = (r["keep_storage_order"], r["consider_morton"], r["consider_polar"])
        check(f"{lbl} の推奨", str(got), str(truth[lbl]))


def verify_tsdf_lfs():
    print("\n[K] TSDF 経由の lfs（正解が分かる形状）")
    from tsdf import surface_from_points
    from tsdf_lfs import shrinking_ball_lfs
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    for R in (0.3, 1.0, 2.0):
        u = rng.normal(size=(20000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        pts = np.ascontiguousarray(u * R)
        d, _ = cKDTree(pts).query(pts, k=2); sp = float(np.median(d[:, 1]))
        V, N, F = surface_from_points(pts, 4 * sp)
        lfs = shrinking_ball_lfs(np.ascontiguousarray(V), np.ascontiguousarray(N),
                                 r_init=4 * R)
        check(f"球 R={R} の lfs 中央値", float(np.median(lfs)), R, tol_rel=0.04)
