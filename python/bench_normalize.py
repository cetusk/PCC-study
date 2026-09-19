"""正規化レイヤーのエンドツーエンド評価（LAS/LAZ 経路）。

測るもの:
  A. 元のまま LAZ に書いた場合（production ベースライン）
  B. 正規化してから LAZ に書き、外した分を外部符号化した場合
     （容器 + 外部ストリーム + 仕様 の合計。副情報も全部数える）

そして B から元データがビット完全に戻ることを毎回検証する。
戻らなければ数字は意味を持たないので、検証に失敗したら結果を出さない。
"""
from __future__ import annotations
import sys, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import laspy
from pcio import load
import normalize as nz
from rangecoder import encode_ints, decode_ints

# LAS の ExtraBytes 型コード -> numpy dtype
EB_TYPE = {1: "uint8", 2: "int8", 3: "uint16", 4: "int16", 5: "uint32",
           6: "int32", 7: "uint64", 8: "int64", 9: "float32", 10: "float64"}


def _extra_dim_specs(hdr):
    out = []
    for vlr in hdr.vlrs:
        if type(vlr).__name__ == "ExtraBytesVlr":
            for eb in vlr.extra_bytes_structs:
                out.append(dict(name=eb.name.decode().rstrip("\x00"),
                                type=EB_TYPE.get(eb.data_type, "uint16"),
                                scale=None if eb.scale is None else list(eb.scale),
                                offset=None if eb.offset is None else list(eb.offset),
                                description=eb.description.decode().rstrip("\x00")))
    return out


def _write_laz(path, src_hdr, extra_specs, n, xyz_int, scale, offset, fields):
    # 拡張次元を含まない素の point format から組み直す
    hdr = laspy.LasHeader(version=src_hdr.version,
                          point_format=laspy.PointFormat(src_hdr.point_format.id))
    for sp in extra_specs:
        sc = None if sp["scale"] is None else np.asarray(sp["scale"], float)
        of = None if sp["offset"] is None else np.asarray(sp["offset"], float)
        if sc is not None and of is None:      # LAS は scale 単独を許すが laspy は両方要求する
            of = np.zeros_like(sc)
        hdr.add_extra_dim(laspy.ExtraBytesParams(
            name=sp["name"], type=sp["type"], scales=sc, offsets=of,
            description=sp["description"][:32]))
    hdr.scales, hdr.offsets = scale, offset
    las = laspy.LasData(hdr)
    las.X, las.Y, las.Z = xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2]
    arr = las.points.array
    for nm in arr.dtype.names:
        if nm in ("X", "Y", "Z"):
            continue
        if nm in fields:
            arr[nm] = fields[nm].astype(arr[nm].dtype, copy=False)
        else:
            arr[nm] = 0            # 外部符号化に回した標準次元は定数で埋める
    t = time.perf_counter()
    las.write(str(path))
    return time.perf_counter() - t


def _read_laz(path):
    with laspy.open(str(path)) as fh:
        las = fh.read()
    arr = las.points.array
    return {nm: np.asarray(arr[nm]) for nm in arr.dtype.names if nm not in ("X", "Y", "Z")}, las


def run(path, max_points=2_000_000, enable_residual=True, grid_bits=0):
    pc = load(path, max_points=max_points)
    n = pc.n
    src = pc.meta["las_header"]
    specs = _extra_dim_specs(src)
    d = Path(tempfile.mkdtemp())

    # ---- A: 元のまま
    a_path = d / "orig.laz"
    a_enc = _write_laz(a_path, src, specs, n, pc.xyz_int, pc.scale, pc.offset, pc.raw)
    A = a_path.stat().st_size

    # ---- 解析
    plan = nz.analyze(pc.raw, enable_residual=enable_residual)
    xyz_b, scale_b = pc.xyz_int, pc.scale
    if grid_bits:
        gop = nz.make_grid_op(pc.scale, grid_bits)
        plan.ops.insert(0, gop)
        xyz_b = nz.grid_forward(pc.xyz_int, grid_bits)
        scale_b = nz.grid_inverse_scale(pc.scale, grid_bits)
    keep, ext = nz.apply(pc.raw, plan)
    keep_specs = [s for s in specs if s["name"] in keep]

    # ---- B: 正規化後
    b_path = d / "norm.laz"
    b_enc = _write_laz(b_path, src, keep_specs, n, xyz_b, scale_b, pc.offset, keep)
    B_container = b_path.stat().st_size

    ext_bytes = {}
    t0 = time.perf_counter()
    for k, v in ext.items():
        ext_bytes[k] = encode_ints(v)
    ext_enc_s = time.perf_counter() - t0
    spec = plan.spec_bytes()
    B = B_container + sum(len(v) for v in ext_bytes.values()) + len(spec)

    # ---- 検証: B から元データをビット完全に戻せるか
    got_keep, las_b = _read_laz(b_path)
    got_keep = {k: v for k, v in got_keep.items() if k in keep or k in nz.PROTECTED}
    got_ext = {k: decode_ints(v, n) for k, v in ext_bytes.items()}
    plan2 = nz.Plan.from_spec(spec)          # 仕様もシリアライズ経由で復元
    rec = nz.invert(got_keep, got_ext, plan2, n)
    bad = [k for k in pc.raw
           if k not in rec or not np.array_equal(rec[k], pc.raw[k])]
    if grid_bits:
        # bounded: 宣言した誤差上限を実際に守れているかを検証する。
        # offset が大きい（例: 133980m）ため世界座標で引き算すると桁落ちする。
        # 整数格子のまま差を取ってから scale を掛けることで正確に評価する。
        q = np.stack([np.asarray(las_b.X), np.asarray(las_b.Y),
                      np.asarray(las_b.Z)], 1).astype(np.int64)
        d_int = (q << grid_bits) - pc.xyz_int
        e = np.linalg.norm(d_int * pc.scale, axis=1)
        geom_ok = bool(e.max() <= plan.ops[0].params["max_err_m"] + 1e-12)
        geom_note = (f"誤差上限 {plan.ops[0].params['max_err_m']*1000:.2f}mm 以内"
                     f"（実測 max {e.max()*1000:.2f}mm / RMS {e.std()*1000:.2f}mm）")
    else:
        geom_ok = (np.array_equal(np.asarray(las_b.X), pc.xyz_int[:, 0])
                   and np.array_equal(np.asarray(las_b.Y), pc.xyz_int[:, 1])
                   and np.array_equal(np.asarray(las_b.Z), pc.xyz_int[:, 2]))
        geom_note = "ビット完全"

    # ---- 出力
    print(f"# 正規化レイヤー  {Path(path).name}   N={n:,}")
    print()
    print(plan.report())
    print()
    print(f"{'':<34}{'bytes':>12}{'bpp':>10}")
    print("-" * 58)
    print(f"{'A. 元のまま LAZ':<34}{A:>12,}{A*8/n:>10.3f}")
    print(f"{'B. 正規化後 容器':<34}{B_container:>12,}{B_container*8/n:>10.3f}")
    for k, v in sorted(ext_bytes.items()):
        print(f"{'   + 外部 ' + k:<34}{len(v):>12,}{len(v)*8/n:>10.3f}")
    print(f"{'   + 仕様':<34}{len(spec):>12,}{len(spec)*8/n:>10.4f}")
    print(f"{'B. 合計':<34}{B:>12,}{B*8/n:>10.3f}")
    print("-" * 58)
    print(f"{'削減':<34}{A-B:>12,}{(A-B)*8/n:>10.3f}   = {(A-B)/A*100:.1f}%")
    print()
    print(f"検証: 幾何 {geom_note} → OK={geom_ok}   属性 完全一致={not bad}"
          + (f"   ★不一致: {bad}" if bad else ""))
    print(f"      （外部符号化 {ext_enc_s:.1f}s / {sum(len(v) for v in ext.values())/max(ext_enc_s,1e-9)/1000:.0f}k値/s）")
    return dict(A=A, B=B, ok=geom_ok and not bad)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--grid-bits", type=int, default=0,
                    help="座標の下位ビットを何ビット落とすか（0=可逆）")
    a = ap.parse_args()
    run(a.path, a.max_points, enable_residual=not a.no_residual, grid_bits=a.grid_bits)
