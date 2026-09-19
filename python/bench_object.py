"""オブジェクトスキャン（PLY）に対する正規化レイヤーの評価。

レンジスキャナの出力は、走査軸が固定刻みの格子に乗っていることがある。
その軸は数百段階しかないのに float32 32 bit で格納されている。
検出して整数に戻し、ビット完全に復元できることを確認したうえで効果を測る。
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import zstandard as zstd
import normalize as nz
import pccfile
from pcio import load
from baselines import byte_split


def _f32_lossless_bytes(col: np.ndarray) -> bytes:
    u = np.ascontiguousarray(col.astype(np.float32)).view(np.uint32).astype(np.int64)
    d = np.diff(u, prepend=np.int64(0))
    z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
    return zstd.ZstdCompressor(level=19).compress(byte_split(z))


def run(path, max_points=None):
    p = Path(path)
    pc = load(p, max_points=max_points)
    xyz = pc.xyz_float.astype(np.float32)
    n = len(xyz)

    ops, streams, spec_ops = [], {}, []
    raw_axis_bytes, new_axis_bytes = {}, {}
    for i, nm in enumerate("xyz"):
        col = np.ascontiguousarray(xyz[:, i])
        raw_axis_bytes[nm] = len(_f32_lossless_bytes(col))
        op = nz.detect_palette(nm, col, max_levels=8192)
        if op is not None:
            ops.append(op)
            streams[nm] = nz.palette_forward(col, op.params["values"])
            spec_ops.append({"kind": "palette", "target": nm,
                             "values": op.params["values"], "dtype": "float32"})
        else:
            # 格子が無い軸は float32 のビットをそのまま整数列として運ぶ
            streams[nm] = col.view(np.uint32).astype(np.int64)
            spec_ops.append({"kind": "raw_f32", "target": nm})

    d = Path(tempfile.mkdtemp())
    out = d / (p.stem + ".pcc")
    B = pccfile.write(out, {"n": n, "ops": spec_ops}, streams)

    # 検証
    spec2, back = pccfile.read(out, n)
    rec = np.empty((n, 3), np.float32)
    for i, o in enumerate(spec2["ops"]):
        nmi = o["target"]
        if o["kind"] == "palette":
            rec[:, i] = nz.palette_inverse(back[nmi], o["values"], np.float32)
        else:
            rec[:, i] = back[nmi].astype(np.uint32).view(np.float32)
    ok = np.array_equal(rec.view(np.uint32), xyz.view(np.uint32))

    A_file = p.stat().st_size
    A_raw = n * 12
    A_zstd = sum(raw_axis_bytes.values())
    print(f"# {p.name}   N={n:,}")
    for o in ops:
        print(f"  検出: {o.describe()}   {o.saved_note}")
    if not ops:
        print("  検出: なし（どの軸も格子に乗っていない）")
    print()
    print(f"  {'':<40}{'bytes':>12}{'bpp':>10}")
    print("  " + "-" * 62)
    print(f"  {'元ファイル（配布形式）':<40}{A_file:>12,}{A_file*8/n:>10.3f}")
    print(f"  {'float32 xyz 無圧縮':<40}{A_raw:>12,}{A_raw*8/n:>10.3f}")
    print(f"  {'float32 xyz 可逆圧縮（軸別 差分+zstd）':<40}{A_zstd:>12,}{A_zstd*8/n:>10.3f}")
    print(f"  {'正規化 + PCC1（仕様込み）':<40}{B:>12,}{B*8/n:>10.3f}")
    print("  " + "-" * 62)
    print(f"  {'可逆圧縮比での削減':<40}{A_zstd-B:>12,}"
          f"{(A_zstd-B)*8/n:>10.3f}   = {(A_zstd-B)/A_zstd*100:.1f}%")
    print(f"\n  検証: float32 ビット完全復元 = {ok}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    for q in a.paths:
        run(q)
        print()
