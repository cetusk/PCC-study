"""フィールド別ビット内訳: 幾何と属性のどちらがビットを食っているか。

ビットの内訳を分解し、属性側に回るべきかを決める材料。
各フィールドを単独で符号化した「孤立コスト」と、LASzip 実測の内訳を両方出す。
孤立コストの和 >= 実測合計 になるのが普通（相関を捨てているため）。その差自体が
「フィールド間相関にどれだけ伸びしろがあるか」の目安になる。
"""
from __future__ import annotations
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baselines import byte_split, CODECS, laszip_bits
from pcio import PointCloud


def _cost_bits(a: np.ndarray, codec="zstd19") -> dict:
    """配列単独の符号長。生 / 差分 の良い方を採る。"""
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        v = a.view(np.uint32 if a.itemsize == 4 else np.uint64)
    else:
        v = a.astype(np.int64)
        d = np.diff(v, prepend=v[:1])
        dz = ((d << 1) ^ (d >> 63)).astype(np.uint64)
        v = v.astype(np.uint64)
    n = len(a)
    out = {}
    raw, _, _ = CODECS[codec](byte_split(v))
    out["plain"] = len(raw) * 8 / n
    if a.dtype.kind != "f":
        raw2, _, _ = CODECS[codec](byte_split(dz))
        out["delta"] = len(raw2) * 8 / n
    out["best"] = min(out.values())
    out["order0_H"] = _order0_entropy(a)
    return out


def _order0_entropy(a: np.ndarray) -> float:
    """順序0エントロピー（値の出現頻度のみ）。上限の目安。"""
    if a.dtype.kind == "f":
        a = a.view(np.uint32 if a.itemsize == 4 else np.uint64)
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def analyse(pc: PointCloud, codec="zstd19") -> str:
    n = pc.n
    L = [f"N = {n:,}", ""]
    # --- 実測の内訳（LASzip）
    tot = laszip_bits(pc, keep_attrs=True)
    geo = laszip_bits(pc, keep_attrs=False)
    L.append(f"LASzip 実測  全体 {tot.bpp:7.3f} bpp  /  幾何のみ {geo.bpp:7.3f} bpp  "
             f"→ 属性ぶん {tot.bpp-geo.bpp:7.3f} bpp ({100*(tot.bpp-geo.bpp)/tot.bpp:4.1f}%)")
    L.append("")
    L.append(f"{'field':<22}{'dtype':<10}{'bits/pt':>9}{'order0 H':>10}{'raw bits':>10}  内訳")
    L.append("-" * 80)
    rows = []
    g = _cost_bits(pc.xyz_int.astype(np.int64).ravel(order="F")
                   if pc.xyz_int is not None else pc.xyz_float.ravel(order="F"), codec)
    for k, v in pc.attrs.items():
        c = _cost_bits(v, codec)
        rows.append((k, str(v.dtype), c["best"], c["order0_H"], v.dtype.itemsize * 8, c))
    rows.sort(key=lambda r: -r[2])
    attr_sum = sum(r[2] for r in rows)
    for k, dt, b, h, rb, c in rows:
        bar = "#" * int(round(40 * b / max(1e-9, rows[0][2])))
        L.append(f"{k:<22}{dt:<10}{b:9.3f}{h:10.3f}{rb:10d}  {bar}")
    L.append("-" * 80)
    L.append(f"{'属性 孤立コスト合計':<22}{'':<10}{attr_sum:9.3f}"
             f"   (LASzip 実測の属性ぶん {tot.bpp-geo.bpp:.3f} bpp / "
             f"相関で回収済み {attr_sum-(tot.bpp-geo.bpp):+.3f})")
    L.append(f"{'幾何 孤立コスト':<22}{'':<10}{g['best']*3:9.3f}   (3軸合計, 参照パイプライン)")
    return "\n".join(L)
