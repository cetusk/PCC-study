"""順序の寄与を測れるかどうかを、先に小さく確かめる予備試行。

仮説: 走査系の符号器（走査v1 / 走査変換）は入力の置換に対してほぼ不変である。
`build_scan_ctx` が符号化の直前に (point_source_id, gps_time) で順序を作り直すため。
一方 幾何v0〜v3 は格納順の逐次予測なので置換で崩れる。

これが正しければ、Δ_order を「PCC2 の最良候補」で定義することはできない。
置換すると最小が必ず走査系に移り、頭打ちになるからである。

条件:
  元順序    そのまま
  Morton    空間充填曲線順。順序非依存の符号器が選べる現実的な並び
  逆順      局所差分の大きさは不変で符号だけ反転。対称な符号器なら Δ≈0
  ランダム  完全置換（種を固定）

使い方:
    $PCCPY python/exp_order_pilot.py [開始点] [点数]
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import tmc13_bits           # noqa: E402

SRC = "data/raw/ahn4/31HZ1_20.LAZ"
PCC = "./cpp/build/pccnorm"
PRE = ("gps_time", "point_source_id", "bit_fields")
SHOW = ("raw64", "range", "delta", "幾何v0", "幾何v1", "幾何v2", "幾何v3", "走査v1", "走査変換")


def morton3(a: np.ndarray) -> np.ndarray:
    """21 bit ずつ 3 軸を交互に並べた Morton 符号。"""
    def spread(v):
        v = v.astype(np.uint64) & np.uint64((1 << 21) - 1)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
        return v
    b = a - a.min(0)
    return spread(b[:, 0]) | (spread(b[:, 1]) << np.uint64(1)) | (spread(b[:, 2]) << np.uint64(2))


def run_pcc(path: str) -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser("~/tools/laszip-install/lib") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([PCC, "pack", path, os.path.join(d, "o.pcc2"), "--trace"],
                           capture_output=True, text=True, env=env)
    cand, pre, sel, tot, emb, inblk = {}, {}, None, None, False, False
    for ln in r.stdout.splitlines():
        if re.match(r"^  X\+Y\+Z\s", ln):
            sel = ln.split()[1]; inblk = True; continue
        if inblk:
            m = re.match(r"^      (\S+)\s+([\d.]+) bpp", ln)
            if m:
                cand[m.group(1)] = float(m.group(2)); continue
            if re.match(r"^  \S", ln):
                inblk = False
        m = re.match(r"^  (\S+)\s+\S+\s+([\d.]+) bpp", ln)
        if m and m.group(1) in PRE:
            pre[m.group(1)] = float(m.group(2))
        m = re.match(r"^PCC2\s+[\d.]+ MB\s+([\d.]+) bpp", ln)
        if m:
            tot = float(m.group(1))
        if ln.startswith("中身") and "包んだ" in ln:
            emb = True
    if sel is None:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:])
    return {"cand": cand, "pre": pre, "sel": sel, "tot": tot, "emb": emb}


def main() -> None:
    off = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000

    with laspy.open(SRC) as fh:
        hdr = fh.header
        pts = fh.read_points(off + cnt)
    pts = pts[off:off + cnt]
    n = len(pts)
    xyz = np.stack([np.asarray(pts["X"]), np.asarray(pts["Y"]),
                    np.asarray(pts["Z"])], 1).astype(np.int64)

    rng = np.random.default_rng(12345)
    orders = {
        "元順序":   np.arange(n),
        "Morton":   np.argsort(morton3(xyz), kind="stable"),
        "逆順":     np.arange(n)[::-1].copy(),
        "ランダム": rng.permutation(n),
    }

    print(f"入力        {SRC}  点 {off}〜{off + cnt}（{n} 点）")
    print()
    tmp = Path(tempfile.mkdtemp())
    res = {}
    for nm, idx in orders.items():
        p = tmp / f"{nm}.laz"
        h2 = laspy.LasHeader(version=hdr.version, point_format=hdr.point_format)
        for v in hdr.vlrs:
            h2.vlrs.append(v)
        h2.scales, h2.offsets = hdr.scales, hdr.offsets
        las = laspy.LasData(h2)
        las.points = pts[idx].copy()
        las.write(str(p))
        g = tmc13_bits(xyz[idx])
        res[nm] = {"pcc": run_pcc(str(p)), "gpcc": g.bytes * 8.0 / n if g.bytes else float("nan"),
                   "gok": g.lossless, "laz": p.stat().st_size * 8.0 / n}
        print(f"{nm} 済", flush=True)
        p.unlink()

    base = res["元順序"]
    print()
    print(f"{'候補':<12}" + "".join(f"{k:>12}" for k in orders) + f"{'ランダム/元':>12}")
    print("-" * (12 + 12 * len(orders) + 12))
    for c in SHOW:
        if c not in base["pcc"]["cand"]:
            continue
        vs = [res[k]["pcc"]["cand"].get(c, float("nan")) for k in orders]
        d = 100 * (vs[-1] / vs[0] - 1) if vs[0] else float("nan")
        print(f"{c:<12}" + "".join(f"{v:>12.3f}" for v in vs) + f"{d:>11.1f}%")
    print()
    for c in PRE:
        vs = [res[k]["pcc"]["pre"].get(c, float("nan")) for k in orders]
        d = 100 * (vs[-1] / vs[0] - 1) if vs[0] else float("nan")
        print(f"{c:<12}" + "".join(f"{v:>12.3f}" for v in vs) + f"{d:>11.1f}%")
    print()
    for lab, key in (("G-PCC", "gpcc"), ("LAZ 全列", "laz")):
        vs = [res[k][key] for k in orders]
        d = 100 * (vs[-1] / vs[0] - 1) if vs[0] else float("nan")
        print(f"{lab:<12}" + "".join(f"{v:>12.3f}" for v in vs) + f"{d:>11.1f}%")
    print()
    print("採択 / 全列 / 退避:")
    for k in orders:
        r = res[k]["pcc"]
        print(f"  {k:<10} 採択 {str(r['sel']):<10} 全列 {r['tot']}  "
              f"{'元の器を包んだ' if r['emb'] else '自前の符号器'}  "
              f"G-PCC 可逆 {res[k]['gok']}")


if __name__ == "__main__":
    main()
