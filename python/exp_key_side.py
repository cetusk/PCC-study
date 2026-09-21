"""走査モデルが使う鍵（gps_time・point_source_id）を計上した場合の幾何 bpp。

台は las 入力に鍵つきのファイルを詰め、X+Y+Z の行だけを取る。走査モデルは
鍵を副次情報として使うので、その費用は幾何の bpp に入っていない。G-PCC と LAZ
は XYZ だけなので、鍵を落とした場合と並べて影響の大きさを出す。
"""
from __future__ import annotations
import concurrent.futures as cf
import re
import sys
import tempfile
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = 200000


def pick(src: Path, tmp: Path):
    r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--fast-attr",
             "--no-fallback", "--no-verify"], ENV)
    m = re.search(r"  X\+Y\+Z\s+(\S+)\s+([0-9.]+) bpp", r["out"])
    ks = [re.search(r"  " + c + r"\s+\S+\s+([0-9.]+) bpp", r["out"])
          for c in ("gps_time", "point_source_id")]
    key = sum(float(k.group(1)) for k in ks if k)
    return (m.group(1), float(m.group(2)), key) if m else ("?", float("nan"), key)


def one(item):
    lab, path, kind = item
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        geom = tmp / "geom.laz"
        M.write_las(geom, xyz, sc, of, None, None)
        a = pick(geom, tmp)
        b = a
        if kind == "las" and g is not None:
            h = laspy.LasHeader(version="1.4", point_format=6)
            h.scales, h.offsets = sc, of
            las = laspy.LasData(h)
            las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            las.gps_time, las.point_source_id = g, sid
            k = tmp / "key.laz"
            las.write(str(k))
            b = pick(k, tmp)
        return {"lab": lab, "n": len(xyz), "nokey": a, "key": b}
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()


def main():
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, INPUTS))
    res.sort(key=lambda r: r["n"])
    print(f"標本 {N} 点。X+Y+Z の bpp。鍵つきは台が実際に詰める構成。")
    print(f"{'データ':<13}{'点':>8}{'鍵なし':>9}{'採択':>11}{'鍵つき':>9}{'採択':>11}"
          f"{'鍵の bpp':>10}{'鍵で得た分':>10}")
    for r in res:
        d = 100 * (r["key"][1] / r["nokey"][1] - 1)
        print(f"{r['lab']:<13}{r['n']:>8}{r['nokey'][1]:>9.3f}{r['nokey'][0]:>11}"
              f"{r['key'][1]:>9.3f}{r['key'][0]:>11}{r['key'][2]:>10.3f}{d:>+9.1f}%")


if __name__ == "__main__":
    main()
