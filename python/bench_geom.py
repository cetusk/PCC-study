"""幾何のみで G-PCC と PCC2 を比べる — 同じ仕事をさせた場合の時間とメモリ。

G-PCC は幾何しか符号化しないので、全 25 列を符号化した PCC2 の時間と
並べるのは比較になっていない。入力を X/Y/Z だけにして揃える。
走査モデルは鍵（gps_time, point_source_id）を要るので、その 2 列も入れた
入力を別に作る（鍵は走査モデルへの入力であって、符号化の対象ではない）。
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, TMC3, ENV, GFLAGS, write_ply

FILES = [("AHN4 _20", "data/raw/ahn4/31HZ1_20.LAZ"),
         ("AHN3 _20", "data/raw/ahn3/31HZ1_20.LAZ"),
         ("AHN5 _20", "data/raw/ahn5/31HZ1_20.LAZ"),
         ("USGS NY", "data/raw/usgs/NY_ClintonEssex_2014.laz"),
         ("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz")]


def one(path: str, n: int, tmp: Path):
    total = M.total_points("las", path)
    start = max(0, total // 2 - n // 2) if total > n else 0
    xyz, g, sid, sc, of, pts, hdr = M.read_block("las", path, start, n)
    geom = tmp / "g.laz"
    M.write_las(geom, xyz, sc, of, None, None)
    h = laspy.LasHeader(version="1.4", point_format=6)
    h.scales, h.offsets = sc, of
    las = laspy.LasData(h)
    las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.gps_time, las.point_source_id = g, sid
    key = tmp / "k.laz"
    las.write(str(key))
    ply = tmp / "g.ply"
    write_ply(ply, xyz)
    out = {}
    r = run([TMC3, "--mode=0", f"--uncompressedDataPath={ply}",
             f"--compressedStreamPath={tmp / 'g.bin'}"] + GFLAGS, ENV)
    bpp = (tmp / "g.bin").stat().st_size * 8.0 / len(xyz)
    out["G-PCC"] = (r["sec"], r["peak_mb"], bpp)
    for lab, src, force in (("幾何v3", geom, "幾何v3"), ("走査v1", key, "走査v1")):
        r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--force-geom", force,
                 "--fast-attr", "--no-fallback", "--no-verify"], ENV)
        b = float("nan")
        for ln in r["out"].splitlines():
            if ln.startswith("  X+Y+Z") and ln.split()[1] == force:
                b = float(ln.split()[2])
        out[lab] = (r["sec"], r["peak_mb"], b)
    for p in tmp.glob("*"):
        p.unlink()
    return out, len(xyz)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    tmp = Path(tempfile.mkdtemp())
    print(f"幾何のみ、{n} 点、符号化のみ（検証と決定性の二度書きを含まない）")
    print(f"{'データ':<13}{'G-PCC':>22}{'PCC2 幾何v3':>22}{'PCC2 走査v1':>22}")
    print(f"{'':13}" + 3 * f"{'秒':>7}{'MB':>7}{'bpp':>8}")
    agg = {k: [] for k in ("G-PCC", "幾何v3", "走査v1")}
    for lab, path in FILES:
        if not Path(path).is_file():
            print(f"{lab:<13} 入力が無い")
            continue
        r, got = one(path, n, tmp)
        line = f"{lab:<13}"
        for k in ("G-PCC", "幾何v3", "走査v1"):
            s, m, b = r[k]
            agg[k].append((s, m, b))
            line += f"{s:>7.2f}{m:>7.0f}{b:>8.3f}"
        print(line)
    print("-" * 79)
    line = f"{'中央値':<13}"
    for k in ("G-PCC", "幾何v3", "走査v1"):
        a = np.array(agg[k])
        line += f"{np.median(a[:, 0]):>7.2f}{np.median(a[:, 1]):>7.0f}{np.median(a[:, 2]):>8.3f}"
    print(line)
    tmp.rmdir()


if __name__ == "__main__":
    main()
