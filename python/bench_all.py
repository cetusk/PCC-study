"""5 軸で全符号器を比べる — bpp / 符号化 / 復号 / ピーク / 可逆性。

同じ仕事をさせる。G-PCC は幾何しか符号化しないので、入力は X/Y/Z だけにする。
走査モデルは鍵（時刻・飛行線）を入力として要るので、その 2 列を足した入力を
別に作る（鍵は走査モデルへの入力であって符号化の対象ではない）。

計測は runpeak の補助プロセスに任せる。親からの fork で測ると、親のメモリが
子の RSS に乗って符号器の値にならない。
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

PY = sys.executable
LAZIO = str(Path(__file__).parent / "lazio.py")
FILES = [("AHN4 _20", "data/raw/ahn4/31HZ1_20.LAZ"),
         ("AHN3 _20", "data/raw/ahn3/31HZ1_20.LAZ"),
         ("AHN5 _20", "data/raw/ahn5/31HZ1_20.LAZ"),
         ("USGS NY", "data/raw/usgs/NY_ClintonEssex_2014.laz"),
         ("USGS AK", "data/raw/usgs/AK_Kenai_2008_000001.laz"),
         ("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz"),
         ("workshop", "data/raw/extrabytes/workshop_TM_551_101.laz")]


def one(path: str, n: int, tmp: Path) -> dict:
    total = M.total_points("las", path)
    start = max(0, total // 2 - n // 2) if total > n else 0
    xyz, g, sid, sc, of, pts, hdr = M.read_block("las", path, start, n)
    m = len(xyz)
    out = {}

    # --- G-PCC（幾何のみ）
    ply = tmp / "g.ply"
    write_ply(ply, xyz)
    bs = tmp / "g.bin"
    e = run([TMC3, "--mode=0", f"--uncompressedDataPath={ply}",
             f"--compressedStreamPath={bs}"] + GFLAGS, ENV)
    d = run([TMC3, "--mode=1", f"--compressedStreamPath={bs}",
             f"--reconstructedDataPath={tmp / 'rec.ply'}", "--outputBinaryPly=1"], ENV)
    out["G-PCC"] = (bs.stat().st_size * 8.0 / m, e["sec"], d["sec"],
                    e["peak_mb"], d["peak_mb"])

    # --- LAZ（幾何のみ）
    npy = tmp / "a.npy"
    np.save(npy, xyz)
    lz = tmp / "a.laz"
    e = run([PY, LAZIO, "enc", str(npy), str(lz)], ENV)
    d = run([PY, LAZIO, "dec", str(lz), str(tmp / "b.npy")], ENV)
    out["LAZ"] = (lz.stat().st_size * 8.0 / m, e["sec"], d["sec"],
                  e["peak_mb"], d["peak_mb"])

    # --- PCC2
    geom = tmp / "geom.laz"
    M.write_las(geom, xyz, sc, of, None, None)
    h = laspy.LasHeader(version="1.4", point_format=6)
    h.scales, h.offsets = sc, of
    las = laspy.LasData(h)
    las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.gps_time, las.point_source_id = g, sid
    key = tmp / "key.laz"
    las.write(str(key))
    for lab, src, force in (("幾何v3", geom, "幾何v3"), ("走査v1", key, "走査v1")):
        pc2 = tmp / "o.pcc2"
        e = run([PCC, "pack", str(src), str(pc2), "--force-geom", force,
                 "--fast-attr", "--no-fallback", "--no-verify"], ENV)
        bpp = float("nan")
        for ln in e["out"].splitlines():
            if ln.startswith("  X+Y+Z") and ln.split()[1] == force:
                bpp = float(ln.split()[2])
        d = run([PCC, "unpack", str(pc2), str(tmp / "back.laz")], ENV)
        out[lab] = (bpp, e["sec"], d["sec"], e["peak_mb"], d["peak_mb"])
    for p in tmp.glob("*"):
        p.unlink()
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    tmp = Path(tempfile.mkdtemp())
    keys = ("G-PCC", "LAZ", "幾何v3", "走査v1")
    agg = {k: [] for k in keys}
    print(f"幾何のみ、{n} 点。符号化は検証と決定性の二度書きを含まない。")
    print(f"{'データ':<13}{'符号器':<9}{'bpp':>9}{'enc 秒':>9}{'dec 秒':>9}"
          f"{'enc MB':>9}{'dec MB':>9}")
    for lab, path in FILES:
        if not Path(path).is_file():
            print(f"{lab:<13} 入力が無い")
            continue
        r = one(path, n, tmp)
        for k in keys:
            v = r[k]
            agg[k].append(v)
            print(f"{lab if k == keys[0] else '':<13}{k:<9}"
                  f"{v[0]:>9.3f}{v[1]:>9.2f}{v[2]:>9.2f}{v[3]:>9.0f}{v[4]:>9.0f}")
    print("-" * 68)
    print(f"{'中央値':<13}{'':<9}{'bpp':>9}{'enc 秒':>9}{'dec 秒':>9}{'enc MB':>9}{'dec MB':>9}")
    base = np.median(np.array(agg["G-PCC"]), axis=0)
    for k in keys:
        a = np.median(np.array(agg[k]), axis=0)
        print(f"{'':<13}{k:<9}" + "".join(
            f"{a[i]:>9.3f}" if i == 0 else f"{a[i]:>9.2f}" if i < 3 else f"{a[i]:>9.0f}"
            for i in range(5)))
    print()
    print("G-PCC を 100 としたとき（小さいほど良い）")
    print(f"{'':<13}{'':<9}{'bpp':>9}{'enc':>9}{'dec':>9}{'enc MB':>9}{'dec MB':>9}")
    for k in keys[1:]:
        a = np.median(np.array(agg[k]), axis=0)
        print(f"{'':<13}{k:<9}" + "".join(f"{100 * a[i] / base[i]:>9.0f}" for i in range(5)))
    tmp.rmdir()


if __name__ == "__main__":
    main()
