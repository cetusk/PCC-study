"""幾何のみ・符号化のみの時間とピーク RSS を、点数を変えて測る（論文 order.tex 用）。

論文の手順どおり**候補を 1 本に固定**する。既定では記号版の対も候補に出るため、
`--force-geom 幾何v3` でも報告されるのが 幾何v3記 になる。PCC_FSYM_CAND=0 で
対を作らないようにすると、指定した候補がそのまま報告される。

測定は runpeak の補助プロセスに任せる（親のメモリが子の RSS に乗るため）。
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, TMC3, ENV, GFLAGS, write_ply
from bench_size import INPUTS

LAB = os.environ.get("LEAN_LAB", "AHN4 _20")
NS = [int(x) for x in os.environ.get("LEAN_NS", "500000,1000000,2000000,4000000").split(",")]
CANDS = os.environ.get("LEAN_CANDS", "幾何v3,走査v1").split(",")

path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == LAB][0]
print(f"{LAB}、幾何のみ、符号化のみ。候補は 1 本に固定（PCC_FSYM_CAND=0）。")
print(f"{'点数':>9}{'G-PCC 秒':>10}{'G-PCC MB':>10}"
      + "".join(f"{c + ' 秒':>10}{c + ' MB':>10}" for c in CANDS)
      + "   bpp")
for n0 in NS:
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(n0, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        M.write_las(tmp / "geom.laz", xyz, sc, of, None, None)
        write_ply(tmp / "in.ply", xyz)
        a = run([TMC3, "--mode=0", f"--uncompressedDataPath={tmp / 'in.ply'}",
                 f"--compressedStreamPath={tmp / 'g.bin'}"] + GFLAGS, ENV)
        gb = (tmp / "g.bin").stat().st_size * 8.0 / n
        cells, bpps = [f"{a['sec']:>10.2f}{a['peak_mb']:>10.0f}"], [f"G-PCC {gb:.3f}"]
        env = dict(ENV); env["PCC_FSYM_CAND"] = "0"
        for c in CANDS:
            r = run([PCC, "pack", str(tmp / "geom.laz"), str(tmp / "o.pcc2"), "--fast-attr",
                     "--no-fallback", "--no-verify", "--force-geom", c], env)
            b = [l for l in r["out"].splitlines() if l.startswith("  X+Y+Z")]
            nm = b[0].split()[1] if b else "?"
            cells.append(f"{r['sec']:>10.2f}{r['peak_mb']:>10.0f}")
            bpps.append(f"{nm} {b[0].split()[2] if b else '?'}")
        print(f"{n:>9}" + "".join(cells) + "   " + " / ".join(bpps))
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()
