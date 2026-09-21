"""ピーク RSS を外から測って G-PCC と比べる（幾何のみ、符号化のみ）。

LASzip は CLI が無くライブラリなので、外から測れる G-PCC を相手にする。
測定は runpeak の補助プロセスに任せる（親のメモリが子の RSS に乗るため）。
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, TMC3, ENV, GFLAGS, write_ply
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "1000000"))
SEL = sys.argv[1:] or ["AHN4 _20", "USGS NY", "autzen-2023", "red-rocks"]

print(f"標本 {N} 点。幾何のみ、符号化のみ。ピーク RSS。")
print(f"{'データ':<13}{'点':>8}{'G-PCC MB':>10}{'PCC2 MB':>9}{'比':>7}"
      f"{'G-PCC 秒':>10}{'PCC2 秒':>9}")
rm, rt = [], []
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        M.write_las(tmp / "in.laz", xyz, sc, of, pts, hdr)
        write_ply(tmp / "in.ply", xyz)
        a = run([TMC3, "--mode=0", f"--uncompressedDataPath={tmp / 'in.ply'}",
                 f"--compressedStreamPath={tmp / 'g.bin'}"] + GFLAGS, ENV)
        b = run([PCC, "pack", str(tmp / "in.laz"), str(tmp / "o.pcc2"),
                 "--fast-attr", "--no-fallback", "--no-verify"], ENV)
        rm.append(b["peak_mb"] / a["peak_mb"])
        rt.append(b["sec"] / a["sec"])
        print(f"{lab:<13}{n:>8}{a['peak_mb']:>10.0f}{b['peak_mb']:>9.0f}"
              f"{rm[-1]:>6.2f}x{a['sec']:>10.2f}{b['sec']:>9.2f}")
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()
print(f"\n  PCC2/G-PCC  ピーク RSS 中央値 {np.median(rm):.2f}x"
      f"   時間 中央値 {np.median(rt):.2f}x")
