"""符号化の時間が列ごとにどう散っているかを見る。"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
ONLY = sys.argv[1:] or None
SEL = [i for i in INPUTS if not ONLY or i[0] in ONLY]

for lab, path, kind in SEL:
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        src = tmp / "a.laz"
        M.write_las(src, xyz, sc, of, pts, hdr)
        r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], ENV)
        print(f"=== {lab}  {n} 点  全体 {r['sec']:.2f}s ===")
        print(r["out"].rstrip())
        print()
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()
