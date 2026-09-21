"""復号の時間が列ごとにどう散っているかを見る。

復号は依存の波に分かれ、波の中は並列に走る。したがって復号の時間は
**各波の最長の列の和**で決まり、それ以外の列は幾らあっても効かない。
どの列が最長かが分かれば、詰める先が決まる。
"""
from __future__ import annotations
import os, re, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
ONLY = sys.argv[1:] or None
LAS = [i for i in INPUTS if i[2] == "las" and (not ONLY or i[0] in ONLY)]

for lab, path, kind in LAS:
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        src = tmp / "a.laz"
        M.write_las(src, xyz, sc, of, pts, hdr)
        env = dict(ENV); env["PCC_DPROF"] = "1"
        r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], env)
        print(f"=== {lab}  {n} 点 ===")
        for ln in r["out"].splitlines():
            if ln.startswith("  [波") or ln.startswith("      "):
                print(ln)
        m = re.search(r"dec ([0-9.]+)s", r["out"])
        if m:
            print(f"  [往復検証が測った復号] {m.group(1)}s")
        print()
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()
