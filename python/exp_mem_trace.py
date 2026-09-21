"""符号化のどの段でメモリが積み上がるかを見る（--mem-trace）。

**ru_maxrss は exec をまたいで引き継がれる**ので、太った親から起動すると
子のピークに親の分が乗る。pccnorm 側で /proc/self/statm を 3 ms 刻みに
取った値を使う。
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "2000000"))
LAB = os.environ.get("MEM_LAB", "AHN4 _20")
EXTRA = sys.argv[1:]

path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == LAB][0]
tmp = Path(tempfile.mkdtemp())
tot = M.total_points(kind, path)
n = min(N, tot)
st = max(0, tot // 2 - n // 2)
xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
M.write_las(tmp / "a.laz", xyz, sc, of, pts, hdr)
r = subprocess.run([PCC, "pack", str(tmp / "a.laz"), str(tmp / "o.pcc2"),
                    "--no-fallback", "--no-verify", "--mem-trace"] + EXTRA,
                   capture_output=True, text=True, env=ENV)
print(f"{LAB}  {n} 点  全列")
for l in r.stderr.splitlines():
    if "メモリ" in l:
        print(l)
for l in r.stdout.splitlines():
    if l.startswith("5 軸"):
        print(" ", l)
for q in tmp.glob("*"):
    q.unlink()
tmp.rmdir()
