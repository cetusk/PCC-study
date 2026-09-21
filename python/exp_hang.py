"""同じ入力を何度も詰めて、止まるものが無いかを見る。

待ち行列の取りこぼしのように、並列の噛み合わせでまれに起きる停止は
1 回走らせても出ない。実行ファイルを 2 つ取って、回数で比べる。
"""
from __future__ import annotations
import os, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "1000000"))
REP = int(os.environ.get("HANG_REP", "20"))
LIM = float(os.environ.get("HANG_LIMIT", "60"))
LAB = os.environ.get("HANG_LAB", "AHN3 _20")
EXES = [e for e in os.environ.get("HANG_EXES", PCC).split(",") if e]

path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == LAB][0]
tmp = Path(tempfile.mkdtemp())
tot = M.total_points(kind, path)
n = min(N, tot)
st = max(0, tot // 2 - n // 2)
xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
M.write_las(tmp / "a.laz", xyz, sc, of, pts, hdr)
print(f"{LAB} {n} 点を {REP} 回ずつ。1 回 {LIM:.0f} 秒で打ち切り。")
for exe in EXES:
    hang = 0
    worst = 0.0
    for i in range(REP):
        t0 = time.perf_counter()
        try:
            subprocess.run([exe, "pack", str(tmp / "a.laz"), str(tmp / "o.pcc2"),
                            "--no-fallback", "--no-verify"],
                           capture_output=True, timeout=LIM, env=ENV)
        except subprocess.TimeoutExpired:
            hang += 1
        worst = max(worst, time.perf_counter() - t0)
    print(f"  {exe.split('/')[-3]:<10} 止まった {hang}/{REP}  最長 {worst:.1f}s")
for q in tmp.glob("*"):
    q.unlink()
tmp.rmdir()
