"""全列を詰めたときの符号化時間を LASzip と比べる。

これまでの exp_speed.py は X+Y+Z の列だけを見ていたが、全体の時間は属性が
占める（AHN3 _20 の 100 万点で 16.4 秒のうち X+Y+Z は 0.34 秒だった）。
1 つずつ順に測る。PCC2 は自分が出す enc、LASzip は laspy の書き出しの実測。
"""
from __future__ import annotations
import os
import re
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
REP = int(os.environ.get("EXP_REP", "2"))
LAS = [i for i in INPUTS if i[2] == "las"]


def main():
    print(f"標本 {N} 点。全列。1 つずつ順に、各 {REP} 回の最小。")
    print(f"{'データ':<13}{'点':>8}{'LASzip':>9}{'PCC2':>8}{'比':>8}")
    rat = []
    for lab, path, kind in LAS:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path)
            n = min(N, tot)
            st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            src = tmp / "a.laz"
            lz = []
            for _ in range(REP):
                t0 = time.perf_counter()
                M.write_las(src, xyz, sc, of, pts, hdr)
                lz.append(time.perf_counter() - t0)
            laz = min(lz)
            v = []
            for _ in range(REP):
                r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback",
                         "--no-verify"], ENV)
                m = re.search(r"enc ([0-9.]+)s", r["out"])
                v.append(float(m.group(1)))
            t = min(v)
            rat.append(t / laz)
            print(f"{lab:<13}{len(xyz):>8}{laz:>9.3f}{t:>8.2f}{t/laz:>7.1f}x")
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()
    a = np.array(rat)
    print(f"\n  PCC2/LASzip  中央値 {np.median(a):.1f}x"
          f"  四分位 [{np.percentile(a,25):.1f}, {np.percentile(a,75):.1f}]"
          f"  幅 {a.min():.1f}〜{a.max():.1f}")


if __name__ == "__main__":
    main()
