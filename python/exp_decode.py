"""復号の時間を LASzip と比べる。

符号化は候補を全部試すが、**復号は選ばれた 1 本しか通らない**。
選択原理の費用は復号には掛からないはずで、そこを測る。
PCC2 は pack が出す dec（往復検証で実際に復号した時間）、
LASzip は laspy で同じファイルを読み直す実測。
"""
from __future__ import annotations
import os, re, sys, tempfile, time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
LAS = [i for i in INPUTS if i[2] == "las"]


def main():
    print(f"標本 {N} 点。全列。1 つずつ順に、各 2 回の最小。")
    print(f"{'データ':<13}{'点':>8}{'LASzip 読込':>12}{'PCC2 復号':>10}{'比':>7}")
    rat = []
    for lab, path, kind in LAS:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path)
            n = min(N, tot)
            st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            src = tmp / "a.laz"
            M.write_las(src, xyz, sc, of, pts, hdr)
            lz = []
            for _ in range(2):
                t0 = time.perf_counter()
                d = laspy.read(str(src))
                _ = np.asarray(d.X)[0]
                lz.append(time.perf_counter() - t0)
            laz = min(lz)
            v = []
            for _ in range(2):
                r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], ENV)
                m = re.search(r"dec ([0-9.]+)s", r["out"])
                if m:
                    v.append(float(m.group(1)))
            if not v:
                continue
            t = min(v)
            rat.append(t / laz)
            print(f"{lab:<13}{n:>8}{laz:>12.3f}{t:>10.2f}{t/laz:>6.1f}x")
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()
    a = np.array(rat)
    print(f"\n  PCC2 復号 / LASzip 読込  中央値 {np.median(a):.1f}x"
          f"  四分位 [{np.percentile(a,25):.1f}, {np.percentile(a,75):.1f}]"
          f"  幅 {a.min():.1f}〜{a.max():.1f}")


if __name__ == "__main__":
    main()
