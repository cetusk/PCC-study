"""競合の無い実時間で PCC2 と LAZ の符号化時間を比べる。

台（bench_size.py）は 6 並列で走るので実時間が競合を受ける。ここは 1 つずつ
順に測る。**入力は台と同じもの**にする（las 入力には gps_time と
point_source_id を入れる）。鍵を落とすと gps_time が定数になり、走査モデルが
掃引を 1 本しか作れなくなって、かえって遅くなる。

PCC2 は X+Y+Z の列の時間、LAZ は同じ XYZ を laspy で書き出した実測。
pccnorm の表示は 0.01 秒刻みなので、小さい入力では分解能が足りない。
"""
from __future__ import annotations
import os
import re
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
REP = int(os.environ.get("EXP_REP", "3"))


def main():
    print(f"標本 {N} 点。1 つずつ順に、各 {REP} 回の最小。台と同じ入力（鍵つき）。")
    print(f"{'データ':<13}{'点':>8}{'LAZ':>8}{'PCC2':>7}{'比':>7}")
    rat = []
    for lab, path, kind in INPUTS:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path)
            n = min(N, tot)
            st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            geom = tmp / "g.laz"
            t0 = time.perf_counter()
            M.write_las(geom, xyz, sc, of, None, None)
            laz = time.perf_counter() - t0
            src = geom
            if kind == "las" and g is not None:
                h = laspy.LasHeader(version="1.4", point_format=6)
                h.scales, h.offsets = sc, of
                las = laspy.LasData(h)
                las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                las.gps_time, las.point_source_id = g, sid
                src = tmp / "k.laz"
                las.write(str(src))
            v = []
            for _ in range(REP):
                r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--fast-attr",
                         "--no-fallback", "--no-verify"], ENV)
                m = re.search(r"  X\+Y\+Z\s+\S+\s+[0-9.]+ bpp\s+([0-9.]+)s", r["out"])
                v.append(float(m.group(1)))
            t = min(v)
            rat.append(t / laz)
            print(f"{lab:<13}{len(xyz):>8}{laz:>8.3f}{t:>7.2f}{t/laz:>6.1f}x")
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()
    a = np.array(rat)
    print(f"\n  PCC2/LAZ  中央値 {np.median(a):.1f}x"
          f"  四分位 [{np.percentile(a,25):.1f}, {np.percentile(a,75):.1f}]"
          f"  幅 {a.min():.1f}〜{a.max():.1f}")


if __name__ == "__main__":
    main()
