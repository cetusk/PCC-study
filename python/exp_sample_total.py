"""--sample-select（散らした標本で計画し、勝者と控えだけ全点で測り直す）を
全列で測る。既定は候補を全部全点で符号化するので、そこが一番重い。"""
from __future__ import annotations
import os, re, sys, tempfile, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
SAMP = os.environ.get("EXP_SAMPLE", "25000")
LAS = [i for i in INPUTS if i[2] == "las"]


def main():
    print(f"標本 {N} 点。全列。1 つずつ順に、各 2 回の最小。散らす標本は {SAMP} 点。")
    print(f"{'データ':<13}{'LASzip':>9}{'既定 bpp':>10}{'標本 bpp':>10}{'差':>7}"
          f"{'既定 s':>8}{'標本 s':>8}")
    ds, r0, r1 = [], [], []
    for lab, path, kind in LAS:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path)
            n = min(N, tot)
            st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            src = tmp / "a.laz"
            t0 = time.perf_counter(); M.write_las(src, xyz, sc, of, pts, hdr)
            laz = time.perf_counter() - t0
            out = []
            for extra in ([], ["--sample-select", SAMP]):
                b = t = None
                for _ in range(2):
                    r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback",
                             "--no-verify"] + extra, ENV)
                    sz = (tmp / "o.pcc2").stat().st_size * 8.0 / n
                    v = float(re.search(r"enc ([0-9.]+)s", r["out"]).group(1))
                    b = sz; t = v if t is None else min(t, v)
                out.append((b, t))
            d = 100 * (out[1][0] / out[0][0] - 1)
            ds.append(d); r0.append(out[0][1] / laz); r1.append(out[1][1] / laz)
            print(f"{lab:<13}{laz:>9.3f}{out[0][0]:>10.3f}{out[1][0]:>10.3f}{d:>+6.2f}%"
                  f"{out[0][1]:>8.2f}{out[1][1]:>8.2f}")
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()
    a = np.array(ds)
    print(f"\n  サイズ 中央値 {np.median(a):+.2f}%  幅 {a.min():+.2f}〜{a.max():+.2f}%")
    for nm, v in (("既定", np.array(r0)), ("標本", np.array(r1))):
        print(f"  {nm} PCC2/LASzip 中央値 {np.median(v):.1f}x"
              f"  四分位 [{np.percentile(v,25):.1f}, {np.percentile(v,75):.1f}]")


if __name__ == "__main__":
    main()
