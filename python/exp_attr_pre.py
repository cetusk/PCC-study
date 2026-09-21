"""属性の列にも事前選別を掛けた場合の、全列サイズと符号化時間。"""
from __future__ import annotations
import concurrent.futures as cf
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
LAS = [i for i in INPUTS if i[2] == "las"]


def one(item):
    lab, path, kind = item
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        src = tmp / "a.laz"
        t0 = time.perf_counter()
        M.write_las(src, xyz, sc, of, pts, hdr)
        laz = time.perf_counter() - t0
        row = {"lab": lab, "n": len(xyz), "laz": laz}
        for key, env in (("off", {}), ("on", {"PCC_PRESELECT_ATTR": "1"})):
            b = t = None
            for _ in range(2):
                r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback",
                         "--no-verify"], dict(ENV, **env))
                mb = re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp", r["out"], re.M)
                mt = re.search(r"enc ([0-9.]+)s", r["out"])
                b = float(mb.group(1)) if mb else float("nan")
                v = float(mt.group(1)) if mt else float("nan")
                t = v if t is None else min(t, v)
            row[key] = (b, t)
        return row
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()


def main():
    with cf.ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, LAS))
    res.sort(key=lambda r: r["lab"])
    print(f"標本 {N} 点。全列。事前選別を属性にも掛けた場合。")
    print(f"{'データ':<13}{'LAZ 書出':>9}{'bpp 前':>9}{'bpp 後':>9}{'差':>7}"
          f"{'enc 前':>8}{'enc 後':>8}{'差':>7}")
    ds, dt = [], []
    for r in res:
        a, b = r["off"], r["on"]
        vs = 100 * (b[0] / a[0] - 1)
        vt = 100 * (b[1] / a[1] - 1) if a[1] > 0 else float("nan")
        ds.append(vs); dt.append(vt)
        print(f"{r['lab']:<13}{r['laz']:>9.3f}{a[0]:>9.3f}{b[0]:>9.3f}{vs:>+6.2f}%"
              f"{a[1]:>8.2f}{b[1]:>8.2f}{vt:>+6.1f}%")
    print(f"\n  サイズ 中央値 {np.median(ds):+.3f}%  幅 {min(ds):+.2f}〜{max(ds):+.2f}%")
    print(f"  時間   中央値 {np.median(dt):+.1f}%  幅 {min(dt):+.1f}〜{max(dt):+.1f}%")


if __name__ == "__main__":
    main()
