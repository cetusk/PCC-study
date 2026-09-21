"""サイズと速度の取引を、選べる設定ごとに測って並べる。

既定は候補を全部全点で符号化して最短を採る。これが一番小さいが一番遅い。
--fast-attr は属性の候補を {delta, ctx} に絞る。
PCC_PRESELECT_ATTR=1 は属性にも標本での事前選別を掛ける。
"""
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
LAS = [i for i in INPUTS if i[2] == "las"]
CFG = [("既定", [], {}),
       ("属性に事前選別", [], {"PCC_PRESELECT_ATTR": "1"}),
       ("--fast-attr", ["--fast-attr"], {})]


def main():
    print(f"標本 {N} 点。全列。1 つずつ順に、各 2 回の最小。基準は LASzip。")
    rows = []
    for lab, path, kind in LAS:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path)
            n = min(N, tot)
            st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            src = tmp / "a.laz"
            t0 = time.perf_counter(); M.write_las(src, xyz, sc, of, pts, hdr)
            laz_t = time.perf_counter() - t0
            laz_b = src.stat().st_size * 8.0 / n
            row = {"lab": lab, "laz": (laz_b, laz_t)}
            for nm, extra, env in CFG:
                t = None
                for _ in range(2):
                    r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback",
                             "--no-verify"] + extra, dict(ENV, **env))
                    v = float(re.search(r"enc ([0-9.]+)s", r["out"]).group(1))
                    t = v if t is None else min(t, v)
                row[nm] = ((tmp / "o.pcc2").stat().st_size * 8.0 / n, t)
            rows.append(row)
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()
    print(f"\n{'設定':<16}{'対 LASzip 中央値':>18}{'小さい件数':>11}{'PCC2/LASzip 時間':>18}{'四分位':>18}")
    for nm, _, _ in CFG:
        sz = np.array([100 * (r[nm][0] / r["laz"][0] - 1) for r in rows])
        tt = np.array([r[nm][1] / r["laz"][1] for r in rows])
        print(f"{nm:<16}{np.median(sz):>+17.1f}%{int((sz<0).sum()):>8}/{len(sz)}"
              f"{np.median(tt):>17.1f}x  [{np.percentile(tt,25):.1f}, {np.percentile(tt,75):.1f}]")


if __name__ == "__main__":
    main()
