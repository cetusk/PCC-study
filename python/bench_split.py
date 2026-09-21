"""圧縮率の優位が幾何から来ているのか属性から来ているのかを分ける。

同じ入力について、全列と幾何のみを両方測り、差を属性ぶんとする。
容器の頭（ヘッダや VLR）は両側に同じだけ乗るので、差を取るとほぼ消える
（20 万点なら 0.01 bpp 程度）。**厳密な分解ではなく、内訳の目安である。**
"""
from __future__ import annotations
import os, re, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
LAS = [i for i in INPUTS if i[2] == "las"]

rows = []
for lab, path, kind in LAS:
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        M.write_las(tmp / "all.laz", xyz, sc, of, pts, hdr)
        r = run([PCC, "pack", str(tmp / "all.laz"), str(tmp / "o.pcc2"),
                 "--no-fallback", "--no-verify"], ENV)
        lz_all = float(re.search(r"基準 LASzip\s+\S+ MB\s+([0-9.]+) bpp", r["out"]).group(1))
        p2_all = float(re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp", r["out"], re.M).group(1))
        p2_geo = float([l for l in r["out"].splitlines()
                        if l.startswith("  X+Y+Z")][0].split()[2])
        lz_geo = M.laz_geom_bpp(xyz, sc, of)[0]
        rows.append((lab, n, lz_geo, p2_geo, lz_all - lz_geo, p2_all - p2_geo,
                     lz_all, p2_all))
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()

print(f"標本 {N} 点。bpp。幾何と属性に分けた内訳（容器の頭のぶんだけ目安）。\n")
print(f"{'データ':<13}{'幾何 LASzip':>12}{'幾何 PCC2':>10}{'差':>8}"
      f"{'属性 LASzip':>12}{'属性 PCC2':>10}{'差':>8}{'全体の差':>10}")
for r in rows:
    dg = (r[3] - r[2]) / r[2] * 100
    da = (r[5] - r[4]) / r[4] * 100 if r[4] > 0 else float("nan")
    dt = (r[7] - r[6]) / r[6] * 100
    print(f"{r[0]:<13}{r[2]:>12.3f}{r[3]:>10.3f}{dg:>7.1f}%"
          f"{r[4]:>12.3f}{r[5]:>10.3f}{da:>7.1f}%{dt:>9.1f}%")
g = np.array([(r[3] - r[2]) / r[2] * 100 for r in rows])
a = np.array([(r[5] - r[4]) / r[4] * 100 for r in rows if r[4] > 0])
t = np.array([(r[7] - r[6]) / r[6] * 100 for r in rows])
print(f"\n  幾何   中央値 {np.median(g):+.1f}%   PCC2 が小さい {int((g<0).sum())}/{len(g)}")
print(f"  属性   中央値 {np.median(a):+.1f}%   PCC2 が小さい {int((a<0).sum())}/{len(a)}")
print(f"  全体   中央値 {np.median(t):+.1f}%   PCC2 が小さい {int((t<0).sum())}/{len(t)}")
