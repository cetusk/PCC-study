"""全列を詰めたときの圧縮後サイズを LASzip と比べる。

bench_size.py は幾何だけを見る（G-PCC が幾何しか符号化しないため）。
利用者が受け取るファイルは全列なので、そちらも見る。基準は LASzip。
"""
from __future__ import annotations
import concurrent.futures as cf
import os
import re
import sys
import tempfile
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
        src = tmp / "all.laz"
        M.write_las(src, xyz, sc, of, pts, hdr)          # 元の列と VLR を保つ
        r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback",
                 "--no-verify"], ENV)
        laz = re.search(r"基準 LASzip\s+\S+ MB\s+([0-9.]+) bpp", r["out"])
        p2 = re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp", r["out"], re.M)
        cols = re.search(r"([0-9]+) 点 / 列 ([0-9]+)", r["out"])
        return {"lab": lab, "n": len(xyz),
                "ncol": int(cols.group(2)) if cols else 0,
                "laz": float(laz.group(1)) if laz else float("nan"),
                "pcc2": float(p2.group(1)) if p2 else float("nan")}
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()


def main():
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, LAS))
    res.sort(key=lambda r: r["lab"])
    print(f"標本 {N} 点。全列。bpp。基準は LASzip（同じ列を詰めたもの）。")
    print(f"{'データ':<13}{'点':>8}{'列':>4}{'LASzip':>10}{'PCC2':>10}{'対 LASzip':>10}")
    d = []
    for r in res:
        v = 100 * (r["pcc2"] / r["laz"] - 1)
        d.append(v)
        print(f"{r['lab']:<13}{r['n']:>8}{r['ncol']:>4}{r['laz']:>10.3f}"
              f"{r['pcc2']:>10.3f}{v:>+9.1f}%")
    a = np.array(d)
    print(f"\n  PCC2 が LASzip より小さい {int((a<0).sum())}/{len(a)}"
          f"   中央値 {np.median(a):+.1f}%"
          f"  四分位 [{np.percentile(a,25):+.1f}, {np.percentile(a,75):+.1f}]"
          f"  幅 {a.min():+.1f}〜{a.max():+.1f}")


if __name__ == "__main__":
    main()
