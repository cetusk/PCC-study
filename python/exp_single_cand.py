"""候補を 1 本に固定したときの符号化時間 — 選別をなくした場合の下限。

幾何の列だけを測る。LAZ は同じ XYZ を laspy で書き出した実測（出力ファイルの
書き込みを含む）。pccnorm の表示は 0.01 秒刻みなので、小さい入力では pccnorm 側が
0.000 と出る（LAZ は perf_counter なので刻みの制約はない）。
ファイルごとの比は回ごとに大きく動くので、中央値だけを読むこと。
"""
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
NAMES = ("幾何v3", "幾何v4W4")


def one(item):
    lab, path, kind = item
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        t0 = time.perf_counter()
        M.write_las(tmp / "geom.laz", xyz, sc, of, None, None)
        row = {"lab": lab, "n": len(xyz), "laz": time.perf_counter() - t0}
        for name in NAMES:
            r = run([PCC, "pack", str(tmp / "geom.laz"), str(tmp / "o.pcc2"), "--fast-attr",
                     "--no-fallback", "--no-verify", "--force-geom", name], ENV)
            m = re.search(r"  X\+Y\+Z\s+\S+\s+[0-9.]+ bpp\s+([0-9.]+)s", r["out"])
            row[name] = float(m.group(1)) if m else float("nan")
        return row
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()


def main():
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, INPUTS))
    res.sort(key=lambda r: r["n"])
    print(f"標本 {N} 点。幾何の列だけ。秒。")
    print(f"{'データ':<13}{'点':>8}{'LAZ':>8}" + "".join(f"{c:>10}" for c in NAMES) + f"{'v3/LAZ':>8}")
    for r in res:
        rt = r[NAMES[0]] / r["laz"] if r["laz"] > 0 else float("nan")
        print(f"{r['lab']:<13}{r['n']:>8}{r['laz']:>8.3f}"
              + "".join(f"{r[c]:>10.3f}" for c in NAMES)
              + (f"{rt:>7.1f}x" if r[NAMES[0]] > 0 else f"{'分解能下':>8}"))
    a = np.array([r[NAMES[0]] / r["laz"] for r in res if r[NAMES[0]] > 0])
    print(f"\n  表示の分解能（0.01 s）より上の {len(a)} 件で 幾何v3 / LAZ"
          f"  中央値 {np.median(a):.1f}x  幅 {a.min():.1f}〜{a.max():.1f}")


if __name__ == "__main__":
    main()
