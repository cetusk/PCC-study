"""2 つの二値を同じ入力で交互に回し、時間とピークメモリを比べる（各 REP 回の最小）。

    python -u python/bench_ab.py <旧 pccnorm> <新 pccnorm>

1 回ずつの測定は ±4〜10% 揺れる（同じ二値を続けて回しても 15 件合計で 34.7〜36.2 s）。
機能の速さの費用はこれで測る。交互に回すので、機械の混み具合の揺れは両方に乗る。
"""
import os, re, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import ENV
from bench_size import INPUTS
A, B = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
REP = int(os.environ.get("BENCH_REP", "3"))
N = int(os.environ.get("BENCH_N", "2000000"))
print(f"標本 中央 {N} 点。各 {REP} 回の最小（交互に回す）。旧 = {A}\n新 = {B}")
print(f"{'データ':<13}{'旧 bpp':>9}{'新 bpp':>9}{'旧 enc':>8}{'新 enc':>8}{'旧 dec':>8}{'新 dec':>8}{'旧 MB':>7}{'新 MB':>7}")
tot = {k: 0.0 for k in ("ea", "eb", "da", "db")}
mem = []
for lab, path, kind in [i for i in INPUTS if i[2] == "las"]:
    tmp = Path(tempfile.mkdtemp())
    try:
        t = M.total_points(kind, path); n = min(N, t); st = max(0, t // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        src = tmp / "a.laz"; M.write_las(src, xyz, sc, of, pts, hdr)
        res = {A: [], B: []}
        for _ in range(REP):
            for exe in (A, B):
                r = run([exe, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], ENV)
                o = r["out"]
                e = re.search(r"enc ([0-9.]+)s / dec ([0-9.]+)s", o)
                b = re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp", o, re.M)
                if e and b: res[exe].append((float(e.group(1)), float(e.group(2)), r["peak_mb"], float(b.group(1))))
        a = res[A]; bb = res[B]
        ea, da, ma = min(x[0] for x in a), min(x[1] for x in a), min(x[2] for x in a)
        eb, db, mb = min(x[0] for x in bb), min(x[1] for x in bb), min(x[2] for x in bb)
        tot["ea"] += ea; tot["eb"] += eb; tot["da"] += da; tot["db"] += db
        mem.append((ma * 1e6 / n, mb * 1e6 / n))
        print(f"{lab:<13}{a[0][3]:>9.3f}{bb[0][3]:>9.3f}{ea:>8.2f}{eb:>8.2f}{da:>8.2f}{db:>8.2f}{ma:>7.0f}{mb:>7.0f}", flush=True)
    finally:
        for q in tmp.glob("*"): q.unlink()
        tmp.rmdir()
import numpy as np
ma = np.median([m[0] for m in mem]); mb = np.median([m[1] for m in mem])
print(f"\n合計 符号化 {tot['ea']:.2f} → {tot['eb']:.2f} s（{(tot['eb']/tot['ea']-1)*100:+.1f}%）"
      f"  復号 {tot['da']:.2f} → {tot['db']:.2f} s（{(tot['db']/tot['da']-1)*100:+.1f}%）")
print(f"1 点あたりピーク 中央値 {ma:.0f} → {mb:.0f} byte")
