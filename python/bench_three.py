"""G-PCC・LASzip・PCC2 を同じ土俵で比べる。

G-PCC は幾何しか符号化しないので、三者が同じ土俵に立てるのは**幾何のみ**である。
サイズ・符号化時間・ピーク RSS をまとめて出す。

LASzip は CLI が無くライブラリなので、書き出しだけを行う補助プロセス
（_lazmem.py）を立て、**import だけして終わる空実行との差**を増分として測る。
G-PCC と PCC2 はそのまま外から測れる。
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, TMC3, ENV, GFLAGS, write_ply
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
SEL = sys.argv[1:] or ["AHN3 _20", "AHN4 _20", "USGS NY", "autzen-2023",
                       "workshop", "red-rocks"]
HELP = str(Path(__file__).parent / "_lazmem.py")
PY_ = sys.executable

base = run([PY_, HELP, "base"], ENV)["peak_mb"]
print(f"標本 {N} 点。**幾何のみ**（G-PCC は幾何しか符号化しない）。符号化のみ。")
print(f"空の補助プロセス（numpy と laspy を import しただけ）は {base:.0f} MB。"
      f" LAZ の RSS はこれを引いた増分。\n")
print(f"{'データ':<13}{'点':>8}"
      f"{'G-PCC':>8}{'LAZ':>8}{'PCC2':>8}"
      f"{'G-PCC 秒':>9}{'LAZ 秒':>8}{'PCC2 秒':>8}"
      f"{'G-PCC MB':>10}{'LAZ MB':>8}{'PCC2 MB':>9}")
rows = []
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        np.save(tmp / "xyz.npy", xyz); np.save(tmp / "sc.npy", np.asarray(sc))
        np.save(tmp / "of.npy", np.asarray(of))
        write_ply(tmp / "in.ply", xyz)
        M.write_las(tmp / "in.laz", xyz, sc, of, pts, hdr)

        a = run([TMC3, "--mode=0", f"--uncompressedDataPath={tmp / 'in.ply'}",
                 f"--compressedStreamPath={tmp / 'g.bin'}"] + GFLAGS, ENV)
        gs = (tmp / "g.bin").stat().st_size * 8.0 / n

        l = run([PY_, HELP, str(tmp / "xyz.npy"), str(tmp / "sc.npy"),
                 str(tmp / "of.npy"), str(tmp / "geom.laz")], ENV)
        ls = (tmp / "geom.laz").stat().st_size * 8.0 / n

        p = run([PCC, "pack", str(tmp / "in.laz"), str(tmp / "o.pcc2"),
                 "--fast-attr", "--no-fallback", "--no-verify"], ENV)
        ps = float("nan")
        for ln in p["out"].splitlines():
            if ln.startswith("  X+Y+Z"):
                ps = float(ln.split()[2])
        rows.append((lab, n, gs, ls, ps, a["sec"], l["sec"], p["sec"],
                     a["peak_mb"], max(0.0, l["peak_mb"] - base), p["peak_mb"]))
        r = rows[-1]
        print(f"{r[0]:<13}{r[1]:>8}{r[2]:>8.3f}{r[3]:>8.3f}{r[4]:>8.3f}"
              f"{r[5]:>9.2f}{r[6]:>8.2f}{r[7]:>8.2f}"
              f"{r[8]:>10.0f}{r[9]:>8.0f}{r[10]:>9.0f}")
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()

a = np.array([[r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]] for r in rows])
nm = ["bpp", "秒", "ピーク MB"]
print()
print("  中央値。比は**ファイルごとに取ってから**中央値を出す"
      "（中央値どうしを割ると別の量になる）。")
for k, lab in enumerate(nm):
    g, l, p = a[:, k * 3], a[:, k * 3 + 1], a[:, k * 3 + 2]
    rg, rl = np.median(p / g), np.median(p / l)
    print(f"  {lab:<10} G-PCC {np.median(g):8.2f}   LASzip {np.median(l):8.2f}   "
          f"PCC2 {np.median(p):8.2f}"
          f"    PCC2/G-PCC {rg:5.2f}x  PCC2/LASzip {rl:6.2f}x")
print()
for k, lab in enumerate(nm):
    g, l, p = a[:, k * 3], a[:, k * 3 + 1], a[:, k * 3 + 2]
    for nm2, v in (("PCC2/G-PCC", p / g), ("PCC2/LASzip", p / l)):
        print(f"  {lab:<10} {nm2:<12} 中央値 {np.median(v):5.2f}x"
              f"  四分位 [{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]"
              f"  幅 {v.min():.2f}〜{v.max():.2f}")
