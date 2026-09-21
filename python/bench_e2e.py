"""利用者から見た end-to-end の比較。

利用者は LAS/LAZ ファイルを 1 本渡して、小さくなったものを受け取り、
元に戻せることを期待する。その土俵で比べる。

- **G-PCC は幾何しか符号化しない。**属性を別立てにしない限り、この土俵に
  そもそも出られない。ここでは「出られない」と書く。
- **PCC2 は一部の次元を読んでいない**（bench_cover.py）。読まない次元を
  LASzip が符号化するのに使っている bit をそのまま足したものを
  「PCC2（欠けを LASzip で埋めた場合）」として見積もる。

  元の .laz の bpp           L_full   … LASzip が全次元に使った量
  pccnorm の「基準 LASzip」   L_kept   … LASzip が PCC2 の読む列に使った量
  PCC2 の bpp                P_kept
  欠けを埋めた見積り          P_kept + (L_full - L_kept)
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
        src = tmp / "all.laz"
        M.write_las(src, xyz, sc, of, pts, hdr)
        lfull = src.stat().st_size * 8.0 / n
        r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"),
                 "--no-fallback", "--no-verify"], ENV)
        lkept = float(re.search(r"基準 LASzip\s+\S+ MB\s+([0-9.]+) bpp", r["out"]).group(1))
        pkept = float(re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp", r["out"], re.M).group(1))
        rows.append((lab, n, lfull, lkept, pkept, pkept + max(0.0, lfull - lkept)))
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()

print(f"標本 {N} 点。bpp。利用者が渡すファイル 1 本を丸ごと小さくする土俵。\n")
print(f"{'データ':<13}{'LASzip':>9}{'PCC2(読む列)':>13}{'欠けの費用':>11}"
      f"{'PCC2(埋めた)':>13}{'対 LASzip':>10}")
for r in rows:
    gap = r[2] - r[3]
    d = (r[5] - r[2]) / r[2] * 100
    print(f"{r[0]:<13}{r[2]:>9.3f}{r[4]:>13.3f}{gap:>11.3f}{r[5]:>13.3f}{d:>9.1f}%")
d = np.array([(r[5] - r[2]) / r[2] * 100 for r in rows])
gapf = np.array([(r[2] - r[3]) / r[2] * 100 for r in rows])
print(f"\n  欠けを埋めた見積りで PCC2 が小さい {int((d<0).sum())}/{len(d)}"
      f"   中央値 {np.median(d):+.1f}%   幅 {d.min():+.1f}〜{d.max():+.1f}")
print(f"  欠けた次元が元の .laz に占める割合  中央値 {np.median(gapf):.1f}%"
      f"   幅 {gapf.min():.1f}〜{gapf.max():.1f}")
print("\n  G-PCC はこの土俵に出られない（幾何しか符号化しない）。")
