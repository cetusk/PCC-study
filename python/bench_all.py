"""サイズ・速度・メモリーを 1 回の実行でまとめて比べる。

指標の優先順位は「可逆を前提に、サイズ > 速度 > メモリ」。その順に並べる。
別々の台の数字を継ぎ合わせると条件が揃わないので、1 ファイルにつき 1 回だけ
符号化して、そこから全部の値を採る。

基準は LASzip。**生の .laz と比べてはいけない。**PCC2 が読まない次元（走査方向
や飛行線の端のビット、波形の 7 種など）が .laz には入っているので、生の大きさと
比べると PCC2 が不当に有利になる。pccnorm が出す「基準 LASzip」は、PCC2 が
読んだ列だけを書き直した LAZ なので、両側が同じ値を詰めている。

LASzip は CLI が無くライブラリなので、ピーク RSS だけは外から測れない。
メモリーは G-PCC と比べる別表（bench_mem.py）にする。
どの次元が戻らないかは bench_cover.py で数える。
"""
from __future__ import annotations
import os, re, sys, tempfile, time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
REP = int(os.environ.get("BENCH_REP", "2"))
LAS = [i for i in INPUTS if i[2] == "las"]


def main():
    rows = []
    for lab, path, kind in LAS:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path)
            n = min(N, tot)
            st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            src = tmp / "a.laz"
            we = []
            for _ in range(REP):
                t0 = time.perf_counter()
                M.write_las(src, xyz, sc, of, pts, hdr)
                we.append(time.perf_counter() - t0)
            rd = []
            for _ in range(REP):
                t0 = time.perf_counter()
                d = laspy.read(str(src))
                _ = np.asarray(d.X)[0]
                rd.append(time.perf_counter() - t0)
            enc = dec = None
            peak = 0.0
            bl = bp = float("nan")
            for _ in range(REP):
                r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], ENV)
                e = re.search(r"enc ([0-9.]+)s / dec ([0-9.]+)s", r["out"])
                if not e:
                    continue
                q = re.search(r"基準 LASzip\s+\S+ MB\s+([0-9.]+) bpp", r["out"])
                w = re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp", r["out"], re.M)
                if q and w: bl, bp = float(q.group(1)), float(w.group(1))
                enc = float(e.group(1)) if enc is None else min(enc, float(e.group(1)))
                dec = float(e.group(2)) if dec is None else min(dec, float(e.group(2)))
                peak = max(peak, r["peak_mb"])
            rows.append({
                "lab": lab, "n": n, "laz": bl, "pcc": bp,
                "wenc": min(we), "wdec": min(rd), "enc": enc, "dec": dec, "peak": peak,
            })
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()

    print(f"標本 {N} 点。全列。各 {REP} 回の最小。基準は LASzip（同じ列）。\n")
    print("== サイズ（第一指標）==")
    print(f"{'データ':<13}{'点':>8}{'LASzip bpp':>11}{'PCC2 bpp':>10}{'差':>9}")
    for r in rows:
        a, b = r["laz"], r["pcc"]
        print(f"{r['lab']:<13}{r['n']:>8}{a:>11.3f}{b:>10.3f}{(b-a)/a*100:>8.1f}%")
    d = np.array([(r["pcc"] - r["laz"]) / r["laz"] * 100 for r in rows])
    print(f"\n  PCC2 が小さい {int((d<0).sum())}/{len(d)}   中央値 {np.median(d):+.1f}%"
          f"   四分位 [{np.percentile(d,25):+.1f}, {np.percentile(d,75):+.1f}]"
          f"   幅 {d.min():+.1f}〜{d.max():+.1f}\n")

    print("== 速度（第二指標）==")
    print(f"{'データ':<13}{'LASzip 符号':>12}{'PCC2 符号':>10}{'比':>7}"
          f"{'LASzip 復号':>12}{'PCC2 復号':>10}{'比':>7}")
    for r in rows:
        re_, rd_ = r["enc"] / r["wenc"], r["dec"] / r["wdec"]
        print(f"{r['lab']:<13}{r['wenc']:>12.3f}{r['enc']:>10.3f}{re_:>6.1f}x"
              f"{r['wdec']:>12.3f}{r['dec']:>10.3f}{rd_:>6.1f}x")
    for nm, k, w in (("符号化", "enc", "wenc"), ("復号", "dec", "wdec")):
        v = np.array([r[k] / r[w] for r in rows])
        print(f"\n  {nm} PCC2/LASzip  中央値 {np.median(v):.1f}x"
              f"  四分位 [{np.percentile(v,25):.1f}, {np.percentile(v,75):.1f}]"
              f"  幅 {v.min():.1f}〜{v.max():.1f}")

    print("\n== メモリー（第三指標）==")
    print(f"{'データ':<13}{'点':>8}{'PCC2 ピーク MB':>15}{'1 点あたり byte':>16}")
    for r in rows:
        print(f"{r['lab']:<13}{r['n']:>8}{r['peak']:>15.0f}{r['peak']*1e6/r['n']:>16.0f}")
    p = np.array([r["peak"] * 1e6 / r["n"] for r in rows])
    print(f"\n  1 点あたり 中央値 {np.median(p):.0f} byte"
          f"  幅 {p.min():.0f}〜{p.max():.0f}")


if __name__ == "__main__":
    main()
