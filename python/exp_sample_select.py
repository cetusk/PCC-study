"""候補選択を「先頭からの連続標本」でなく「全体に散らした塊」で行った場合の A/B。

pccnorm には既に --sample-select がある（sample_frame で塊を散らして取り、
標本用の副次情報も一緒に作る）。best_stream の中の事前選別は先頭からの連続標本
なので、族の保護で候補を上乗せしている。両者を同じ入力で突き合わせる。

--sample-select を付けると内側の事前選別は門 ncols_n > PRE_SAMP * 2 で無効になる
（標本フレーム 25000 点に対し PRE_SAMP の下限は 20000）。重なるのではなく
置き換わる。

サイズは出力ファイルのバイト数で比べる。--sample-select の経路では列ごとの
bpp が標本での値になるため、列の行は使えない。bpp 3 桁では 20 万点で 25 バイトの
差が見えない。

入力は台（bench_size.py）が実際に詰めるものと同じ、鍵つきのファイルにする。
走査モデルは gps_time と point_source_id を副次情報に使うので、鍵を落とすと
この比較の意味が変わる。
"""
from __future__ import annotations
import concurrent.futures as cf
import os
import re
import sys
import tempfile
from pathlib import Path
import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
SAMP = os.environ.get("EXP_SAMPLE", "25000")


def pack(src: Path, dst: Path, extra):
    r = run([PCC, "pack", str(src), str(dst), "--fast-attr", "--no-fallback",
             "--no-verify"] + extra, ENV)
    t = re.search(r"enc ([0-9.]+)s", r["out"])
    g = re.search(r"X\+Y\+Z\s+(\S+)\s", r["out"])
    return (dst.stat().st_size if dst.exists() else -1,
            float(t.group(1)) if t else float("nan"),
            g.group(1) if g else "?")


def one(item):
    lab, path, kind = item
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        src = tmp / "geom.laz"
        M.write_las(src, xyz, sc, of, None, None)
        if kind == "las" and g is not None:
            h = laspy.LasHeader(version="1.4", point_format=6)
            h.scales, h.offsets = sc, of
            las = laspy.LasData(h)
            las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            las.gps_time, las.point_source_id = g, sid
            src = tmp / "key.laz"
            las.write(str(src))
        a = pack(src, tmp / "a.pcc2", [])
        b = pack(src, tmp / "b.pcc2", ["--sample-select", SAMP])
        return {"lab": lab, "n": len(xyz), "a": a, "b": b}
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()


def main():
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, INPUTS))
    res.sort(key=lambda r: r["n"])
    print(f"標本 {N} 点。出力のバイト数と符号化時間。散らす標本は {SAMP} 点。"
          f"「採択(散)」は標本での 1 位で、全点の勝者とは限らない。")
    print(f"{'データ':<13}{'点':>8}{'いま byte':>11}{'散標本 byte':>12}{'差':>7}"
          f"{'いま s':>8}{'散標本 s':>9}{'採択(いま)':>11}{'採択(散)':>10}")
    d, t0, t1 = [], [], []
    for r in res:
        dd = 100 * (r["b"][0] / r["a"][0] - 1)
        d.append(dd); t0.append(r["a"][1]); t1.append(r["b"][1])
        print(f"{r['lab']:<13}{r['n']:>8}{r['a'][0]:>11}{r['b'][0]:>12}{dd:>+6.1f}%"
              f"{r['a'][1]:>8.2f}{r['b'][1]:>9.2f}{r['a'][2]:>11}{r['b'][2]:>10}")
    print(f"\n  サイズ差 中央値 {np.median(d):+.2f}%  幅 {min(d):+.1f}〜{max(d):+.1f}"
          f"   時間 中央値 {np.median(t0):.2f}s → {np.median(t1):.2f}s")


if __name__ == "__main__":
    main()
