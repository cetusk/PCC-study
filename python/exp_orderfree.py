"""点の順序を捨ててよいとしたら何が変わるかを測る（既存の検討とは独立）。

**この実験は既存の符号器を一切変えない。**入力を Morton 順に並べ替えた複製を作り、
同じ符号器に通すだけである。並べ替えた出力は元のファイルには戻らないので、
ここで出る数字は**いまの PCC2 の性能ではない**。別の製品の見積りである。

測るもの:
  元の順序  … いまの PCC2（入力ファイルを再現する）
  Morton順  … 順序を捨ててよいとしたときの、同じ符号器の値
  基準      … LASzip（順序を保つ。並べ替えた側と直接は比べられない）

幾何と属性を分けて出す。**属性は順序を捨てると得をする可能性がある**
（空間的に近い点が隣り合うので、空間予測と参照残差が当たりやすくなる）。
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

N = int(os.environ.get("BENCH_N", "2000000"))
SEL = sys.argv[1:] or [i[0] for i in INPUTS if i[2] == "las"]


def morton_key(q: np.ndarray) -> np.ndarray:
    u = (q - q.min(axis=0)).astype(np.uint64)
    m = int(u.max()) if u.size else 0
    sh = max(0, m.bit_length() - 21)
    u = u >> np.uint64(sh)
    out = np.zeros(len(u), dtype=np.uint64)
    for b in range(21):
        for a in range(3):
            out |= ((u[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return out


def measure(path: Path) -> tuple[float, float, float]:
    """(基準 LASzip, 全列 PCC2, 幾何 PCC2) を返す。"""
    r = run([PCC, "pack", str(path), str(path) + ".pcc2", "--no-verify"], ENV)
    q = re.search(r"基準 LASzip\s+\S+ MB\s+([0-9.]+) bpp", r["out"])
    w = re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp", r["out"], re.M)
    g = re.search(r"^  X\+Y\+Z\s+\S+\s+([0-9.]+) bpp", r["out"], re.M)
    return (float(q.group(1)) if q else float("nan"),
            float(w.group(1)) if w else float("nan"),
            float(g.group(1)) if g else float("nan"))


print(f"標本 {N} 点。**並べ替えた側は元のファイルに戻らない。**\n")
print(f"{'データ':<13}{'基準LASzip':>11}{'元の順序':>10}{'Morton順':>10}{'差':>8}"
      f"{'幾何(元)':>10}{'幾何(M)':>10}{'差':>8}")
rows = []
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g0, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        xyz = np.asarray(xyz)
        a = tmp / "a.laz"
        M.write_las(a, xyz, sc, of, pts, hdr)
        lz, full_o, geo_o = measure(a)
        o = np.argsort(morton_key(xyz), kind="stable")
        b = tmp / "b.laz"
        M.write_las(b, xyz[o], sc, of, pts[o] if pts is not None else None, hdr)
        _, full_m, geo_m = measure(b)
        rows.append((lab, lz, full_o, full_m, geo_o, geo_m))
        print(f"{lab:<13}{lz:>11.3f}{full_o:>10.3f}{full_m:>10.3f}"
              f"{100*(full_m-full_o)/full_o:>7.1f}%"
              f"{geo_o:>10.3f}{geo_m:>10.3f}{100*(geo_m-geo_o)/geo_o:>7.1f}%")
    finally:
        for q2 in tmp.glob("*"):
            q2.unlink()
        tmp.rmdir()

if rows:
    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    print()
    print(f"  全列  Morton 順が小さいファイル {int((a[:,1]<a[:,0]).sum())}/{len(rows)}"
          f"   比の中央値 {np.median(a[:,1]/a[:,0]):.4f}")
    print(f"  幾何  Morton 順が小さいファイル {int((a[:,3]<a[:,2]).sum())}/{len(rows)}"
          f"   比の中央値 {np.median(a[:,3]/a[:,2]):.4f}")
