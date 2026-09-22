"""順序を捨てたら幾何がどこまで縮むかの上限を測る。

G-PCC は点を多重集合として扱い、格納順を捨てる。こちらは順序を保つので、
負けているぶんのどれだけが「順序を保つ代償」なのかを先に切り分ける。

やり方: 同じ点集合を Morton 順に並べ替えて書き直し、同じ符号器に通す。
**これは可逆ではない**（元の順序は失われる）。上限を知るためだけの測定であり、
この値を成果として報告してはいけない。順序を戻すには置換を送る必要があり、
その費用（下の「置換の下限」）を足して初めて比較できる。
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
SEL = sys.argv[1:] or ["USGS NY", "autzen-2023", "extra", "plane", "vegetation", "KITTI"]


def morton_key(q: np.ndarray) -> np.ndarray:
    """21 bit × 3 軸を交互に差し込む（int64 に収まる）。"""
    u = (q - q.min(axis=0)).astype(np.uint64)
    m = u.max() if u.size else 0
    sh = max(0, int(m).bit_length() - 21)
    u >>= np.uint64(sh)
    out = np.zeros(len(u), dtype=np.uint64)
    for b in range(21):
        for a in range(3):
            out |= ((u[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return out


def geom_bpp(path: str) -> float:
    r = run([PCC, "pack", path, path + ".pcc2", "--fast-attr", "--no-fallback",
             "--no-verify"], ENV)
    for ln in r["out"].splitlines():
        if ln.startswith("  X+Y+Z"):
            return float(ln.split()[2])
    return float("nan")


def perm_lower_bound(idx: np.ndarray) -> float:
    """置換を送る費用の下限 [bit/点]。

    **順序を戻すのに要る情報量そのもの**ではなく、順位の 1 次差分を
    経験分布で符号化したときの下限。並べ替えが恒等に近いほど小さくなる。
    """
    d = np.diff(np.concatenate(([0], idx.astype(np.int64))))
    _, c = np.unique(d, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


print(f"標本 {N} 点。**幾何のみ**。Morton 順は可逆ではない（上限を知るための測定）。\n")
print(f"{'データ':<13}{'点':>8}{'格納順':>9}{'Morton順':>10}{'差':>8}"
      f"{'置換の下限':>11}{'合計':>9}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        a = tmp / "a.laz"
        M.write_las(a, xyz, sc, of, pts, hdr)
        b0 = geom_bpp(str(a))

        k = morton_key(xyz)
        o = np.argsort(k, kind="stable")
        b = tmp / "b.laz"
        M.write_las(b, xyz[o], sc, of, pts[o] if pts is not None else None, hdr)
        b1 = geom_bpp(str(b))
        # Morton 順から元の順に戻すための順位
        inv = np.empty(n, dtype=np.int64); inv[o] = np.arange(n)
        pb = perm_lower_bound(inv)
        print(f"{lab:<13}{n:>8}{b0:>9.3f}{b1:>10.3f}{b1-b0:>8.3f}{pb:>11.3f}{b1+pb:>9.3f}")
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()
