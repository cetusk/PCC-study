"""G-PCC に「元の順序を復元する」仕事もさせたときの大きさを測る。

G-PCC は点を多重集合として扱い、出力の並びは自分の八分木の走査順になる。
LAS/LAZ を 1 本復元する用途では、属性はファイル上の点順に並んでいるので
**元の順序を復元できなければ同じファイルにならない**。
そこで tmc3 で符号化→復号し、復号の並びから元の並びへの置換を実際に符号化して足す。

置換の符号化は 2 通り測って**短いほうを採る**（G-PCC に有利な側に倒す）:
  順位差   … 元の位置の 1 次差分を「桁数＋下位ビット」で送る
  残り順位 … まだ出ていない点の中での順位（Lehmer）を同じ形で送る
どちらも経験分布での値で、学習の費用は数えていない（G-PCC に有利）。
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

N = int(os.environ.get("BENCH_N", "2000000"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra", "USGS NY", "autzen-2023", "KITTI"]


def read_ply_xyz(p: Path) -> np.ndarray:
    """頂点の x y z だけを読む。**face の property を混ぜないこと**
    （tmc3 の出力は `element face 0` を持ち、その property list まで数えると
    1 点あたりの大きさを取り違えて読み込みが落ちる）。"""
    dt = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
          "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
          "uchar": "u1", "ushort": "u2", "short": "i2", "int16": "i2",
          "uint16": "u2", "uint8": "u1", "char": "i1", "int8": "i1"}
    with open(p, "rb") as f:
        n = 0
        fmt = "ascii"
        cur = None
        types: list[str] = []
        while True:
            ln = f.readline().decode("ascii", "replace").strip()
            if not ln:
                raise ValueError("PLY の頭が終わらない")
            w = ln.split()
            if w[0] == "format":
                fmt = w[1]
            elif w[0] == "element":
                cur = w[1]
                if cur == "vertex":
                    n = int(w[2])
            elif w[0] == "property" and cur == "vertex":
                if w[1] == "list":
                    raise ValueError("頂点に list 型がある")
                types.append(dt.get(w[1], "f4"))
            elif ln == "end_header":
                break
        if fmt == "ascii":
            a = np.loadtxt(f, max_rows=n, usecols=(0, 1, 2))
            return np.rint(a).astype(np.int64)
        rec = np.dtype([(f"c{i}", t) for i, t in enumerate(types)])
        arr = np.frombuffer(f.read(n * rec.itemsize), dtype=rec, count=n)
        return np.stack([np.rint(arr[f"c{i}"].astype(np.float64)).astype(np.int64)
                         for i in range(3)], axis=1)


def code_bits(v: np.ndarray) -> float:
    """zigzag して「桁数を経験分布で + 下位ビットは生」で数えた bit/点。"""
    z = np.where(v >= 0, v.astype(np.int64) * 2, -v.astype(np.int64) * 2 - 1)
    k = np.zeros(len(z), dtype=np.int64)
    t = (z + 1).astype(np.uint64)
    while True:
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1
        t[m] >>= np.uint64(1)
    vals, c = np.unique(k, return_counts=True)
    p = c / c.sum()
    h = float(-(p * np.log2(p)).sum())
    return h + float(k.mean())


def lehmer_bits(idx: np.ndarray) -> float:
    """まだ出ていない点の中での順位。Fenwick 木で数える。"""
    n = len(idx)
    bit = np.zeros(n + 1, dtype=np.int64)

    def add(i, v):
        i += 1
        while i <= n:
            bit[i] += v
            i += i & (-i)

    def pref(i):
        i += 1
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    for i in range(n):
        add(i, 1)
    r = np.empty(n, dtype=np.int64)
    for j, v in enumerate(idx.tolist()):
        r[j] = pref(v) - 1
        add(v, -1)
    return code_bits(r)


print(f"標本 {N} 点。**幾何のみ**。G-PCC に順序の復元もさせたときの大きさ。\n")
print(f"{'データ':<13}{'点':>8}{'PCC2':>9}{'G-PCC':>9}{'置換':>9}{'G-PCC+順序':>12}{'勝敗':>8}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tmp = Path(tempfile.mkdtemp())
    try:
        tot = M.total_points(kind, path)
        n = min(N, tot)
        st = max(0, tot // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        xyz = np.asarray(xyz).astype(np.int64)
        write_ply(tmp / "in.ply", xyz)
        run([TMC3, "--mode=0", f"--uncompressedDataPath={tmp / 'in.ply'}",
             f"--compressedStreamPath={tmp / 'g.bin'}"] + GFLAGS, ENV)
        gs = (tmp / "g.bin").stat().st_size * 8.0 / n
        run([TMC3, "--mode=1", f"--compressedStreamPath={tmp / 'g.bin'}",
             f"--reconstructedDataPath={tmp / 'out.ply'}"], ENV)
        dec = read_ply_xyz(tmp / "out.ply")
        if len(dec) != n:
            print(f"{lab:<13}{n:>8}  復号の点数が違う（{len(dec)}）— 比較できない")
            continue
        # **G-PCC は最小値へ平行移動して出す。**戻してから突き合わせる。
        dec = dec - dec.min(axis=0) + xyz.min(axis=0)
        # 復号の並び → 元の並び。3 列をそのまま辞書順に並べて突き合わせる
        # （鍵をビットで畳むと負の値や広い範囲で衝突する）。
        oo = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        od = np.lexsort((dec[:, 2], dec[:, 1], dec[:, 0]))
        if not np.array_equal(xyz[oo], dec[od]):
            print(f"{lab:<13}{n:>8}  点集合が一致しない — 比較できない")
            continue
        orig_of_dec = np.empty(n, dtype=np.int64)
        orig_of_dec[od] = oo                      # 復号の j 番目は元の何番目か
        a = code_bits(np.diff(np.concatenate(([0], orig_of_dec))))
        b = lehmer_bits(orig_of_dec) if n <= 300000 else float("inf")
        pb = min(a, b)
        r = run([PCC, "pack", str(tmp / "p.laz"), str(tmp / "o.pcc2")], ENV) if False else None
        M.write_las(tmp / "p.laz", xyz, sc, of, pts, hdr)
        rr = run([PCC, "pack", str(tmp / "p.laz"), str(tmp / "o.pcc2"),
                  "--fast-attr", "--no-fallback", "--no-verify"], ENV)
        ps = float("nan")
        for ln in rr["out"].splitlines():
            if ln.startswith("  X+Y+Z"):
                ps = float(ln.split()[2])
        tot_g = gs + pb
        print(f"{lab:<13}{n:>8}{ps:>9.3f}{gs:>9.3f}{pb:>9.3f}{tot_g:>12.3f}"
              f"{'PCC2' if ps < tot_g else 'G-PCC':>8}")
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()
