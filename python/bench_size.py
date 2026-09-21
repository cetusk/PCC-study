"""全データで圧縮後サイズを素早く比べる — 改善ループを回すための台。

指標の優先順位は「可逆を前提に、サイズ > 速度 > メモリ」。この台はサイズを
先頭に出す。速く回すために各入力から標本を取り、ファイルをまたいで並列に走らせる。

幾何のみで比べる。G-PCC は幾何しか符号化しないので、属性を含めると比較にならない。
走査モデルは鍵（時刻・飛行線）を入力に要るので、鍵つきの入力を別に作る。

「連続比」は格納順で隣り合う 2 点の距離の中央値を点間隔で割った値。
逐次予測が効くかどうかを符号化の前に予測できる（1 前後なら効く）。
"""
from __future__ import annotations
import concurrent.futures as cf
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, TMC3, ENV, GFLAGS, write_ply

N = int(os.environ.get("BENCH_N", "200000"))
INPUTS = [
    ("AHN3 _20",    "data/raw/ahn3/31HZ1_20.LAZ", "las"),
    ("AHN4 _20",    "data/raw/ahn4/31HZ1_20.LAZ", "las"),
    ("AHN4 _21",    "data/raw/ahn4/31HZ1_21.LAZ", "las"),
    ("AHN5 _20",    "data/raw/ahn5/31HZ1_20.LAZ", "las"),
    ("USGS AK",     "data/raw/usgs/AK_Kenai_2008_000001.laz", "las"),
    ("USGS NY",     "data/raw/usgs/NY_ClintonEssex_2014.laz", "las"),
    ("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz", "las"),
    ("autzen_trim", "data/raw/small/autzen_trim.laz", "las"),
    ("workshop",    "data/raw/extrabytes/workshop_TM_551_101.laz", "las"),
    ("red-rocks",   "data/raw/extrabytes/entwine_data_red-rocks.laz", "las"),
    # extrabytes.las は extra.laz と XYZ が完全一致する（1065 点、3 列とも同値）。
    # 幾何だけを比べるこの台では同じ 1 件なので、片方だけを数える。
    ("extra",       "data/raw/extrabytes/extra.laz", "las"),
    ("simple1_4",   "data/raw/small/simple1_4.las", "las"),
    ("plane",       "data/raw/small/plane.laz", "las"),
    ("vegetation",  "data/raw/small/vegetation_1_3.las", "las"),
    ("fullwave",    "data/raw/small/fullwave.laz", "las"),
    ("KITTI",       "data/raw/kitti", "kitti"),
    ("TLS p1",      "data/raw/tls/lecturehall/lecturehall1.pose1.object1.label.csv", "tls"),
]
CODERS = ("G-PCC", "LAZ", "PCC2")   # PCC2 は候補を自由に選ばせた結果


def continuity(xyz: np.ndarray, sc) -> float:
    w = xyz * np.asarray(sc)
    bb = w.max(0) - w.min(0)
    area = bb[0] * bb[1]
    if area <= 0 or len(w) < 2:
        return float("nan")
    spacing = float(np.sqrt(area / len(w)))
    d = np.linalg.norm(np.diff(w, axis=0), axis=1)
    return float(np.median(d)) / spacing if spacing > 0 else float("nan")


def enc_sec(out: str, pat: str) -> float:
    """符号器が自分で出した符号化時間を取る。ファイル読込や別工程を含めない。"""
    m = re.search(pat, out)
    return float(m.group(1)) if m else float("nan")


def one(item) -> dict:
    lab, path, kind = item
    tmp = Path(tempfile.mkdtemp())
    try:
        total = M.total_points(kind, path)
        n = min(N, total)
        start = max(0, total // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, start, n)
        m = len(xyz)
        out = {"n": m, "cont": continuity(xyz, sc), "key": kind == "las"}
        ply = tmp / "g.ply"
        write_ply(ply, xyz)
        bs = tmp / "g.bin"
        r = run([TMC3, "--mode=0", f"--uncompressedDataPath={ply}",
                 f"--compressedStreamPath={bs}"] + GFLAGS, ENV)
        t = enc_sec(r["out"], r"Processing time \(user\):\s*([0-9.]+)")
        out["G-PCC"] = (bs.stat().st_size * 8.0 / m if bs.exists() else float("nan"),
                        t if t == t else r["sec"])
        geom = tmp / "geom.laz"
        t0 = time.perf_counter()
        M.write_las(geom, xyz, sc, of, None, None)
        out["LAZ"] = (geom.stat().st_size * 8.0 / m, time.perf_counter() - t0)
        src = geom
        if kind == "las" and g is not None:
            h = laspy.LasHeader(version="1.4", point_format=6)
            h.scales, h.offsets = sc, of
            las = laspy.LasData(h)
            las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            las.gps_time, las.point_source_id = g, sid
            src = tmp / "key.laz"
            las.write(str(src))
        # 候補を固定せず、選択原理に選ばせる（PCC2 が実際に出す値）
        r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"),
                 "--fast-attr", "--no-fallback", "--no-verify"], ENV)
        # 時間は X+Y+Z の列だけを取る。enc は鍵の列も含むので LAZ（XYZ だけの
        # 書き出し）と釣り合わない（autzen-2023 で幾何 0.05s に対し鍵つき 0.15s）。
        b, who, t = float("nan"), "?", float("nan")
        for ln in r["out"].splitlines():
            if ln.startswith("  X+Y+Z"):
                f = ln.split()
                b, who, t = float(f[2]), f[1], float(f[4].rstrip("s"))
        out["PCC2"] = (b, t)
        out["who"] = who
        return {"lab": lab, **out}
    finally:
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()


def main() -> None:
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, INPUTS))
    res.sort(key=lambda r: (np.isnan(r["cont"]), r["cont"]))
    print(f"標本 {N} 点。幾何のみ、符号化のみ。bpp が小さいほど良い。")
    print("  時間は各符号器が自分で出した符号化時間（LAZ は書き出しの実測）。"
          "ファイル読込は含めない。")
    print(f"{'データ':<13}{'点':>8}{'連続比':>7}" + "".join(f"{c:>10}" for c in CODERS)
          + f"{'採択':>10}{'対G-PCC':>9}")
    wins = 0
    rel = []
    lose_laz = 0
    for r in res:
        # LAZ との min を取ると、PCC2 が LAZ に負けている件が見えなくなる。
        # 測っているのは PCC2 なので PCC2 単独で出す。
        d = 100 * (r["PCC2"][0] / r["G-PCC"][0] - 1)
        rel.append(d)
        wins += d < 0
        lose_laz += r["PCC2"][0] > r["LAZ"][0]
        print(f"{r['lab']:<13}{r['n']:>8}{r['cont']:>7.2f}"
              + "".join(f"{r[c][0]:>10.3f}" if r[c][0] == r[c][0] else f"{'—':>10}"
                        for c in CODERS)
              + f"{r['who']:>10}{d:>+8.1f}%")
    a = np.array(rel)
    print(f"\n  PCC2 が G-PCC に勝つ {wins}/{len(res)}   LAZ に負ける {lose_laz}/{len(res)}"
          f"   対 G-PCC 中央値 {np.median(a):+.1f}%"
          f"  四分位 [{np.percentile(a,25):+.1f}, {np.percentile(a,75):+.1f}]"
          f"  幅 {a.min():+.1f}〜{a.max():+.1f}")
    for c in CODERS:
        v = np.array([r[c][0] / r["G-PCC"][0] for r in res if r[c][0] == r[c][0]])
        t = np.array([r[c][1] for r in res if r[c][1] == r[c][1]])
        print(f"    {c:<7} 対 G-PCC 中央値 {100*(np.median(v)-1):+6.1f}%"
              f"  ばらつき（四分位幅）{100*(np.percentile(v,75)-np.percentile(v,25)):5.1f}%点"
              f"  符号化 中央値 {np.median(t):.2f}s")
    # 速度は「ファイルごとの比」で見る。中央値どうしの割り算は、小さい入力が
    # 表示の分解能より下に沈むぶんだけ楽な値になる。
    rr = np.array([r["PCC2"][1] / r["LAZ"][1] for r in res
                   if r["LAZ"][1] > 0 and r["PCC2"][1] > 0])
    nres = sum(1 for r in res if r["PCC2"][1] <= 0)
    print(f"    PCC2/LAZ の比（{len(rr)} 件、pccnorm の表示が 0.00s の {nres} 件は除く）"
          f"  中央値 {np.median(rr):.1f}x  四分位 [{np.percentile(rr,25):.1f}, "
          f"{np.percentile(rr,75):.1f}]  幅 {rr.min():.1f}〜{rr.max():.1f}")
    Path("data/work").mkdir(parents=True, exist_ok=True)
    Path("data/work/bench_size.json").write_text(json.dumps(
        {r["lab"]: {c: r[c][0] for c in CODERS} for r in res}, indent=1))


if __name__ == "__main__":
    main()
