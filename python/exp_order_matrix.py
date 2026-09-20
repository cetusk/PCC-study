"""順序感度の本測定。入力順を統制して、符号器ごとの費用を測る。

問い: 可逆符号器の費用は入力順にどれだけ依存するか。その依存は
      符号器の性質か、それとも順序を再構成する鍵の性質か。

符号器 4 つ:
  G-PCC/TMC13   多重集合を復元する。順序不変が期待値（帰無対照）
  LAZ 全列      系列を復元する。順序依存
  PCC2 幾何v3   系列を復元する。格納順の逐次予測
  PCC2 走査v1   系列を復元する。(psid, gps_time) で順序を作り直す

条件 8 つ:
  恒等 / 逆順 / Morton / 局所シャッフル w=100, 1000, 10000 / 完全ランダム 2 種
  逆順は健全性検査（対称な差分符号器なら Δ≈0）。

ブロックは 100 万点。TMC13 の sliceMaxPoints 既定 1,100,000 を下回るので
G-PCC が単一スライスになり、厳密に順序不変な対照になる。

PCC2 の値は --force-geom で候補を固定し、--no-fallback で退避路を切って測る。
退避路に落ちると検証の対象が埋め込んだ器になり、符号器を検証しなくなるため。

使い方:
    $PCCPY python/exp_order_matrix.py <出力先>
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import tmc13_bits           # noqa: E402

PCC = "./cpp/build/pccnorm"
BLOCK = 1_000_000

# 名前 | 入力 | 開始点（負なら中央付近を自動）
FILES = [
    ("red-rocks",   "data/raw/extrabytes/entwine_data_red-rocks.laz", 0),
    ("simple1_4",   "data/raw/small/simple1_4.las", 0),
    ("plane",       "data/raw/small/plane.laz", 0),
    ("vegetation",  "data/raw/small/vegetation_1_3.las", 0),
    ("fullwave",    "data/raw/small/fullwave.laz", 0),
    ("autzen_trim", "data/raw/small/autzen_trim.laz", 0),
    ("workshop",    "data/raw/extrabytes/workshop_TM_551_101.laz", -1),
    ("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz", -1),
    ("AHN3 _20",    "data/raw/ahn3/31HZ1_20.LAZ", -1),
    ("AHN4 _20",    "data/raw/ahn4/31HZ1_20.LAZ", -1),
    ("AHN4 _21",    "data/raw/ahn4/31HZ1_21.LAZ", -1),
    ("AHN5 _20",    "data/raw/ahn5/31HZ1_20.LAZ", -1),
]
FORCE = ("幾何v3", "走査v1")


def morton3(a: np.ndarray) -> np.ndarray:
    """21 bit ずつ 3 軸を交互に並べた Morton 符号。

    座標を各軸の範囲で 21 bit に正規化してから詰める。生の値をマスクすると
    範囲が 2^21 を超える軸（例: simple1_4 は X 2.25 億 / Y 9.9 億）で
    上位が黙って捨てられ、空間局所性と無関係な並びになる。
    """
    b = a - a.min(0)
    r = np.maximum(b.max(0), 1)
    a = ((b.astype(np.float64) * ((1 << 21) - 1)) / r).astype(np.int64)

    def spread(v):
        v = v.astype(np.uint64) & np.uint64((1 << 21) - 1)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
        return v
    return spread(a[:, 0]) | (spread(a[:, 1]) << np.uint64(1)) | (spread(a[:, 2]) << np.uint64(2))


def block_shuffle(n: int, w: int, rng) -> np.ndarray:
    """窓幅 w の中だけを置換する。w=n で完全ランダムに一致する。"""
    idx = np.arange(n)
    for a in range(0, n, w):
        b = min(a + w, n)
        idx[a:b] = rng.permutation(idx[a:b])
    return idx


def laz_geom_bpp(xyz: np.ndarray, scale, offset) -> tuple[float, bool, float, float]:
    """幾何のみの LAZ。属性を含めると他の 3 列（幾何のみ）と比較できない。"""
    n = len(xyz)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.laz"
        hdr = laspy.LasHeader(version="1.4", point_format=6)
        hdr.scales, hdr.offsets = scale, offset
        las = laspy.LasData(hdr)
        las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        t0 = time.perf_counter(); las.write(str(p)); enc = time.perf_counter() - t0
        b = p.stat().st_size
        t0 = time.perf_counter()
        with laspy.open(str(p)) as fh:
            back = fh.read()
        dec = time.perf_counter() - t0
        ok = bool(np.array_equal(np.asarray(back.X), xyz[:, 0]) and
                  np.array_equal(np.asarray(back.Y), xyz[:, 1]) and
                  np.array_equal(np.asarray(back.Z), xyz[:, 2]))
    return b * 8.0 / n, ok, enc, dec


def run_pcc(path: str, force: str) -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser("~/tools/laszip-install/lib") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([PCC, "pack", path, os.path.join(d, "o.pcc2"),
                            "--force-geom", force, "--fast-attr", "--no-fallback"],
                           capture_output=True, text=True, env=env)
    out = {"bpp": float("nan"), "ok": None, "det": None, "avail": True,
           "sel": None, "emb": False, "enc": float("nan"), "dec": float("nan"),
           "peak": float("nan"), "total": float("nan")}
    for ln in r.stdout.splitlines():
        if "候補にならない" in ln:
            out["avail"] = False
        m = re.match(r"^  X\+Y\+Z\s+(\S+)\s+([\d.]+) bpp", ln)
        if m:
            out["sel"] = m.group(1); out["bpp"] = float(m.group(2))
        if "全列一致" in ln:
            out["ok"] = "true" in ln
        if "決定性" in ln:
            out["det"] = "バイト一致" in ln
        if ln.startswith("中身") and "包んだ" in ln:
            out["emb"] = True
        m = re.match(r"^PCC2\s+[\d.]+ MB\s+([\d.]+) bpp", ln)
        if m:
            out["total"] = float(m.group(1))       # 容器込みの全列合計（幾何ストリームとの差が固定費）
        m = re.search(r"enc ([\d.]+)s / dec ([\d.]+)s .*ピーク ([\d.]+) GB", ln)
        if m:
            out["enc"] = float(m.group(1)); out["dec"] = float(m.group(2))
            out["peak"] = float(m.group(3)) * 1024.0
    if out["sel"] != force:
        out["avail"] = False
    return out


def main() -> None:
    dest = open(sys.argv[1], "w", encoding="utf-8") if len(sys.argv) > 1 else sys.stdout

    def say(s=""):
        print(s, file=dest, flush=True)

    say("順序感度の測定 — 入力順を統制して符号器ごとの費用を測る")
    say(f"大きいファイルはブロック {BLOCK} 点、小さいファイルは全点。")
    say("PCC2 の値は --force-geom で候補を固定し、--no-fallback で退避路を切って測る。")
    say("退避路に落ちた行は検証の対象が符号器でないので NG にする。")
    say()
    hdr = (f"{'データ':<13}{'条件':<10}{'点数':>9}{'鍵重複':>7}{'重複点':>7}"
           f"{'G-PCC':>9}{'LAZ幾何':>9}{'幾何v3':>9}{'走査v1':>9}{'全列計':>9}"
           f"{'Genc':>7}{'Gdec':>7}{'GpkMB':>8}{'Penc':>7}{'Pdec':>7}{'PpkMB':>8}{'検証':>5}")
    say(hdr)
    say("-" * len(hdr))

    done = set()
    if len(sys.argv) > 2 and Path(sys.argv[2]).is_file():
        for ln in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
            for nm, _, _ in FILES:
                if ln.startswith(nm):
                    done.add(nm)

    tmp = Path(tempfile.mkdtemp())
    for name, path, off in FILES:
        if name in done:
            continue
        if not Path(path).is_file():
            say(f"{name:<13}  入力が無い: {path}")
            continue
        with laspy.open(path) as fh:
            hdr_src = fh.header
            sc, of = list(hdr_src.scales), list(hdr_src.offsets)
            total = hdr_src.point_count
            start = max(0, total // 2 - BLOCK // 2) if off < 0 else off
            if start + BLOCK > total:
                start = 0
            pts = fh.read_points(start + min(BLOCK, total))
        pts = pts[start:start + BLOCK]
        n = len(pts)
        xyz = np.stack([np.asarray(pts["X"]), np.asarray(pts["Y"]),
                        np.asarray(pts["Z"])], 1).astype(np.int64)
        g = np.asarray(pts["gps_time"]).astype(np.float64)
        sid = np.asarray(pts["point_source_id"]).astype(np.int64)
        # 走査モデルが実際に見るのは鍵の重複率である。隣接同値率ではない。
        key = np.stack([sid.astype(np.float64), g], 1)
        dup_key = 100.0 * (1.0 - len(np.unique(key, axis=0)) / n)
        dup_pt = int(n - len(np.unique(xyz, axis=0)))   # 重複点の数（可逆判定の前提）

        rng = np.random.default_rng(20260920)
        conds = [("恒等", np.arange(n)), ("逆順", np.arange(n)[::-1].copy()),
                 ("Morton", np.argsort(morton3(xyz), kind="stable"))]
        for w in (100, 1000, 10000):
            # w が n に近いと完全ランダムと区別できず、単調性を見かけ上強める。
            if w * 10 <= n:
                conds.append((f"窓{w}", block_shuffle(n, w, rng)))
        conds.append(("ランダム1", rng.permutation(n)))
        conds.append(("ランダム2", rng.permutation(n)))

        for cname, idx in conds:
            p = tmp / "x.laz"
            h2 = laspy.LasHeader(version=hdr_src.version, point_format=hdr_src.point_format)
            for v in hdr_src.vlrs:
                tn = type(v).__name__
                # ExtraBytes の記述子は laspy が点形式から自前で書き出すので、
                # ここで複製すると VLR が二重になり追加バイトのオフセットが狂う。
                # COPC の VLR は laspy が書き戻せない（空間索引で中身に無関係）。
                if tn.startswith("ExtraBytes") or tn.startswith("Copc"):
                    continue
                try:
                    v.record_data_bytes()
                except Exception:
                    continue
                h2.vlrs.append(v)
            h2.scales, h2.offsets = hdr_src.scales, hdr_src.offsets
            las = laspy.LasData(h2)
            las.points = pts[idx].copy()
            las.write(str(p))
            lz, lzok, lz_enc, lz_dec = laz_geom_bpp(xyz[idx], sc, of)
            gp = tmc13_bits(xyz[idx])
            gb = gp.bytes * 8.0 / n if gp.bytes else float("nan")
            rs = {f: run_pcc(str(p), f) for f in FORCE}
            p.unlink()
            # 退避路に落ちた行は、検証の対象が符号器ではないので失格にする
            ver = all(r["ok"] and r["det"] and not r["emb"]
                      for r in rs.values() if r["avail"])
            ver = ver and (gp.lossless is not False)
            vals = []
            for f in FORCE:
                vals.append(rs[f]["bpp"] if rs[f]["avail"] else float("nan"))
            pk = [rs[f]["peak"] for f in FORCE]
            ver = ver and lzok
            say(f"{name:<13}{cname:<10}{n:>9}{dup_key:>6.1f}%{dup_pt:>7}"
                f"{gb:>9.3f}{lz:>9.3f}{vals[0]:>9.3f}{vals[1]:>9.3f}"
                f"{rs[FORCE[0]]['total']:>9.3f}"
                f"{gp.enc_s:>7.1f}{gp.dec_s:>7.1f}{gp.peak_mb:>8.0f}"
                f"{max(r['enc'] for r in rs.values()):>7.1f}"
                f"{max(r['dec'] for r in rs.values()):>7.1f}{max(pk):>8.0f}"
                f"{'ok' if ver else 'NG':>5}")
        say()
    if dest is not sys.stdout:
        dest.close()


if __name__ == "__main__":
    main()
