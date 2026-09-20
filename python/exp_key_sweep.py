"""鍵の縮退を 1 ファイル内で制御して掃引する。

問い: 走査モデルの順序不変性は、符号器の性質か、それとも順序を再構成する
      鍵 (point_source_id, gps_time) の一意性の性質か。

これまでは鍵の重複率を**観測共変量**として 12 ファイル間で相関を取っていた。
独立サイトの問題・ファイル固有の逆順オフセット・1 件への依存が残る。
ここでは**同一ファイル・同一ブロック・同一符号器で、鍵だけを段階的に壊す**。
x 軸が制御変数になるので、それらの問題が消え、因果を直接試せる。

操作: 入力順は恒等に固定したまま gps_time を段階的に粗く量子化し、
      鍵の重複率 0 / 10 / 25 / 50 / 75 / 100% を人工的に作る。

条件: 各水準で 恒等 / 逆順 / ランダム置換。
指標: **生の Δ = ランダム置換 / 恒等 の増分**。当初は逆順を雑音床として
      引いていたが、逆順は `stable_sort` のタイを決定的に逆向きに解く
      構造化された処置であり雑音ではない（36 節）。逆順は別条件として併記する。

ディザ条件: 量子化は鍵の単射性と走査モデルの時間解像度を同時に壊すので、
      「鍵だけを壊した」ことにならない。同じ量子化幅のまま、群内の点に
      量子化幅の 100 万分の 1 の連番を足して**単射性だけを回復**した条件を
      併せて測る。恒等の bpp が量子化前に戻るなら漂移は時間解像度の劣化であり、
      Δ(ランダム) が 0 に落ちるなら「単射性が原因」が交絡なしで示せる。

事前に決めたこと（走らせる前に書く）:
  予測 — 走査v1 の Δ補正 は重複率とともに増え、100% で幾何v3 に一致する。
  反証 — 幾何v3 は鍵を使わないので、その Δ補正 が水準間で 2% 以上動いたら
         測定系の不具合として中止する。
  plane は使わない（飽和の機構が別種の疑いがあり、別途診断が要る）。

使い方:
    $PCCPY python/exp_key_sweep.py > data/work/key_sweep.txt
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

PCC = "./cpp/build/pccnorm"
BLOCK = 1_000_000
FORCE = ("幾何v3", "走査v1")
TARGETS = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0)
FILES = [("vegetation", "data/raw/small/vegetation_1_3.las"),
         ("AHN4 _20", "data/raw/ahn4/31HZ1_20.LAZ")]


def dup_rate(sid: np.ndarray, g: np.ndarray) -> float:
    key = np.stack([sid.astype(np.float64), g], 1)
    return 100.0 * (1.0 - len(np.unique(key, axis=0)) / len(g))


def quantize_for(sid: np.ndarray, g: np.ndarray, target: float) -> tuple[np.ndarray, float, float]:
    """目標の重複率に最も近くなる量子化幅を二分探索で選ぶ。"""
    if target <= 0:
        return g.copy(), 0.0, dup_rate(sid, g)
    span = float(g.max() - g.min())
    if span <= 0:
        return g.copy(), 0.0, dup_rate(sid, g)
    lo, hi = span * 1e-12, span * 2.0
    best = (None, None, 1e9)
    for _ in range(40):
        q = (lo * hi) ** 0.5                      # 対数の中点
        gq = np.round(g / q) * q
        r = dup_rate(sid, gq)
        if abs(r - target) < best[2]:
            best = (gq, q, abs(r - target))
        if r < target:
            lo = q
        else:
            hi = q
    gq, q, _ = best
    return gq, q, dup_rate(sid, gq)


def run_pcc(path: str, force: str) -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser("~/tools/laszip-install/lib") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([PCC, "pack", path, os.path.join(d, "o.pcc2"),
                            "--force-geom", force, "--fast-attr", "--no-fallback"],
                           capture_output=True, text=True, env=env)
    out = {"bpp": float("nan"), "ok": None, "det": None, "sel": None}
    for ln in r.stdout.splitlines():
        m = re.match(r"^  X\+Y\+Z\s+(\S+)\s+([\d.]+) bpp", ln)
        if m:
            out["sel"] = m.group(1); out["bpp"] = float(m.group(2))
        if "全列一致" in ln:
            out["ok"] = "true" in ln
        if "決定性" in ln:
            out["det"] = "バイト一致" in ln
    return out


def dither(sid: np.ndarray, gq: np.ndarray, q: float) -> np.ndarray:
    """量子化幅はそのままに、鍵の単射性だけを回復する。

    同じ (psid, gq) を持つ群の中で、点に群内の連番 × eps を足す。
    eps は「1 群が量子化幅を跨がない最大」に取る（q / (最大群サイズ + 2)）。
    小さすぎると float64 の ulp に埋もれて効かない。実際、当初 q*1e-6 に
    したところ gps_time の桁（約 5e5、ulp 5.8e-11）より小さくなり、
    中間水準でまったく効いていなかった。
    """
    if q <= 0:
        return gq.copy()
    key = np.stack([sid.astype(np.float64), gq], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    order = np.lexsort((np.arange(len(inv)), inv))
    grp = inv[order]
    idx = np.arange(len(grp))
    start = np.r_[True, grp[1:] != grp[:-1]]
    rank = np.empty(len(inv), dtype=np.int64)
    rank[order] = idx - np.maximum.accumulate(np.where(start, idx, 0))
    eps = q / (rank.max() + 2)
    ulp = float(np.spacing(np.abs(gq).max()))
    if eps < ulp * 4:
        # 量子化幅が ulp に近すぎて、群を分けるだけの余地が無い
        return gq.copy()
    return gq + rank * eps


def measure(pts, gq, orders, hdr, tmp) -> tuple[dict, bool]:
    """与えた gps_time で、各順序・各符号器の bpp を測る。"""
    vals, allok = {}, True
    for cond, idx in orders.items():
        p = tmp / "x.laz"
        h2 = laspy.LasHeader(version=hdr.version, point_format=hdr.point_format)
        for v in hdr.vlrs:
            tn = type(v).__name__
            if tn.startswith("ExtraBytes") or tn.startswith("Copc"):
                continue
            try:
                v.record_data_bytes()
            except Exception:
                continue
            h2.vlrs.append(v)
        h2.scales, h2.offsets = hdr.scales, hdr.offsets
        las = laspy.LasData(h2)
        las.points = pts[idx].copy()
        las.gps_time = gq[idx]
        las.write(str(p))
        for f in FORCE:
            r = run_pcc(str(p), f)
            vals[(cond, f)] = r["bpp"]
            allok = allok and bool(r["ok"]) and bool(r["det"]) and r["sel"] == f
        p.unlink()
    return vals, allok


def main() -> None:
    print("鍵の縮退を制御して掃引する")
    print("入力順は恒等に固定し、gps_time の量子化だけで鍵の重複率を作る。")
    print("ディザ条件は同じ量子化幅のまま鍵の単射性だけを戻したもの。")
    print()
    tmp = Path(tempfile.mkdtemp())
    for label, src in FILES:
        if not Path(src).is_file():
            print(f"{label}: 入力が無い {src}")
            continue
        with laspy.open(src) as fh:
            hdr = fh.header
            total = hdr.point_count
            start = max(0, total // 2 - BLOCK // 2) if total > BLOCK else 0
            pts = fh.read_points(start + min(BLOCK, total))
        pts = pts[start:start + BLOCK]
        n = len(pts)
        g0 = np.asarray(pts["gps_time"]).astype(np.float64)
        sid = np.asarray(pts["point_source_id"]).astype(np.int64)
        print(f"### {label}  {n} 点  元の鍵の重複率 {dup_rate(sid, g0):.1f}%")
        print(f"{'目標':>6}{'条件':>8}{'重複率':>8}{'群サイズ':>10}{'量子化幅':>12}"
              f"{'幾何v3':>9}{'走査v1':>9}{'走査v1 Δ':>10}{'逆順Δ':>9}{'検証':>6}")
        rng = np.random.default_rng(20260920)
        orders = {"恒等": np.arange(n), "逆順": np.arange(n)[::-1].copy(),
                  "ランダム": rng.permutation(n)}
        for target in TARGETS:
            gq0, q, actual = quantize_for(sid, g0, target)
            variants = [("量子化", gq0, actual)]
            if target > 0:
                gd = dither(sid, gq0, q)
                variants.append(("ディザ", gd, dup_rate(sid, gd)))
            for vname, gq, rate in variants:
                vals, allok = measure(pts, gq, orders, hdr, tmp)
                b3 = vals[("恒等", "幾何v3")]
                b1 = vals[("恒等", "走査v1")]
                d = 100 * (vals[("ランダム", "走査v1")] / b1 - 1)
                dr = 100 * (vals[("逆順", "走査v1")] / b1 - 1)
                grp = 1.0 / max(1e-9, 1.0 - rate / 100.0)
                print(f"{target:>5.0f}%{vname:>8}{rate:>7.1f}%{grp:>10.1f}{q:>12.2e}"
                      f"{b3:>9.3f}{b1:>9.3f}{d:>9.1f}%{dr:>8.1f}%"
                      f"{'ok' if allok else 'NG':>6}", flush=True)
        print()
    print("Δ = ランダム置換 / 恒等 の増分。逆順Δ は同じ基準での逆順の増分（引かない）。")
    print("ディザは量子化幅を変えずに鍵の単射性だけを戻した条件。")


if __name__ == "__main__":
    main()
