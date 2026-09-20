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
指標: Δ補正 = Δ(ランダム) − Δ(逆順)。逆順はその水準での雑音床である。

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


def main() -> None:
    print("鍵の縮退を制御して掃引する")
    print("入力順は恒等に固定し、gps_time の量子化だけで鍵の重複率を作る。")
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
        print(f"{'目標':>6}{'実際':>8}{'量子化幅':>13}"
              + "".join(f"{c:>26}" for c in FORCE) + f"{'検証':>6}")
        rng = np.random.default_rng(20260920)
        orders = {"恒等": np.arange(n), "逆順": np.arange(n)[::-1].copy(),
                  "ランダム": rng.permutation(n)}
        for target in TARGETS:
            gq, q, actual = quantize_for(sid, g0, target)
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
            cell = []
            for f in FORCE:
                b, rv, rd = vals[("恒等", f)], vals[("逆順", f)], vals[("ランダム", f)]
                d_rand = 100 * (rd / b - 1)
                d_rev = 100 * (rv / b - 1)
                cell.append(f"{b:8.3f} Δ{d_rand:+6.1f} 補正{d_rand - d_rev:+6.1f}")
            print(f"{target:>5.0f}%{actual:>7.1f}%{q:>13.3e}"
                  + "".join(f"{c:>26}" for c in cell)
                  + f"{'ok' if allok else 'NG':>6}")
        print()
    print("Δ = ランダム / 恒等 の増分、補正 = Δ − 逆順の Δ（その水準の雑音床を引く）")


if __name__ == "__main__":
    main()
