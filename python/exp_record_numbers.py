"""記録（results/scan_model_fitting.md）で引用している導出値を、すべて生ログから作り直す。

24 節で「引用する数字はリポジトリ内のスクリプトで生成し出力を残す」と定めたが、
統計量と和は、その場で書いた使い捨ての python で出して保存していなかった。
記録中の 3 桁小数 183 個のうち 33 個が出所ファイルに存在しない状態になっていた。
この script の出力（data/work/record_numbers.txt）が、それらの出所になる。

使い方:
    $PCCPY python/exp_record_numbers.py > data/work/record_numbers.txt
"""
from __future__ import annotations
import math
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BATCH = Path("data/work/batch")
PRE = ("gps_time", "point_source_id", "bit_fields")


def batch_cols(log: str):
    """バッチのログから、列ごとの bpp と X+Y+Z の候補を取り出す。"""
    txt = (BATCH / (log + ".log")).read_text(encoding="utf-8", errors="replace")
    cols, cand, inblk = {}, {}, False
    total = None
    for ln in txt.splitlines():
        if re.match(r"^  X\+Y\+Z\s", ln):
            inblk = True
        elif inblk and re.match(r"^      (\S+)\s+([\d.]+) bpp", ln):
            m = re.match(r"^      (\S+)\s+([\d.]+) bpp", ln)
            cand[m.group(1)] = float(m.group(2))
            continue
        elif inblk and re.match(r"^  \S", ln):
            inblk = False
        m = re.match(r"^  (\S+)\s+\S+\s+([\d.]+) bpp", ln)
        if m:
            cols[m.group(1)] = float(m.group(2))
        m = re.match(r"^PCC2\s+[\d.]+ MB\s+([\d.]+) bpp", ln)
        if m:
            total = float(m.group(1))
    return cols, cand, total


def main() -> None:
    print("記録で引用している導出値（すべて生ログから再計算）")
    print()

    # --- 19 節: 退避した 3 件の「自前符号器のみ」の合計 ---
    print("== 19 節: 退避した 3 件の符号器のみの合計 ==")
    print(f"{'データ':<20}{'符号器のみ':>11}{'包んだ出力':>11}{'LASzip':>9}{'符号器の負け':>12}")
    for log, base in (("small_autzen_trim", 43.507), ("ahn3_31HZ1_20", 50.059),
                      ("small_vegetation", 49.845)):
        cols, _, total = batch_cols(log)
        s = sum(cols.values())
        print(f"{log:<20}{s:>11.3f}{total:>11.3f}{base:>9.3f}{100 * (s / base - 1):>11.1f}%")
    print()

    # --- 27 節: 列ごとの bpp は厳密には加法的でない ---
    print("== 27 節: 列の和と合計の差 ==")
    for log in ("small_plane", "ahn4_tile"):
        cols, _, total = batch_cols(log)
        s = sum(cols.values())
        print(f"  {log:<16} 列の和 {s:8.3f}  合計 {total:8.3f}  差 {total - s:+.3f} bpp")
    print()

    # --- 26/29 節: 幾何対幾何と、副列を課した比較 ---
    print("== 26・29 節: 幾何v3 対 走査（副列を課す場合と課さない場合）==")
    JOBS = [("autzen_trim", "small_autzen_trim"), ("plane", "small_plane"),
            ("fullwave", "small_fullwave"), ("vegetation", "small_vegetation"),
            ("workshop", "workshop_TerraScan"), ("autzen-2023", "autzen2023_LasMonkey"),
            ("AHN5 _20", "ahn5_31HZ1_20"), ("AHN3 _20", "ahn3_31HZ1_20"),
            ("AHN4 _20", "ahn4_tile"), ("AHN4 _21", "ahn4_tile21")]
    print(f"{'データ':<14}{'幾何v3':>9}{'走査最良':>10}{'副列':>8}{'走査+副列':>11}{'幾何対幾何':>12}")
    w_geom = w_sub = 0
    for nm, log in JOBS:
        cols, cand, _ = batch_cols(log)
        g3 = cand.get("幾何v3", float("nan"))
        sc = min([cand[k] for k in ("走査v1", "走査変換") if k in cand], default=float("nan"))
        pre = sum(cols.get(k, 0.0) for k in PRE)
        w_geom += sc < g3
        w_sub += (sc + pre) < g3
        print(f"{nm:<14}{g3:>9.3f}{sc:>10.3f}{pre:>8.3f}{sc + pre:>11.3f}"
              f"{'走査' if sc < g3 else '幾何v3':>12}")
    print(f"  走査が勝つ件数: 幾何対幾何 {w_geom} / 10、副列を課すと {w_sub} / 10")
    print()

    # --- 28 節: 予備試行の副列の和 ---
    print("== 28 節: 予備試行（AHN4 中間 100 万点）の副列の和 ==")
    pil = Path("data/work/order_pilot.txt").read_text(encoding="utf-8")
    vals = {}
    for key in ("gps_time", "point_source_id", "bit_fields", "走査v1", "幾何v3"):
        m = re.search(rf"^{re.escape(key)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", pil, re.M)
        if m:
            vals[key] = [float(m.group(i)) for i in (1, 2, 3, 4)]
    conds = ("元順序", "Morton", "逆順", "ランダム")
    if len(vals) == 5:
        print(f"{'条件':<10}{'走査v1':>9}{'副列':>9}{'走査+副列':>11}{'幾何v3':>9}")
        for i, c in enumerate(conds):
            pre = sum(vals[k][i] for k in PRE)
            print(f"{c:<10}{vals['走査v1'][i]:>9.3f}{pre:>9.3f}"
                  f"{vals['走査v1'][i] + pre:>11.3f}{vals['幾何v3'][i]:>9.3f}")
        p0 = sum(vals[k][0] for k in PRE); p3 = sum(vals[k][3] for k in PRE)
        s0, s3 = vals["走査v1"][0], vals["走査v1"][3]
        print(f"  Δ(ランダム/元順序)  幾何のみ {100 * (s3 / s0 - 1):+.1f}%  "
              f"副列込み {100 * ((s3 + p3) / (s0 + p0) - 1):+.1f}%")
        n = 1_000_000
        print(f"  置換の情報量 log2(n!)/n = {math.log2(n) - math.log2(math.e):.3f} bit/点 (n={n})")
        print(f"  副列の実測増分 = {(p3 - p0):+.2f} bpp")
    print()

    # --- 17・21 節: 少数点の順序から読んだ法則の検定 ---
    print("== 17・21 節: 順序から読んだ法則の検定 ==")
    # 17 節: 面外（退避）と 対幾何v3
    d17 = [("AHN4 _21", 1.93, -12.5), ("AHN4 2000万", 2.06, -2.9),
           ("autzen-2023", 2.36, -18.9), ("AHN4 _20", 2.18, 2.2),
           ("workshop", 4.76, 17.4), ("AHN5", 6.52, 1.3), ("AHN3", 19.31, 71.4)]
    r, pv = spearmanr([a for _, a, _ in d17], [b for _, _, b in d17])
    print(f"  17 節 面外 対 幾何v3 比      rho={r:+.3f}  p={pv:.4f}  n={len(d17)}")
    # 21 節: 採用率と (走査v1 − 走査変換)/走査v1
    d21 = [("AHN4 _21", .528, 18.488, 19.566), ("AHN4 2000万", .520, 21.817, 23.048),
           ("AHN4 _20", .477, 22.940, 24.133), ("AHN5", .035, 25.614, 25.540),
           ("autzen-2023", .027, 20.874, 20.849), ("AHN3", .025, 22.491, 22.436),
           ("workshop", .009, 26.702, 26.671)]
    r, pv = spearmanr([a for _, a, _, _ in d21],
                      [100 * (v1 - v5) / v1 for _, _, v1, v5 in d21])
    print(f"  21 節 採用率 対 当てはめの利得 rho={r:+.3f}  p={pv:.4f}  n={len(d21)}")
    print("  21 節の 2 群の分離:")
    for nm, a, v1, v5 in sorted(d21, key=lambda t: -t[1]):
        print(f"    採用率 {a:.3f}  当てはめの利得 {100 * (v1 - v5) / v1:+6.2f}%  {nm}")
    print()

    # --- 30 節: 順序感度の相関（すべての標本の取り方）---
    print("== 30 節: タイ率との順位相関 ==")
    rows = []
    for ln in Path("data/work/order_matrix.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+(?: \S+)?)\s+(恒等|逆順|Morton|窓\d+|ランダム\d)\s+([\d.]+)%"
                     r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(ok|NG)\s*$", ln)
        if m:
            rows.append((m.group(1), m.group(2), float(m.group(3)),
                         [float(m.group(i)) for i in (4, 5, 6, 7)]))
    files = []
    for r in rows:
        if r[0] not in files:
            files.append(r[0])
    COD = ["G-PCC", "LAZ", "幾何v3", "走査v1"]
    tie, dlt, rev = [], {c: [] for c in COD}, {c: [] for c in COD}
    for nm in files:
        b = next(r for r in rows if r[0] == nm and r[1] == "恒等")
        rn = [r for r in rows if r[0] == nm and r[1].startswith("ランダム")]
        rv = next(r for r in rows if r[0] == nm and r[1] == "逆順")
        tie.append(b[2])
        for i, c in enumerate(COD):
            dlt[c].append(100 * (float(np.mean([x[3][i] for x in rn])) / b[3][i] - 1))
            rev[c].append(100 * (rv[3][i] / b[3][i] - 1))
    a = np.array(tie)
    print(f"{'符号器':<9}{'n=12':>20}{'n=10（縮退2件除く）':>24}{'逆順補正 n=10':>20}")
    for c in COD[1:]:
        v, w = np.array(dlt[c]), np.array(rev[c])
        m = a < 100
        r1, p1 = spearmanr(a, v)
        r2, p2 = spearmanr(a[m], v[m])
        r3, p3 = spearmanr(a[m], v[m] - w[m])
        print(f"{c:<9}{f'{r1:+.3f} (p={p1:.4f})':>20}{f'{r2:+.3f} (p={p2:.4f})':>24}"
              f"{f'{r3:+.3f} (p={p3:.4f})':>20}")
    print()
    print("  逆順のファイル別の幅（この研究自身の雑音床）:")
    for c in COD[1:]:
        print(f"    {c:<8} 最小 {min(rev[c]):+.2f}%  最大 {max(rev[c]):+.2f}%")
    print()
    print("  窓幅の単調性（窓100 → 窓1000 → 窓10000 → ランダム1）:")
    wc = ["窓100", "窓1000", "窓10000", "ランダム1"]
    for i, c in enumerate(COD):
        bad = []
        for nm in files:
            v = [next((r[3][i] for r in rows if r[0] == nm and r[1] == w), None) for w in wc]
            if all(x is not None for x in v) and not all(v[j] <= v[j + 1] + 1e-9 for j in range(3)):
                bad.append(nm)
        print(f"    {c:<8} 単調なファイル {len(files) - len(bad)} / {len(files)}"
              + (f"  反例: {', '.join(bad)}" if bad else ""))
    print()
    print("  条件ごとの中央値（恒等比 %）:")
    conds2 = ["逆順", "Morton", "窓100", "窓1000", "窓10000", "ランダム1"]
    print(f"    {'条件':<10}" + "".join(f"{c:>10}" for c in COD))
    for cn in conds2:
        out = []
        for i, c in enumerate(COD):
            ds = []
            for nm in files:
                b = next(r for r in rows if r[0] == nm and r[1] == "恒等")
                x = next((r for r in rows if r[0] == nm and r[1] == cn), None)
                if x and b[3][i]:
                    ds.append(100 * (x[3][i] / b[3][i] - 1))
            out.append(float(np.median(ds)) if ds else float("nan"))
        print(f"    {cn:<10}" + "".join(f"{v:>9.1f}%" for v in out))


if __name__ == "__main__":
    main()
