"""順序感度の測定を要約する。data/work/order_matrix.txt を読む。

各データについて、恒等順を基準にした条件ごとの増分を符号器別に出し、
鍵（point_source_id, gps_time）のタイ率との関係を見る。
"""
from __future__ import annotations
import math
import re
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

SRC = Path("data/work/order_matrix.txt")
COD = ["G-PCC", "LAZ", "幾何v3", "走査v1"]


def main() -> None:
    rows = []
    for ln in SRC.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+(?: \S+)?)\s+(恒等|逆順|Morton|窓\d+|ランダム\d)\s+"
                     r"([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(ok|NG)\s*$", ln)
        if m:
            rows.append((m.group(1), m.group(2), float(m.group(3)),
                         [float(m.group(i)) for i in (4, 5, 6, 7)], m.group(8)))
    files = []
    for nm, c, t, v, ok in rows:
        if nm not in files:
            files.append(nm)
    print(f"読み込んだ行 {len(rows)}   データ {len(files)}   NG {sum(1 for r in rows if r[4] != 'ok')}")
    print()

    # 恒等を基準にした増分
    print("=== ランダム置換 / 恒等（%） ===")
    print(f"{'データ':<13}{'タイ率':>7}" + "".join(f"{c:>10}" for c in COD))
    print("-" * (20 + 10 * len(COD)))
    tie, dlt = [], {c: [] for c in COD}
    for nm in files:
        base = next((r for r in rows if r[0] == nm and r[1] == "恒等"), None)
        rnd = [r for r in rows if r[0] == nm and r[1].startswith("ランダム")]
        if not base or not rnd:
            continue
        t = base[2]
        tie.append(t)
        out = []
        for i, c in enumerate(COD):
            m = float(np.mean([r[3][i] for r in rnd]))
            d = 100 * (m / base[3][i] - 1) if base[3][i] else float("nan")
            dlt[c].append(d)
            out.append(d)
        print(f"{nm:<13}{t:>6.1f}%" + "".join(f"{x:>9.1f}%" for x in out))
    print()
    print("=== タイ率との順位相関（n = %d）===" % len(tie))
    for c in COD:
        r, p = spearmanr(tie, dlt[c])
        print(f"  {c:<8} rho = {r:+.3f}  p = {p:.4f}")
    print()
    print("=== 条件ごとの中央値（全データ、恒等比 %）===")
    conds = ["逆順", "Morton", "窓100", "窓1000", "窓10000", "ランダム1"]
    print(f"{'条件':<10}" + "".join(f"{c:>10}" for c in COD))
    for cn in conds:
        vals = []
        for i, c in enumerate(COD):
            ds = []
            for nm in files:
                b = next((r for r in rows if r[0] == nm and r[1] == "恒等"), None)
                x = next((r for r in rows if r[0] == nm and r[1] == cn), None)
                if b and x and b[3][i]:
                    ds.append(100 * (x[3][i] / b[3][i] - 1))
            vals.append(float(np.median(ds)) if ds else float("nan"))
        print(f"{cn:<10}" + "".join(f"{v:>9.1f}%" for v in vals))
    print()
    n = 1_000_000
    print(f"参考: log2(n!)/n = {math.log2(n) - math.log2(math.e):.3f} bit/点（n = {n}）")


if __name__ == "__main__":
    main()
