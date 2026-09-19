"""Phase 1 レポート生成器: 1ファイル投げれば「ビット内訳」が全部出る。

usage: $PCCPY python/run_report.py <file> [--max-points N] [--kmax 10] [--out results/x.md]
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from pcio import load
from baselines import laszip_bits, tmc13_bits, ref_geometry_bits, Result
from lsbsweep import sweep, marginal_table
from fieldshare import analyse as field_analyse


def order_redundancy(pc, sample=None) -> str:
    """点の順序が持つ冗長 (survey 攻め筋③ / Q2)。

    log2(N!)/N は「集合として送れば節約できる」理論上限。ただし取得順そのものが
    予測可能（走査順）なら、実際の符号長は既にそれを回収している。
    そこで理論値と、並べ替えたときの実測差を並べて見る。
    """
    n = pc.n
    theo = (np.log2(np.arange(1, min(n, 1) + 1)).sum() if n < 2 else
            float(n * np.log2(n / np.e) + 0.5 * np.log2(2 * np.pi * n)) / n)
    a = pc.xyz_int if pc.xyz_int is not None else None
    L = [f"理論上限 log2(N!)/N        = {theo:8.3f} bits/点  "
         f"（集合として符号化すれば原理上不要な量）"]
    if a is not None:
        o = ref_geometry_bits(a, "original").bpp
        m = ref_geometry_bits(a, "morton").bpp
        L.append(f"参照PL: 取得順のまま         = {o:8.3f} bpp")
        L.append(f"参照PL: Morton 並べ替え後    = {m:8.3f} bpp   "
                 f"({'並べ替えで改善 ' if m < o else '★並べ替えで悪化 '}{o-m:+.3f} bpp)")
        t_o = tmc13_bits(a).bpp
        L.append(f"G-PCC (順序非依存・集合符号化) = {t_o:8.3f} bpp")
        if m > o:
            L.append("  → 取得順（走査線順）は Morton より予測しやすい。"
                     "「順序は捨ててよい冗長」という素朴な前提はこのデータでは成立しない。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    pc = load(a.path, max_points=a.max_points)
    S = []
    def P(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True); S.append(s)

    P(f"# ビット内訳レポート  —  {Path(a.path).name}")
    P(f"\n生成 {time.strftime('%Y-%m-%d %H:%M')}   ")
    P("\n## 0. データ概要\n```")
    P(pc.summary())
    P("```")

    if pc.xyz_int is None:
        P("\n（float 原本のため整数格子解析は quantize 後に実施）")
        return

    P("\n## 1. ベースライン（5軸同時計測）\n```")
    P(f"{'codec':<28}{'size':>12}{'bpp':>12}{'enc':>11}{'dec':>11}{'peak':>10}")
    for r in [Result("raw int32 xyz", pc.n, pc.n*12, 0, 0, 0, True),
              laszip_bits(pc, keep_attrs=True),
              laszip_bits(pc, keep_attrs=False),
              tmc13_bits(pc.xyz_int),
              ref_geometry_bits(pc.xyz_int, "original")]:
        P(r.row())
    P("```")

    P("\n## 2. フィールド別ビット内訳（幾何 vs 属性）\n```")
    P(field_analyse(pc))
    P("```")

    P("\n## 3. 点の順序が持つ冗長\n```")
    P(order_redundancy(pc))
    P("```")

    P("\n## 4. LSB スイープ（ノイズ床はどこか）\n```")
    rows = sweep(pc, ks=range(0, a.kmax + 1), verbose=True)
    P("")
    for c in ("laszip_geom", "tmc13"):
        P(f"--- {c} ---")
        P(marginal_table(rows, c))
        P("")
    P("```")
    P(f"\n_経過 {time.time()-t0:.1f}s_")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(S), encoding="utf-8")
        json.dump(rows, open(str(a.out) + ".sweep.json", "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
