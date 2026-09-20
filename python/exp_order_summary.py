"""順序感度の測定を要約する。data/work/order_matrix.txt を読む。

読み取りは ordermatrix_io に任せる（形式を知る場所は 1 つにする）。

30 節が自分で挙げた欠陥に対応する:
  * 標本数 1 で分散推定が無い → 大きいファイルは 3 ブロックあるので
    ファイル内のばらつきを併記する
  * 逆順を雑音の下限に使っていた → 逆順は構造のある処置であって雑音ではない。
    種だけが違うランダム1 と ランダム2 の差を雑音の下限にする
  * 中央値が散らばりを隠す → 中央値には必ず四分位範囲を付ける
  * タイ率は鍵の重複率ではない → 鍵の重複率そのものを使う
  * 相関はブロック単位では独立でない → 独立なのはファイル単位
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from ordermatrix_io import read_matrix, check

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/work/order_matrix.txt")
COD = [("gpcc", "G-PCC"), ("laz", "LAZ"), ("geom3", "幾何v3"), ("scan1", "走査v1")]
CONDS = ["逆順", "Morton", "窓100", "窓1000", "窓10000", "ランダム1", "ランダム2"]


def fname(block: str) -> str:
    return block.split("#")[0].strip()


def delta(rows: list[dict], block: str, cond: str, key: str) -> float | None:
    """恒等順を基準にした増分 [%]。"""
    b = next((r for r in rows if r["name"] == block and r["cond"] == "恒等"), None)
    x = next((r for r in rows if r["name"] == block and r["cond"] == cond), None)
    if not b or not x:
        return None
    # nan は真なので `not b[key]` では落ちない。鍵を持たない入力（KITTI/TLS）は
    # 走査列が nan なので、ここで落とさないと中央値も相関も全部 nan になる。
    import math
    if not b[key] or math.isnan(b[key]) or math.isnan(x[key]):
        return None
    return 100.0 * (x[key] / b[key] - 1.0)


def med_iqr(v: list[float]) -> str:
    if not v:
        return "     —    "
    a = np.asarray(v, float)
    q1, q3 = np.percentile(a, [25, 75])
    return f"{np.median(a):+6.1f} [{q1:+.1f},{q3:+.1f}]"


def main() -> None:
    rows = read_matrix(str(SRC))
    print(check(rows))
    blocks, files = [], []
    for r in rows:
        if r["name"] not in blocks:
            blocks.append(r["name"])
        if fname(r["name"]) not in files:
            files.append(fname(r["name"]))
    print(f"  ブロック {len(blocks)} / 独立ファイル {len(files)}")
    print()

    print("=== 雑音の下限 — 2 つの対照 [%点] ===")
    print("  (a) 種だけが違う 2 つのランダム置換の差。同一処置内の種ゆらぎだけを測る。")
    print("  (b) 逆順条件。Δ=0 が理想の、構造のある帰無処置。局所性を壊さない並べ替えで")
    print("      どれだけ動くかを測るので、こちらが実際に効く下限である。")
    print(f"  {'符号器':<8}{'(a) 中央値':>12}{'(a) 最大':>10}{'(b) 最小':>10}{'(b) 最大':>10}{'(b) |Δ|>(a)最大':>16}")
    floors = {}
    for key, lab in COD:
        d = [abs(u - v) for b in blocks
             if (u := delta(rows, b, "ランダム1", key)) is not None
             and (v := delta(rows, b, "ランダム2", key)) is not None]
        rv = [x for b in blocks if (x := delta(rows, b, "逆順", key)) is not None]
        a = np.asarray(d, float) if d else np.zeros(1)
        rr = np.asarray(rv, float) if rv else np.zeros(1)
        floors[key] = max(a.max(), np.abs(rr).max())
        over = int((np.abs(rr) > a.max()).sum())
        print(f"  {lab:<8}{np.median(a):>12.3f}{a.max():>10.3f}"
              f"{rr.min():>+10.3f}{rr.max():>+10.3f}{f'{over}/{len(rr)}':>16}")
    print("  実効下限（(a) 最大と |(b)| 最大の大きいほう）: "
          + " / ".join(f"{lab} {floors[key]:.3f}" for key, lab in COD))
    print()

    print("=== ランダム置換 / 恒等（ファイルごと、%）===")
    print("  括弧内はファイル内のブロック間の幅（最小〜最大）。1 ブロックのファイルは —")
    print(f"{'データ':<13}{'鍵重複':>7}  " + "".join(f"{l:>22}" for _, l in COD))
    print("-" * (22 + 22 * len(COD)))
    dup, dlt = [], {k: [] for k, _ in COD}
    for f in files:
        bs = [b for b in blocks if fname(b) == f]
        # ファイル内で鍵重複率は大きく振れる（AHN4 _21 は 14.6〜45.0%）。
        # 先頭ブロックは系統的に低めに出るので中央値を使う。
        # dup と Δ を別々に中央値にすると、どのブロックにも存在しない組になる
        # （7 ファイル中 3 件でそうなっていた）。鍵重複が中央値のブロックを選び、
        # そのブロックの Δ を対にする。
        dks = [next(r["dup_key"] for r in rows if r["name"] == b) for b in bs]
        rep_i = int(np.argsort(dks)[len(bs) // 2])
        rep_b = bs[rep_i]
        dk = dks[rep_i]
        dup.append(dk)
        cells = []
        for key, _ in COD:
            per = []
            for b in bs:
                v = [delta(rows, b, c, key) for c in ("ランダム1", "ランダム2")]
                v = [x for x in v if x is not None]
                if v:
                    per.append(float(np.mean(v)))
            if not per:
                cells.append(f"{'—':>22}"); dlt[key].append(float("nan")); continue
            rv = [delta(rows, rep_b, c, key) for c in ("ランダム1", "ランダム2")]
            m = float(np.mean([x for x in rv if x is not None]))
            dlt[key].append(m)
            rng = f"({min(per):+.1f}〜{max(per):+.1f})" if len(per) > 1 else "(—)"
            cells.append(f"{m:>+9.1f}% {rng:>11}")
        dkl = f"{dk:.1f}" if len(dks) == 1 else f"{dk:.1f}({min(dks):.0f}〜{max(dks):.0f})"
        print(f"{f:<13}{dkl:>13}%  " + "".join(cells))
    print()

    print(f"=== 鍵の重複率との順位相関（独立ファイル単位、鍵重複はファイル内中央値、n = {len(files)}）===")
    dup_a = np.asarray(dup, float)
    for label, keep in (("全 12 件", dup_a < 101), ("鍵重複 100% の 2 件を除く", dup_a < 100)):
        print(f"  [{label}]")
        for key, lab in COD:
            v = np.asarray(dlt[key], float)
            ok = (~np.isnan(v)) & keep
            if np.ptp(v[ok]) == 0 or np.ptp(dup_a[ok]) == 0:
                print(f"    {lab:<8} 定義されない（片方が定数）  (n={int(ok.sum())})")
                continue
            r, p = spearmanr(dup_a[ok], v[ok])
            star = "  *p<0.05" if p < 0.05 else ""
            print(f"    {lab:<8} rho = {r:+.3f}  p = {p:.4f}  p×9 = {min(p * 9, 1):.3f}"
                  f"  (n={int(ok.sum())}){star}")

    bdup, bdlt = [], {k: [] for k, _ in COD}
    for b in blocks:
        bdup.append(next(r["dup_key"] for r in rows if r["name"] == b))
        for key, _ in COD:
            v = [delta(rows, b, c, key) for c in ("ランダム1", "ランダム2")]
            v = [x for x in v if x is not None]
            bdlt[key].append(float(np.mean(v)) if v else float("nan"))
    print(f"  [ブロック単位 n = {len(blocks)}（同一ファイルの 3 ブロックは独立でない。参考値）]")
    ba = np.asarray(bdup, float)
    for key, lab in COD:
        v = np.asarray(bdlt[key], float)
        # 鍵を持たない入力は dup_key も nan。Δ 側だけ見ると nan が残って rho が nan になる。
        ok = (~np.isnan(v)) & (~np.isnan(ba))
        if np.ptp(v[ok]) == 0:
            print(f"    {lab:<8} 定義されない（片方が定数）  (n={int(ok.sum())})")
            continue
        r, _ = spearmanr(ba[ok], v[ok])
        print(f"    {lab:<8} rho = {r:+.3f}  (n={int(ok.sum())}) — p 値は出さない"
              " （同一ファイルの 3 ブロックが独立でなく、検定の前提を満たさない）")
    print()
    print("  検定は 3 符号器 × 3 単位で 9 本ある。p×9 はボンフェローニ補正後の上限。")
    print()
    print("  [非縮退 10 件から 1 件ずつ抜いたとき（走査v1）]")
    v = np.asarray(dlt["scan1"], float)
    keep = dup_a < 100
    fa, va = dup_a[keep], v[keep]
    names = [f for f, k in zip(files, keep) if k]
    r0, p0 = spearmanr(fa, va)
    print(f"    全 10 件  rho = {r0:+.3f}  p = {p0:.4f}")
    for i, nm in enumerate(names):
        m = np.ones(len(names), bool); m[i] = False
        rr, pp = spearmanr(fa[m], va[m])
        mark = "  ← 有意でなくなる" if pp >= 0.05 else ""
        print(f"    −{nm:<12} rho = {rr:+.3f}  p = {pp:.4f}{mark}")
    print()


    print("=== 条件ごとの増分（全ブロック、恒等比 %）中央値 [第1四分位, 第3四分位] ===")
    print(f"{'条件':<10}" + "".join(f"{l:>22}" for _, l in COD))
    for c in CONDS:
        cells = []
        for key, _ in COD:
            d = [x for b in blocks if (x := delta(rows, b, c, key)) is not None]
            cells.append(f"{med_iqr(d):>22}")
        print(f"{c:<10}" + "".join(cells))
    print()
    print("  窓10000 は小さい 4 ブロック（1〜2.8 万点）では作れないので n が 4 少ない。")
    print("=== 窓幅に対する単調性（ブロックごと、窓100 < 窓1000 < 窓10000 < ランダム）===")
    for key, lab in COD:
        mono = tot = 0
        for b in blocks:
            v = [delta(rows, b, c, key) for c in ("窓100", "窓1000", "窓10000", "ランダム1")]
            if any(x is None for x in v):
                continue
            tot += 1
            mono += all(v[i] < v[i + 1] for i in range(3))
        note = "（値が定数なので厳密な単調増加は成立しない）" if lab == "G-PCC" else ""
        print(f"  {lab:<8} {mono} / {tot}{note}")
    print()

    print("=== Morton の符号の割れ（ブロックごと）===")
    for key, lab in COD:
        d = [x for b in blocks if (x := delta(rows, b, "Morton", key)) is not None]
        neg = [x for x in d if x < 0]
        print(f"  {lab:<8} 負 {len(neg)} / {len(d)}" +
              (f"  最小 {min(d):+.1f}%  最大 {max(d):+.1f}%" if d else ""))
    print()

    print("=== 走査v1 の増分が実効下限を超えるブロック ===")
    seed = max((abs(u - v) for b in blocks
                if (u := delta(rows, b, "ランダム1", "scan1")) is not None
                and (v := delta(rows, b, "ランダム2", "scan1")) is not None),
               default=0.0)
    revm = max((abs(x) for b in blocks
                if (x := delta(rows, b, "逆順", "scan1")) is not None), default=0.0)
    floor = max(seed, revm)
    print(f"  種違いの差の最大 {seed:.3f}%点、逆順の |Δ| 最大 {revm:.3f}%点 "
          f"→ 実効下限 {floor:.3f}%点")
    over, meas = [], 0
    for b in blocks:
        v = [x for c in ("ランダム1", "ランダム2")
             if (x := delta(rows, b, c, "scan1")) is not None]
        if not v:
            continue          # 走査列が無いブロックは分母にも入れない
        meas += 1
        m = float(np.mean(v))
        if abs(m) > floor:
            dk = next(r["dup_key"] for r in rows if r["name"] == b)
            over.append((b, dk, m))
    for b, dk, m in sorted(over, key=lambda x: -abs(x[2])):
        print(f"    {b:<13} 鍵重複 {dk:>5.1f}%  Δ {m:+8.2f}%")
    print(f"  {len(over)} / {meas} ブロック（走査列が測れたブロックのみ）")
    print()

    n = 1_000_000
    print(f"参考: log2(n!)/n = {math.log2(n) - math.log2(math.e):.3f} bit/点（n = {n}）。")
    print("  これは置換そのものの情報量であって、上の増分と直接は比べられない。")
    print("  増分は符号器が順序の乱れで失う量であり、順序を復元する費用ではない。")


if __name__ == "__main__":
    main()
