"""G-PCC の可逆判定が本当に厳しいかを、人工データで確かめる。

査読で 2 点指摘された。
  * `_ply_matches` が `np.rint` で無条件に丸めるので、±0.5 未満のずれを
    「可逆」と判定しうる。→ 復号値が整数そのものであることを先に検査するよう直した。
  * `--mergeDuplicatedPoints=0` の正しさが実証されていない。実データの
    検査範囲に重複点が 0 件（200 万点）〜2 件（800 万点）しかないため。
    → 重複を人工的に入れて確かめる。

使い方:
    $PCCPY python/exp_gpcc_sanity.py > data/work/gpcc_sanity.txt
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import tmc13_bits           # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20260920)
    print("G-PCC の可逆判定の健全性")
    print()
    base = rng.integers(0, 50_000, size=(200_000, 3)).astype(np.int64)

    cases = [
        ("重複なし", base),
        ("半数が重複（10 万組）", np.vstack([base[:100_000], base[:100_000]])),
        ("全点が 4 重", np.vstack([base[:50_000]] * 4)),
    ]
    print(f"{'条件':<24}{'点数':>9}{'相異なる点':>11}{'bpp':>9}{'可逆':>7}")
    for name, a in cases:
        u = len(np.unique(a, axis=0))
        r = tmc13_bits(a)
        bpp = r.bytes * 8.0 / len(a) if r.bytes else float("nan")
        print(f"{name:<24}{len(a):>9}{u:>11}{bpp:>9.3f}{str(r.lossless):>7}")
        if r.note:
            print(f"  注記 {r.note[:70]}")
    print()
    print("多重集合として照合しているので、重複が落ちれば false になる。")
    print("整数性の検査を入れたので、丸めで隠れるずれも false になる。")


if __name__ == "__main__":
    main()
