"""取得順の置換を実際に符号化したときの費用を測る。

多重集合を復元する符号器（G-PCC）を系列可逆にするには、出力順から
入力順への置換を送る必要がある。その上界は log2(n!)/n だが、取得順は
空間的に相関しているので実際にはもっと安い。ここでは実際に符号化して
「達成できる値」を出す。上界ではなく、ある符号器が現に達成した値である。

対照として一様乱数の置換も同じ経路で符号化する。これが log2(n!)/n に
近い値になれば、符号器が置換に対して極端に弱くないことの確認になる。
"""
from __future__ import annotations
import lzma
import sys
from math import lgamma, log
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp_order_matrix import FILES, BLOCK, NBLOCK, morton3, read_block, total_points


def perm_bits(n: int) -> float:
    """log2(n!)/n。一様な置換の情報量。"""
    return lgamma(n + 1) / log(2) / n


def encode_perm(rank: np.ndarray) -> float:
    """置換を符号化して bit/点を返す。

    順位の 1 次差分を zigzag して可変長整数にし、LZMA に通す。
    取得順が空間的に連続なら差分が小さくまとまるので短くなる。
    """
    d = np.diff(np.r_[0, rank.astype(np.int64)])
    z = np.where(d >= 0, 2 * d, -2 * d - 1).astype(np.uint64)
    out = bytearray()
    for v in z:
        v = int(v)
        while True:
            b = v & 0x7F
            v >>= 7
            out.append(b | (0x80 if v else 0))
            if not v:
                break
    c = lzma.compress(bytes(out), preset=9 | lzma.PRESET_EXTREME)
    return len(c) * 8.0 / len(rank)


def main() -> None:
    print("取得順の置換を符号化したときの費用")
    print("多重集合符号器を系列可逆にするために送る必要のある量。")
    print("基準順は Morton 順（オクツリーが自然に出す順）。")
    print()
    print(f"{'ブロック':<14}{'点数':>9}{'log2(n!)/n':>12}{'取得順':>10}"
          f"{'一様乱数':>10}{'取得順/上界':>12}")
    rows = []
    for name, path, off, kind in FILES:
        if not (Path(path).is_file() or (kind == "kitti" and Path(path).is_dir())):
            continue
        total = total_points(kind, path)
        nb = NBLOCK if total >= BLOCK * NBLOCK else 1
        st = [0] if nb == 1 else [i * ((total - BLOCK) // (nb - 1)) for i in range(nb)]
        for bi, start in enumerate(st):
            xyz, *_ = read_block(kind, path, start, BLOCK)
            n = len(xyz)
            ub = perm_bits(n)
            # Morton 順に並べたときの、各点の取得順での位置
            order = np.argsort(morton3(xyz), kind="stable")
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(n)
            got = encode_perm(rank)
            rng = np.random.default_rng(20260920)
            ctl = encode_perm(rng.permutation(n))
            lab = name if nb == 1 else f"{name}#{bi}"
            print(f"{lab:<14}{n:>9}{ub:>12.3f}{got:>10.3f}{ctl:>10.3f}"
                  f"{100 * got / ub:>11.1f}%", flush=True)
            rows.append((lab, n, ub, got, ctl))
    if rows:
        print()
        a = np.array([[r[2], r[3], r[4]] for r in rows])
        print(f"中央値: 上界 {np.median(a[:, 0]):.3f} / 取得順 {np.median(a[:, 1]):.3f}"
              f" / 一様乱数 {np.median(a[:, 2]):.3f} bit/点")
        print(f"取得順は上界の {np.median(a[:, 1] / a[:, 0]) * 100:.1f}%（中央値）")


if __name__ == "__main__":
    main()
