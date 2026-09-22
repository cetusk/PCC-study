"""幾何の予測子を並べて、どこまで短くなりうるかを見る。

符号長は離散ラプラスの見積り（3 軸の合計）。実際の符号化器の値ではないので、
候補の順位を見るためだけに使う。
"""
import sys, math
import numpy as np
import laspy
from scipy.spatial import cKDTree


def bits(r):
    r = np.asarray(r, dtype=np.float64).ravel()
    b = np.mean(np.abs(r - np.median(r)))
    return math.log2(2 * math.e * b) if b > 1e-9 else 0.0


def main(path, n=0):
    f = laspy.read(path)
    m = len(f.X) if not n else min(n, len(f.X))
    P = np.column_stack([np.asarray(f.X[:m]), np.asarray(f.Y[:m]),
                         np.asarray(f.Z[:m])]).astype(np.float64)
    print(f'{path.split("/")[-1]}  {m} 点')
    tot = lambda R: sum(bits(R[:, k]) for k in range(3))

    # 1. 直前の点
    print(f'  {"直前の点":<28}{tot(P[1:] - P[:-1]):7.3f} bit/点')
    # 2. 直前 3 点の差の中央値
    d = np.diff(P, axis=0)
    med = np.median(np.stack([d[2:], d[1:-1], d[:-2]]), axis=0)
    print(f'  {"直近 3 差分の中央値":<28}{tot(d[3:] - med[:-1]):7.3f} bit/点')
    # 3. 直近 W 点のうち最も近い点
    for W in (4, 16, 64):
        pred = np.zeros_like(P)
        for i in range(1, m):
            lo = max(0, i - W)
            q = P[lo:i]
            j = int(np.argmin(((q - P[i]) ** 2).sum(1)))
            pred[i] = q[j]
        print(f'  {f"直近 {W} 点の最近傍":<28}{tot(P[1:] - pred[1:]):7.3f} bit/点')
    # 4. 既出の全点のうち最近傍（上限。復号側も同じものを作れる）
    pred = np.zeros_like(P)
    step = max(1, m // 20000)
    for i in range(1, m):
        lo = 0
        q = P[lo:i]
        if len(q) > 4096:
            q = q[-4096:]
            lo = i - 4096
        j = int(np.argmin(((q - P[i]) ** 2).sum(1)))
        pred[i] = q[j]
    print(f'  {"既出 4096 点の最近傍":<28}{tot(P[1:] - pred[1:]):7.3f} bit/点')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
