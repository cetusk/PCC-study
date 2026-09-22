"""予測子を「選ぶ」のではなく「混ぜる」ときの符号長を見積もる。

いまの設計は列ごとに符号器を 1 本選ぶ。だが**点ごとに最良の予測子は入れ替わる**
（plane では「直前」が 73%、残り 27% は別の予測子が勝つ）。1 本に固定すると
plane で 0.49、vegetation で 1.76 bit/点を捨てている。

どれが勝ったかを送ると log2(K) 掛かって割に合わない。**送らずに混ぜる。**
各予測子に「中心 = その予測、幅 = 直近の誤差」の分布を持たせ、
重みを直近の当たり具合で更新して足し合わせる。復号側も同じものを作れる。

これは見積りであり、実装の符号長ではない。ただし**適応的に重みを更新する
逐次符号なので到達可能**である（過去だけを見て次を符号化している）。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "2000000"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra"]


def med3(a, b, c):
    return a + b + c - min(a, b, c) - max(a, b, c)


class Pred:
    """予測子 1 本。中心と、直近の誤差から決まる幅を持つ。"""

    def __init__(self, kind):
        self.kind = kind
        self.h = [0, 0, 0, 0]        # 直近の値
        self.d = [0, 0, 0]           # 直近の差分
        self.k = 0
        self.mad = 1.0               # 平均絶対誤差（指数移動平均）

    def predict(self):
        h, d, k = self.h, self.d, self.k
        if k == 0:
            return 0
        p = h[0]
        if k < 2 or self.kind == "prev":
            return p
        d1 = h[0] - h[1]
        if self.kind == "lin":
            return p + d1
        if self.kind == "half":
            return p + d1 // 2
        if k < 4:
            return p
        d2, d3 = h[1] - h[2], h[2] - h[3]
        if self.kind == "med3":
            return p + med3(d1, d2, d3)
        return p + (d1 + d2 + d3) // 3          # avg3

    def push(self, v, err):
        self.mad += (abs(err) - self.mad) * 0.06
        if self.mad < 0.5:
            self.mad = 0.5
        self.h = [v] + self.h[:3]
        self.k += 1


KINDS = ["prev", "lin", "half", "med3", "avg3"]


def mix_bits(col: np.ndarray, mixing: bool) -> float:
    """整数 1 列の符号長 [bit/値]。mixing=False なら最良の 1 本に固定したときの値。"""
    n = len(col)
    ps = [Pred(k) for k in KINDS]
    K = len(ps)
    w = np.full(K, 1.0 / K)
    tot = 0.0
    # 混ぜない場合は各予測子の合計を出して最小を採る
    each = np.zeros(K)
    for i in range(n):
        v = int(col[i])
        mu = np.array([p.predict() for p in ps], dtype=np.float64)
        b = np.array([p.mad for p in ps], dtype=np.float64)
        # 離散ラプラス: P(v) = (1-r)/(1+r) * r^|v-mu|,  r = exp(-1/b)
        r = np.exp(-1.0 / b)
        lp = np.log2((1 - r) / (1 + r)) + np.abs(v - mu) * np.log2(r)
        each += -lp
        if mixing:
            pv = float(np.dot(w, np.exp2(lp)))
            pv = max(pv, 1e-12)
            tot += -np.log2(pv)
            # 重みを当たり具合で更新（指数重み付け）
            post = w * np.exp2(lp - lp.max())
            s = post.sum()
            if s > 0:
                w = 0.92 * w + 0.08 * (post / s)
                w /= w.sum()
        for j, p in enumerate(ps):
            p.push(v, v - mu[j])
    return (tot / n) if mixing else (each.min() / n)


print(f"標本 {N} 点。幾何 3 軸の合計 [bit/点]。離散ラプラスでの見積り。\n")
print(f"{'データ':<13}{'1 本に固定':>12}{'混ぜる':>10}{'差':>9}{'減り':>8}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    a = sum(mix_bits(xyz[:, c], False) for c in range(3))
    b = sum(mix_bits(xyz[:, c], True) for c in range(3))
    print(f"{lab:<13}{a:>12.3f}{b:>10.3f}{b-a:>9.3f}{100*(b-a)/a:>7.1f}%")
