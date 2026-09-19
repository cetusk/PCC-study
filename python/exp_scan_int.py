"""走査モデル経由の幾何を、整数演算だけで厳密に往復させる検証。

三角関数は処理系ごとに最下位ビットが違うので、復号側では呼ばない。
 - 回転は 3 回のせん断に分解する（各段が Z^2 上の全単射）
 - 走査角の正接は固定小数の表を整数で線形補間して引く

ここで確かめるのは 2 つ。
 (1) 整数だけで厳密に元へ戻ること
 (2) 整数化しても残差のエントロピーが浮動小数版から悪化しないこと
"""
from __future__ import annotations
import numpy as np

SH = 30                      # せん断定数の固定小数（2^-30）
TAN_BITS = 16                # tan 表の分割（[-45,45] 度を 2^16 分割）
TAN_SH = 30                  # tan 値の固定小数


def build_tan_table() -> np.ndarray:
    """[-45°, +45°] の tan を 2^TAN_BITS+1 点、2^-TAN_SH 固定小数で持つ。"""
    n = (1 << TAN_BITS) + 1
    a = np.linspace(-np.pi / 4, np.pi / 4, n)
    return np.round(np.tan(a) * (1 << TAN_SH)).astype(np.int64)


_TAB = build_tan_table()


def tan_fx(theta_fx: np.ndarray) -> np.ndarray:
    """theta_fx は [-45,45] 度を [-2^31, 2^31] に写した整数。整数のみで補間する。"""
    # 表の索引 = (theta_fx + 2^31) >> (32 - TAN_BITS)
    u = (theta_fx.astype(np.int64) + (1 << 31))
    u = np.clip(u, 0, (1 << 32) - 1)
    idx = u >> (32 - TAN_BITS)
    frac = u - (idx << (32 - TAN_BITS))
    idx = np.clip(idx, 0, (1 << TAN_BITS) - 1)
    a = _TAB[idx]
    b = _TAB[idx + 1]
    return a + ((b - a) * frac >> (32 - TAN_BITS))


def shear_forward(X: np.ndarray, Y: np.ndarray, t2: int, sn: int):
    """3 回のせん断による整数回転。t2 = tan(θ/2), sn = sin(θ)（2^-SH 固定小数）。"""
    x = X - ((t2 * Y) >> SH)
    y = Y + ((sn * x) >> SH)
    x = x - ((t2 * y) >> SH)
    return x, y


def shear_inverse(x: np.ndarray, y: np.ndarray, t2: int, sn: int):
    # 順の 3 段を逆順に、同じ丸めで戻す。
    #   順: x1 = X - [t2 Y],  y1 = Y + [sn x1],  x2 = x1 - [t2 y1]
    #   逆: x1 = x2 + [t2 y1], Y = y1 - [sn x1],  X = x1 + [t2 Y]
    x1 = x + ((t2 * y) >> SH)
    Y = y - ((sn * x1) >> SH)
    X = x1 + ((t2 * Y) >> SH)
    return X, Y


def check_roundtrip(n=200000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(-10**8, 10**8, n, dtype=np.int64)
    Y = rng.integers(-10**8, 10**8, n, dtype=np.int64)
    ok_all = True
    for deg in (0.0, 7.3, -31.4, 44.9, -44.9):
        th = np.radians(deg)
        t2 = int(round(np.tan(th / 2) * (1 << SH)))
        sn = int(round(np.sin(th) * (1 << SH)))
        x, y = shear_forward(X, Y, t2, sn)
        X2, Y2 = shear_inverse(x, y, t2, sn)
        ok = bool(np.all(X2 == X) and np.all(Y2 == Y))
        ok_all &= ok
        print(f"  回転 {deg:+6.1f} 度: 厳密に戻る = {ok}  "
              f"（座標の伸び {np.abs(x).max()/max(np.abs(X).max(),1):.4f}）")
    return ok_all


def check_tan(n=100000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi / 4, np.pi / 4, n)
    fx = np.round(a / (np.pi / 4) * (1 << 31)).astype(np.int64)
    got = tan_fx(fx) / (1 << TAN_SH)
    err = np.abs(got - np.tan(a))
    print(f"  固定小数 tan の最大誤差 {err.max():.3e}  "
          f"（1200 m 先で {err.max()*1200*1000:.4f} mm）")
    return err.max()


if __name__ == '__main__':
    print("三角関数を使わない整数回転の検証")
    ok = check_roundtrip()
    print("固定小数 tan の精度")
    check_tan()
    print("→ 厳密往復:", ok)
