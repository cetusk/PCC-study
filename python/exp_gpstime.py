"""gps_time が符号器の想定する整数格子に乗っているかを調べる。

AHN3 は gps_time だけで 9.693 bpp を使い、差分も文脈も効かない。
AHN4 は同じ候補で 2.18〜2.29 bpp に落ちる。何が違うのかを見る。

  (1) 値の刻み      連続する相異なる値の差の GCD。ulp の何倍か
  (2) 並び          時刻が単調か、どれだけ後戻りするか
  (3) 0 次の下限    生の差分・格子に直した差分・並べ替えた差分の各エントロピー

使い方:
    $PCCPY python/exp_gpstime.py <in.las|laz> [最大点数]
"""
from __future__ import annotations
import sys
import math
import numpy as np
import laspy

ULP = 59.6046e-9


def h0(v: np.ndarray) -> float:
    """zigzag ビット長の 0 次エントロピー [bit/点]。列の符号長の目安。"""
    if len(v) == 0:
        return 0.0
    z = (v.astype(np.int64) << 1) ^ (v.astype(np.int64) >> 63)
    b = np.zeros(len(z), dtype=np.int64)
    m = z > 0
    b[m] = np.floor(np.log2(z[m].astype(np.float64))).astype(np.int64) + 1
    cnt = np.bincount(b)
    p = cnt[cnt > 0] / len(b)
    # 長さの符号 + 仮数部のビット
    return float(-(p * np.log2(p)).sum() + (b.astype(np.float64)).mean())


def gcd_of(d: np.ndarray, cap: int = 200000) -> int:
    d = d[d != 0]
    if len(d) == 0:
        return 0
    g = 0
    for x in d[:cap]:
        g = math.gcd(g, int(abs(x)))
        if g == 1:
            break
    return g


def main() -> None:
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with laspy.open(path) as fh:
        n_all = fh.header.point_count
        pts = fh.read()
    t = np.asarray(pts.gps_time, dtype=np.float64)
    if cap and cap < len(t):
        t = t[:cap]
    n = len(t)

    print(f"入力        {path}")
    print(f"            {n} 点（全 {n_all}）")
    print(f"            範囲 {t.min():.6f} 〜 {t.max():.6f}  幅 {t.max() - t.min():.3f} s")

    # 値をビットパターンの整数として見る（符号器と同じ見方）
    q = t.view(np.int64)
    d = np.diff(q)

    print()
    print("--- (1) 刻み ---")
    u = np.unique(t)
    print(f"相異なる値   {len(u)} / {n}  ({len(u) / n:.4f})")
    du = np.diff(u)
    print(f"隣の値の差   最小 {du.min():.6e} s  中央 {np.median(du):.6e} s")
    print(f"             ulp 換算 最小 {du.min() / ULP:.3f}  中央 {np.median(du) / ULP:.3f}")
    gi = gcd_of(np.diff(u.view(np.int64)))
    print(f"ビットパターン差の GCD  {gi}")

    print()
    print("--- (2) 並び ---")
    back = int((d < 0).sum())
    print(f"時刻が後戻りする箇所  {back} / {n - 1}  ({back / max(1, n - 1):.4f})")
    if back:
        print(f"             後戻り幅 中央 {np.median(-np.diff(t)[np.diff(t) < 0]):.6e} s")
    print(f"前進の差 中央 {np.median(np.diff(t)[np.diff(t) > 0]):.6e} s")

    print()
    print("--- (3) 0 次の下限 [bit/点] ---")
    print(f"生の値（ビットパターン）        {h0(q - q[0]):8.3f}")
    print(f"1 次差分                        {h0(d):8.3f}")
    ts = np.sort(t)
    print(f"並べ替えてから差分              {h0(np.diff(ts.view(np.int64))):8.3f}")
    # 格子に直す: 最小刻みで割る
    step = du.min()
    if step > 0:
        k = np.rint((t - t.min()) / step).astype(np.int64)
        err = np.abs((t - t.min()) / step - k).max()
        print(f"最小刻み {step:.6e} s で格子化   {h0(np.diff(k)):8.3f}   丸め誤差 {err:.3e}")
        print(f"  （格子化の可逆性: {'可' if err < 1e-6 else '不可 — 端数が残る'}）")

    print()
    print("--- (4) 可逆な予測子の比較 [bit/点] ---")
    print(f"ulp 換算の 1 刻み  {np.spacing(abs(t).max()):.6e} s")
    print(f"1 次差分                        {h0(d):8.3f}")
    print(f"2 次差分                        {h0(np.diff(d)):8.3f}")
    # 直前の差分を予測に使う（ビットパターンのまま）
    pred = q[1:-1] + d[:-1]
    print(f"前の差分を繰り越す              {h0(q[2:] - pred):8.3f}")
    # 直近 3 差分の中央値
    if len(d) > 4:
        m3 = np.median(np.stack([d[:-3], d[1:-2], d[2:-1]]), axis=0).astype(np.int64)
        print(f"直近 3 差分の中央値             {h0(q[4:] - (q[3:-1] + m3[:len(q) - 4])):8.3f}")
    # 実時間の刻みを固定して外挿し、ビットパターン差を送る
    step = float(np.median(np.diff(t)[np.diff(t) > 0]))
    pf = (t[:-1] + step).view(np.int64)
    print(f"実時間で +{step:.3e} s 外挿    {h0(q[1:] - pf):8.3f}")


if __name__ == "__main__":
    main()
