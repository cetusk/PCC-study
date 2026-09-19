"""符号器が使っているモデル族での静的最適費用を測り、実測との差を出す。

PCC2 の UIntCoder は値をこう分解して、判定ごとに文脈つきの適応 2 値モデルで符号化する。

    z = zigzag(r),  k = floor(log2(z+1)),  rem = z+1-2^k
    接頭辞  pm[0..k-1] <- 1,  pm[k] <- 0
    仮数    sm[k-1..0] <- rem の各ビット

判定を (文脈, 種別, 位置) のバケツに集め、バケツごとの経験 2 値エントロピーを
足したものが、このモデル族での静的最適費用になる。実測との差は適応の学習コスト。

文脈を他の列のビット長まで広げると、列をまたぐ文脈で取れる利得が直接出る。

使い方:
    $PCCPY python/exp_entropy_bound.py <in.las|laz> [点数]
"""
from __future__ import annotations
import sys
import numpy as np
import laspy

NCTX = 24          # PCC2 と同じ文脈数
MAXK = 64


def zigzag(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64, copy=False)
    return ((v << 1) ^ (v >> 63)).astype(np.uint64)


def klen(z: np.ndarray) -> np.ndarray:
    """floor(log2(z+1))。z = 2^64-1 は 64 とする（符号器の特例と対）。"""
    k = np.zeros(len(z), dtype=np.int64)
    t = z + np.uint64(1)
    full = (z == np.uint64(0xFFFFFFFFFFFFFFFF))
    nz = ~full
    if nz.any():
        k[nz] = np.floor(np.log2(t[nz].astype(np.float64))).astype(np.int64)
        # 浮動小数の丸めで 1 ずれることがあるので詰め直す
        over = nz & (k > 0) & ((t >> k.astype(np.uint64)) == np.uint64(0))
        k[over] -= 1
        und = nz & ((t >> (k + 1).astype(np.uint64)) != np.uint64(0))
        k[und] += 1
    k[full] = 64
    return k


def ctx_of(z: np.ndarray) -> np.ndarray:
    """直前の残差の zigzag からビット長文脈（0..NCTX-1）を作る。C++ の ctx_of と同じ。"""
    b = np.zeros(len(z), dtype=np.int64)
    m = z > np.uint64(0)
    if m.any():
        b[m] = np.floor(np.log2(z[m].astype(np.float64))).astype(np.int64) + 1
    return np.minimum(b, NCTX - 1)


def static_bits(z: np.ndarray, ctx: np.ndarray, nctx: int):
    """このモデル族での静的最適費用［bit］と、使ったバケツ数を返す。"""
    k = klen(z)
    rem = (z + np.uint64(1)) - (np.uint64(1) << k.astype(np.uint64))
    rem[k == 64] = z[k == 64]
    kmax = int(k.max()) if len(k) else 0

    tot = 0.0
    nbucket = 0

    def acc(n1: np.ndarray, n0: np.ndarray):
        """バケツごとの 1/0 の個数から経験 2 値エントロピーの総和を足す。"""
        nonlocal tot, nbucket
        n = n1 + n0
        use = n > 0
        if not use.any():
            return
        nbucket += int(use.sum())
        a, b_, t = n1[use].astype(np.float64), n0[use].astype(np.float64), n[use].astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(a > 0, a * np.log2(t / np.maximum(a, 1)), 0.0) + \
                np.where(b_ > 0, b_ * np.log2(t / np.maximum(b_, 1)), 0.0)
        tot += float(s.sum())

    # 接頭辞: 位置 i の判定は k > i の点に 1、k == i の点に 0（i < MAXK のとき）
    for i in range(kmax + 1):
        m1 = k > i
        m0 = (k == i)
        n1 = np.bincount(ctx[m1], minlength=nctx) if m1.any() else np.zeros(nctx, dtype=np.int64)
        n0 = np.bincount(ctx[m0], minlength=nctx) if m0.any() else np.zeros(nctx, dtype=np.int64)
        acc(n1, n0)

    # 仮数: 位置 i の判定は k > i の点にだけある
    for i in range(kmax):
        m = k > i
        if not m.any():
            continue
        bit = ((rem[m] >> np.uint64(i)) & np.uint64(1)).astype(np.int64)
        c = ctx[m]
        n1 = np.bincount(c[bit == 1], minlength=nctx)
        n0 = np.bincount(c[bit == 0], minlength=nctx)
        acc(n1, n0)

    return tot, nbucket


def residuals(v: np.ndarray):
    """PCC2 のスカラ候補と同じ 3 つの予測子の残差を返す。"""
    d1 = np.empty_like(v); d1[0] = v[0]; d1[1:] = v[1:] - v[:-1]
    d2 = np.empty_like(d1); d2[0] = d1[0]; d2[1:] = d1[1:] - d1[:-1]
    # 直近 3 差分の中央値（MedPred）
    med = np.empty_like(v)
    med[0] = v[0]
    if len(v) > 1:
        med[1] = v[1] - v[0]
    if len(v) > 2:
        med[2] = v[2] - v[1] - d1[1]
    if len(v) > 3:
        a, b_, c = d1[1:-3], d1[2:-2], d1[3:-1]
        m3 = np.median(np.stack([a, b_, c]), axis=0).astype(np.int64)
        med[4:] = v[4:] - (v[3:-1] + m3)
        med[3] = v[3] - (v[2] + d1[2])
    return {"delta": d1, "delta2": d2, "med3": med}


def load_cols(path: str, cap: int):
    with laspy.open(path) as fh:
        pts = fh.read()
    n = min(cap, len(pts.X)) if cap else len(pts.X)
    cols = {}
    cols["X"] = np.asarray(pts.X, dtype=np.int64)[:n]
    cols["Y"] = np.asarray(pts.Y, dtype=np.int64)[:n]
    cols["Z"] = np.asarray(pts.Z, dtype=np.int64)[:n]
    raw = getattr(pts.points, "array", None)
    for nm in pts.point_format.dimension_names:
        if nm in ("X", "Y", "Z"):
            continue
        # 追加バイトは laspy がスケールを適用して float64 で返す。符号器が見るのは
        # ファイル中の整数なので、生の配列があるときはそちらを使う。
        a = None
        if raw is not None and nm in raw.dtype.names:
            a = np.asarray(raw[nm])
        if a is None or not (np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_):
            try:
                a = np.asarray(pts[nm])
            except Exception:
                continue
        if a.ndim != 1 or len(a) < n:
            continue
        if a.dtype == np.float64:
            cols[nm] = a[:n].view(np.int64).copy()     # ビットパターンを整数と見る
        elif np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_:
            cols[nm] = a[:n].astype(np.int64)
    return cols, n


def main() -> None:
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
    cols, n = load_cols(path, cap)
    print(f"入力        {path}")
    print(f"            {n} 点 / 列 {len(cols)}")

    # --- (2) 列ごとの静的最適費用 ---
    print()
    print("=== (2) 符号器のモデル族での静的最適費用 [bit/点] ===")
    print(f"{'列':<20}{'予測子':>8}{'静的最適':>10}{'バケツ':>9}")
    chosen = {}
    for nm, v in cols.items():
        best = None
        for pn, r in residuals(v).items():
            z = zigzag(r)
            c = np.zeros(len(z), dtype=np.int64)
            c[1:] = ctx_of(z[:-1])
            b, nb = static_bits(z, c, NCTX)
            if best is None or b < best[1]:
                best = (pn, b, nb, z, c)
        pn, b, nb, z, c = best
        chosen[nm] = (pn, b / n, z, c)
        print(f"{nm:<20}{pn:>8}{b / n:>10.3f}{nb:>9}")

    # --- (1) 列をまたぐ文脈で取れる利得 ---
    print()
    print("=== (1) 列をまたぐ文脈で取れる利得 [bit/点] ===")
    print(f"{'列':<20}{'単独':>9}{'最良の相手':>14}{'併用':>9}{'利得':>8}")
    names = list(cols.keys())
    kcls = {nm: ctx_of(chosen[nm][2]) for nm in names}   # 同じ点でのビット長クラス
    total_gain = 0.0
    for nm in names:
        pn, base, z, c = chosen[nm]
        bestp, bestb = None, base
        for om in names:
            if om == nm:
                continue
            c2 = c * NCTX + kcls[om]
            b2, _ = static_bits(z, c2, NCTX * NCTX)
            if b2 / n < bestb:
                bestb, bestp = b2 / n, om
        g = base - bestb
        total_gain += max(g, 0.0)
        print(f"{nm:<20}{base:>9.3f}{(bestp or '-'):>14}{bestb:>9.3f}{g:>8.3f}")
    print(f"{'合計の利得':<20}{'':>9}{'':>14}{'':>9}{total_gain:>8.3f}")

    # --- (3) 収束とバイアス ---
    print()
    print("=== (3) 標本数を変えたときの収束 [bit/点] ===")
    sizes = [s for s in (125000, 250000, 500000, 1000000, 2000000) if s <= n]
    print(f"{'点数':>10}{'合計(静的最適)':>16}{'バケツ':>9}{'MM補正':>9}{'補正後':>10}")
    for s in sizes:
        tot, nb = 0.0, 0
        for nm, v in cols.items():
            pn = chosen[nm][0]
            r = residuals(v[:s])[pn]
            z = zigzag(r)
            c = np.zeros(len(z), dtype=np.int64)
            c[1:] = ctx_of(z[:-1])
            b, k = static_bits(z, c, NCTX)
            tot += b; nb += k
        mm = nb / (2.0 * s * np.log(2.0))
        print(f"{s:>10}{tot / s:>16.3f}{nb:>9}{mm:>9.4f}{tot / s + mm:>10.3f}")


if __name__ == "__main__":
    main()
