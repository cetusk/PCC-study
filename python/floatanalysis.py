"""float32 で保存された点群の「本当の自由度」を測る。

仮説:
  回転式 LiDAR の点は本来 (レーザID, 方位角, 距離) から導出された量であり、
  デカルト float32 xyz (96 bit/点) は派生表現にすぎない。
  もし距離・方位角が固定ステップで量子化されているなら、真の自由度は
  6 + ~16 + ~16 = 40 bit 程度しかなく、96 bit との差は表現の無駄である。

  これが正しければ「2/3 にしかならない」の直接の原因になりうる。
  表現そのものを疑うための、最も安上がりな検証。
"""
from __future__ import annotations
import numpy as np


# ------------------------------------------------------- float32 のビット構造

def float32_bit_profile(a: np.ndarray) -> str:
    """32 ビット各位置の使われ方。エントロピー ~1.0 は「ランダム＝圧縮不能」。"""
    assert a.dtype == np.float32
    u = a.view(np.uint32).ravel()
    L = [f"N = {u.size:,}",
         f"{'bit':>4} {'意味':<10} {'H0':>7} {'H1|prev':>9}  分布"]
    L.append("-" * 62)
    tot0 = tot1 = 0.0
    for b in range(31, -1, -1):
        v = (u >> np.uint32(b)) & np.uint32(1)
        p = v.mean()
        h0 = 0.0 if p in (0.0, 1.0) else float(-(p*np.log2(p) + (1-p)*np.log2(1-p)))
        # 直前の点の同じビットで条件付け（走査順の相関を見る）
        h1 = _cond_entropy(v[1:], v[:-1])
        tot0 += h0; tot1 += h1
        kind = "sign" if b == 31 else ("exp" if b >= 23 else "mantissa")
        bar = "#" * int(round(h0 * 30))
        L.append(f"{b:>4} {kind:<10} {h0:7.3f} {h1:9.3f}  {bar}")
    L.append("-" * 62)
    L.append(f"{'合計':>4} {'':<10} {tot0:7.3f} {tot1:9.3f}   (32 bit 中、実際に使われている量)")
    return "\n".join(L)


def _cond_entropy(x: np.ndarray, cond: np.ndarray) -> float:
    """H(x | cond)。どちらも小さな非負整数配列。"""
    k = int(cond.max()) + 1
    j = cond.astype(np.int64) * 2 + x.astype(np.int64)
    c = np.bincount(j, minlength=2 * k).reshape(k, 2).astype(float)
    n = c.sum()
    if n == 0:
        return 0.0
    out = 0.0
    for row in c:
        s = row.sum()
        if s == 0:
            continue
        p = row / s
        p = p[p > 0]
        out += (s / n) * float(-(p * np.log2(p)).sum())
    return out


# ------------------------------------------------------------- 量子化ステップ

def detect_quantum(v: np.ndarray, max_check: int = 200_000) -> dict:
    """値が固定ステップの整数倍になっているかを調べる。

    返り値の on_grid_frac が 1.0 に近ければ、その値は実質的に整数格子上にある。
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size > max_check:
        v = v[:max_check]
    u = np.unique(v)
    if u.size < 3:
        return {"step": None, "on_grid_frac": 1.0, "n_unique": int(u.size)}
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return {"step": None, "on_grid_frac": 1.0, "n_unique": int(u.size)}
    step = float(np.percentile(d, 1))      # 最頻の最小間隔に近い頑健な推定
    if step <= 0:
        return {"step": None, "on_grid_frac": 0.0, "n_unique": int(u.size)}
    r = v / step
    frac = np.abs(r - np.rint(r))
    return {"step": step,
            "on_grid_frac": float((frac < 1e-3).mean()),
            "n_unique": int(u.size),
            "min_gap": float(d.min())}


# --------------------------------------------------------------- 極座標分解

def polar_decompose(xyz: np.ndarray) -> dict:
    x, y, z = xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64), xyz[:, 2].astype(np.float64)
    r = np.sqrt(x*x + y*y + z*z)
    az = np.arctan2(y, x)
    el = np.arcsin(np.clip(z / np.maximum(r, 1e-9), -1, 1))
    return {"r": r, "az": az, "el": el}


def ring_histogram(el: np.ndarray, nbins: int = 4000) -> tuple[int, np.ndarray]:
    """仰角のヒストグラムから「レーザ本数」を推定する。"""
    h, edges = np.histogram(el, bins=nbins)
    # 疎なビンを落として、連続する山の数を数える
    thr = max(1, int(0.05 * h.max()))
    occ = h > thr
    peaks = int(np.sum(occ[1:] & ~occ[:-1])) + int(occ[0])
    centers = 0.5 * (edges[1:] + edges[:-1])
    return peaks, centers[occ]


def analyse_lidar_frame(xyz: np.ndarray, name: str = "") -> str:
    L = [f"### {name}  N={len(xyz):,}  保存形式 float32 x3 = {len(xyz)*12/1e3:.0f} KB "
         f"({96:.0f} bit/点)", ""]
    L.append("[1] デカルト float32 のビット使用状況（x 軸）")
    L.append(float32_bit_profile(np.ascontiguousarray(xyz[:, 0])))
    p = polar_decompose(xyz)
    L.append("")
    L.append("[2] 極座標に戻したときの量子化構造")
    for k, v in (("距離 r [m]", p["r"]), ("方位角 az [rad]", p["az"]), ("仰角 el [rad]", p["el"])):
        q = detect_quantum(v)
        s = q["step"]
        L.append(f"  {k:<16} unique={q['n_unique']:>8,}  推定ステップ="
                 f"{('%.3e' % s) if s else 'なし':>10}  格子上の点の割合={q['on_grid_frac']*100:5.1f}%")
    peaks, centers = ring_histogram(p["el"])
    L.append(f"  仰角のクラスタ数（≒レーザ本数の推定） = {peaks}")
    if peaks:
        L.append(f"    仰角レンジ {np.degrees(centers.min()):.2f}° 〜 {np.degrees(centers.max()):.2f}°")
    return "\n".join(L)
