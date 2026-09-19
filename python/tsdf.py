"""本来の意味での TSDF — 正則ボクセル格子上の符号付き距離場。

点ごとの平面当てはめで平滑化する版は、平坦な領域で近傍の取り方に敏感すぎて
lfs 分布の上側が安定しなかった（標本数と平滑化スケールの両方に強く依存）。
正則格子で符号付き距離を作り、marching cubes で陽に曲面を取り出せば、
平滑化がボクセルサイズだけで決まり、近傍の取り方に依存しなくなる。

  1. 点に法線を付ける
  2. 曲面の近傍だけ（narrow band）ボクセルを立てる
  3. 各ボクセル中心で、近傍点の法線への射影の重み付き平均 = 符号付き距離
  4. marching cubes で零等位面を取り出す
  5. その曲面の頂点と法線で lfs を測る
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree
from skimage import measure


def estimate_normals(pts: np.ndarray, k: int = 24,
                     up: np.ndarray | None = None) -> np.ndarray:
    tree = cKDTree(pts)
    _, nb = tree.query(pts, k=min(k + 1, len(pts)))
    Q = pts[nb[:, 1:]]
    c = Q.mean(1, keepdims=True)
    M = Q - c
    C = np.einsum("mki,mkj->mij", M, M) / M.shape[1]
    _, evec = np.linalg.eigh(C)
    N = evec[:, :, 0]
    ref = (pts - pts.mean(0)) if up is None else np.broadcast_to(up, pts.shape)
    N[np.einsum("ij,ij->i", N, ref) < 0] *= -1
    return N


def build_tsdf(pts: np.ndarray, normals: np.ndarray, v: float,
               trunc_mult: float = 3.0, knn: int = 12, pad: int = 3):
    """narrow band の符号付き距離場を密配列で返す。

    v がそのまま平滑化スケールになる（近傍の取り方には依存しない）。
    """
    trunc = trunc_mult * v
    lo = pts.min(0) - pad * v
    hi = pts.max(0) + pad * v
    dims = np.maximum(np.ceil((hi - lo) / v).astype(int) + 1, 2)
    if int(np.prod(dims)) > 60_000_000:
        raise MemoryError(f"格子が大きすぎる: {dims} = {np.prod(dims):,.0f} ボクセル。"
                          f"v を大きくするか領域を切ること")
    sdf = np.full(dims, np.nan, np.float32)

    # 曲面の近傍のボクセルだけを候補にする
    idx = np.rint((pts - lo) / v).astype(np.int64)
    off = np.arange(-int(np.ceil(trunc / v)), int(np.ceil(trunc / v)) + 1)
    O = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    cand = (idx[:, None, :] + O[None, :, :]).reshape(-1, 3)
    cand = np.unique(cand, axis=0)
    cand = cand[np.all((cand >= 0) & (cand < dims), axis=1)]
    centers = lo + cand * v

    tree = cKDTree(pts)
    d, j = tree.query(centers, k=min(knn, len(pts)))
    if d.ndim == 1:
        d, j = d[:, None], j[:, None]
    w = np.exp(-(d / max(v, 1e-30)) ** 2)
    w = np.where(d <= trunc, w, 0.0)
    s = np.einsum("nk,nki->nk", np.ones_like(d),
                  (centers[:, None, :] - pts[j]) * normals[j])
    proj = s.sum(-1) if s.ndim == 3 else np.einsum(
        "nki,nki->nk", centers[:, None, :] - pts[j], normals[j])
    wsum = w.sum(1)
    ok = wsum > 1e-12
    val = np.where(ok, (w * proj).sum(1) / np.maximum(wsum, 1e-30), np.nan)
    val = np.clip(val, -trunc, trunc)
    sdf[cand[ok, 0], cand[ok, 1], cand[ok, 2]] = val[ok]

    # 未定義ボクセルの符号を決める。一律 +trunc で埋めると、物体内部の
    # 未定義領域との境界に**偽の零交差**ができて内側に余分な曲面が生まれる
    # （実測: 球の lfs 誤差が 1.7% から 23% に悪化した）。
    # 格子の外周から未定義領域を塗り分け、外に繋がる方を +、残りを − にする。
    from scipy import ndimage
    unknown = np.isnan(sdf)
    lab, n_lab = ndimage.label(unknown)
    outside = np.zeros(n_lab + 1, bool)
    for sl in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
        outside[np.unique(sl)] = True
    outside[0] = False
    fill = np.where(outside[lab], trunc, -trunc).astype(np.float32)
    sdf = np.where(unknown, fill, sdf)
    return sdf, lo, v


def extract_surface(sdf: np.ndarray, lo: np.ndarray, v: float):
    """零等位面の頂点と、そこでの外向き法線（勾配）を返す。"""
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=0.0, spacing=(v, v, v))
    verts = verts + lo
    n = np.asarray(normals, float)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return verts, -n, faces        # marching_cubes の法線は勾配の逆向き


def surface_from_points(pts: np.ndarray, v: float, up=None, **kw):
    N = estimate_normals(pts, up=up)
    sdf, lo, v = build_tsdf(pts, N, v, **kw)
    return extract_surface(sdf, lo, v)
