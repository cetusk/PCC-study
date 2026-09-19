"""Euler 特性変換（ECT）による形状の署名。

(3) の検討: パーシステンス図から 3D 構造を復元できるという話は、
単一の図ではなく**全方向について取った図の族**（persistent homology transform /
Euler characteristic transform）のことで、これは単射だが表現は元より大きい。
つまり圧縮にはならない。

使えるとすれば形状の比較・検索・重複検出で、
「同じ物体を何度もスキャンしたデータで重複を見つけ、差分だけ送る」
という経路なら圧縮に繋がりうる。その可否を測る。

実装: アルファ複体を 1 回だけ作り、方向 v ごとに
  χ(t) = Σ_{σ: h(σ) ≤ t} (-1)^dim(σ),   h(σ) = max_{頂点 x∈σ} <x, v>
を計算する。方向 × 閾値 の行列が署名になる。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gudhi
from scipy.spatial import cKDTree


def fibonacci_directions(n: int = 32) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], 1)


def ect(pts: np.ndarray, n_dir: int = 32, n_thresh: int = 24,
        alpha_mult: float = 2.0) -> np.ndarray:
    """方向 × 閾値 の Euler 特性行列を返す。正規化済み（スケール・位置に不変）。"""
    c = pts.mean(0)
    p = pts - c
    scale = float(np.linalg.norm(p, axis=1).max())
    p = p / max(scale, 1e-30)
    d, _ = cKDTree(p).query(p, k=2)
    sp = float(np.median(d[:, 1]))
    st = gudhi.AlphaComplex(points=p).create_simplex_tree(
        max_alpha_square=(alpha_mult * sp) ** 2)
    simplices = [(s, len(s) - 1) for s, _ in st.get_simplices()]
    dirs = fibonacci_directions(n_dir)
    ts = np.linspace(-1.0, 1.0, n_thresh)
    out = np.zeros((n_dir, n_thresh))
    sign = np.array([(-1) ** dim for _, dim in simplices], float)
    verts = [np.asarray(s) for s, _ in simplices]
    for k, v in enumerate(dirs):
        h = p @ v
        hs = np.array([h[s].max() for s in verts])
        order = np.argsort(hs)
        cum = np.cumsum(sign[order])
        idx = np.searchsorted(hs[order], ts, side="right") - 1
        out[k] = np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)
    return out / max(len(pts), 1)


def ect_distance(a: np.ndarray, b: np.ndarray) -> float:
    """署名どうしの距離（署名は既に同じ基準系で計算されている前提）。"""
    return float(np.abs(a - b).mean())


def pca_frame(pts: np.ndarray) -> np.ndarray:
    """主成分による基準系。符号の任意性は 3 次モーメントで固定する。"""
    c = pts.mean(0)
    p = pts - c
    _, _, vt = np.linalg.svd(p[::max(1, len(p) // 50_000)], full_matrices=False)
    R = vt                                   # 行が主軸
    proj = p @ R.T
    for i in range(3):
        if np.mean(proj[:, i] ** 3) < 0:     # 歪度の符号で向きを決める
            R[i] = -R[i]
    if np.linalg.det(R) < 0:
        R[2] = -R[2]
    return R


def ect_aligned(pts: np.ndarray, **kw) -> np.ndarray:
    """PCA で正準化してから署名を取る。回転に（近似的に）不変になる。"""
    R = pca_frame(pts)
    return ect((pts - pts.mean(0)) @ R.T, **kw)
