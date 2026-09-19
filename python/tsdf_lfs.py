"""平滑化した曲面からの局所特徴サイズ（lfs）。

生の点に Voronoi 極を当てる方法は使えなかった（P1/標本間隔 = 3.2 で一定、
標本密度に比例して収束しない）。中軸はノイズに対して不安定で、測定ノイズが
表面のすぐ近くに作る偽の枝を測ってしまうためである。

対策は「中軸を計算する前に平滑化する」こと。TSDF がやっているのはこれなので、
同じ効果を点群のまま得る:
  1. 半径 h の近傍に平面を当て、点をその平面へ射影する（h は**絶対値で固定**）
  2. 平滑化した点で shrinking ball を回し、最大内接球の半径 = lfs を得る

h を標本間隔に紐付けてはいけない。紐付けると平滑化のスケールが標本とともに
動き、また標本密度を測ることになる。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree


def smooth_and_normals(pts: np.ndarray, h: float, k_max: int = 48,
                       up: np.ndarray | None = None):
    """半径 h の近傍に平面を当て、射影した点と法線を返す。

    h は絶対値で与える（標本間隔に紐付けない）。
    """
    tree = cKDTree(pts)
    nb = tree.query_ball_point(pts, h)
    P = np.empty_like(pts)
    N = np.zeros_like(pts)
    ok = np.zeros(len(pts), bool)
    for i, idx in enumerate(nb):
        if len(idx) < 6:
            P[i] = pts[i]; N[i] = (0, 0, 1.0); continue
        if len(idx) > k_max:
            idx = idx[:k_max]
        Q = pts[idx]
        c = Q.mean(0)
        M = Q - c
        ev, evec = np.linalg.eigh(M.T @ M / len(M))
        n = evec[:, 0]
        P[i] = pts[i] - n * float(n @ (pts[i] - c))     # 平面へ射影
        N[i] = n
        ok[i] = True
    # 法線の向きを揃える（閉じた物体なら外向き、地形なら上向き）
    ref = (pts - pts.mean(0)) if up is None else np.broadcast_to(up, pts.shape)
    flip = np.einsum("ij,ij->i", N, ref) < 0
    N[flip] *= -1
    return P, N, ok


def shrinking_ball_lfs(pts: np.ndarray, normals: np.ndarray,
                       r_init: float, n_iter: int = 60,
                       sep_angle_deg: float = 32.0, rel_tol: float = 1e-3,
                       eps: float = 1e-12) -> np.ndarray:
    """最大内接球の半径（= 中軸までの距離）を反復で求める。

    素朴に実装すると潰れる。球の中心に達したあと、同じ面上のわずかに近い点が
    次々と球を縮め、最後は p の直近の近傍に収束してしまう（実測: 正解 1.0 の
    球で 0.577、ノイズを入れると 0.016 まで落ちた）。

    対策は **分離角** の閾値。球の中心から見た p と q の成す角が小さい q は、
    p と同じ面の上にいるだけなので中軸を決める資格が無い。閾値は Ma らの
    実装にならい 32°。加えて半径の相対変化が小さくなったら止める。

    ただし**初回の縮小には分離角を課さない**。初期半径が大きいと中心が物体の
    遥か外にあり、そこから見た p と q はほぼ同じ方向に見えて必ず弾かれるため、
    球が一度も縮まなくなる（実測: 全ケースが r_init のまま返った）。
    初回は幾何的に正しい半径を与えるので、保護は 2 回目以降で足りる。
    """
    n = len(pts)
    tree = cKDTree(pts)
    r = np.full(n, float(r_init))
    has_shrunk = np.zeros(n, bool)
    cos_min = np.cos(np.radians(sep_angle_deg))
    idx_all = np.arange(n)
    for _ in range(n_iter):
        c = pts - normals * r[:, None]
        d, j = tree.query(c, k=2)
        same = (j[:, 0] == idx_all)
        q = np.where(same[:, None], pts[j[:, 1]], pts[j[:, 0]])
        dq = np.where(same, d[:, 1], d[:, 0])

        # 中心から見た p と q の分離角
        vp = pts - c
        vq = q - c
        npv = np.linalg.norm(vp, axis=1)
        nqv = np.linalg.norm(vq, axis=1)
        cos_sep = np.einsum("ij,ij->i", vp, vq) / np.maximum(npv * nqv, eps)
        separated = cos_sep < cos_min          # 角が閾値より大きい＝別の面

        diff = pts - q
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist = np.sqrt(np.maximum(dist2, eps))
        cosang = np.einsum("ij,ij->i", normals, diff) / dist
        with np.errstate(divide="ignore", invalid="ignore"):
            r_new = dist2 / (2.0 * np.maximum(cosang, eps) * dist)

        good = ((dq < r - eps) & (separated | ~has_shrunk) & np.isfinite(r_new)
                & (r_new > 0) & (r_new < r * (1 - rel_tol)))
        if not good.any():
            break
        r = np.where(good, r_new, r)
        has_shrunk |= good
    return r
