"""多スケールの秩序を測る。

(1) 短距離・中距離・長距離の秩序   近傍距離の分布と動径分布関数 g(r)
(2) 結合配向秩序                  Steinhardt の q4 / q6
(5) 変換に対する応答              回転させたときの符号長の変化

狙いは「圧縮方向のガイド（エンベロープ）」になりうるかの当たりを付けること。
秩序があるということは予測できるということなので、どのスケールに秩序が
残っているかが分かれば、そこを攻めればよい、という道具立てになる。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y
import zstandard as zstd
from baselines import byte_split


# ------------------------------------------------ (1) 短距離秩序: 近傍距離の分布

def neighbor_order(w: np.ndarray, kmax: int = 6, sample: int = 50_000,
                   seed: int = 0) -> dict:
    """第 k 近傍距離の「揃い方」。変動係数が小さいほど規則的な配置。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    d, _ = cKDTree(w).query(w[idx], k=kmax + 1)
    out = []
    for k in range(1, kmax + 1):
        v = d[:, k]
        out.append(dict(k=k, median=float(np.median(v)),
                        cv=float(v.std() / max(v.mean(), 1e-30))))
    return dict(ranks=out, spacing=out[0]["median"])


# ---------------------------------------------- (1) 中距離秩序: 動径分布関数 g(r)

def radial_distribution(w: np.ndarray, sp: float, r_max_mult: float = 8.0,
                        nbins: int = 80, sample: int = 4000, seed: int = 0) -> dict:
    """面上の点群を想定し、2 次元の殻（2πr dr）で規格化する。

    規則的な走査格子があれば g(r) に明瞭なピークが立つ。
    ランダム標本なら 1 に張り付く。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    tree = cKDTree(w)
    r_max = sp * r_max_mult
    edges = np.linspace(0, r_max, nbins + 1)
    counts = np.zeros(nbins)
    for i in idx:
        nb = tree.query_ball_point(w[i], r_max)
        if len(nb) < 2:
            continue
        dd = np.linalg.norm(w[nb] - w[i], axis=1)
        dd = dd[dd > 0]
        counts += np.histogram(dd, bins=edges)[0]
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell = 2 * np.pi * centers * (edges[1] - edges[0])     # 面上の環の面積
    dens = counts / (len(idx) * np.maximum(shell, 1e-30))
    g = dens / max(np.median(dens[nbins // 2:]), 1e-30)
    return dict(r=centers / sp, g=g, peak=float(g.max()),
                peak_at=float(centers[int(np.argmax(g))] / sp))


# ------------------------------------------------------- (2) 結合配向秩序 q_l

def steinhardt(w: np.ndarray, l: int = 6, k: int = 8, sample: int = 20_000,
               seed: int = 0) -> float:
    """Steinhardt の結合配向秩序パラメータ q_l。

    近傍への方向ベクトルを球面調和関数に展開し、その大きさを見る。
    配置が規則的なら大きく、等方ランダムなら小さくなる。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    _, nb = cKDTree(w).query(w[idx], k=k + 1)
    v = w[nb[:, 1:]] - w[idx][:, None, :]
    r = np.linalg.norm(v, axis=2)
    theta = np.arccos(np.clip(v[:, :, 2] / np.maximum(r, 1e-30), -1, 1))
    phi = np.arctan2(v[:, :, 1], v[:, :, 0])
    acc = 0.0
    for m in range(-l, l + 1):
        y = sph_harm_y(l, m, theta, phi)
        acc += np.abs(y.mean(axis=1)) ** 2
    q = np.sqrt(4 * np.pi / (2 * l + 1) * acc)
    return float(np.mean(q))


# ------------------------------------------- (5) 変換に対する応答: 回転感度

def _bits_grid(w: np.ndarray, v: float) -> float:
    q = np.rint((w - w.min(0)) / v).astype(np.int64)
    t = 0
    for i in range(3):
        d = np.diff(q[:, i], prepend=np.int64(0))
        z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
        t += len(zstd.ZstdCompressor(level=12).compress(byte_split(z)))
    return t * 8 / len(w)


def _rand_rot(rng) -> np.ndarray:
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def rotation_sensitivity(w: np.ndarray, v: float, n_rot: int = 12,
                         seed: int = 0) -> dict:
    """座標軸に対する向きで符号長がどれだけ変わるか。

    格子も octree も軸に揃っているので、物体の向きは符号長に効く。
    効くなら、回転を 4 つの数で仕様に書くだけで得ができる。
    """
    rng = np.random.default_rng(seed)
    c = w.mean(0)
    base = _bits_grid(w, v)
    # 主成分に揃えた向き
    u, s, vt = np.linalg.svd((w - c)[::max(1, len(w)//200_000)], full_matrices=False)
    pca = _bits_grid((w - c) @ vt.T, v)
    vals = []
    for _ in range(n_rot):
        vals.append(_bits_grid((w - c) @ _rand_rot(rng).T, v))
    vals = np.array(vals)
    return dict(base=base, pca=pca, rot_min=float(vals.min()),
                rot_max=float(vals.max()), rot_mean=float(vals.mean()),
                spread=float((vals.max() - vals.min()) / vals.mean()))


def hexatic(w: np.ndarray, m: int = 6, k: int = 6, sample: int = 20_000,
            seed: int = 0) -> dict:
    """接平面内の m 回対称な結合配向秩序 ψ_m。

    点群は面上にあるので、3 次元の球面調和で測る q_l は「面が平らである」ことに
    支配されて面内の配置を見ない。局所平面に射影してから 2 次元で測る。
      ψ_m(i) = | (1/k) Σ_j exp(i m θ_ij) |
    規則的な格子なら 1 に近づき、ランダムなら 1/√k 程度になる。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    _, nb = cKDTree(w).query(w[idx], k=k + 1)
    P = w[nb[:, 1:]]
    c = w[idx][:, None, :]
    v = P - c
    # 近傍の共分散から接平面の基底を作る
    Q = v - v.mean(1, keepdims=True)
    C = np.einsum("mki,mkj->mij", Q, Q) / Q.shape[1]
    _, evec = np.linalg.eigh(C)
    e1, e2 = evec[:, :, 2], evec[:, :, 1]          # 大きい方 2 つが接平面
    x = np.einsum("mki,mi->mk", v, e1)
    y = np.einsum("mki,mi->mk", v, e2)
    th = np.arctan2(y, x)
    psi = np.abs(np.exp(1j * m * th).mean(axis=1))
    return dict(psi=float(psi.mean()), psi_med=float(np.median(psi)),
                random_ref=float(1 / np.sqrt(k)))
