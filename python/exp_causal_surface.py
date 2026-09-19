"""点間隔以下の構造は「因果的な」予測器でも取れるのか。

前回の測定 (exp_subspacing.py) は全近傍から平面を当てていたが、実際の復号器は
まだ復号していない点を使えない。符号化順を固定し、**自分より前の点だけ**から
平面を当てたときに、法線方向の残差がどれだけ小さくなるかを測り直す。

ここで利得が消えるなら、この路線は実装する価値がない。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree
from pcio import load
from baselines import morton3


def causal_plane_residual(w: np.ndarray, order: np.ndarray, k: int = 8,
                          pool: int = 48, warmup: int = 2000, sample: int = 120_000,
                          seed: int = 0):
    """符号化順 order に沿って、各点を「自分より前の点」だけから予測する。

    返り値: (法線方向残差, 接平面方向残差1, 接平面方向残差2, 使えた近傍数)
    """
    wo = w[order]
    tree = cKDTree(wo)
    rng = np.random.default_rng(seed)
    cand = rng.choice(np.arange(warmup, len(wo)), min(sample, len(wo) - warmup),
                      replace=False)
    cand.sort()
    _, nb = tree.query(wo[cand], k=pool)

    dn, dt1, dt2, used = [], [], [], []
    for row, i in zip(nb, cand):
        prev = row[(row < i)][:k]          # 既に復号済みの点だけ
        if len(prev) < 4:
            continue
        P = wo[prev]
        c = P.mean(0)
        Q = P - c
        C = Q.T @ Q / len(Q)
        ev, evec = np.linalg.eigh(C)
        v = wo[i] - c
        dn.append(float(v @ evec[:, 0]))
        dt1.append(float(v @ evec[:, 1]))
        dt2.append(float(v @ evec[:, 2]))
        used.append(len(prev))
    return (np.array(dn), np.array(dt1), np.array(dt2), np.array(used))


def noncausal_plane_residual(w: np.ndarray, k: int = 8, sample: int = 120_000,
                             seed: int = 0):
    tree = cKDTree(w)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    _, nb = tree.query(w[idx], k=k + 1)
    P = w[nb[:, 1:]]
    c = P.mean(1, keepdims=True)
    Q = P - c
    C = np.einsum("mki,mkj->mij", Q, Q) / Q.shape[1]
    _, evec = np.linalg.eigh(C)
    v = w[idx] - c[:, 0, :]
    return (np.einsum("mi,mi->m", v, evec[:, :, 0]),
            np.einsum("mi,mi->m", v, evec[:, :, 1]),
            np.einsum("mi,mi->m", v, evec[:, :, 2]))


def run(path, max_points=1_200_000, k=8, cls_filter=None):
    pc = load(path, max_points=max_points)
    w = pc.world(); w = w - w.mean(0)
    if cls_filter is not None and "classification" in pc.raw:
        m = pc.raw["classification"] == cls_filter
        w = np.ascontiguousarray(w[m])
    tree = cKDTree(w)
    sp = float(np.median(tree.query(w[::37], k=2)[0][:, 1]))

    orders = {
        "取得順（走査線順）": np.arange(len(w)),
        "Morton 順": np.argsort(morton3((w - w.min(0)) / max(sp / 8, 1e-9))),
    }
    print(f"# 因果的な表面予測  {Path(path).name}  N={len(w):,}  k={k}"
          + (f"  class={cls_filter}" if cls_filter is not None else ""))
    print(f"最近傍距離 中央値 = {sp*1000:.1f} mm\n")

    nn, nt1, nt2 = noncausal_plane_residual(w, k=k)
    t_nc = 0.5 * (nt1.std() + nt2.std())
    print(f"{'予測器':<26}{'法線σ':>10}{'接線σ':>10}{'比':>8}{'利得':>9}{'近傍数':>8}")
    print("-" * 74)
    print(f"{'非因果（全近傍）※参考':<26}{nn.std()*1000:>9.1f}mm{t_nc*1000:>9.1f}mm"
          f"{nn.std()/t_nc:>8.3f}{np.log2(t_nc/nn.std()):>9.2f}{k:>8}")
    for name, o in orders.items():
        dn, d1, d2, used = causal_plane_residual(w, o, k=k)
        if len(dn) == 0:
            print(f"{name:<26}  （有効な近傍が得られず）")
            continue
        t = 0.5 * (d1.std() + d2.std())
        print(f"{name:<26}{dn.std()*1000:>9.1f}mm{t*1000:>9.1f}mm"
              f"{dn.std()/t:>8.3f}{np.log2(t/dn.std()):>9.2f}{used.mean():>8.1f}")
    print("-" * 74)
    print("\n『利得』= 法線方向を接平面方向と同じ精度で送る場合に浮くビット数の目安。")
    print("非因果の値は実現できない上限。因果の値との差が、順序の制約で失う分。")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/raw/ahn4/31HZ1_20.LAZ")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=1_200_000)
    ap.add_argument("--cls", type=int, default=None)
    a = ap.parse_args()
    run(a.path, a.max_points, a.k, a.cls)
