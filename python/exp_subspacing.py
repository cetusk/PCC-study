"""点間隔以下のスケールに構造はあるか（次テーマの事前検証）。

LSB スイープで分かったこと:
  octree は点間隔より細かいスケールでは 1 レベルあたり 3 ビット丸ごと掛かる。
  つまり「点のサブボクセル位置」は octree から見て一様ランダムである。

仮説:
  点群は表面のサンプルである。ならばサブボクセル位置は等方ではないはずで、
    ・法線方向  … 表面の位置で決まる（低エントロピー）
    ・接平面内  … どこをサンプリングしたかで決まる（高エントロピー）
  に分解できる。3 自由度のうち 1 つだけが圧縮可能、という構造になる。

測るもの: 近傍から当てた局所平面に対する、法線方向と接平面方向の散らばりの比。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree
from pcio import load


def local_frame_decompose(w: np.ndarray, k: int = 12, sample: int = 100_000,
                          seed: int = 0):
    """各点について近傍 k 点から平面を当て、その点の法線／接線方向成分を返す。

    自分自身は平面の当てはめから除く（自己参照で当たり前に 0 になるのを防ぐ）。
    """
    tree = cKDTree(w)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    d, nb = tree.query(w[idx], k=k + 1)
    nb = nb[:, 1:]                      # 自分自身を除外
    P = w[nb]                           # (M, k, 3)
    c = P.mean(1, keepdims=True)
    Q = P - c
    # 各点の近傍共分散の最小固有ベクトル = 法線
    C = np.einsum("mki,mkj->mij", Q, Q) / Q.shape[1]
    evals, evecs = np.linalg.eigh(C)
    n = evecs[:, :, 0]                  # 最小固有値の固有ベクトル
    t1 = evecs[:, :, 1]
    t2 = evecs[:, :, 2]
    v = w[idx] - c[:, 0, :]             # 近傍重心から見た自分の位置
    return (np.einsum("mi,mi->m", v, n),
            np.einsum("mi,mi->m", v, t1),
            np.einsum("mi,mi->m", v, t2),
            d[:, 1], evals)


def run(path, max_points=400_000, k=12):
    pc = load(path, max_points=max_points)
    w = pc.world()
    w = w - w.mean(0)
    dn, dt1, dt2, nn, evals = local_frame_decompose(w, k=k)
    sp = float(np.median(nn))
    print(f"# 点間隔以下の構造  {Path(path).name}  N={pc.n:,}  近傍数 k={k}")
    print(f"\n最近傍距離 中央値 = {sp*1000:.1f} mm  （octree の折れ点はこの付近に来る）\n")
    print(f"{'方向':<22}{'標準偏差':>12}{'P95(|·|)':>12}{'点間隔比':>10}")
    print("-" * 58)
    for nm, v in (("法線方向", dn), ("接平面 方向1", dt1), ("接平面 方向2", dt2)):
        print(f"{nm:<22}{v.std()*1000:>11.2f}mm{np.percentile(np.abs(v),95)*1000:>11.2f}mm"
              f"{v.std()/sp:>10.3f}")
    print("-" * 58)
    ratio = dn.std() / max(1e-12, 0.5 * (dt1.std() + dt2.std()))
    print(f"\n法線方向 / 接平面方向 の散らばり比 = {ratio:.3f}")
    print(f"  → 1 に近ければ等方（構造なし）。小さいほど「表面らしさ」が強い。")
    # 符号化に使える利得の目安（微分エントロピーの差）
    gain = np.log2(0.5 * (dt1.std() + dt2.std()) / max(1e-12, dn.std()))
    print(f"\n法線方向を接平面方向と同じ精度で送る場合に浮くビット数 "
          f"≈ {gain:.2f} bit/点")
    print(f"  （3 自由度のうち 1 つぶんが上限。octree は現状これを 0 と見なしている）")
    # 平面性の分布
    lin = evals[:, 0] / np.maximum(evals.sum(1), 1e-30)
    print(f"\n局所平面性（最小固有値の比）: 中央値 {np.median(lin):.4f}  "
          f"平面上なら 0 に近い")
    frac = float((lin < 0.02).mean())
    print(f"  平面的と見なせる点の割合（比 < 0.02）= {frac*100:.1f}%")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/ahn4/31HZ1_20.LAZ")
