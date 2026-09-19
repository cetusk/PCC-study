"""決定実験: 局所表面から octree の子ノード占有を予測できるか。

LSB スイープが示したこと:
  点間隔以下では 1 レベルあたり 3.04 bit 掛かる。= 8 個の子のどれかが
  一様ランダムに見えている。

問い:
  粗いレベルの占有（＝復号器が既に知っている情報）から局所平面を当て、
  その平面が親セルのどこを通るかで、子オクタントを当てられるか。

  当てられるなら、点間隔以下のレベルにも文脈がある。
  当てられないなら、この路線は打ち止めで、3 bit は本当に情報である。

前回までの『法線方向の残差が小さい』は等方性のずれを測っていただけで、
octree に対する絶対的な優位を示していなかった。ここでは
H(子オクタント | 平面からの予測) を直接測る。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pcio import load


def _entropy(a: np.ndarray) -> float:
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _cond_entropy(a: np.ndarray, ctx: np.ndarray) -> float:
    ab = np.stack([np.asarray(a, np.int64), np.asarray(ctx, np.int64)], 1)
    _, c = np.unique(ab, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum()) - _entropy(ctx)


def octant_experiment(w: np.ndarray, v: float, radius: int = 2,
                      sample: int = 150_000, seed: int = 0) -> dict:
    """セルサイズ v のレベルで、子オクタントが粗レベル占有から予測できるか。"""
    parent = np.floor(w / (2 * v)).astype(np.int64)
    child = np.floor(w / v).astype(np.int64) & 1          # (N,3) 0/1
    oct_id = (child[:, 0] | (child[:, 1] << 1) | (child[:, 2] << 2))

    # 粗レベル（親セル）の占有集合。復号器がこのレベルまでは持っている。
    key = {}
    pk = (parent[:, 0].astype(np.int64) * 73856093
          ^ parent[:, 1].astype(np.int64) * 19349663
          ^ parent[:, 2].astype(np.int64) * 83492791)
    uniq_p, first = np.unique(pk, return_index=True)
    occ = set(uniq_p.tolist())

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)

    offs = [(dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
            if not (dx == dy == dz == 0)]
    offs_arr = np.array(offs, dtype=np.int64)

    pred_oct = np.full(len(idx), -1, np.int64)
    n_used = np.zeros(len(idx), np.int64)
    for t, i in enumerate(idx):
        p0 = parent[i]
        cand = p0 + offs_arr
        ck = (cand[:, 0] * 73856093 ^ cand[:, 1] * 19349663 ^ cand[:, 2] * 83492791)
        hit = np.fromiter((k in occ for k in ck.tolist()), bool, len(ck))
        if hit.sum() < 6:
            continue
        # 占有している近傍親セルの中心に平面を当てる
        C = (cand[hit].astype(np.float64) + 0.5) * (2 * v)
        c = C.mean(0)
        Q = C - c
        ev, evec = np.linalg.eigh(Q.T @ Q / len(Q))
        nvec = evec[:, 0]
        # 親セル内の 8 子中心のうち、平面に最も近いものを予測とする
        base = p0.astype(np.float64) * (2 * v)
        cc = base + (np.array([[x, y, z] for z in (0, 1) for y in (0, 1)
                               for x in (0, 1)], float) + 0.5) * v
        dist = np.abs((cc - c) @ nvec)
        j = int(np.argmin(dist))
        # 上の並びは x が最内なので oct_id と同じ規約に直す
        zz, yy, xx = j // 4, (j // 2) % 2, j % 2
        pred_oct[t] = xx | (yy << 1) | (zz << 2)
        n_used[t] = int(hit.sum())

    m = pred_oct >= 0
    a = oct_id[idx][m]
    p = pred_oct[m]
    return dict(v=v, n=int(m.sum()), coverage=float(m.mean()),
                H=_entropy(a), H_cond=_cond_entropy(a, p),
                acc=float((a == p).mean()), used=float(n_used[m].mean()))


def run(path, max_points=1_000_000, cells_mm=(16, 32, 64, 128, 256)):
    pc = load(path, max_points=max_points)
    w = pc.world(); w = w - w.min(0)
    print(f"# 子オクタントは局所平面から予測できるか  {Path(path).name}  N={pc.n:,}\n")
    print(f"{'セル':>8}{'対象点':>10}{'被覆':>8}{'H(子)':>9}{'H(子|予測)':>12}"
          f"{'情報量':>9}{'的中率':>9}{'近傍':>7}")
    print("-" * 74)
    for mm in cells_mm:
        r = octant_experiment(w, mm / 1000.0)
        mi = r["H"] - r["H_cond"]
        print(f"{mm:>6}mm{r['n']:>10,}{r['coverage']*100:>7.0f}%{r['H']:>9.3f}"
              f"{r['H_cond']:>12.3f}{mi:>9.3f}{r['acc']*100:>8.1f}%{r['used']:>7.1f}")
    print("-" * 74)
    print("\n『情報量』= 平面予測が子オクタントについて与えるビット数。")
    print("  0 に近ければ、そのスケールに文脈は無く 3 bit は本当に情報。")
    print("  的中率の基準は 1/8 = 12.5%。")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/raw/ahn4/31HZ1_20.LAZ")
    ap.add_argument("--max-points", type=int, default=1_000_000)
    a = ap.parse_args()
    run(a.path, a.max_points)
