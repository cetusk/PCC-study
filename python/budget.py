"""誤差予算の決定。

これまで bounded 操作の誤差上限は、センサのノイズ床（計測側の都合）だけを
根拠にしていた。位相は物体側の根拠を与えるので、両方を出して厳しい方を採る。

  eps_sensor   ノイズ床。ここより細かい桁は測れていない
  eps_topology 位相が壊れない限界。ここより粗いと物体が別物になる
  eps          min(eps_sensor, eps_topology)

注意: 位相側は標本を間引いて計算するため、間引くほど細かい構造を見落として
      **甘い（大きい）値**になる。点数を変えて安定性を確認し、揺れていれば警告する。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree
import gudhi
from exp_topology import persistence


def point_spacing_of(w: np.ndarray, sample: int = 50_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    d, _ = cKDTree(w).query(w[idx], k=2)
    return float(np.median(d[:, 1]))


def _bottleneck_ratio(pts: np.ndarray, base, sp: float, v: float) -> float:
    q = np.unique(np.rint(pts / v).astype(np.int64), axis=0) * v
    if len(q) < 50:
        return float("inf")
    iv = persistence(q)
    return max(gudhi.bottleneck_distance(base[k], iv[k]) for k in (1, 2)) / sp


def topological_budget(w: np.ndarray, n_sample: int = 12_000, threshold: float = 0.5,
                       lo_m: float = 1 / 16, hi_m: float = 8.0, iters: int = 8,
                       seed: int = 0, verbose: bool = False) -> dict:
    """位相が壊れない最大の格子サイズを二分探索で求め、その誤差上限を返す。

    ボトルネック距離は格子サイズに対して単調に増えるので二分探索が使える
    （4 データセットで単調性を確認済み）。
    """
    rng = np.random.default_rng(seed)
    pts = w[rng.choice(len(w), min(n_sample, len(w)), replace=False)]
    d, _ = cKDTree(pts).query(pts, k=2)
    sp = float(np.median(d[:, 1]))
    base = persistence(pts)

    lo, hi = lo_m, hi_m
    if _bottleneck_ratio(pts, base, sp, sp * lo) > threshold:
        best = lo                                   # 最小候補でも駄目
    else:
        best = lo
        for _ in range(iters):
            mid = np.sqrt(lo * hi)                  # 対数中点
            r = _bottleneck_ratio(pts, base, sp, sp * mid)
            if verbose:
                print(f"      v/点間隔 {mid:6.3f} -> 距離/点間隔 {r:6.3f}")
            if r <= threshold:
                lo, best = mid, mid
            else:
                hi = mid
    v = sp * best
    return dict(n_sample=len(pts), spacing=sp, v_over_spacing=best,
                v=v, eps=float(np.sqrt(3) / 2 * v), threshold=threshold)


def noise_floor_estimate(w: np.ndarray, k: int = 8, sample: int = 100_000,
                         seed: int = 0) -> dict:
    """センサ側の目安。近傍から当てた局所平面からの残差の標準偏差。

    面が取れる対象（地面・建物・オブジェクト表面）では測距ノイズの目安になるが、
    植生のように本質的に 3 次元的な対象では過大評価になる。あくまで proxy。
    """
    tree = cKDTree(w)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    _, nb = tree.query(w[idx], k=k + 1)
    P = w[nb[:, 1:]]
    c = P.mean(1, keepdims=True)
    Q = P - c
    C = np.einsum("mki,mkj->mij", Q, Q) / Q.shape[1]
    ev, evec = np.linalg.eigh(C)
    resid = np.einsum("mi,mi->m", w[idx] - c[:, 0, :], evec[:, :, 0])
    planar = ev[:, 0] / np.maximum(ev.sum(1), 1e-30) < 0.02
    sigma_all = float(resid.std())
    sigma_planar = float(resid[planar].std()) if planar.sum() > 100 else sigma_all
    return dict(sigma=sigma_planar, sigma_all=sigma_all,
                planar_frac=float(planar.mean()), eps=sigma_planar)


def error_budget(w: np.ndarray, *, sensor_eps: float | None = None,
                 use_topology=True, use_noise_floor=True, keep_top: int = 6,
                 n_sample=20_000, threshold=0.5, stability_check=True,
                 verbose=False) -> dict:
    """両側の根拠を出して、厳しい方を採る。

    sensor_eps を渡せばセンサ側はその値を使う。データからの推定（局所平面残差）は
    面が支配的な対象でしか当てにならないので、仕様値が分かっているなら渡すこと。
    KITTI のような市街地シーンでは、植生や柱の実構造を残差に含んでしまい
    推定値が甘くなる（実測 30mm / HDL-64E の測距精度 ±20mm）。
    """
    out = {}
    if sensor_eps is not None:
        out["sensor"] = dict(eps=float(sensor_eps), source="指定値")
    elif use_noise_floor:
        nf = noise_floor_estimate(w)
        nf["source"] = "局所平面残差からの推定（proxy）"
        out["sensor"] = nf
    if use_topology:
        # 安定性定理にもとづく版。保存する特徴の数（既定 6 個）で決まる。
        t = topological_budget_v2(w, keep_top=keep_top, n_sample=n_sample)
        t["source"] = f"寿命 {t.get('L_min', 0):.5g} の特徴を保存（上位 {t.get('n_kept')} 個）"
        out["topology"] = t
    cands = [v["eps"] for v in out.values() if v.get("eps")]
    out["eps"] = float(min(cands)) if cands else None
    out["binding"] = min(((v["eps"], k) for k, v in out.items()
                          if isinstance(v, dict) and v.get("eps")),
                         default=(None, None))[1]
    return out


def normalize_geometry(w: np.ndarray, *, eps=None, budget_kw=None,
                       verbose=False) -> dict:
    """誤差予算を決めてから幾何表現を選ぶ、一連の流れ。

    予算は「ノイズ床」と「位相の限界」の厳しい方。表現は候補を実際に
    符号化して短い方。どちらも推測に頼らず、決めた結果は仕様に書く。
    """
    import normalize as nz
    out = {}
    if eps is None:
        b = error_budget(w, verbose=verbose, **(budget_kw or {}))
        eps = b["eps"]
        out["budget"] = b
        if verbose:
            print(f"    誤差予算 {eps:.5g}（律速: {b['binding']}）")
            if "topology" in b and not b["topology"].get("stable", True):
                print(f"      ※ 位相側は標本数で揺れている"
                      f"（{b['topology']['stability']:.2f}倍）。"
                      f"位相が律速でないなら影響しない")
    op, streams = nz.choose_geometry(w, eps, verbose=verbose)
    out["eps"] = eps
    out["op"] = op
    out["streams"] = streams
    return out


# ===================================================== 位相からの予算（訂正版）

"""【訂正】ボトルネック距離を直接使う方法は、物体側の基準になっていなかった。

パーシステントホモロジーの安定性定理により、点集合を δ だけ動かしたとき
パーシステンス図のボトルネック距離は δ 以下に抑えられる。量子化 v の摂動は
δ = (√3/2)v なので、距離は v に比例して増えるだけで、物体が何であるかを
ほとんど反映しない。実測でも「限界 v ≈ その標本自身の点間隔」が常に成り立ち、
標本を増やすと限界がそのまま縮んだ（収束しない）。測っていたのは
『標本より細かい構造は見えない』という同語反復だった。

物体側の量は別のところにある。同じ安定定理から、
  寿命 L の特徴は、摂動 δ < L/2 なら必ず生き残る
が言える。したがって物体固有の基準は**保存したい特徴の最小寿命**であり、

  eps_topology = L_min / 2

となる。L_min は「どの特徴まで残すか」という要件そのものなので、位相は
数値を一つ返すのではなく、**特徴とその寿命の一覧（メニュー）**を返すのが正しい。
"""


def feature_spectrum(w: np.ndarray, n_sample: int = 20_000, seed: int = 0,
                     top: int = 8) -> dict:
    """物体が持つ位相的特徴と、その寿命を列挙する。

    寿命 L の特徴を保存したければ量子化の誤差を L/2 未満に抑えればよい
    （安定性定理）。どの特徴まで残すかは用途の判断なので、ここでは決めない。
    """
    rng = np.random.default_rng(seed)
    pts = w[rng.choice(len(w), min(n_sample, len(w)), replace=False)]
    d, _ = cKDTree(pts).query(pts, k=2)
    sp = float(np.median(d[:, 1]))
    iv = persistence(pts)
    feats = []
    for dim in (1, 2):
        a = iv[dim]
        if len(a) == 0:
            continue
        fin = a[np.isfinite(a[:, 1])]
        for b_, d_ in fin:
            feats.append(dict(dim=dim, birth=float(b_), death=float(d_),
                              life=float(d_ - b_)))
    feats.sort(key=lambda f: -f["life"])
    return dict(spacing=sp, n_sample=len(pts), features=feats[:top],
                n_features=len(feats))


def report_spectrum(w: np.ndarray, label: str, n_sample: int = 20_000, top: int = 6):
    s = feature_spectrum(w, n_sample=n_sample, top=top)
    sp = s["spacing"]
    print(f"## {label}   標本 {s['n_sample']:,}   点間隔 {sp:.5g}   "
          f"有限寿命の特徴 {s['n_features']:,} 個")
    print(f"   {'次元':>4}{'誕生':>11}{'消滅':>11}{'寿命 L':>11}{'L/点間隔':>10}"
          f"{'保存に必要な ε = L/2':>21}")
    for f in s["features"]:
        print(f"   {'H'+str(f['dim']):>4}{f['birth']:>11.5g}{f['death']:>11.5g}"
              f"{f['life']:>11.5g}{f['life']/sp:>10.2f}{f['life']/2:>21.5g}")
    return s


def topological_budget_v2(w: np.ndarray, *, keep_top: int | None = None,
                          min_life: float | None = None, n_sample: int = 20_000,
                          seed: int = 0) -> dict:
    """保存したい特徴を指定すると、必要な誤差上限を返す（安定性定理による）。

    keep_top=k   寿命の長い順に k 個を保存する
    min_life=L   寿命 L 以上の特徴をすべて保存する
    どちらも指定しなければ、最も長寿命の特徴だけを保存する（最も緩い基準）。
    """
    s = feature_spectrum(w, n_sample=n_sample, seed=seed, top=10_000)
    feats = s["features"]
    if not feats:
        return dict(eps=None, reason="有限寿命の特徴が無い", spacing=s["spacing"])
    if min_life is not None:
        kept = [f for f in feats if f["life"] >= min_life]
    elif keep_top is not None:
        kept = feats[:max(1, keep_top)]
    else:
        kept = feats[:1]
    L_min = min(f["life"] for f in kept)
    return dict(eps=float(L_min / 2), L_min=float(L_min), n_kept=len(kept),
                spacing=s["spacing"], n_sample=s["n_sample"],
                L_over_spacing=float(L_min / s["spacing"]),
                top_life=float(feats[0]["life"]))
