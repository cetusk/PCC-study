"""取得構造スコア — センサ由来の構造がどれだけ生き残っているか。

ここまでの計測で、圧縮に効く支配的な要因は「取得プロセスの構造が
データに残っているか」だった。サンプリング位置が幾何のビットの 78〜89% を
占め、その位置を説明できるのは物体ではなく走査幾何だからである。

そして同じことを別の角度から測る指標が複数あることが分かった。

  距離の規則性   第1近傍距離の変動係数 CV
  角度の規則性   接平面内の結合配向秩序 ψ6
  向きの特権性   回転させたときの符号長のばらつき
  順序の意味     Morton に並べ替えたときの符号長の変化
  走査の痕跡     方位角の単調性 / 軸が固定格子に乗っているか

【設計方針】このスコアは判定を下さない。候補集合を絞って試行時間を減らし、
診断メッセージを出すだけで、最終的な表現の選択は必ず実測で決める
（当たって +13%、外して −174% という非対称性があるため）。
"""
from __future__ import annotations
import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree
import zstandard as zstd
from baselines import byte_split, morton3
from exp_order import hexatic, neighbor_order, rotation_sensitivity


def _bits(cols, n) -> float:
    t = 0
    for v in cols:
        d = np.diff(np.asarray(v, np.int64), prepend=np.int64(0))
        z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
        t += len(zstd.ZstdCompressor(level=12).compress(byte_split(z)))
    return t * 8 / n


def axis_grid_signature(w: np.ndarray, spacing: float,
                        sample: int = 200_000) -> list[dict]:
    """各軸が固定刻みの格子に乗っているかを調べる。

    走査軸と格納格子を区別する必要がある。どちらも「格子に乗っている」が、
      走査軸  … 刻みが点間隔と同程度（センサが実際にその刻みで走査した）
      格納格子… 刻みが点間隔よりずっと細かい（LAS の scale など、単なる量子化）
    刻み / 点間隔 で切り分ける。
    """
    out = []
    for i in range(3):
        v = w[:sample, i]
        u = np.unique(v)
        rec = dict(levels=int(len(u)), step=0.0, on_grid=0.0,
                   step_over_spacing=0.0, kind="none")
        if len(u) < 3 or len(u) > 0.5 * len(v):
            out.append(rec); continue
        d = np.diff(u); d = d[d > 0]
        if len(d) == 0:
            out.append(rec); continue
        step = float(np.median(d))
        # 刻みが細かすぎると v/step が巨大になり、整数判定が丸めで壊れる
        if step <= 0 or np.abs(v).max() / step > 1e9:
            out.append(rec); continue
        k = v / step
        on = float((np.abs(k - np.rint(k)) < 0.01).mean())
        r = step / max(spacing, 1e-30)
        rec.update(step=step, on_grid=on, step_over_spacing=float(r))
        if on > 0.99:
            rec["kind"] = "走査軸" if r > 0.1 else "格納格子"
        out.append(rec)
    return out


def azimuth_monotonicity(w: np.ndarray) -> float:
    az = np.arctan2(w[:, 1], w[:, 0])
    d = np.diff(az)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float((d > 0).mean())


def morton_delta(w: np.ndarray, v: float) -> float:
    """Morton に並べ替えると符号長がどう変わるか。正なら格納順の方が良い。"""
    q = np.rint((w - w.min(0)) / v).astype(np.int64)
    a = _bits([q[:, 0], q[:, 1], q[:, 2]], len(q))
    o = np.argsort(morton3(q))
    qm = q[o]
    b = _bits([qm[:, 0], qm[:, 1], qm[:, 2]], len(q))
    return float(b - a)


def acquisition_score(w_world: np.ndarray, *, sample: int = 60_000,
                      seed: int = 0, verbose: bool = False) -> dict:
    """取得構造の残り具合を測り、候補集合の絞り込み方を返す。"""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(w_world)
    # 並び順そのものを見る指標があるので、連続した区間を取る（ランダム抽出しない）
    s = slice(0, min(sample, n))
    w_raw = np.ascontiguousarray(w_world[s])
    c = w_raw.mean(0)
    w = np.ascontiguousarray(w_raw - c)

    no = neighbor_order(w)
    sp = no["spacing"]
    cv = no["ranks"][0]["cv"]
    h = hexatic(w, 6, sample=min(20_000, len(w)))
    psi6, ref = h["psi"], h["random_ref"]
    rot = rotation_sensitivity(w, sp / 8, n_rot=6, seed=seed)
    # 「元の向きが、適当な向きよりどれだけ得か」を直接使う
    frame_gain = float((rot["rot_mean"] - rot["base"]) / max(rot["rot_mean"], 1e-30))
    rot_gain = float((rot["base"] - min(rot["pca"], rot["rot_min"]))
                     / max(rot["base"], 1e-30))
    md = morton_delta(w, sp / 8)
    grid = axis_grid_signature(w_raw, sp)
    extent = float(np.linalg.norm(w_raw.max(0) - w_raw.min(0)))
    origin_dist = float(np.linalg.norm(c)) / max(extent, 1e-30)
    # 方位角の単調性は、座標原点がセンサ位置と解釈できるときだけ意味がある
    azm = azimuth_monotonicity(w_raw) if origin_dist <= 10.0 else float("nan")

    # --- 0..1 に正規化した成分
    scan_axis = max((g["on_grid"] if g["kind"] == "走査軸" else 0.0) for g in grid)
    comp = {
        "距離の規則性": float(np.clip((0.52 - cv) / 0.45, 0, 1)),      # ポアソン 0.52 を基準
        "角度の規則性": float(np.clip((psi6 - ref) / (1.0 - ref), 0, 1)),
        "向きの特権性": float(np.clip(frame_gain / 0.30, 0, 1)),
        "順序の意味":   float(np.clip(md / 4.0, 0, 1)),                # +4bpp で満点
        "走査軸の痕跡": float(scan_axis),
        "回転走査の痕跡": (0.0 if azm != azm
                     else float(np.clip((azm - 0.5) / 0.45, 0, 1))),
    }
    score = float(np.mean(list(comp.values())))

    # --- 候補集合の絞り込み（最終判断は実測に委ねる）
    keep_order = md > 0.3                       # 格納順の方が明確に良い
    rec = dict(
        keep_storage_order=bool(keep_order),
        consider_morton=bool(md <= 0.3),
        consider_polar=bool(origin_dist <= 10.0 and azm == azm and azm > 0.95),
        consider_rotation=bool(rot_gain > 0.01),   # 実際に得になるときだけ
        notes=[],
    )
    if keep_order:
        rec["notes"].append(f"格納順が Morton より {md:.2f} bpp 良い。並べ替えないこと")
    else:
        rec["notes"].append(f"格納順に意味が見えない（Morton 差 {md:+.2f} bpp）。Morton を候補に")
    for i, g in enumerate(grid):
        if g["kind"] == "走査軸":
            rec["notes"].append(
                f"{'xyz'[i]} 軸が {g['levels']:,} 段階の格子に {g['on_grid']*100:.0f}% 乗り、"
                f"刻みが点間隔の {g['step_over_spacing']:.2f} 倍。走査軸の可能性")
        elif g["kind"] == "格納格子":
            rec["notes"].append(
                f"{'xyz'[i]} 軸は刻み {g['step']:.3g} の格納格子（点間隔の "
                f"{g['step_over_spacing']:.3f} 倍）。走査構造ではなく既存の量子化")
    if origin_dist > 10.0:
        rec["notes"].append(f"座標原点が広がりの {origin_dist:.0f} 倍遠い。"
                            f"極座標は縮退するので除外（方位角の判定も無効）")
    elif azm == azm and azm > 0.95:
        rec["notes"].append(f"方位角が {azm*100:.0f}% 単調。回転式センサの 1 スイープの可能性")
    if rot_gain > 0.01:
        rec["notes"].append(f"回転すると {rot_gain*100:.1f}% 得になる。回転探索の余地")
    else:
        rec["notes"].append(f"元の向きが最良（適当な向きより {frame_gain*100:.1f}% 良い）。"
                            f"回転させないこと")

    out = dict(score=score, components=comp,
               signals=dict(cv_1nn=cv, psi6=psi6, psi6_ref=ref,
                            frame_gain=frame_gain, rot_gain=rot_gain,
                            morton_delta=md, azimuth_monotonic=azm,
                            axis_grid=grid, origin_over_extent=origin_dist,
                            spacing=sp),
               recommend=rec, n_used=len(w_raw), seconds=time.perf_counter() - t0)
    if verbose:
        print(f"  取得構造スコア {score:.2f}   （{out['seconds']:.1f}s, {len(w_raw):,}点）")
        for k, v in comp.items():
            bar = "#" * int(round(v * 24))
            print(f"    {k:<12}{v:5.2f}  {bar}")
        for nte in rec["notes"]:
            print(f"    · {nte}")
    return out
