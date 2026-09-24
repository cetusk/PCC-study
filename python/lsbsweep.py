"""LSB スイープ: 「下位何ビットが本当にノイズか」を仮定なしで測る。

原理:
  座標の下位 k ビットを捨てたときの符号長の減り方を見る。
  1ビット捨てて符号長がちょうど 1 bit/点/軸 減るなら、そのビットは
  完全にランダム＝圧縮不能＝ノイズ。減りが 1 未満ならそのビットには
  構造があり、モデル化の余地が残っている。

  これでビット内訳の分解とノイズ床の位置が同時に決まる。要件を「ビット完全」から「ノイズ床基準」に移したときの
  利得の上限も、この曲線の積分としてそのまま読める。
"""
from __future__ import annotations
import numpy as np, dataclasses, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baselines import laszip_bits, tmc13_bits, ref_geometry_bits, Result
from pcio import PointCloud


def drop_lsb(xyz_int: np.ndarray, k: int) -> np.ndarray:
    """下位 k ビットを丸めて落とす（最近傍量子化）。返り値は縮んだ格子上の整数。"""
    if k == 0:
        return xyz_int.copy()
    step = 1 << k
    return ((xyz_int + (step >> 1)) >> k)


def sweep(pc: PointCloud, ks=range(0, 13), codecs=("laszip_geom", "tmc13", "ref_orig"),
          verbose=True) -> list[dict]:
    rows = []
    for k in ks:
        q = drop_lsb(pc.xyz_int, k)
        step_m = float(pc.scale[0]) * (1 << k)          # 実効ボクセルサイズ [m]
        rec = {"k": k, "step_m": step_m, "n": pc.n,
               "uniq": int(len(np.unique(q, axis=0)))}
        if "laszip_geom" in codecs:
            sub = PointCloud(xyz_int=q, scale=pc.scale * (1 << k), offset=pc.offset)
            rec["laszip_geom"] = laszip_bits(sub, keep_attrs=False).bpp
        if "tmc13" in codecs:
            r = tmc13_bits(q)
            rec["tmc13"] = r.bpp
            rec["tmc13_lossless"] = r.lossless
        if "ref_orig" in codecs:
            rec["ref_orig"] = ref_geometry_bits(q, "original").bpp
        rows.append(rec)
        if verbose:
            print(f"k={k:2d} step={step_m*1000:8.2f}mm uniq={rec['uniq']:>9,} " +
                  "  ".join(f"{c}={rec.get(c,float('nan')):7.3f}" for c in codecs), flush=True)
    return rows


def marginal_table(rows: list[dict], codec: str) -> str:
    """1ビット落とすごとに実際に何ビット減ったか。3.000 = 完全ノイズ(3軸ぶん)。"""
    L = [f"{'k':>3} {'step[mm]':>10} {'bpp':>9} {'Δbpp/LSB':>10} {'ノイズ度':>9}  判定",
         "-" * 72]
    for i, r in enumerate(rows):
        if codec not in r:
            continue
        d = (rows[i-1][codec] - r[codec]) if i else float("nan")
        ratio = d / 3.0
        if i == 0:
            judge = "(基準)"
        elif ratio > 0.92:
            judge = "■ ほぼ純ノイズ（圧縮不能）"
        elif ratio > 0.70:
            judge = "▨ ほぼノイズ・弱い構造"
        elif ratio > 0.40:
            judge = "▤ 構造あり"
        else:
            judge = "□ 強い構造（モデル化余地）"
        L.append(f"{r['k']:>3} {r['step_m']*1000:>10.2f} {r[codec]:>9.3f} "
                 f"{d:>10.3f} {ratio:>9.2f}  {judge}")
    return "\n".join(L)
