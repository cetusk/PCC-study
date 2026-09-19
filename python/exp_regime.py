"""レジーム曲線: 「1 レベルあたり何ビット掛かるか」を 格子/点間隔 比の関数として測る。

【注意・この指標は信用しすぎないこと】
  TMC13 の bpp 差分を「1 レベルあたりのコスト」とみなす方法は、
  コーデックの深さごとの符号化効率の差に汚染される。
  実際 Armadillo では 1.8 bit/レベルと出るが、子オクタントの
  エントロピーを直接測ると 3.000（＝一様）で、整合しない。
  合成データ（一様ランダム・球面）では 3.040 と正しく出る。
  結論を出すときは exp_octant_predict.py の直接測定を使うこと。


これまでに分かったこと:
  ・octree は点間隔より細かいスケールでは 1 レベル 3.0 bit 掛かる（＝情報ゼロ）
  ・点間隔付近から構造が出てくる

仮説:
  この折れ点は「密なオブジェクト点群 vs 疎な LiDAR」という
  データの種類で決まるのではなく、**格子サイズと点間隔の比**だけで決まる。
  だとすれば、全く違うデータセットの曲線が横軸を正規化すると重なるはずである。

これが正しければ、8iVFB で学習型が −40% 出るのは「密だから」ではなく
「誰かが点間隔に合わせてボクセル化しておいたから」ということになる。
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scipy.spatial import cKDTree
from baselines import tmc13_bits


def point_spacing(w: np.ndarray, sample: int = 50_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), min(sample, len(w)), replace=False)
    tree = cKDTree(w)
    d, _ = tree.query(w[idx], k=2)
    return float(np.median(d[:, 1]))


def regime_curve(w: np.ndarray, name: str, n_levels: int = 14,
                 finest_ratio: float = 1 / 512) -> list[dict]:
    """点間隔の finest_ratio 倍で 1 回だけ量子化し、あとはビットシフトで粗くする。

    重要: v を変えるたびに rint(w/v) をやり直してはいけない。
    それでは連続する行が入れ子の octree レベルにならず、元の座標格子との
    干渉（エイリアシング）が bpp 差に混入して、1 レベルあたりのビット数を
    過小評価する。ここでは一度整数化してから >>k で厳密に入れ子にする。
    """
    sp = point_spacing(w)
    w = w - w.min(0)
    base = sp * finest_ratio
    q0 = np.rint(w / base).astype(np.int64)
    # TMC13 の実用的な深さに収まるところまで下げる
    shift0 = 0
    while int((q0 >> shift0).max()) >= (1 << 21):
        shift0 += 1
    rows = []
    prev = None
    for i in range(n_levels):
        k = shift0 + i
        q = q0 >> k
        if int((q.max(0) - q.min(0)).max()) < 2:
            break
        r = tmc13_bits(q)
        if r.bytes == 0:
            continue
        v = base * (1 << k)
        row = dict(ratio=v / sp, cell=v, bpp=r.bpp, lossless=r.lossless,
                   uniq=int(len(np.unique(q, axis=0))))
        if prev is not None:
            row["per_level"] = prev["bpp"] - r.bpp
        rows.append(row)
        prev = row
    for r in rows:
        r["name"] = name
        r["spacing"] = sp
    return rows


def print_curve(rows: list[dict]) -> None:
    if not rows:
        print("  （測定できず）")
        return
    print(f"  点間隔 = {rows[0]['spacing']*1000:.3f} mm")
    print(f"  {'格子/点間隔':>12}{'格子':>12}{'bpp':>9}{'1レベル当たり':>14}  判定")
    print("  " + "-" * 66)
    for r in rows:
        pl = r.get("per_level")
        if pl is None:
            j = "(基準)"
        elif pl > 2.85:
            j = "■ 情報ゼロ"
        elif pl > 2.2:
            j = "▨ ほぼゼロ"
        elif pl > 1.2:
            j = "▤ 構造あり"
        else:
            j = "□ 強い構造"
        cell_mm = r["cell"] * 1000
        print(f"  {r['ratio']:>12.4f}{cell_mm:>11.4f}m{r['bpp']:>9.3f}"
              f"{(pl if pl is not None else float('nan')):>14.3f}  {j}")
