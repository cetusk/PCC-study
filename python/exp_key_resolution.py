"""鍵の縮退を「作った」場合と「自然に生じた」場合で、時刻の分解能を比べる。

掃引（exp_key_sweep.py）は gps_time を量子化して縮退を作るので、
単射性と同時に時間分解能も壊す。観察データ（exp_order_matrix.py）の
縮退は分解能が保たれたまま値が重複している。両者が同じ処置でないことを
数字で示すために、発射間隔に対する量子化幅の比を出す。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
from exp_key_sweep import BLOCK, dup_rate, quantize_for

FILES = [("vegetation", "data/raw/small/vegetation_1_3.las"),
         ("AHN4 _20", "data/raw/ahn4/31HZ1_20.LAZ"),
         ("USGS NY", "data/raw/usgs/NY_ClintonEssex_2014.laz")]
TARGETS = (50.0, 60.0, 70.0, 80.0, 90.0, 100.0)


def main() -> None:
    print("時刻の分解能 — 作った縮退と自然に生じた縮退の違い")
    print("発射間隔は、同じ飛行線の中で連続する点の時刻差の正の中央値。")
    print()
    print(f"{'データ':<11}{'点数':>9}{'鍵重複':>8}{'発射間隔':>12}{'量子化幅':>12}{'間隔比':>9}")
    for label, src in FILES:
        if not Path(src).is_file():
            print(f"  {label}: 入力が無い {src}")
            continue
        with laspy.open(src) as fh:
            total = fh.header.point_count
            start = max(0, total // 2 - BLOCK // 2) if total > BLOCK else 0
            pts = fh.read_points(start + min(BLOCK, total))
        pts = pts[start:start + BLOCK]
        g = np.asarray(pts["gps_time"]).astype(np.float64)
        g = g - g.min()
        sid = np.asarray(pts["point_source_id"]).astype(np.int64)
        d = np.diff(g)
        step = float(np.median(d[d > 0])) if np.any(d > 0) else float("nan")
        print(f"{label:<11}{len(g):>9}{dup_rate(sid, g):>7.1f}%{step:>12.2e}"
              f"{'—':>12}{'（自然）':>9}")
        for t in TARGETS:
            _, q, act = quantize_for(sid, g, t)
            print(f"{'  目標 ' + f'{t:.0f}%':<11}{'':>9}{act:>7.1f}%{step:>12.2e}"
                  f"{q:>12.2e}{q / step:>9.2f}")
        print()
    print("量子化幅が発射間隔を超えると、走査モデルが当てはめに使う刻みが失われる。")
    print("自然に生じた縮退では刻みは保たれ、値だけが重複する。")


if __name__ == "__main__":
    main()
