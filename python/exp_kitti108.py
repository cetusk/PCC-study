"""KITTI の 108 frame（強度込み、float32 × 4 = 128 bit/点）を可逆で pack し、bpp の中央値・最小・最大を出す。

    python -u python/exp_kitti108.py      # 出力は data/work/grid/kitti108_<版>.log に保存する

自己検証つきで回す（全列一致を確かめる）。standing.md の「KITTI 108 frame 中央値」の出どころ。
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PCC = os.path.abspath("cpp/build/pccnorm")
frames = sorted(Path("data/raw/kitti").rglob("*.bin"))
tmp = Path(tempfile.mkdtemp())
out = tmp / "k.pcc2"
bpp = []
bad = 0
for k in frames:
    n = k.stat().st_size // 16
    r = subprocess.run([PCC, "pack", str(k), str(out), "--no-fallback"], capture_output=True, text=True)
    if r.returncode != 0 or "全列一致 = true" not in r.stdout:
        bad += 1
        print(f"落ちた: {k.name} rc={r.returncode}", flush=True)
        continue
    bpp.append(out.stat().st_size * 8 / n)
v = np.array(bpp)
print(f"KITTI {len(frames)} frame 中央値 {np.median(v):.3f} 最小 {v.min():.3f} 最大 {v.max():.3f}"
      f"  無圧縮（128 bit）比 {100 * (np.median(v) / 128 - 1):+.1f}%  落ちたもの {bad}")
sys.exit(1 if bad else 0)
