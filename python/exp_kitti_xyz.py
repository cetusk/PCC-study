"""KITTI の座標 3 成分だけを、いまの器で可逆と誤差上限つきで測る（pcc.tex の
「Returning to the acquisition representation」の節を測り直すため）。

元の節は、float32 の座標 3 成分（96 bit/点）に対して「取得順差分 + zstd」の可逆 46.183 bpp と、
極座標格子（2 mm・0.002°、同じ差分 + zstd）の 14.184 bpp（最大誤差 2.11 mm）を並べていた。
ここでは同じ 108 frame の座標だけを pccnorm に渡す。強度の列を 0 にすると定数の列として
落ちるので、器に残るのは座標と、定数の値を書いた計画だけになる。

    python -u python/exp_kitti_xyz.py            # 出力を data/work/grid/kitti_xyz_v22.log に保存した

pack は自己検証つき（可逆は全列一致、誤差上限つきは誤差上限の中）で回す。可逆はさらに
unpack --bin で戻した座標のビット列を、強度を 0 にした入力と比べる（float32 のまま一致するか）。
pack が見つけた格子（「格子 X 刻み …」の行）を数え、中央値のほかに走行全体の合計
（全 frame の器の bit の和 ÷ 全点数）も出す。元の節の 46.183・14.184 は 108 frame をまとめた値（`kitti_polar_finding.md`）である。
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PCC = "cpp/build/pccnorm"
CASES = [("可逆", []), ("誤差上限 2 mm", ["--eps", "0.002"]), ("誤差上限 2.11 mm", ["--eps", "0.00211"])]
frames = sorted(Path("data/raw/kitti").rglob("*.bin"))
tmp = Path(tempfile.mkdtemp())
res = {c[0]: [] for c in CASES}
tot_bits = {c[0]: 0 for c in CASES}
tot_n = 0
grid_seen = {}
bits_equal = 0
kind = {c[0]: {} for c in CASES}
bad = 0
print(f"KITTI {len(frames)} frame。座標 3 成分だけ（強度は 0 にして定数の列として落とす）。基準は float32 × 3 = 96 bit/点。")
for k in frames:
    a = np.fromfile(k, "<f4").reshape(-1, 4).copy()
    a[:, 3] = 0.0
    src = tmp / "xyz.bin"
    a.tofile(src)
    n = len(a)
    tot_n += n
    for name, args in CASES:
        out = tmp / "o.pcc2"
        r = subprocess.run([PCC, "pack", str(src), str(out), "--no-fallback", *args],
                           capture_output=True, text=True)
        ok = r.returncode == 0 and ("全列一致 = true" in r.stdout or "誤差上限の中 = true" in r.stdout)
        if not ok:
            bad += 1
            print(f"  落ちた: {k.name} {name} rc={r.returncode}", flush=True)
            continue
        res[name].append(out.stat().st_size * 8 / n)
        tot_bits[name] += out.stat().st_size * 8
        if not args:
            gl = re.search(r"^格子\s+(.*)$", r.stdout, re.M)
            key = re.sub(r"（-0\.0 が \d+ 点）", "", gl.group(1)) if gl else "格子なし"
            grid_seen[key] = grid_seen.get(key, 0) + 1
            back = tmp / "b.bin"
            u = subprocess.run([PCC, "unpack", str(out), "--bin", str(back)], capture_output=True, text=True)
            if u.returncode == 0 and back.read_bytes() == src.read_bytes():
                bits_equal += 1
            else:
                bad += 1
                print(f"  ビット列が戻らない: {k.name}", flush=True)
        m = re.search(r"幾何を(\S+?)に量子化", r.stdout)
        g = m.group(1) if m else "非可逆の量子化なし"
        kind[name][g] = kind[name].get(g, 0) + 1
for name, _ in CASES:
    v = np.array(res[name])
    print(f"{name:<16} 中央値 {np.median(v):7.3f} bpp  平均 {v.mean():7.3f}  幅 {v.min():.3f}〜{v.max():.3f}"
          f"  96 bit 比 {100 * (np.median(v) / 96 - 1):+.1f}%  n={len(v)}  格子 {kind[name]}")
for name, _ in CASES:
    t = tot_bits[name] / tot_n
    print(f"{name:<16} 走行全体の合計 {t:7.3f} bpp  96 bit 比 {100 * (t / 96 - 1):+.1f}%")
print(f"全点数 {tot_n}")
print(f"可逆で見つけた格子: {grid_seen}")
print(f"可逆で unpack --bin が入力とバイト一致: {bits_equal}/{len(frames)}")
print(f"\n落ちたもの {bad} 件")
sys.exit(1 if bad else 0)
