"""G-PCC（TMC13）の設定を振って、比較に使う構成が公正かを確かめる。

比較相手を手加減していないことを示すための測定である。出力は
data/work/gpcc_config.txt に残し、論文の脚注の裏付けにする。

入力の妥当性も同時に検査する。TMC13 には float32 の PLY で渡すので、
最小を引いた後の座標が 2^24 を超えると黙って丸められる。超える入力は
可逆判定が false になるだけで止まらないため、先に落とす。

使い方:
    $PCCPY python/exp_gpcc_config.py <in.las|laz> [点数]
"""
from __future__ import annotations
import os
import subprocess
import sys
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import tmc13_bits, TMC3          # noqa: E402

# 既定の調整フラグ（baselines.tmc13_bits が渡すもの）を打ち消すための値
PLAIN = ["--neighbourAvailBoundaryLog2=0", "--intra_pred_max_node_size_log2=0",
         "--maxNumQtBtBeforeOt=0", "--minQtbtSizeLog2=0", "--planarEnabled=0",
         "--inferredDirectCodingMode=0"]

CFG = [
    ("調整済み（比較に使う設定）", []),
    ("調整なし（既定に近い）", PLAIN),
    ("IDCM=0", ["--inferredDirectCodingMode=0"]),
    ("IDCM=2", ["--inferredDirectCodingMode=2"]),
    ("planar 無効", ["--planarEnabled=0"]),
    ("単一スライス", ["--partitionMethod=0"]),
    ("重複点を併合", ["--mergeDuplicatedPoints=1"]),
    ("予測幾何", ["--geomTreeType=1"]),
    ("予測幾何 + 予測数 15", ["--geomTreeType=1", "--predGeomMaxPredIdx=15"]),
]


def main() -> None:
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000

    ver = subprocess.run([TMC3, "--help"], capture_output=True, text=True)
    tag = ""
    for ln in (ver.stdout + ver.stderr).splitlines()[:5]:
        if "release" in ln or "version" in ln.lower():
            tag = ln.strip(); break

    with laspy.open(path) as fh:
        pts = fh.read_points(cap)
    xyz = np.stack([np.asarray(pts["X"]), np.asarray(pts["Y"]),
                    np.asarray(pts["Z"])], 1).astype(np.int64)
    n = len(xyz)
    a = xyz - xyz.min(0)
    if a.max() >= 2 ** 24:
        sys.exit(f"座標が float32 の整数域を超える（最大 {a.max()} >= 2^24）。"
                 "PLY を float64 にしない限りこの比較は成立しない。")
    dup = n - len(np.unique(a, axis=0))

    print(f"入力        {path}")
    print(f"            {n} 点 / 最小を引いた後の最大 {a.max()} (< 2^24) / 重複点 {dup}")
    print(f"TMC13       {tag or '版の取得に失敗'}")
    print()
    print(f"{'設定':<28}{'bpp':>10}{'可逆':>7}{'enc[s]':>9}{'対 比較設定':>12}")

    base = None
    for name, extra in CFG:
        r = tmc13_bits(xyz, extra=extra)
        if not r.bytes:
            print(f"{name:<28}{'失敗':>10}   {r.note[:50]}")
            continue
        bpp = r.bytes * 8.0 / n
        if base is None:
            base = bpp
        print(f"{name:<28}{bpp:>10.3f}{str(r.lossless):>7}{r.enc_s:>9.1f}"
              f"{100 * (bpp / base - 1):>11.2f}%", flush=True)


if __name__ == "__main__":
    main()
