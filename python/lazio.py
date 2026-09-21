"""LAZ の符号化／復号だけを行う補助。外部からピーク RSS を測るために使う。

laspy を親プロセスで呼ぶと、親が抱えている点群が子の RSS に乗る（runpeak を参照）。
この小さなスクリプトを新しいプロセスとして起動すれば、測るのは LAZ の分だけになる。
"""
from __future__ import annotations
import sys
import numpy as np
import laspy


def main() -> None:
    mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "enc":
        a = np.load(src)
        h = laspy.LasHeader(version="1.4", point_format=6)
        h.scales, h.offsets = [0.001] * 3, [0.0] * 3
        las = laspy.LasData(h)
        las.X, las.Y, las.Z = a[:, 0], a[:, 1], a[:, 2]
        las.write(dst)
    elif mode == "dec":
        with laspy.open(src) as fh:
            p = fh.read()
        np.save(dst, np.stack([np.asarray(p.X), np.asarray(p.Y), np.asarray(p.Z)], 1))
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
