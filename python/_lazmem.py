"""幾何のみの LAZ を書き出すだけの補助プロセス。runpeak から起動して RSS を測る。

引数が "base" のときは import だけして終わる。その差を書き出しの増分とする。
**書き方は exp_order_matrix.laz_geom_bpp と同じでなければならない。**
xyz は整数座標（生の X/Y/Z）であって、尺度を掛けた実数ではない。
取り違えると LAZ の差分符号化が壊れ、AHN3 で 19.4 → 35.9 bpp と 1.8 倍に見える。
"""
import sys
import numpy as np
import laspy

if len(sys.argv) > 1 and sys.argv[1] == "base":
    sys.exit(0)
xyz = np.load(sys.argv[1])
sc = np.load(sys.argv[2])
of = np.load(sys.argv[3])
h = laspy.LasHeader(version="1.4", point_format=6)
h.scales, h.offsets = sc, of
d = laspy.LasData(h)
d.X, d.Y, d.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
d.write(sys.argv[4])
