"""KITTI の velodyne .bin を、可逆に整数化して LAS に書く。

.bin の float32 は 1 mm（強度は 0.01）の整数倍であり、
整数から float32 を作り直すとビット一致する。よってこの変換は可逆である。
"""
import sys, os, glob
import numpy as np
import laspy


def convert(binpath, lasdir, name=None):
    a = np.fromfile(binpath, dtype=np.float32).reshape(-1, 4)
    xyz = a[:, :3].astype(np.float64)
    k = np.round(xyz / 0.001).astype(np.int64)
    # 非連続な 2 次元の切り出しに view を掛けるとビット列の並びが変わるので、
    # 軸ごとに連続化してから比べる。
    # **負のゼロだけは整数を経由すると符号が消える。**値は等しいがビット列は違う。
    # 数を数えて返す。ビット完全にするならこの点だけ例外表に載せればよい。
    nzero = 0
    for ax in range(3):
        v = np.ascontiguousarray(a[:, ax])
        back = np.ascontiguousarray((k[:, ax] * 0.001).astype(np.float32))
        bad = back.view(np.uint32) != v.view(np.uint32)
        nz = bad & (v == 0)
        assert np.array_equal(bad, nz), '負のゼロ以外の不一致がある'
        nzero += int(nz.sum())
    inten = np.round(a[:, 3].astype(np.float64) / 0.01).astype(np.int64)
    assert inten.min() >= 0 and inten.max() <= 65535

    h = laspy.LasHeader(version='1.2', point_format=0)
    h.scales = np.array([0.001, 0.001, 0.001])
    h.offsets = np.array([0.0, 0.0, 0.0])
    d = laspy.LasData(h)
    d.X, d.Y, d.Z = k[:, 0], k[:, 1], k[:, 2]
    d.intensity = inten
    out = os.path.join(lasdir, (name or os.path.basename(binpath)[:-4]) + '.las')
    d.write(out)
    return out, len(a), nzero


def main(bindir, lasdir, nframe=1, merge=False):
    os.makedirs(lasdir, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(bindir, '*.bin')))[:nframe]
    if not merge:
        for f in fs:
            o, n, nz = convert(f, lasdir)
            print(f'{os.path.basename(f)} → {os.path.basename(o)}  {n} 点  '
                  f'負のゼロ {nz} 点（{100*nz/(3*n):.4f}%）')
        return
    K, I = [], []
    nzero = 0
    for f in fs:
        a = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        kk = np.round(a[:, :3].astype(np.float64) / 0.001).astype(np.int64)
        # まとめる経路でも 1 枚ずつ可逆性を確かめる。
        # （確かめない経路を残すと、検証したつもりの測定が混ざる）
        for ax in range(3):
            v = np.ascontiguousarray(a[:, ax])
            back = np.ascontiguousarray((kk[:, ax] * 0.001).astype(np.float32))
            bad = back.view(np.uint32) != v.view(np.uint32)
            nz = bad & (v == 0)
            assert np.array_equal(bad, nz), f'{f}: 負のゼロ以外の不一致がある'
            nzero += int(nz.sum())
        K.append(kk)
        I.append(np.round(a[:, 3].astype(np.float64) / 0.01).astype(np.int64))
    k = np.vstack(K); inten = np.concatenate(I)
    h = laspy.LasHeader(version='1.2', point_format=0)
    h.scales = np.array([0.001, 0.001, 0.001]); h.offsets = np.array([0.0, 0.0, 0.0])
    d = laspy.LasData(h)
    d.X, d.Y, d.Z = k[:, 0], k[:, 1], k[:, 2]
    d.intensity = inten
    out = os.path.join(lasdir, f'kitti_{len(fs)}frames.las')
    d.write(out)
    print(f'{len(fs)} frame → {os.path.basename(out)}  {len(k)} 点  '
          f'負のゼロ {nzero} 点（{100*nzero/(3*len(k)):.4f}%）')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         len(sys.argv) > 4 and sys.argv[4] == 'merge')
