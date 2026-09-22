"""掃引の切り出しを 2 通りで比べる。

  A. 時刻の隙間（今の符号化器）
  B. 走査角の向きの反転（掃引はミラーが端から端へ振れる 1 往復の片道）

走査角は LAS の列なので、復号器も同じ手掛かりを持つ（副情報は要らない）。
"""
import sys
import numpy as np
import laspy


def stats(lens, n, label):
    lens = np.asarray(lens)
    lens = lens[lens > 0]
    if len(lens) == 0:
        print(f'  {label:<22}—'); return
    cov = lens[lens >= 80].sum() / n * 100
    print(f'  {label:<22}{len(lens):>8}{int(np.median(lens)):>10}{cov:>12.1f}%')


def main(path, n=300000):
    f = laspy.read(path)
    m = min(n, len(f.x))
    g = np.asarray(f.gps_time[:m], dtype=np.float64)
    gi = g.view(np.int64)
    src = np.asarray(f.point_source_id[:m], dtype=np.int64)
    sa = np.asarray(f.scan_angle[:m] if hasattr(f, 'scan_angle')
                    else f.scan_angle_rank[:m], dtype=np.int64)

    o = np.lexsort((gi, src))
    gs, ss, sas = gi[o], src[o], sa[o]
    span = g.max() - g.min()
    print(f'  点 {m}  時刻の幅 {span:.4f} s  走査角 {sas.min()}〜{sas.max()}')
    print(f'  {"切り方":<22}{"掃引数":>8}{"中央値 n":>10}{"n>=80 被覆":>12}')

    # A: 時刻の隙間
    same = ss[1:] == ss[:-1]
    d = gs[1:] - gs[:-1]
    pos = d[same & (d > 0)]
    med = int(np.median(pos)) if len(pos) else 10
    thr = med * 20
    brk = (~same) | (d > thr)
    cuts = np.flatnonzero(brk) + 1
    stats(np.diff(np.concatenate([[0], cuts, [m]])), m, 'A 時刻の隙間×20')

    # B: 走査角の向きの反転
    ds = np.diff(sas.astype(np.int64))
    sign = np.sign(ds)
    # 0 は直前の向きを引き継ぐ
    last = 0
    fill = np.empty_like(sign)
    for i, v in enumerate(sign):
        if v != 0:
            last = v
        fill[i] = last
    rev = np.zeros(m - 1, dtype=bool)
    rev[1:] = (fill[1:] != fill[:-1]) & (fill[1:] != 0) & (fill[:-1] != 0)
    brkB = (~same) | rev
    cutsB = np.flatnonzero(brkB) + 1
    stats(np.diff(np.concatenate([[0], cutsB, [m]])), m, 'B 走査角の反転')

    # B': 反転 + 大きな時刻の隙間
    brkC = brkB | (d > med * 1000)
    cutsC = np.flatnonzero(brkC) + 1
    stats(np.diff(np.concatenate([[0], cutsC, [m]])), m, 'B+ 反転と大きな隙間')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300000)
