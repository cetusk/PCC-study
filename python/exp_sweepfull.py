"""符号化器の掃引切り出しを全点に対して再現し、標本と比べる。

符号化器の手順:
  1. (point_source_id, gps_time) で安定整列
  2. 正の時刻差の中央値 × 20 をしきい値に、それを超える隙間で切る

先頭 N 点だけを取ると掃引が砕けるのか、全点でも砕けるのかを分ける。
"""
import sys
import numpy as np
import laspy


def cut(gi, src):
    o = np.lexsort((gi, src))
    gs, ss = gi[o], src[o]
    same = ss[1:] == ss[:-1]
    d = gs[1:] - gs[:-1]
    pos = d[same & (d > 0)]
    med = int(np.median(pos)) if len(pos) else 10
    thr = med * 20
    brk = (~same) | (d > thr)
    cuts = np.flatnonzero(brk) + 1
    lens = np.diff(np.concatenate([[0], cuts, [len(gs)]]))
    return lens[lens > 0], med


def show(lens, n, label):
    cov = lens[lens >= 80].sum() / n * 100
    print(f'  {label:<26}{len(lens):>9}{int(np.median(lens)):>10}{cov:>12.1f}%')


def main(path, n=300000):
    f = laspy.read(path)
    tot = len(f.x)
    gi = np.asarray(f.gps_time, dtype=np.float64).view(np.int64)
    src = np.asarray(f.point_source_id, dtype=np.int64)
    print(f'  全点 {tot}')
    print(f'  {"標本の取り方":<26}{"掃引数":>9}{"中央値 n":>10}{"n>=80 被覆":>12}')

    m = min(n, tot)
    l1, _ = cut(gi[:m], src[:m])
    show(l1, m, f'先頭 {m} 点')

    # 取得順に並べてから連続した m 点を取る
    o = np.lexsort((gi, src))
    st = (tot - m) // 2
    sl = o[st:st + m]
    l2, _ = cut(gi[sl], src[sl])
    show(l2, m, f'取得順で連続な {m} 点')

    l3, _ = cut(gi, src)
    show(l3, tot, '全点')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300000)
