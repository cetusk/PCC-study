"""格納順が取得順かどうかを測る。

走査モデルは点が取得順に並んでいることを前提にしている。
空間順で格納されたファイルでは、先頭 N 点を取ると時間的にばらばらな標本になり、
掃引が砕ける。それが符号化器の欠陥なのか標本の取り方の問題なのかを分ける。
"""
import sys
import numpy as np
import laspy


def main(path, n=300000):
    f = laspy.read(path)
    tot = len(f.x)
    g = np.asarray(f.gps_time, dtype=np.float64)
    src = np.asarray(f.point_source_id, dtype=np.int64)
    idx = np.arange(tot)

    # 格納順と時刻の順位相関（全点）
    rg = np.argsort(np.argsort(g))
    rho = np.corrcoef(idx, rg)[0, 1]

    m = min(n, tot)
    span_pre = g[:m].max() - g[:m].min()
    span_all = g.max() - g.min()

    # 先頭 N 点は全体の時間のどれだけを覆うか
    print(f'  全点 {tot}  source 種 {len(np.unique(src))}')
    print(f'  格納順と時刻の順位相関   {rho:+.4f}')
    print(f'  全体の時刻の幅           {span_all:.2f} s')
    print(f'  先頭 {m} 点の時刻の幅   {span_pre:.2f} s  '
          f'（全体の {100*span_pre/max(span_all,1e-9):.1f}%）')
    # 先頭 N 点を時刻で並べたときの発射の抜け具合
    gs = np.sort(g[:m])
    d = np.diff(gs)
    d = d[d > 0]
    ga = np.sort(g)
    da = np.diff(ga); da = da[da > 0]
    if len(d) and len(da):
        print(f'  発射間隔の中央値: 先頭 {np.median(d)*1e6:.3f} us / 全点 {np.median(da)*1e6:.3f} us'
              f'  → 標本は {np.median(d)/np.median(da):.1f} 倍に間引かれている')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300000)
