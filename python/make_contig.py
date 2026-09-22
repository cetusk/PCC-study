"""取得順に連続した N 点を切り出して LAZ に書く。

--max-points N は格納順の先頭を取る。空間順に格納されたファイルや、
飛行ごとに固まったファイルでは、これは時間的にばらばらな標本になり、
取得構造を使う候補（走査モデル）だけが不当に不利になる。
取得順 (point_source_id, gps_time) で並べ、真ん中から連続した N 点を取る。
"""
import sys
import numpy as np
import laspy


def main(src, dst, n=2000000):
    f = laspy.read(src)
    tot = len(f.points)
    gi = np.asarray(f.gps_time, dtype=np.float64).view(np.int64)
    sid = np.asarray(f.point_source_id, dtype=np.int64)
    o = np.lexsort((gi, sid))
    m = min(n, tot)
    st = (tot - m) // 2
    sel = np.sort(o[st:st + m])          # 格納順は保つ（順序の意味を壊さない）
    # COPC の VLR を引き継ぐと書けないので、頭を作り直して版だけ合わせる
    h = laspy.LasHeader(version=f.header.version,
                        point_format=f.header.point_format)
    h.scales, h.offsets = f.header.scales, f.header.offsets
    for v in f.header.vlrs:
        if v.user_id not in ('copc', 'entwine'):
            try:
                h.vlrs.append(v)
            except Exception:
                pass
    out = laspy.LasData(h)
    out.points = f.points[sel]
    out.write(dst)
    g = np.asarray(f.gps_time)[sel]
    print(f'{src} → {dst}  {m} 点  時刻の幅 {g.max()-g.min():.2f} s')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000000)
