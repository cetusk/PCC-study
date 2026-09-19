"""走査モデルの当てはめがなぜ当たらないのかを、3 つの角度から一度に見る。

C++ 側で PCC_SCAN_DUMP=<path> を付けて符号化すると、走査線ごとの表と、
50 本に 1 本の生データが書き出される。それを読んで次を出す。

  (1) C++ の当てはめは収束不足か   同じ走査線に scipy を当てて最終残差を比べる
  (2) 反復は足りているか           反復ごとの残差がまだ減っているか
  (3) 当たらない線の性質は何か     点数・角度幅・面の薄さ・高度の分布

使い方:
    $PCCPY python/exp_scan_diag.py <dump.tsv>
"""
from __future__ import annotations
import sys
import pathlib
import numpy as np
from scipy.optimize import least_squares

SH, TAN_SH = 30, 30
ULP = 59.6046e-9          # gps_time の 1 刻み [s]


def blen(v: np.ndarray) -> np.ndarray:
    """zigzag のビット長。C++ 側の費用の測り方と揃える。"""
    z = (v.astype(np.int64) << 1) ^ (v.astype(np.int64) >> 63)
    out = np.zeros(len(z), dtype=np.int64)
    m = z > 0
    out[m] = np.floor(np.log2(z[m])).astype(np.int64) + 1
    return out


def load_rows(path: str):
    return np.genfromtxt(path, delimiter='\t', names=True)


def load_raw(path: str):
    """生データを走査線ごとの配列に分ける。"""
    lines, cur, cid = [], [], None
    with open(path, encoding='utf-8') as f:
        for ln in f:
            if ln.startswith('# s'):
                continue
            if ln.startswith('# line'):
                if cur:
                    lines.append((cid, np.array(cur, dtype=np.float64)))
                cid = int(ln.split()[2]); cur = []
            else:
                cur.append([float(x) for x in ln.split()])
    if cur:
        lines.append((cid, np.array(cur, dtype=np.float64)))
    return lines


def scipy_fit(s, z, shot, sa):
    """Python 試作と同じ当てはめ。戻り値は残差のビット長／点。"""
    ang = np.radians(sa * 0.006)
    c = np.polyfit(shot, ang, 1) if np.ptp(shot) > 0 else np.array([0.0, ang.mean()])
    ta = np.tan(ang)
    Sz0 = z.mean() + np.ptp(s) / max(np.ptp(ta), 1e-6)
    Sz0 = min(max(Sz0, z.mean() + 1e5), z.mean() + 3.9e6)

    def f(q):
        return s - (q[0] + (q[1] - z) * np.tan(q[2] + q[3] * shot))

    try:
        r = least_squares(f, [s.mean(), Sz0, c[1], c[0]], method='trf',
                          bounds=([s.mean() - 2e6, z.mean() + 5e4, -np.pi / 4 + 1e-6, -1e-2],
                                  [s.mean() + 2e6, z.mean() + 4e6, np.pi / 4 - 1e-6, 1e-2]),
                          max_nfev=200)
        return float(blen(np.round(f(r.x))).mean()), r.x
    except Exception:
        return float('inf'), None


def main(dump: str):
    d = load_rows(dump)
    ncol = sum(1 for nm in d.dtype.names if nm.startswith('i') and nm[1:].isdigit())
    it = np.vstack([d[f'i{k}'] for k in range(ncol)]).T
    ok = d['ok'] > 0.5
    print(f"走査線 {len(d)} 本  採用 {ok.sum()} ({ok.mean():.3f}) / 不採用 {(~ok).sum()}")

    print("\n(3) 当たる線と当たらない線の性質")
    print(f"{'':12}{'点数':>9}{'角度幅[度]':>11}{'薄さ[mm]':>11}{'高度[m]':>10}{'角速度[度/s]':>13}")
    for nm, m in (("採用", ok), ("不採用", ~ok)):
        if m.sum() == 0:
            continue
        print(f"{nm:12}{np.median(d['n'][m]):9.0f}{np.median(d['span_deg'][m]):11.1f}"
              f"{np.median(d['thin_mm'][m]):11.2f}{np.median(d['height_m'][m]):10.1f}"
              f"{np.median(d['omega_deg_s'][m]):13.0f}")
    print("  高度が 100 m 未満か 3000 m 超は縮退の疑い: "
          f"採用 {np.mean((d['height_m'][ok] < 100) | (d['height_m'][ok] > 3000)):.3f} / "
          f"不採用 {np.mean((d['height_m'][~ok] < 100) | (d['height_m'][~ok] > 3000)):.3f}")

    print(f"\n(2) 反復ごとの面内残差のビット長／点（中央値、{ncol} 回）")
    step = max(1, ncol // 12)
    ks = list(range(0, ncol, step))
    print("   " + "  ".join(f"{k:>5}" for k in ks))
    print("   " + "  ".join(f"{np.median(it[:, k]):5.2f}" for k in ks))
    drop = it[:, -2] - it[:, -1]
    print(f"  最後の 1 回でまだ減っている線: {np.mean(drop > 0.01):.3f}"
          f"  （減り幅の中央値 {np.median(drop):+.4f} bit）")
    print(f"  反復 0 から {ncol-1} への改善: 中央 {np.median(it[:, 0] - it[:, -1]):+.3f} bit")

    print("\n(1) 同じ走査線での C++ と scipy の比較")
    raw = load_raw(dump + '.raw')
    rows = {int(r['id']): r for r in d}
    n_better, diffs = 0, []
    for cid, arr in raw:
        if cid not in rows or len(arr) < 20:
            continue
        s, z, shot, sa = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        py_bits, _ = scipy_fit(s, z, shot, sa)
        cpp_bits = float(rows[cid]['mdl_bits'])
        diffs.append(cpp_bits - py_bits)
        if py_bits < cpp_bits - 0.01:
            n_better += 1
    diffs = np.array(diffs)
    if len(diffs) == 0:
        print("  比較できる走査線が無い")
        return
    print(f"  比較した走査線 {len(diffs)} 本")
    print(f"  C++ − scipy の残差差 [bit/点]: 中央 {np.median(diffs):+.3f}  "
          f"平均 {diffs.mean():+.3f}  P90 {np.percentile(diffs, 90):+.3f}")
    print(f"  scipy のほうが良かった線: {n_better / len(diffs):.3f}")


def compare(dumps):
    """複数の書き出しを並べ、退避側の費用と LM が止まった線を比べる。"""
    print(f"\n(4) 退避側の費用と、当てはめが動かなかった線")
    print(f"{'書き出し':<14}{'線':>7}{'採用率':>8}{'退避のalt_bits':>15}"
          f"{'ω=0の線':>10}{'LM不動':>9}")
    per = []
    for path in dumps:
        d = load_rows(path)
        ncol = sum(1 for nm in d.dtype.names if nm.startswith('i') and nm[1:].isdigit())
        it = np.vstack([d[f'i{k}'] for k in range(ncol)]).T
        ok = d['ok'] > 0.5
        big = d['n'] >= 20                       # 極小の線は当てはめ以前の問題
        stuck = (np.abs(it[:, 0] - it[:, -1]) < 1e-9) & big
        zero_om = (d['omega_deg_s'] == 0) & big
        alt_ng = d['alt_bits'][(~ok) & big]
        print(f"{pathlib.Path(path).stem:<14}{big.sum():>7}{ok[big].mean():>8.3f}"
              f"{np.median(alt_ng):>15.2f}{zero_om.sum()/max(big.sum(),1):>10.3f}"
              f"{stuck.sum()/max(big.sum(),1):>9.3f}")
        per.append((pathlib.Path(path).stem, d, big, ok))

    print("\n  退避側の alt_bits を走査線の長さで分けた中央値")
    edges = [20, 100, 300, 600, 1200, 10**9]
    hdr = "  ".join(f"{a}-{b if b < 10**8 else ''}".rjust(10)
                    for a, b in zip(edges[:-1], edges[1:]))
    print(f"{'':<14}{hdr}")
    for nm, d, big, ok in per:
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = big & (~ok) & (d['n'] >= a) & (d['n'] < b)
            cells.append(f"{np.median(d['alt_bits'][m]):10.2f}" if m.sum() >= 5 else f"{'—':>10}")
        print(f"{nm:<14}" + "  ".join(cells))

    print("\n  ω=0 の線の性質（中央値）")
    for nm, d, big, ok in per:
        m = big & (d['omega_deg_s'] == 0)
        if m.sum() < 5:
            print(f"{nm:<14}該当 {m.sum()} 本")
            continue
        print(f"{nm:<14}該当 {m.sum():5d} 本  点数 {np.median(d['n'][m]):5.0f}  "
              f"角度幅 {np.median(d['span_deg'][m]):4.1f} 度  "
              f"薄さ {np.median(d['thin_mm'][m]):7.2f} mm  "
              f"採用率 {ok[m].mean():.3f}")


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 1:
        compare(args)
    else:
        main(args[0] if args else 'scan_diag.tsv')
