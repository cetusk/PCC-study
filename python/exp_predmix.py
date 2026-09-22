"""予測子を「選ぶ」のと「混ぜる」のを、実際の値で比べる。

面内には予測子が 2 つある。当てはめ（走査モデル）と中央値予測（退避路）。
今は走査線ごとにどちらか一方を選んでいる。
点ごとに w·model + (1-w)·alt と混ぜたら短くなるかを測る。

符号長は離散ラプラスで見積もる（裾に効く形）。
"""
import sys, math
import numpy as np


def bits(r):
    r = np.asarray(r, dtype=np.float64)
    b = np.mean(np.abs(r - np.median(r)))
    return math.log2(2 * math.e * b) if b > 1e-9 else 0.0


def main(path):
    a = np.loadtxt(path, skiprows=1, dtype=np.int64)
    ok = a[:, 1]
    r_used = a[:, 5].astype(np.float64)
    r_mdl = a[:, 8].astype(np.float64)
    r_alt = a[:, 9].astype(np.float64)
    n = len(a)
    line = a[:, 0]
    brk = np.flatnonzero(np.diff(line) != 0) + 1
    segs = np.split(np.arange(n), brk)

    # **すべて走査線ごとに見積もって点数で重み付ける。**
    # 全体で見積もると線ごとの原点の違いが混ざり、別の量になる。
    acc = dict(mdl=0.0, alt=0.0, used=0.0, pick=0.0, gmix=0.0, lmix=0.0)
    ws = np.linspace(0, 1, 21)
    # 大域の比率は先に決める（線ごとに見積もった合計を最小にする w）
    tot_w = []
    for w in ws:
        t = 0.0
        for sgm in segs:
            if len(sgm) < 32:
                t += len(sgm) * bits(r_used[sgm]); continue
            t += len(sgm) * bits(w * r_mdl[sgm] + (1 - w) * r_alt[sgm])
        tot_w.append(t)
    wg = ws[int(np.argmin(tot_w))]
    acc['gmix'] = min(tot_w)

    for sgm in segs:
        L = len(sgm)
        acc['mdl'] += L * bits(r_mdl[sgm])
        acc['alt'] += L * bits(r_alt[sgm])
        acc['used'] += L * bits(r_used[sgm])
        pb = np.where(np.abs(r_mdl[sgm]) < np.abs(r_alt[sgm]), r_mdl[sgm], r_alt[sgm])
        acc['pick'] += L * bits(pb)
        if L < 32:
            acc['lmix'] += L * bits(r_used[sgm]); continue
        acc['lmix'] += L * min(bits(w * r_mdl[sgm] + (1 - w) * r_alt[sgm]) for w in ws)

    print(f'  点 {n}  走査線 {len(segs)} 本  モデル採用率 {ok.mean():.3f}'
          f'  （すべて走査線ごとに見積もり、点数で重み付け）')
    print(f'    {"当てはめだけ":<24}{acc["mdl"]/n:7.3f} bit/点')
    print(f'    {"退避路だけ":<24}{acc["alt"]/n:7.3f} bit/点')
    print(f'    {"今（線ごとに選ぶ）":<24}{acc["used"]/n:7.3f} bit/点')
    print(f'    {"混ぜる（大域の比率）":<24}{acc["gmix"]/n:7.3f} bit/点   w={wg:.2f}')
    print(f'    {"混ぜる（線ごとの比率）":<24}{acc["lmix"]/n:7.3f} bit/点'
          f'   （-{(acc["used"]-acc["lmix"])/n:.3f}、'
          f'{100*(acc["used"]-acc["lmix"])/max(acc["used"],1e-9):.1f}%）')
    print(f'    {"点ごとに短いほう（上限）":<24}{acc["pick"]/n:7.3f} bit/点'
          f'   ※復号側は選べないので実現不可')


if __name__ == '__main__':
    main(sys.argv[1])
