"""差分の値が繰り返す列を、直近の値の表で指す案を符号化前に見積もる。

gps_time の差分は値の種類が少ないのに桁が大きい。いまの符号器は
「bit 長を文脈つきで符号化し、下位ビットは素通し」なので、同じ値が何度出ても
毎回 bit 長ぶん払う。直近 K 個の相異なる差分を表に持ち、当たれば添字だけを
送れば、この繰り返しを拾える。

逐次推定（KT）で符号長を出す。実際の適応符号器と同じで、当たらない場合の
逃げ道（escape + いまの方式）も込みで数える。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
K = int(os.environ.get("EXP_K", "64"))
NCTX = 24


def klen(z: int) -> int:
    return min(int(z).bit_length(), NCTX - 1)


def cost_now(d: np.ndarray) -> float:
    """いまの方式に近い見積り: bit 長を直前の bit 長で条件づけ、下位は素通し。"""
    cnt = np.full((NCTX, NCTX), 0.5)
    tot = np.full(NCTX, 0.5 * NCTX)
    bits, prev = 0.0, 0
    for v in d:
        z = abs(int(v)) * 2
        k = klen(z)
        bits -= np.log2(cnt[prev, k] / tot[prev])
        cnt[prev, k] += 1.0
        tot[prev] += 1.0
        bits += max(k - 1, 0)
        prev = k
    return bits / len(d)


def cost_cache(d: np.ndarray, k: int) -> float:
    """直近 k 個の相異なる差分の表。当たれば添字、外れれば escape + いまの方式。"""
    idx_cnt = np.full(k + 1, 0.5)          # k は escape
    idx_tot = 0.5 * (k + 1)
    cnt = np.full((NCTX, NCTX), 0.5)
    tot = np.full(NCTX, 0.5 * NCTX)
    cache: list[int] = []
    bits, prev = 0.0, 0
    for v in d:
        x = int(v)
        pos = cache.index(x) if x in cache else k
        bits -= np.log2(idx_cnt[pos] / idx_tot)
        idx_cnt[pos] += 1.0
        idx_tot += 1.0
        if pos == k:
            z = abs(x) * 2
            kk = klen(z)
            bits -= np.log2(cnt[prev, kk] / tot[prev])
            cnt[prev, kk] += 1.0
            tot[prev] += 1.0
            bits += max(kk - 1, 0)
            prev = kk
        if x in cache:
            cache.remove(x)
        cache.insert(0, x)                  # 移動して先頭へ
        if len(cache) > k:
            cache.pop()
    return bits / len(d)


def main():
    print(f"標本 {N} 点。gps_time の差分。bit/点。表の大きさ K={K}。")
    print(f"{'データ':<13}{'点':>8}{'Δ の H0':>9}{'いまの方式':>11}{'表':>8}{'差':>8}{'当たり率':>9}")
    for lab, path, kind in INPUTS:
        if kind != "las":
            continue
        try:
            a = laspy.read(path)
            g = np.asarray(a.gps_time)
        except Exception:
            continue
        if len(np.unique(g)) <= 1:
            continue
        tot_n = len(g)
        n = min(N, tot_n)
        st = max(0, tot_n // 2 - n // 2)
        u = g[st:st + n].view(np.int64)
        d = np.diff(u)
        if len(d) < 10:
            continue
        vals, c = np.unique(d, return_counts=True)
        p = c / c.sum()
        h0 = float(-(p * np.log2(p)).sum())
        a0 = cost_now(d)
        a1 = cost_cache(d, K)
        # 当たり率
        cache: list[int] = []
        hit = 0
        for v in d:
            x = int(v)
            if x in cache:
                hit += 1
                cache.remove(x)
            cache.insert(0, x)
            if len(cache) > K:
                cache.pop()
        print(f"{lab:<13}{n:>8}{h0:>9.2f}{a0:>11.2f}{a1:>8.2f}{100*(a1/a0-1):>+7.1f}%"
              f"{100*hit/len(d):>8.1f}%")


if __name__ == "__main__":
    main()
