"""外挿でなく内挿で符号化したときの符号長を見積もる（多段の持ち上げ変換）。

いまの符号器は格納順に厳密に因果的で、各点は「それまでに復号した点」しか見ない。
つまり**外挿**している。八分木系や PACE の post-causal は逆で、粗い層を先に
確定させてから細かい層の文脈にする。我々の系列に落とすと**内挿**になる。

  第 0 段: 偶数番の点を送る
  第 1 段: 奇数番の点を、前後の偶数番の点で挟んで内挿し、残差だけ送る
  これを再帰する（点数が半分ずつになる）

滑らかな系列では内挿の残差は外挿の残差よりずっと小さい。
整数だけで書けて決定的なので、我々の制約に収まる。

見積りは符号器と同じ形で数える（桁数を適応模型 + 仮数部は生）。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "2000000"))
SEL = sys.argv[1:] or ["plane", "vegetation", "extra", "USGS NY", "autzen-2023", "KITTI"]
NCTX = 24


def bitlen(z: np.ndarray) -> np.ndarray:
    k = np.zeros(len(z), dtype=np.int64)
    t = (z.astype(np.int64) + 1).astype(np.uint64)
    while True:
        m = t > np.uint64(1)
        if not m.any():
            break
        k[m] += 1
        t[m] >>= np.uint64(1)
    return k


def zz(v: np.ndarray) -> np.ndarray:
    return np.where(v >= 0, v * 2, -v * 2 - 1).astype(np.int64)


def cost(res: np.ndarray, ctx: np.ndarray | None = None, nctx: int = 1) -> float:
    """桁数を文脈つき適応模型で、仮数部は生。到達可能な符号長。"""
    if len(res) == 0:
        return 0.0
    k = bitlen(zz(res))
    c = np.zeros(len(k), dtype=np.int64) if ctx is None else np.minimum(ctx, nctx - 1)
    f = np.ones((nctx, 65)); tot = f.sum(axis=1)
    b = 0.0
    for kk, cc in zip(k.tolist(), c.tolist()):
        b += -np.log2(f[cc, kk] / tot[cc])
        f[cc, kk] += 32.0; tot[cc] += 32.0
    return b + float(k.sum())


def extrap(col: np.ndarray) -> float:
    """いまの形: 直前の値からの差（無傾）。文脈は直前の桁数。"""
    d = col - np.concatenate(([0], col[:-1]))
    k = bitlen(zz(d))
    ctx = np.concatenate(([0], k[:-1]))
    return cost(d, ctx, NCTX)


def lifting(col: np.ndarray, levels: int) -> float:
    """多段の内挿。粗い層から順に送り、細かい層は前後で挟んで内挿する。"""
    total = 0.0
    idx = np.arange(len(col))
    for _ in range(levels):
        if len(idx) < 4:
            break
        ev = idx[0::2]          # 次の段へ持ち越す（粗い層）
        od = idx[1::2]          # この段で送る（細かい層）
        # 奇数番を、前後の偶数番で挟んで内挿
        a = col[ev[:len(od)]]
        bnx = col[ev[1:len(od) + 1]] if len(ev) > len(od) else np.concatenate(
            (col[ev[1:]], col[ev[-1:]]))
        if len(bnx) < len(od):
            bnx = np.concatenate((bnx, np.full(len(od) - len(bnx), col[ev[-1]])))
        pred = (a + bnx[:len(od)]) // 2
        r = col[od] - pred
        k = bitlen(zz(r))
        ctx = np.minimum(bitlen(zz(bnx[:len(od)] - a)), NCTX - 1)   # 挟んだ幅を文脈に
        total += cost(r, ctx, NCTX)
        idx = ev
    total += extrap(col[idx])   # 最後に残った粗い層は従来どおり外挿で送る
    return total


print(f"標本 {N} 点。幾何 3 軸の合計 [bit/点]。到達可能な符号長の見積り。\n")
LV = [int(x) for x in os.environ.get("LIFT_LV", "3,6").split(",")]
print(f"{'データ':<13}{'いま(外挿)':>12}" + "".join(f"{('内挿'+str(l)+'段'):>11}" for l in LV) + f"{'最良':>10}")
for lab in SEL:
    path, kind = [(i[1], i[2]) for i in INPUTS if i[0] == lab][0]
    tot = M.total_points(kind, path)
    n = min(N, tot)
    st = max(0, tot // 2 - n // 2)
    xyz, *_ = M.read_block(kind, path, st, n)
    xyz = np.asarray(xyz).astype(np.int64)
    a = sum(extrap(xyz[:, c]) for c in range(3)) / n
    bs = [sum(lifting(xyz[:, c], l) for c in range(3)) / n for l in LV]
    best = min(bs)
    print(f"{lab:<13}{a:>12.3f}" + "".join(f"{v:>11.3f}" for v in bs) + f"{100*(best-a)/a:>9.1f}%")
