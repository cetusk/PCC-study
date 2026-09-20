"""G-PCC 比較の表を、バッチのログと LAS ヘッダから組み直す。

exp_gpcc_matrix.py は PCC2 側を 幾何v3 と 走査+副列 の 2 つでしか比べていない。
ここでは純幾何の全候補（raw64/range/delta/幾何v0〜v3）から最小を取り、
log2(V/N)（オクツリーが 1 点を特定するのに要するビット数の下限）との
順位相関を、そのまま / plane 除外 / 独立サイト単位 の 3 通りで出す。

G-PCC の値は data/work/gpcc_matrix.txt の測定値を写している。

使い方:
    $PCCPY python/exp_gpcc_summary.py
"""
import re, math, numpy as np, laspy
from pathlib import Path
from scipy.stats import spearmanr

JOBS = [
 ("autzen_trim","data/raw/small/autzen_trim.laz","small_autzen_trim",21.011),
 ("plane","data/raw/small/plane.laz","small_plane",2.775),
 ("fullwave","data/raw/small/fullwave.laz","small_fullwave",26.762),
 ("vegetation","data/raw/small/vegetation_1_3.las","small_vegetation",20.791),
 ("workshop","data/raw/extrabytes/workshop_TM_551_101.laz","workshop_TerraScan",18.350),
 ("autzen-2023","data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz","autzen2023_LasMonkey",17.779),
 ("AHN5 _20","data/raw/ahn5/31HZ1_20.LAZ","ahn5_31HZ1_20",24.964),
 ("AHN3 _20","data/raw/ahn3/31HZ1_20.LAZ","ahn3_31HZ1_20",24.535),
 ("AHN4 _20","data/raw/ahn4/31HZ1_20.LAZ","ahn4_tile",24.217),
 ("AHN4 _21","data/raw/ahn4/31HZ1_21.LAZ","ahn4_tile21",23.802),
]
PURE = ("raw64","range","delta","幾何v0","幾何v1","幾何v2","幾何v3")
PRE  = ("gps_time","point_source_id","bit_fields")

print(f"{'データ':<14}{'log2(V/N)':>10}{'G-PCC':>9}{'純幾何最良':>11}{'走査+副列':>10}{'PCC2最良':>9}{'比':>9}")
rows=[]
for nm,p,lg,g in JOBS:
    txt = Path(f"data/work/batch/{lg}.log").read_text(encoding="utf-8",errors="replace")
    # X+Y+Z ブロックの候補だけを拾う
    cand, inblk = {}, False
    for ln in txt.splitlines():
        if re.match(r"^  X\+Y\+Z\s", ln):
            inblk = True; continue
        if inblk:
            m = re.match(r"^      (\S+)\s+([\d.]+) bpp", ln)
            if m:
                cand[m.group(1)] = float(m.group(2)); continue
            if re.match(r"^  \S", ln):
                inblk = False
    pre = sum(float(m.group(2)) for m in re.finditer(r"^  (\S+)\s+\S+\s+([\d.]+) bpp", txt, re.M) if m.group(1) in PRE)
    pure = min(v for k,v in cand.items() if k in PURE)
    scan = min([cand[k] for k in ("走査v1","走査変換") if k in cand], default=float("inf"))
    stot = scan + pre
    best = min(pure, stot)
    with laspy.open(p) as fh:
        h=fh.header; n=h.point_count
        ext=[max((h.maxs[i]-h.mins[i])/h.scales[i],1.0) for i in range(3)]
        v=sum(math.log2(e) for e in ext)-math.log2(n)
    r=100*(best/g-1)
    rows.append((nm,v,r))
    print(f"{nm:<14}{v:>10.2f}{g:>9.3f}{pure:>11.3f}{stot:>10.3f}{best:>9.3f}{r:>8.1f}%")

v=np.array([x[1] for x in rows]); r=np.array([x[2] for x in rows])
rho,pv=spearmanr(v,r); print(f"\nSpearman ρ = {rho:+.3f}  (p={pv:.4f}, n=10)")
m=[i for i,x in enumerate(rows) if x[1]>10]
rho2,pv2=spearmanr(v[m],r[m]); print(f"plane を除く   ρ = {rho2:+.3f}  (p={pv2:.4f}, n={len(m)})")
# 独立サイト単位（AHN 4 タイル→1、autzen 2→1、他 4）
grp={"AHN3 _20":"AHN","AHN4 _20":"AHN","AHN4 _21":"AHN","AHN5 _20":"AHN",
     "autzen_trim":"autzen","autzen-2023":"autzen"}
import collections
agg=collections.defaultdict(list)
for nm,a,b in rows: agg[grp.get(nm,nm)].append((a,b))
gv=np.array([np.mean([x[0] for x in vs]) for vs in agg.values()])
gr=np.array([np.mean([x[1] for x in vs]) for vs in agg.values()])
rho3,pv3=spearmanr(gv,gr); print(f"独立サイト単位 ρ = {rho3:+.3f}  (p={pv3:.4f}, n={len(gv)})")
