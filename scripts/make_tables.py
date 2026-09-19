# 一斉実行の結果を 4 枚の比較表にまとめる
import re, os, sys
D = 'results/matrix'
ORDER = [
 ("AHN4_航空LiDAR","AHN4 航空LiDAR","航空"),
 ("autzen__PDAL_","autzen (PDAL)","航空"),
 ("plane__PDAL_","plane (PDAL)","航空"),
 ("fullwave__YellowScan_","fullwave (YellowScan)","航空"),
 ("simple1_4__GlobalMapper_","simple1_4 (GlobalMapper)","その他"),
 ("vegetation__RS_Survey_","vegetation (RS Survey)","航空"),
 ("KITTI_車載LiDAR","KITTI 車載LiDAR","車載"),
 ("Bunny_生スキャン","Bunny 生スキャン","三角測量 生"),
 ("Bunny_再構成","Bunny 再構成","三角測量 再構成"),
 ("Dragon_生スキャン","Dragon 生スキャン","三角測量 生"),
 ("Dragon_再構成","Dragon 再構成","三角測量 再構成"),
 ("Armadillo_生スキャン","Armadillo 生スキャン","三角測量 生"),
 ("Armadillo_再構成","Armadillo 再構成","三角測量 再構成"),
]
def rd(slug, kind):
    p = f'{D}/{slug}__{kind}.txt'
    return open(p, encoding='utf-8', errors='replace').read() if os.path.exists(p) else ''
def f1(pat, s, d=None, g=1):
    m = re.search(pat, s); return float(m.group(g)) if m else d
def el(s):
    m = re.search(r'__elapsed\s+([\d.]+)', s); return float(m.group(1)) if m else 0.0

W = 150
print("="*W); print("【表1】取得構造スコア — 生スキャンと再構成で構造はどう変わるか"); print("="*W)
print(f"{'データ':<26}{'種別':<16}{'スコア':>7}{'距離':>7}{'角度':>7}{'向き':>7}{'順序':>7}{'走査軸':>8}{'回転':>7}{'Morton差':>10}{'推奨':>10}{'秒':>7}")
print("-"*W)
for slug, name, kind in ORDER:
    t = rd(slug,'score')
    if not t.strip(): continue
    comp = dict(re.findall(r'^\s+(\S+?)\s+([\d.]+)\s', t, re.M))
    sc = f1(r'取得構造スコア ([\d.]+)', t, 0)
    md = f1(r'Morton (?:より|差) ([+-]?[\d.]+) bpp', t, 0)
    rec = re.search(r'格納順維持=(\d).*?Morton候補=(\d).*?極座標候補=(\d)', t)
    g = lambda k: comp.get(k,'—')
    print(f"{name:<26}{kind:<16}{sc:>7.2f}{g('距離の規則性'):>7}{g('角度の規則性'):>7}{g('向きの特権性'):>7}"
          f"{g('順序の意味'):>7}{g('走査軸の痕跡'):>8}{g('回転走査の痕跡'):>7}{md:>10.2f}"
          f"{(','.join(rec.groups()) if rec else '—'):>10}{el(t):>7.2f}")

print()
print("="*W); print("【表2】属性の冗長 — 重複は AHN4 固有か（bench の検出結果, --no-delta）"); print("="*W)
print(f"{'データ':<26}{'A bpp':>9}{'B bpp':>9}{'削減':>8}{'検出した冗長':>46}{'検証':>7}{'秒':>7}")
print("-"*W)
for slug, name, kind in ORDER:
    t = rd(slug,'benchnd')
    if not t.strip(): continue
    A = f1(r'A\. 元のまま LAZ\s+\d+\s+([\d.]+)', t)
    B = f1(r'B\. 合計\s+\d+\s+([\d.]+)', t)
    red = f1(r'=\s*([\d.-]+)%', t, 0)
    ops = re.findall(r'\[可逆\] (\S+) (.+)', t)
    desc = "; ".join(f"{a}←{b.split(' と')[0].split(' との')[0]}" for a,b in ops[:3]) or "なし"
    ok = "OK" if 'OK=true' in t and '完全一致=true' in t else "NG"
    if A is None: continue
    print(f"{name:<26}{A:>9.2f}{B:>9.2f}{red:>7.1f}%{desc[:44]:>46}{ok:>7}{el(t):>7.2f}")

print()
print("="*W); print("【表3】オクタント予測 — 点間隔以下の壁は普遍か"); print("="*W)
print(f"{'データ':<26}{'点間隔':>12}{'被覆@1/4':>10}{'被覆@1':>9}{'H(子)@1':>10}{'情報量@1':>10}{'的中@1':>9}{'秒':>7}")
print("-"*W)
for slug, name, kind in ORDER:
    t = rd(slug,'octant')
    if not t.strip(): continue
    sp = f1(r'点間隔 ([\d.eE+-]+)', t)
    rows = re.findall(r'^\s+([\d.]+)\s+(\d+)%\s+([\d.]+|—)\s+([\d.]+|—)\s+([\d.]+|—)\s+([\d.]+)%', t, re.M)
    g14 = next((r for r in rows if abs(float(r[0])-0.25)<1e-6), None)
    g1  = next((r for r in rows if abs(float(r[0])-1.0)<1e-6), None)
    if sp is None: continue
    print(f"{name:<26}{sp:>12.5g}{(g14[1]+'%') if g14 else '—':>10}{(g1[1]+'%') if g1 else '—':>9}"
          f"{(g1[2]) if g1 else '—':>10}{(g1[4]) if g1 else '—':>10}{(g1[5]+'%') if g1 else '—':>9}{el(t):>7.2f}")

print()
print("="*W); print("【表4】誤差予算 — 位相が律速になる場面はあるか"); print("="*W)
print(f"{'データ':<26}{'点間隔':>12}{'最長寿命 L':>13}{'L/点間隔':>10}{'ε=L/2':>13}{'第6特徴 ε':>13}{'幾何の採用':>14}{'秒':>7}")
print("-"*W)
for slug, name, kind in ORDER:
    tp = rd(slug,'persist'); tg = rd(slug,'geom')
    if not tp.strip(): continue
    sp = f1(r'点間隔 ([\d.eE+-]+)', tp)
    rows = re.findall(r'^\s+H\d\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)\s+([\d.eE+-]+)', tp, re.M)
    if not rows or sp is None: continue
    L1 = float(rows[0][2]); e1 = float(rows[0][4])
    e6 = float(rows[min(5,len(rows)-1)][4])
    sel = re.search(r'選択 (\S+)', tg)
    print(f"{name:<26}{sp:>12.5g}{L1:>13.5g}{L1/sp:>10.2f}{e1:>13.5g}{e6:>13.5g}"
          f"{(sel.group(1) if sel else '—'):>14}{el(tp):>7.2f}")
