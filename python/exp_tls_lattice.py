"""TLS の講堂のスキャン（tls_scan1.ply、400 万点）で、同じ誤差上限の極座標格子と直交格子を器で符号化して比べる。

極座標は pack --eps（器が選ぶ。自己検証つき）。直交格子は刻み 2ε/√3 の整数格子に丸めた LAS を可逆で pack する。
どちらも器の頭まで全部数える。results/tls_polar_finding.md の訂正の出どころ。

    python -u python/exp_tls_lattice.py <作業用ディレクトリ>   # 出力は data/work/grid/tls_lattice_v22.log に保存した
"""
import re, subprocess, sys, numpy as np, laspy
from pathlib import Path
S = Path(sys.argv[1]); PCC = "cpp/build/pccnorm"
raw = open("data/work/tls_scan1.ply", "rb").read(); h = raw.index(b"end_header\n") + 11
xyz = np.frombuffer(raw[h:], "<f4").reshape(-1, 3).astype(np.float64); n = len(xyz)
print(f"tls_scan1 {n} 点（講堂、RIEGL VZ-400 の 400 万点）。器で両方の格子を符号化して比べる")
for eps in (0.0001, 0.0015, 0.005):
    r = subprocess.run([PCC, "pack", "data/work/tls_scan1.ply", str(S / "p.pcc2"), "--no-fallback", "--eps", str(eps)],
                       capture_output=True, text=True)
    kind = re.search(r"幾何を(\S+?)に量子化", r.stdout); okp = "誤差上限の中 = true" in r.stdout
    bp = (S / "p.pcc2").stat().st_size * 8 / n
    step = 2 * eps / np.sqrt(3)                      # 立方格子の最大誤差は step*√3/2
    k = np.round(xyz / step).astype(np.int64)
    err = np.sqrt(((k * step - xyz) ** 2).sum(1)).max()
    hd = laspy.LasHeader(point_format=0, version="1.2"); hd.scales = [step] * 3; hd.offsets = [0, 0, 0]
    d = laspy.LasData(hd); d.X = k[:, 0]; d.Y = k[:, 1]; d.Z = k[:, 2]; d.write(S / "c.las")
    r2 = subprocess.run([PCC, "pack", str(S / "c.las"), str(S / "c.pcc2"), "--no-fallback"], capture_output=True, text=True)
    okc = "全列一致 = true" in r2.stdout
    bc = (S / "c.pcc2").stat().st_size * 8 / n
    print(f"誤差上限 {eps*1000:5.1f} mm  器が選ぶ {kind.group(1) if kind else '?'} {bp:7.3f} bpp（検証 {okp}）"
          f"  直交格子を器で可逆 {bc:7.3f} bpp（最大誤差 {err*1000:.3f} mm、検証 {okc}）  極/直 {100*(bp/bc-1):+.1f}%")
