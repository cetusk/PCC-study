"""推定で切り落としている所の損を測る（approaches.md §G の S2）。

既定の pack と、環境変数で切り落としを外した pack を、15 件の中央 200 万点で比べる。
器の長さ（byte）と符号化の時間（pack が表示する enc）をファイルごとに出す。

    python -u python/exp_select_loss.py 名前=変数=値[,変数=値] ...
    例: python -u python/exp_select_loss.py 予備選別なし=PCC_PRESELECT=0 参照全部=PCC_XREF_KEEP=99

どれも符号化側だけの選択なので、器の読み方は変わらない（--no-verify で回す。採るなら検証一式を別に回す）。
"""
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_pcc2 import ENV
from bench_size import INPUTS
from runpeak import run

PCC = os.path.abspath("cpp/build/pccnorm")
N = int(os.environ.get("BENCH_N", "2000000"))
variants = []
for a in sys.argv[1:]:
    name, rest = a.split("=", 1)
    env = {}
    for kv in rest.split(","):
        k, v = kv.split("=", 1)
        env[k] = v
    variants.append((name, env))
print(f"標本 中央 {N} 点。器の長さ（byte）と enc（秒）。基準は既定の pack。")
print(f"{'データ':<13}{'既定 byte':>12}{'enc':>7}" + "".join(f"{n:>22}" for n, _ in variants))
tot = {n: [0, 0.0] for n, _ in variants}
tot0 = [0, 0.0]
changed = {n: 0 for n, _ in variants}
worse = {n: 0 for n, _ in variants}
for lab, path, kind in [i for i in INPUTS if i[2] == "las"]:
    tmp = Path(tempfile.mkdtemp())
    t = M.total_points(kind, path); n = min(N, t); st = max(0, t // 2 - n // 2)
    xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
    src = tmp / "s.laz"; M.write_las(src, xyz, sc, of, pts, hdr)

    def one(env):
        e = dict(ENV); e.update(env)
        o = tmp / "o.pcc2"
        if o.exists(): o.unlink()
        r = run([PCC, "pack", str(src), str(o), "--no-fallback", "--no-verify"], e)
        m = re.search(r"enc ([0-9.]+)s", r["out"])
        return (o.stat().st_size if o.exists() else -1), (float(m.group(1)) if m else float("nan"))

    b0, e0 = one({})
    tot0[0] += b0; tot0[1] += e0
    row = f"{lab:<13}{b0:>12}{e0:>7.2f}"
    for name, env in variants:
        b, e = one(env)
        tot[name][0] += b; tot[name][1] += e
        d = b - b0
        changed[name] += d != 0
        worse[name] += d > 0
        row += f"{d:>+10} ({100 * d / b0:+.3f}%) {e:5.1f}s"
    print(row, flush=True)
    for p in tmp.iterdir(): p.unlink()
    tmp.rmdir()
print()
print(f"合計 既定 {tot0[0]} byte  enc {tot0[1]:.2f} s")
for name, _ in variants:
    b, e = tot[name]
    print(f"{name:<16} 合計 {b} byte（{100 * (b - tot0[0]) / tot0[0]:+.4f}%）  enc {e:.2f} s（{100 * (e - tot0[1]) / tot0[1]:+.1f}%）"
          f"  長さが変わった {changed[name]}/15（伸びた {worse[name]}）")
