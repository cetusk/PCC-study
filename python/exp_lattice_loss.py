"""非可逆の格子の選択（代理の符号長による推定）がどれだけ損をしているかを測る（approaches.md §G の S2）。

pack --eps が自動で選ぶ格子と、PCC_GEOM_KIND で 3 種類（grid / polar/origin / polar/centroid）に固定した
ときの器の長さを比べる。固定した種類が誤差上限を守れない（候補に無い）ときは自動と同じになる。

    python -u python/exp_lattice_loss.py      # 出力は data/work/grid/lattice_loss_v22.log に保存した
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PCC = os.path.abspath("cpp/build/pccnorm")
KINDS = ["grid", "polar/origin", "polar/centroid"]
cases = []
for k in sorted(Path("data/raw/kitti").rglob("*.bin"))[:20]:
    for e in ("0.002", "0.005", "0.02"):
        cases.append((f"KITTI {k.stem}", k, e))
for f in ["data/raw/stanford/Armadillo.ply", "data/raw/stanford/bunny/data/bun000.ply",
          "data/raw/stanford/dragon_stand/dragonStandRight_0.ply", "data/work/tls_scan1.ply"]:
    for e in ("0.001", "0.01"):
        cases.append((Path(f).name, Path(f), e))
tmp = Path(tempfile.mkdtemp())
out = tmp / "o.pcc2"


def size(src, eps, kind):
    env = dict(os.environ)
    if kind:
        env["PCC_GEOM_KIND"] = kind
    if out.exists():
        out.unlink()
    subprocess.run([PCC, "pack", str(src), str(out), "--no-fallback", "--no-verify", "--eps", eps],
                   capture_output=True, text=True, env=env)
    return out.stat().st_size if out.exists() else None


print("器の長さ（byte）。自動 = pack --eps が代理の符号長で選んだ格子。損 = 自動 − 3 種類の最小")
print(f"{'入力':<26}{'誤差上限':>8}{'自動':>10}" + "".join(f"{k:>16}" for k in KINDS) + f"{'損':>9}")
tot_auto = tot_best = 0
lost = 0
for lab, src, eps in cases:
    a = size(src, eps, None)
    ks = [size(src, eps, k) for k in KINDS]
    best = min([x for x in ks if x] + [a])
    tot_auto += a; tot_best += best
    lost += a > best
    print(f"{lab:<26}{eps:>8}{a:>10}" + "".join(f"{(x if x else '-'):>16}" for x in ks) +
          f"{a - best:>+9}", flush=True)
print(f"\n合計 自動 {tot_auto}  3 種類の最小 {tot_best}  損 {tot_auto - tot_best} byte"
      f"（{100 * (tot_auto - tot_best) / tot_auto:.3f}%）  自動が最短でなかった件 {lost}/{len(cases)}")
