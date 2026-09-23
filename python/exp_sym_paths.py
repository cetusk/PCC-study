"""字母の旗の変更（c05db51）が既定以外の経路の出力を変えるか。
変更前（pcc_base: 決定性の検査だけ入った版）と最終（pcc_final）で、15 件の中央 200 万点を
--sample-select 25000 と PCC_PRESELECT_ATTR=1 で pack し、器の長さを比べる（器の版の違いは長さを変えない）。

    python -u python/exp_sym_paths.py <変更前の pccnorm> <変更後の pccnorm>
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import ENV
from bench_size import INPUTS
A, B = sys.argv[1], sys.argv[2]
CASES = [("--sample-select 25000", ["--sample-select", "25000"], {}),
         ("PCC_PRESELECT_ATTR=1", [], {"PCC_PRESELECT_ATTR": "1"})]
tmp = Path(tempfile.mkdtemp())
print(f"変更前 = {A}\n変更後 = {B}")
print("データ          経路                     変更前 byte    最終 byte      差")
chg = {c[0]: 0 for c in CASES}
for lab, path, kind in [i for i in INPUTS if i[2] == "las"]:
    t = M.total_points(kind, path); n = min(2000000, t); st = max(0, t // 2 - n // 2)
    xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
    src = tmp / "s.laz"; M.write_las(src, xyz, sc, of, pts, hdr)
    for name, args, env in CASES:
        e = dict(ENV); e.update(env)
        sz = []
        for exe in (A, B):
            o = tmp / "o.pcc2"
            if o.exists(): o.unlink()
            r = run([exe, "pack", str(src), str(o), "--no-fallback", "--no-verify", *args], e)
            sz.append(o.stat().st_size if o.exists() else -1)
        d = sz[1] - sz[0]; chg[name] += d != 0
        print(f"{lab:<15} {name:<22} {sz[0]:>12} {sz[1]:>12} {d:>+8} ({100*d/sz[0]:+.3f}%)", flush=True)
print("\n長さが変わったもの: " + "  ".join(f"{k} {v}/15" for k, v in chg.items()))
