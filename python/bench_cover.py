"""往復させて、元の LAS のどの次元が戻ってくるかを全データで数える。

pack の「全列一致」は **PCC2 が列として持ったもの**しか見ない。列にしていない
次元は、一致も不一致も報告されずに消える。それを外から数える。
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
LAS = [i for i in INPUTS if i[2] == "las"]

print(f"標本 {N} 点。pack → unpack --las で戻したものを元と比べる。")
print(f"{'データ':<13}{'点形式':>7}{'次元':>5}{'一致':>5}{'違う':>5}{'VLR':>9}  戻らなかった次元")
tot_bad = 0
for lab, path, kind in LAS:
    tmp = Path(tempfile.mkdtemp())
    try:
        t = M.total_points(kind, path)
        n = min(N, t)
        st = max(0, t // 2 - n // 2)
        xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
        M.write_las(tmp / "a.laz", xyz, sc, of, pts, hdr)
        subprocess.run([PCC, "pack", str(tmp / "a.laz"), str(tmp / "o.pcc2"),
                        "--no-fallback", "--no-verify"], capture_output=True, env=ENV)
        r = subprocess.run([PCC, "unpack", str(tmp / "o.pcc2"), "--las", str(tmp / "b.laz")],
                           capture_output=True, text=True, env=ENV)
        if not (tmp / "b.laz").exists():
            print(f"{lab:<13}  unpack 失敗 {(r.stdout + r.stderr)[-80:]!r}")
            continue
        da, db = laspy.read(str(tmp / "a.laz")), laspy.read(str(tmp / "b.laz"))
        na = [d.name for d in da.point_format.dimensions]
        nb = {d.name for d in db.point_format.dimensions}
        ok, bad = 0, []
        for x in na:
            if x in nb and np.array_equal(np.asarray(da[x]), np.asarray(db[x])):
                ok += 1
            else:
                bad.append(x)
        tot_bad += bool(bad)
        vl = f"{len(da.vlrs)}→{len(db.vlrs)}"
        print(f"{lab:<13}{da.point_format.id:>7}{len(na):>5}{ok:>5}{len(bad):>5}{vl:>9}"
              f"  {', '.join(bad) if bad else '—'}")
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        tmp.rmdir()
print(f"\n  点の次元が 1 つでも戻らないファイル {tot_bad} 件 / {len(LAS)} 件")
