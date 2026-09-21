"""スレッド数を変えても同じ .pcc2 が出るかを、全データで確かめる。

KD木の並列構築のように、並列の分け方が結果に漏れる改変を捕まえるための試験。
"""
from __future__ import annotations
import hashlib, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV
from bench_size import INPUTS

N = int(os.environ.get("BENCH_N", "200000"))
THREADS = (1, 3, 7, 16)


def main():
    print(f"標本 {N} 点。PCC_THREADS を {THREADS} と変えて md5 を比べる。")
    bad = 0
    for lab, path, kind in INPUTS:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path)
            n = min(N, tot)
            st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            src = tmp / "a.laz"
            M.write_las(src, xyz, sc, of, pts, hdr)
            h = []
            for t in THREADS:
                env = dict(ENV); env["PCC_THREADS"] = str(t)
                r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], env)
                h.append("失敗" if r["rc"] != 0
                         else hashlib.md5((tmp / "o.pcc2").read_bytes()).hexdigest())
            ok = len(set(h)) == 1
            bad += not ok
            print(f"{lab:<13}{n:>8}  {h[0][:12]}  {'一致' if ok else '不一致 ' + str(h)}")
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()
    print(f"\n  スレッド数で出力が動いたもの {bad} 件 / {len(INPUTS)} 件")


if __name__ == "__main__":
    main()
