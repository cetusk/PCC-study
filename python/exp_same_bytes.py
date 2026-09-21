"""2 つの実行ファイルが同じ .pcc2 を出すかを、全データで突き合わせる。

速度だけを変えた改変が、出力を 1 bit も動かしていないことの確認に使う。
PCC_REF に基準の実行ファイルの場所を入れて呼ぶ。
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
REF = os.environ["PCC_REF"]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    print(f"標本 {N} 点。基準 {REF}")
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
            for exe in (REF, PCC):
                r = run([exe, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], ENV)
                if r["rc"] != 0:
                    h.append("失敗:" + r["out"][-200:])
                else:
                    h.append(md5(tmp / "o.pcc2"))
            ok = h[0] == h[1]
            bad += not ok
            sz = (tmp / "o.pcc2").stat().st_size
            print(f"{lab:<13}{n:>8}{sz:>11}  {'一致' if ok else '不一致 ' + h[0][:8] + ' / ' + h[1][:8]}")
        finally:
            for q in tmp.glob("*"):
                q.unlink()
            tmp.rmdir()
    print(f"\n  出力が動いたもの {bad} 件 / {len(INPUTS)} 件")


if __name__ == "__main__":
    main()
