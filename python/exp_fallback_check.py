"""退避路（元の器を包む経路）に落ちる条件を確かめる。

`pack` は PCC2 の合計が元の器を超えると出力を埋め込み LAZ に差し替える。
その場合の「全列一致」は埋め込んだ器を検証しており、符号器を検証していない。
30 節の測定はこれを記録していなかったので、代表例で確認した記録を残す。

使い方:
    $PCCPY python/exp_fallback_check.py > data/work/fallback_check.txt
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import laspy

CASES = [("red-rocks", "data/raw/extrabytes/entwine_data_red-rocks.laz"),
         ("simple1_4", "data/raw/small/simple1_4.las"),
         ("AHN4 _20", "data/raw/ahn4/31HZ1_20.LAZ")]
B = 1_000_000


def main() -> None:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser("~/tools/laszip-install/lib") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))
    print("退避路に落ちるかの確認（--no-fallback なし）")
    print(f"{'データ':<12}{'条件':<10}{'幾何v3':>9}{'PCC2合計':>10}{'検証':>7}  中身")
    for label, src in CASES:
        with laspy.open(src) as fh:
            h = fh.header
            t = h.point_count
            st = max(0, t // 2 - B // 2) if t > B else 0
            pts = fh.read_points(min(st + B, t))
        pts = pts[st:st + B]
        n = len(pts)
        rng = np.random.default_rng(20260920)
        for cond, idx in (("恒等", np.arange(n)), ("ランダム", rng.permutation(n))):
            d = Path(tempfile.mkdtemp())
            p = d / "x.laz"
            h2 = laspy.LasHeader(version=h.version, point_format=h.point_format)
            for v in h.vlrs:
                tn = type(v).__name__
                if tn.startswith("ExtraBytes") or tn.startswith("Copc"):
                    continue
                try:
                    v.record_data_bytes()
                except Exception:
                    continue
                h2.vlrs.append(v)
            h2.scales, h2.offsets = h.scales, h.offsets
            las = laspy.LasData(h2)
            las.points = pts[idx].copy()
            las.write(str(p))
            r = subprocess.run(["./cpp/build/pccnorm", "pack", str(p), str(d / "o.pcc2"),
                                "--force-geom", "幾何v3", "--fast-attr"],
                               capture_output=True, text=True, env=env)
            got = {"X": "", "中身": "", "PCC2": "", "検証": ""}
            for ln in r.stdout.splitlines():
                if "X+Y+Z" in ln:
                    got["X"] = ln.split()[2]
                if ln.startswith("中身"):
                    got["中身"] = ln.split(None, 1)[1].strip()
                if ln.startswith("PCC2"):
                    got["PCC2"] = ln.split()[3]
                if "全列一致" in ln:
                    got["検証"] = "true" if "true" in ln else "false"
            print(f"{label:<12}{cond:<10}{got['X']:>9}{got['PCC2']:>10}"
                  f"{got['検証']:>7}  {got['中身']}")


if __name__ == "__main__":
    main()
