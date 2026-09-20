"""PCC2 の幾何符号器を G-PCC（TMC13）と LASzip と同じ土俵で比べる。

土俵の揃え方:
  * G-PCC は点の出力順を保存しない。PCC2 と LASzip は保存する。
    順序を保つぶん不利なので、PCC2 側が勝てばその差は下限になる。
  * 走査モデルは gps_time / point_source_id / bit_fields が先に復号されている
    ことを前提にするので、その 3 列の符号長も走査側に計上する。
    幾何v3 は副列を要しないので計上しない。
  * G-PCC は幾何のみ・可逆（trisoup なし、量子化なし、重複点を残す）。
    復号して全点照合する。

使い方:
    $PCCPY python/exp_gpcc_compare.py <in.las|laz> [点数]
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import tmc13_bits           # noqa: E402

PCC = "./cpp/build/pccnorm"
PRE = ("gps_time", "point_source_id", "bit_fields")


def run_pcc2(path: str, cap: int):
    """PCC2 を走らせ、幾何候補と副列の bpp を取り出す。"""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser("~/tools/laszip-install/lib") +
                              ":" + env.get("LD_LIBRARY_PATH", ""))
    with tempfile.TemporaryDirectory() as d:
        cmd = [PCC, "pack", path, os.path.join(d, "o.pcc2"), "--trace"]
        if cap:
            cmd += ["--max-points", str(cap)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = r.stdout
    cand, pre, sel, total, ok = {}, {}, None, None, None
    for ln in out.splitlines():
        m = re.match(r"^  X\+Y\+Z\s+(\S+)\s+([\d.]+) bpp", ln)
        if m:
            sel = m.group(1)
        m = re.match(r"^      (\S+)\s+([\d.]+) bpp", ln)
        if m:
            cand[m.group(1)] = float(m.group(2))
        m = re.match(r"^  (\S+)\s+\S+\s+([\d.]+) bpp", ln)
        if m and m.group(1) in PRE:
            pre[m.group(1)] = float(m.group(2))
        m = re.match(r"^PCC2\s+[\d.]+ MB\s+([\d.]+) bpp", ln)
        if m:
            total = float(m.group(1))
        if "全列一致" in ln:
            ok = "true" in ln
    if sel is None:
        sys.stderr.write(out[-2000:] + "\n" + r.stderr[-2000:] + "\n")
    return cand, pre, sel, total, ok


def main() -> None:
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000

    with laspy.open(path) as fh:
        pts = fh.read()
    n = min(cap, len(pts.X)) if cap else len(pts.X)
    xyz = np.stack([np.asarray(pts.X)[:n], np.asarray(pts.Y)[:n],
                    np.asarray(pts.Z)[:n]], axis=1).astype(np.int64)
    del pts

    print(f"入力        {path}")
    print(f"            {n} 点")
    print()

    print("G-PCC を走らせています…", flush=True)
    g = tmc13_bits(xyz)
    gb = g.bytes * 8.0 / n if g.bytes else float("nan")
    print(f"G-PCC/TMC13  {gb:8.3f} bpp   可逆 {g.lossless}   enc {g.enc_s:.1f}s / dec {g.dec_s:.1f}s")
    if g.note:
        print(f"             注記 {g.note}")
    print()

    print("PCC2 を走らせています…", flush=True)
    cand, pre, sel, total, ok = run_pcc2(path, cap)
    pre_sum = sum(pre.values())
    print(f"PCC2 採択 {sel} / 全列 {total} bpp / 検証 {ok}")
    print()

    print("=== 幾何のみの比較 [bit/点] ===")
    print(f"{'方式':<22}{'幾何':>9}{'副列':>9}{'合計':>9}{'順序':>7}")
    print(f"{'G-PCC/TMC13':<22}{gb:>9.3f}{0.0:>9.3f}{gb:>9.3f}{'保存せず':>7}")
    for nm in ("幾何v0", "幾何v1", "幾何v2", "幾何v3"):
        if nm in cand:
            print(f"{'PCC2 ' + nm:<22}{cand[nm]:>9.3f}{0.0:>9.3f}{cand[nm]:>9.3f}{'保存':>7}")
    for nm in ("走査v1", "走査変換"):
        if nm in cand:
            print(f"{'PCC2 ' + nm:<22}{cand[nm]:>9.3f}{pre_sum:>9.3f}"
                  f"{cand[nm] + pre_sum:>9.3f}{'保存':>7}")
    print()
    print("副列の内訳: " + " / ".join(f"{k} {v:.3f}" for k, v in pre.items()))

    best = min([cand[k] for k in ("幾何v0", "幾何v1", "幾何v2", "幾何v3") if k in cand] +
               [cand[k] + pre_sum for k in ("走査v1", "走査変換") if k in cand])
    if gb == gb:
        print()
        print(f"PCC2 の最良 {best:.3f} 対 G-PCC {gb:.3f} → {100 * (best / gb - 1):+.1f}%")


if __name__ == "__main__":
    main()
