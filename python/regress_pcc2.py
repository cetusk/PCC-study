"""PCC2 の回帰試験 — 可逆性・決定性・bpp を複数ファイルで確かめる。

速度とメモリを詰めるたびに走らせる。可逆性か決定性が落ちたら即座に止める。
bpp は基準からの変化を出すだけで、良し悪しの判断はしない。
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run
from bench_pcc2 import PCC, ENV

FILES = [("autzen_trim", "data/raw/small/autzen_trim.laz", 0),
         ("vegetation", "data/raw/small/vegetation_1_3.las", 0),
         ("plane", "data/raw/small/plane.laz", 0),
         ("fullwave", "data/raw/small/fullwave.laz", 0),
         ("simple1_4", "data/raw/small/simple1_4.las", 0),
         ("AHN4 _20", "data/raw/ahn4/31HZ1_20.LAZ", 300000),
         ("AHN3 _20", "data/raw/ahn3/31HZ1_20.LAZ", 300000),
         ("USGS NY", "data/raw/usgs/NY_ClintonEssex_2014.laz", 300000),
         ("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz", 300000)]
BASE = Path("data/work/regress_pcc2.json")


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    old = json.loads(BASE.read_text()) if BASE.is_file() else {}
    cur, bad = {}, 0
    print(f"{'データ':<14}{'点':>8}{'秒':>7}{'MB':>6}{'全列 bpp':>10}{'変化':>9}  検証 決定性")
    for lab, path, n in FILES:
        if not Path(path).is_file():
            print(f"{lab:<14} 入力が無い")
            continue
        tot = M.total_points("las", path)
        k = min(n, tot) if n else tot
        xyz, g, sid, sc, of, pts, hdr = M.read_block("las", path, 0, k)
        src = tmp / "in.laz"
        M.write_las(src, xyz, sc, of, pts, hdr)
        r = run([PCC, "pack", str(src), str(tmp / "o.pcc2")], ENV)
        bpp, ok, det = float("nan"), None, None
        for ln in r["out"].splitlines():
            if ln.startswith("PCC2 "):
                bpp = float(ln.split()[3])
            if "全列一致" in ln:
                ok = "true" in ln
            if "決定性" in ln:
                det = "バイト一致" in ln
        cur[lab] = bpp
        d = ""
        if lab in old and old[lab] == old[lab]:
            d = f"{100 * (bpp / old[lab] - 1):+8.3f}%"
        flag = "" if (ok and det) else "  ← 退行"
        bad += 0 if (ok and det) else 1
        print(f"{lab:<14}{len(xyz):>8}{r['sec']:>7.2f}{r['peak_mb']:>6.0f}{bpp:>10.3f}{d:>9}"
              f"  {str(ok):<5}{str(det):<5}{flag}")
        src.unlink()
    BASE.parent.mkdir(parents=True, exist_ok=True)
    BASE.write_text(json.dumps(cur, indent=1))
    print(f"\n  可逆または決定性が落ちたもの {bad} 件")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
