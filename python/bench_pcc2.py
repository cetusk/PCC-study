"""PCC2 と G-PCC を同じ土俵で測る — 符号化のみ、外部から時間とピーク RSS。

計測は runpeak の補助プロセスに任せる（親のメモリが子の RSS に乗るため）。
PCC2 は --no-verify で符号化だけを走らせる。G-PCC の 1.8 秒も符号化だけなので、
検証・決定性の二度書きを含めると比較にならない。
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from runpeak import run

PCC = os.path.abspath("cpp/build/pccnorm")
TMC3 = os.environ.get("TMC3", os.path.expanduser("~/tools/tmc13/build/tmc3/tmc3"))
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = (os.path.expanduser("~/tools/laszip-install/lib") + ":"
                          + ENV.get("LD_LIBRARY_PATH", ""))
GFLAGS = ["--trisoupNodeSizeLog2=0", "--mergeDuplicatedPoints=0",
          "--inferredDirectCodingMode=1", "--neighbourAvailBoundaryLog2=8",
          "--intra_pred_max_node_size_log2=6", "--planarEnabled=1",
          "--maxNumQtBtBeforeOt=4", "--minQtbtSizeLog2=0"]


def write_ply(path: Path, xyz: np.ndarray) -> None:
    a = (xyz - xyz.min(0)).astype(np.float32)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(a)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n".encode())
        f.write(np.ascontiguousarray(a).tobytes())


def bench(path: str, n: int, forces=("幾何v3", "走査v1")) -> dict:
    tmp = Path(tempfile.mkdtemp())
    total = M.total_points("las", path)
    start = max(0, total // 2 - n // 2) if total > n else 0
    xyz, g, sid, sc, of, pts, hdr = M.read_block("las", path, start, n)
    las = tmp / "in.laz"
    M.write_las(las, xyz, sc, of, pts, hdr)
    ply = tmp / "in.ply"
    write_ply(ply, xyz)
    res = {}
    print(f"  {Path(path).stem} {len(xyz)} 点")
    r = run([TMC3, "--mode=0", f"--uncompressedDataPath={ply}",
             f"--compressedStreamPath={tmp / 'g.bin'}"] + GFLAGS, ENV)
    gb = (tmp / "g.bin").stat().st_size * 8.0 / len(xyz) if (tmp / "g.bin").exists() else float("nan")
    res["G-PCC"] = (r["sec"], r["peak_mb"], gb)
    print(f"    {'G-PCC 符号化':<18} {r['sec']:6.2f}s  peak {r['peak_mb']:6.0f} MB  {gb:7.3f} bpp")
    for f in forces:
        r = run([PCC, "pack", str(las), str(tmp / "o.pcc2"), "--force-geom", f,
                 "--fast-attr", "--no-fallback", "--no-verify"], ENV)
        bpp = float("nan")
        for ln in r["out"].splitlines():
            if ln.startswith("  X+Y+Z") and ln.split()[1] == f:
                bpp = float(ln.split()[2])
        res[f] = (r["sec"], r["peak_mb"], bpp)
        print(f"    {'PCC2 ' + f:<18} {r['sec']:6.2f}s  peak {r['peak_mb']:6.0f} MB  {bpp:7.3f} bpp"
              + ("" if r["rc"] == 0 else f"  rc={r['rc']} {r['out'][-120:]}"))
    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()
    return res


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ahn4/31HZ1_20.LAZ"
    for n in (int(x) for x in (sys.argv[2:] or ["1000000"])):
        bench(src, n)
