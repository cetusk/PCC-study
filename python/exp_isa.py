"""命令セットの違う 2 つの二値で、器と復号結果が同じになるかを確かめる。

    python -u python/exp_isa.py <二値 A> <二値 B>

別の機械（別の命令セット）で符号化・復号しても同じ値になるかを、既定の二値
（PCC_MARCH=x86-64、SSE2 だけ）と native（この機械の AVX-512 まで）で建てた二値で比べて確かめる。
  1) 同じ入力を A と B で pack し、器の md5 を比べる
  2) A の器を A と B で unpack し、出力の md5 を比べる（相互の復号）。PLY は書き出しの口が
     無いので、器の一致に加えて両方の二値で検証つきの pack を通す
入力は 15 件の中央 200 万点（LAS）、KITTI 20 frame（可逆・--eps 0.005・--eps 0.1）、PLY 4 件と
TLS の PLY の --eps 0.01、Armadillo の --eps 0.001。KITTI と TLS の非可逆は極座標格子（復号が
sin / cos を通る）、Armadillo はデカルト格子が選ばれる（lattice_kind_v21.log）。
"""
import hashlib, os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exp_order_matrix as M
from bench_size import INPUTS

if len(sys.argv) < 3:
    sys.exit(__doc__)
A, B = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
N = int(os.environ.get("BENCH_N", "2000000"))
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "-"
bad = 0; tot = 0


def pcc(exe, *a):
    return subprocess.run([exe, *a], capture_output=True, text=True)


def case(label, src, pack_extra, out_flag, tmp):
    global bad, tot
    pa, pb = tmp / "a.pcc2", tmp / "b.pcc2"
    for p in (pa, pb):
        if p.exists(): p.unlink()
    ra = pcc(A, "pack", str(src), str(pa), "--no-fallback", "--no-verify", *pack_extra)
    rb = pcc(B, "pack", str(src), str(pb), "--no-fallback", "--no-verify", *pack_extra)
    same_pack = ra.returncode == 0 and rb.returncode == 0 and md5(pa) == md5(pb)
    if out_flag is None:
        # PLY には書き出しの口が無い。器が同じバイト列なら「B で A の器を復号する」は
        # 「B が自分の器を検証する」と同じなので、両方の二値で検証つきの pack を回す。
        va = pcc(A, "pack", str(src), str(pa), "--no-fallback", *pack_extra)
        vb = pcc(B, "pack", str(src), str(pb), "--no-fallback", *pack_extra)
        okv = lambda o: "全列一致 = true" in o or "誤差上限の中 = true" in o   # 可逆 / 非可逆の表示
        same_dec = va.returncode == 0 and vb.returncode == 0 and okv(va.stdout) and okv(vb.stdout)
        tot += 1; bad += not (same_pack and same_dec)
        print(f"{label:<34} 器 {'一致' if same_pack else '違う'}  両方で自己検証 {'通過' if same_dec else '落ちた'}",
              flush=True)
        return
    ext = ".las" if out_flag == "--las" else ".bin"
    oa, ob = tmp / ("ua" + ext), tmp / ("ub" + ext)
    ua = pcc(A, "unpack", str(pa), out_flag, str(oa))
    ub = pcc(B, "unpack", str(pa), out_flag, str(ob))
    same_dec = ua.returncode == 0 and ub.returncode == 0 and md5(oa) == md5(ob)
    tot += 1; bad += not (same_pack and same_dec)
    print(f"{label:<34} 器 {'一致' if same_pack else '違う'}  相互の復号 {'一致' if same_dec else '違う'}"
          + ("" if same_pack and same_dec else f"  rc={ra.returncode},{rb.returncode},{ua.returncode},{ub.returncode}"),
          flush=True)


tmp = Path(tempfile.mkdtemp())
print(f"A = {A}\nB = {B}")
for lab, path, kind in [i for i in INPUTS if i[2] == "las"]:
    t = M.total_points(kind, path); n = min(N, t); st = max(0, t // 2 - n // 2)
    xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
    src = tmp / "s.laz"; M.write_las(src, xyz, sc, of, pts, hdr)
    case(f"{lab} {n}", src, [], "--las", tmp)
frames = sorted(Path("data/raw/kitti").rglob("*.bin"))[:20]
for k in frames:
    case(f"KITTI {k.stem} 可逆", k, [], "--bin", tmp)
    case(f"KITTI {k.stem} eps 5mm", k, ["--eps", "0.005"], "--bin", tmp)
    # KITTI の非可逆はどの誤差上限でも極座標格子（5 mm と 0.1 m で 2 通り試す）
    case(f"KITTI {k.stem} eps 0.1m", k, ["--eps", "0.1"], "--bin", tmp)
for f in ["data/raw/stanford/bunny/data/bun000.ply", "data/raw/stanford/dragon_stand/dragonStandRight_0.ply",
          "data/raw/stanford/Armadillo.ply", "data/work/tls_scan1.ply"]:
    case(Path(f).name, Path(f), [], None, tmp)
case("tls_scan1.ply eps 1cm 極座標", Path("data/work/tls_scan1.ply"), ["--eps", "0.01"], None, tmp)
case("Armadillo.ply eps 1mm デカルト", Path("data/raw/stanford/Armadillo.ply"), ["--eps", "0.001"], None, tmp)
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n一致しなかったもの {bad} / {tot} 件")
sys.exit(1 if bad else 0)
