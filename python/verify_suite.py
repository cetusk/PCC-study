"""検証一式を順に回す（**測定と同時に走らせないこと**。見せかけの往復失敗が出る）。

    python -u python/verify_suite.py [tag]      # ログは data/work/grid/<項目>_<tag>.log

項目（どれも「落ちたもの 0 件」で通過）
  regress   作り物で不具合修理を 1 つずつ当てる（regress_fixes.py）
  size      15 件の中央 200 万点: bpp・時間・ピーク、全列一致・LAS 照合・決定性
  threads   PCC_THREADS を変えても同じ器が出るか（exp_threads.py、20 万点）
  cover     unpack --las で戻らない次元が無いか（bench_cover.py、20 万点）
  unpack    unpack --las を元のファイルと欄ごと・生バイトで比べる（先頭 30 万点）
  ply       PLY 4 件の往復
  kitti     KITTI 108 frame の .bin → PCC2 → .bin のバイト比較
  force     幾何の基底候補を 1 本ずつ --force-geom で往復（5 ファイル）
  lossy     KITTI 20 frame の非可逆（誤差上限 3 通り）で上限を守るか

以前はこのうち後半 4 つがその場のコマンドで、手順がリポジトリに残っていなかった。
"""
from __future__ import annotations
import os, re, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
PY = sys.executable
PCC = os.path.abspath("cpp/build/pccnorm")
TAG = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%m%d")
OUT = Path("data/work/grid")
ONLY = set(os.environ.get("SUITE_ONLY", "").split(",")) - {""}


def pcc(*a, env=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run([PCC, *a], capture_output=True, text=True, env=e)


def section(name, fn):
    if ONLY and name not in ONLY:
        return
    t0 = time.time()
    log = OUT / f"{name}_{TAG}.log"
    lines = []
    def say(s=""):
        lines.append(s); print(s, flush=True)
    say(f"# {time.strftime('%F %T')} {name}")
    bad = fn(say)
    say(f"\n落ちたもの {bad} 件（{time.time() - t0:.0f} 秒）")
    log.write_text("\n".join(lines) + "\n")
    SUMMARY.append((name, bad))


def run_script(say, args, env=None):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([PY, "-u", *args], capture_output=True, text=True, env=e)
    for l in r.stdout.splitlines():
        say(l)
    if r.returncode:
        say(r.stderr[-800:])
    return r


def t_regress(say):
    r = run_script(say, ["python/regress_fixes.py"])
    return sum(1 for l in r.stdout.splitlines() if l.startswith("NG")) or int(r.returncode != 0)


def t_size(say):
    import exp_order_matrix as M
    from runpeak import run
    from bench_pcc2 import ENV
    from bench_size import INPUTS
    N = int(os.environ.get("BENCH_N", "2000000"))
    say(f"標本 中央 {N} 点。全列。1 回。")
    say(f"{'データ':<13}{'点':>8}{'LASzip':>9}{'PCC2':>9}{'差':>8}{'enc':>7}{'dec':>7}{'MB':>6}  全列 LAS 決定性")
    bad = 0; d = []
    for lab, path, kind in [i for i in INPUTS if i[2] == "las"]:
        tmp = Path(tempfile.mkdtemp())
        try:
            tot = M.total_points(kind, path); n = min(N, tot); st = max(0, tot // 2 - n // 2)
            xyz, g, sid, sc, of, pts, hdr = M.read_block(kind, path, st, n)
            src = tmp / "a.laz"; M.write_las(src, xyz, sc, of, pts, hdr)
            r = run([PCC, "pack", str(src), str(tmp / "o.pcc2"), "--no-fallback"], ENV)
            o = r["out"]
            g_ = lambda pat: (re.search(pat, o, re.M) or [None, None])[1]
            laz, pc = g_(r"基準 LASzip\s+\S+ MB\s+([0-9.]+) bpp"), g_(r"^PCC2\s+\S+ MB\s+([0-9.]+) bpp")
            eq, las, det = g_(r"全列一致 = (\w+)"), g_(r"LAS に書き戻して元と一致 = (\w+)"), g_(r"決定性 (\S+)")
            enc, dec = g_(r"enc ([0-9.]+)s"), g_(r"dec ([0-9.]+)s")
            ok = r["rc"] == 0 and eq == "true" and las == "true" and det == "バイト一致"
            bad += not ok
            f = lambda v: float(v) if v else float("nan")
            d.append((f(pc) - f(laz)) / f(laz) * 100)
            say(f"{lab:<13}{n:>8}{f(laz):>9.3f}{f(pc):>9.3f}{d[-1]:>+7.1f}%{f(enc):>7.2f}{f(dec):>7.2f}"
                f"{r['peak_mb']:>6.0f}  {eq} {las} {det}" + ("" if ok else "  ← " + o[-300:].replace("\n", " ")))
        finally:
            for q in tmp.glob("*"): q.unlink()
            tmp.rmdir()
    d = np.array(d)
    say(f"\n  PCC2 が小さい {int((d < 0).sum())}/{len(d)}  中央値 {np.median(d):+.1f}%"
        f"  四分位 [{np.percentile(d, 25):+.1f}, {np.percentile(d, 75):+.1f}]  幅 {d.min():+.1f}〜{d.max():+.1f}")
    return bad


def t_threads(say):
    r = run_script(say, ["python/exp_threads.py"], {"BENCH_N": "200000"})
    m = re.search(r"動いたもの (\d+) 件", r.stdout)
    return int(m.group(1)) if m and r.returncode == 0 else 1


def t_cover(say):
    r = run_script(say, ["python/bench_cover.py"], {"BENCH_N": "200000"})
    # 「違う」列が 0 でない行・戻らなかった次元が「—」でない行を数える
    bad = 0
    for l in r.stdout.splitlines():
        m = re.match(r"^(\S.*?)\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)→(\d+)\s+(.*)$", l)
        if m and (m.group(4) != "0" or m.group(5) != m.group(6) or m.group(7).strip() != "—"):
            bad += 1
    return bad + int(r.returncode != 0)


def t_unpack(say):
    r = run_script(say, ["python/verify_unpack.py"], {"VERIFY_MAXP": "300000"})
    m = re.search(r"一致したもの (\d+) 件 / (\d+) 件", r.stdout)
    return int(m.group(2)) - int(m.group(1)) if m and r.returncode == 0 else 1


def t_ply(say):
    bad = 0
    for f in ["data/raw/stanford/bunny/data/bun000.ply", "data/raw/stanford/dragon_stand/dragonStandRight_0.ply",
              "data/raw/stanford/Armadillo.ply", "data/work/tls_scan1.ply"]:
        with tempfile.TemporaryDirectory() as t:
            r = pcc("pack", f, f"{t}/p.pcc2", "--no-fallback")
        ok = r.returncode == 0 and "全列一致 = true" in r.stdout
        bpp = (re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+)", r.stdout, re.M) or [None, "?"])[1]
        say(f"{Path(f).name:<32} {bpp:>9} bpp  {'全列一致' if ok else '落ちた ' + r.stderr[-200:]}")
        bad += not ok
    return bad


def t_kitti(say):
    bad = n = 0
    with tempfile.TemporaryDirectory() as t:
        for k in sorted(Path("data/raw/kitti").rglob("*.bin")):
            n += 1
            pcc("pack", str(k), f"{t}/k.pcc2", "--no-fallback", "--no-verify")
            pcc("unpack", f"{t}/k.pcc2", "--bin", f"{t}/k.bin")
            same = Path(f"{t}/k.bin").exists() and Path(f"{t}/k.bin").read_bytes() == k.read_bytes()
            if not same:
                bad += 1; say(f"不一致: {k}")
            Path(f"{t}/k.bin").unlink(missing_ok=True)
    say(f"{n} frame を比べた")
    return bad


def t_force(say):
    with tempfile.TemporaryDirectory() as t:
        r = pcc("pack", "data/raw/small/vegetation_1_3.las", f"{t}/g.pcc2", "--trace", "--no-fallback",
                env={"PCC_PRESELECT": "0"})
    names, on = set(), False
    for l in r.stdout.splitlines():
        if re.match(r"^  X\+Y\+Z", l): on = True; continue
        if on and re.match(r"^      \S", l): names.add(re.sub(r"(記|生|束|符[124]|速)+$", "", l.split()[0]))
        elif on and re.match(r"^  \S", l): break
    say(f"基底候補 {len(names)} 本")
    bad = tot = 0
    # 基底の名前だけを強制する（固定した候補には旗を重ねないので、旗つきの版はここでは
    # 通らない。旗つきの版は regress_fixes.py の「旗つきの強制」で名前を指定して通す）
    for f in ["data/raw/small/autzen_trim.laz", "data/raw/small/vegetation_1_3.las",
              "data/raw/small/fullwave.laz", "data/work/tls_scan1.ply",
              "data/raw/usgs/NY_ClintonEssex_2014.laz"]:
        for g in sorted(names):
            tot += 1
            with tempfile.TemporaryDirectory() as t:
                r = pcc("pack", f, f"{t}/g.pcc2", "--force-geom", g, "--no-fallback", "--max-points", "200000")
            if "全列一致 = true" not in r.stdout:
                bad += 1; say(f"往復失敗: {Path(f).name} {g}")
    say(f"{tot} 件を通した")
    return bad


def t_lossy(say):
    frames = sorted(Path("data/raw/kitti").rglob("*.bin"))[:20]
    say(f"KITTI {len(frames)} frame。誤差上限ごとに pack（自己検証）→ unpack --bin で元と比べる。")
    bad = 0
    for eps in (0.0, 0.002, 0.005, 0.02):
        bpps, errs = [], []
        for k in frames:
            with tempfile.TemporaryDirectory() as t:
                a = ["pack", str(k), f"{t}/k.pcc2", "--no-fallback"] + (["--eps", str(eps)] if eps else [])
                r = pcc(*a)
                ok = r.returncode == 0 and re.search(r"= true", r.stdout)
                u = pcc("unpack", f"{t}/k.pcc2", "--bin", f"{t}/k.bin")
                if ok and u.returncode == 0:
                    x = np.fromfile(k, "<f4").reshape(-1, 4); y = np.fromfile(f"{t}/k.bin", "<f4").reshape(-1, 4)
                    e = float(np.sqrt(((x[:, :3].astype(float) - y[:, :3]) ** 2).sum(1)).max()) if len(x) == len(y) else 1e9
                    same_attr = np.array_equal(x[:, 3].view("<u4"), y[:, 3].view("<u4"))
                    ok = same_attr and (e <= eps * (1 + 1e-9) if eps else e == 0 and x.tobytes() == y.tobytes())
                    errs.append(e)
                if not ok:
                    bad += 1; say(f"  落ちた: {k.name} eps={eps}")
                bpps.append(float((re.search(r"^PCC2\s+\S+ MB\s+([0-9.]+)", r.stdout, re.M) or [0, "nan"])[1]))
        say(f"誤差上限 {eps:<6} 中央値 {np.median(bpps):7.3f} bpp  [{min(bpps):.3f}, {max(bpps):.3f}]"
            f"  実測最大誤差 {max(errs) if errs else float('nan'):.4g} m")
    return bad


SUMMARY: list = []
section("regress", t_regress)
section("size", t_size)
section("threads", t_threads)
section("cover", t_cover)
section("unpack", t_unpack)
section("ply", t_ply)
section("kitti", t_kitti)
section("force", t_force)
section("lossy", t_lossy)
print("\n== まとめ ==")
for n, b in SUMMARY:
    print(f"  {n:<8} 落ちたもの {b} 件")
print("全部通過" if all(b == 0 for _, b in SUMMARY) else "落ちた項目がある")
