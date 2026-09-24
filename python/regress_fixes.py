"""不具合修理の回帰（小さな作り物で、直した不具合ごとに 1 つ以上）。

2026-09-23 の修理（封筒のヘッダ欄・未記述バイト・壊れた器・KITTI/PLY の端の場合・
失敗時に器を残さない・--force-geom の旗つき名）を 1 つずつ当てる。
作り物の LAS/PLY は cpp/tests/fixtures/ にある（mklas.py の write_las で作った）。
壊れた器は pcc2_craft.py で正しい器を読み、欄を書き換えて CRC を付け直して作る。

    python python/regress_fixes.py        # リポジトリの根で。全項目「OK」なら通過
"""
import hashlib, os, sys, struct, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcc2_craft import parse, build, crc64
R = "cpp/tests/fixtures"
PCC = "./cpp/build/pccnorm"
T = tempfile.mkdtemp()
res = []
def run(*a, env=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run([PCC, *a], capture_output=True, text=True, env=e)
def check(name, cond, info=""):
    res.append((name, bool(cond))); print(("OK  " if cond else "NG  ") + name, info)

# 1. 作り物の LAS: pack が「LAS に書き戻して元と一致 = true」、unpack が生バイト一致
for f in ["l1_undoc", "l2_gap", "l3_dupname", "l5_neg", "l5_pos", "l6_X", "l6_intensity", "las1", "las2"]:
    src = os.path.join(R, f + ".las"); out = os.path.join(T, f + ".pcc2")
    r = run("pack", src, out)
    ok = r.returncode == 0 and "LAS に書き戻して元と一致 = true" in r.stdout and os.path.exists(out)
    check(f"LAS 往復 {f}", ok, "" if ok else (r.stdout[-300:] + r.stderr[-300:]))
# 2. 点 0 個は断る・器を残さない
out = os.path.join(T, "las0.pcc2")
if os.path.exists(out): os.remove(out)
r = run("pack", os.path.join(R, "las0.las"), out)
check("点 0 個を断る", r.returncode != 0 and not os.path.exists(out) and not os.path.exists(out + ".part"), r.stderr.strip()[:120])
# 3. 負の値の粗い格子（A-10）: l5_neg が l5_pos と同程度に縮む
def bpp(log):
    for l in log.splitlines():
        if l.startswith("PCC2"): return float(l.split()[3])
bn = bpp(run("pack", os.path.join(R, "l5_neg.las"), os.path.join(T, "n.pcc2")).stdout)
bp = bpp(run("pack", os.path.join(R, "l5_pos.las"), os.path.join(T, "p.pcc2")).stdout)
check("負の値の粗い格子", bn is not None and bp is not None and bn <= bp * 1.05, f"neg {bn} / pos {bp} bpp")
# 4. KITTI
rng = np.random.default_rng(1)
def kitti(path, arr, tail=b""):
    open(path, "wb").write(arr.astype("<f4").tobytes() + tail)
def roundtrip_bin(name, arr, extra=()):
    p = os.path.join(T, name + ".bin"); kitti(p, arr)
    c = os.path.join(T, name + ".pcc2"); b = os.path.join(T, name + ".out.bin")
    r = run("pack", p, c, *extra)
    if r.returncode: return False, r.stdout[-200:] + r.stderr[-200:]
    r = run("unpack", c, "--bin", b)
    if r.returncode: return False, r.stderr[-200:]
    return True, np.fromfile(b, "<f4").reshape(-1, 4)
ok, got = roundtrip_bin("krand", rng.standard_normal((50, 4)) * 10)       # 格子なし・100 点未満（A-4）
want = np.fromfile(os.path.join(T, "krand.bin"), "<u4").reshape(-1, 4)
check("KITTI 格子なし 50 点の往復", ok and np.array_equal(got.view("<u4"), want), "" if ok else got)
xyz = np.round(np.random.default_rng(2).uniform(-40, 40, (3000, 3)), 3)     # 1 mm 格子
arr = np.concatenate([xyz, np.random.default_rng(3).uniform(0, 1, (3000, 1))], 1)
ok, got = roundtrip_bin("keps", arr, ("--eps", "0.05"))                  # 非可逆の粗い格子（A-2）
if ok:
    orig = np.fromfile(os.path.join(T, "keps.bin"), "<f4").reshape(-1, 4)
    e = np.sqrt(((got[:, :3].astype(float) - orig[:, :3]) ** 2).sum(1)).max()
    check("KITTI --eps の誤差が上限の中", e <= 0.05 * 1.001, f"最大誤差 {e:.4g} m")
else:
    check("KITTI --eps の誤差が上限の中", False, got)
p = os.path.join(T, "ktail.bin"); kitti(p, arr[:100], b"\x01\x02\x03")        # 端数（A-5）
r = run("pack", p, os.path.join(T, "ktail.pcc2"))
check("KITTI の端数を断る", r.returncode != 0 and "16 byte" in r.stderr, r.stderr.strip()[:100])
arr2 = arr.copy(); arr2[5, 0] = np.nan                                     # NaN と --eps（A-3）
p = os.path.join(T, "knan.bin"); kitti(p, arr2)
r = run("pack", p, os.path.join(T, "knan.pcc2"), "--eps", "0.05")
check("NaN と --eps を断る", r.returncode != 0 and "NaN" in r.stderr, r.stderr.strip()[:100])
# 5. PLY の整数座標（A-11）
def ply(path, props, rows, fmt):
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(rows)
    hdr += "".join(f"property {t} {n}\n" for t, n in props) + "end_header\n"
    open(path, "wb").write(hdr.encode() + b"".join(struct.pack(fmt, *r) for r in rows))
rows = [(int(a), int(b), int(c)) for a, b, c in np.random.default_rng(4).integers(-100000, 100000, (300, 3))]
p = os.path.join(T, "pint.ply"); ply(p, [("int", "x"), ("int", "y"), ("int", "z")], rows, "<iii")
r = run("pack", p, os.path.join(T, "pint.pcc2"))
check("PLY int32 座標", r.returncode == 0 and "幾何 int" in r.stdout, r.stdout[:200] + r.stderr[-200:])
rows16 = [(a % 30000, b % 30000, c % 30000) for a, b, c in rows]
p = os.path.join(T, "pshort.ply"); ply(p, [("short", "x"), ("short", "y"), ("short", "z")], rows16, "<hhh")
r = run("pack", p, os.path.join(T, "pshort.pcc2"))
check("PLY int16 座標", r.returncode == 0 and "全列一致 = true" in r.stdout, r.stderr[-200:])
r = run("pack", os.path.join(R, "wide300.ply"), os.path.join(T, "w300.pcc2"))
check("PLY ascii 300 性質", r.returncode == 0 and "全列一致 = true" in r.stdout, r.stderr[-200:])
# 6. 壊れた器（A-8）: 版 2 の器を作り、各所を壊す。落ちずに（シグナル無しで）失敗する
base = os.path.join(T, "evbase.pcc2"); run("pack", "data/raw/small/plane.laz", base, "--max-points", "3000", "--no-fallback")
c = parse(open(base, "rb").read())
def evil(name, data):
    p = os.path.join(T, name + ".pcc2"); open(p, "wb").write(data)
    r = run("unpack", p)
    check(f"壊れた器 {name}", r.returncode == 1, (r.stderr.strip() or f"rc={r.returncode}")[:120])
evil("ns", build(c, ns_override=2**31))
evil("n", build(c, n_override=2**40))
evil("dlen", build(c, dlen_override=lambda i, pos: 2**64 - 3))
bad = dict(c); bad["tags"] = [(t, (b"\x00" + struct.pack("<I", 60) + b'{"grid":0,"ops":[{"kind":"drop_constant","target":"x","value":"--"}]}'[:60]) if t == 6 else body) for t, body in c["tags"]]
evil("json", build(bad))
bad = dict(c); js = b'{"grid":0,"grid_bits":0,"max_err_m":0,"ops":[{"kind":"drop_affine","target":"classification","source":"intensity","value":0,"num":1,"den":0,"b":0}]}'
bad["tags"] = [(t, (b"\x00" + struct.pack("<I", len(js)) + js) if t == 6 else body) for t, body in c["tags"]]
evil("den0", build(bad))
v1 = bytearray(build(c)); v1[4:6] = struct.pack("<H", 1)
v1 = bytes(v1[:-8]) + struct.pack("<Q", crc64(bytes(v1[:-8])))
p = os.path.join(T, "v1.pcc2"); open(p, "wb").write(v1)
r = run("unpack", p)
check("版 1 の器を断る", r.returncode == 1 and "版 1" in r.stderr, r.stderr.strip()[:120])
# 7. 検証に落ちたら器を残さない（A-9）: 復号が既定値を使う実験用の変数で符号化する
out = os.path.join(T, "fail.pcc2")
for q in (out, out + ".part"):
    if os.path.exists(q): os.remove(q)
r = run("pack", "data/raw/small/plane.laz", out, "--max-points", "20000",
        env={"PCC_RAWKEEP": "3", "PCC_ALLOW_EXPERIMENT": "1"})
check("検証に落ちたら器を残さない", r.returncode == 2 and not os.path.exists(out) and not os.path.exists(out + ".part"),
      f"rc={r.returncode}")
# 8. --force-geom に旗つきの名前（A4）
r = run("pack", "data/raw/small/vegetation_1_3.las", os.path.join(T, "fg.pcc2"), "--force-geom", "幾何v4W16記符2")
check("--force-geom 旗つきの名前", r.returncode == 0 and "候補にならない" not in r.stdout and "幾何v4W16記符2 " in r.stdout, r.stdout[-400:])
# 9. ヘッダの点数・波形の欄が食い違う LAS（査読の指摘）。バイト列を直に作る。
def las_raw(path, vmin, fmt, pts, legacy_n=None, by_ret=None, ext_n=None, ext_by_ret=None,
            wave=None, evlr=None, vlrs=b"", nvlr=0):
    hs = {2: 227, 3: 235, 4: 375}[vmin]
    rl = {1: 28, 4: 57, 6: 30}[fmt]
    n = len(pts)
    body = bytearray()
    for i, (x, y, z, rn, nr) in enumerate(pts):
        if fmt in (1, 4):
            body += struct.pack("<iiiHBBbBHd", x, y, z, 100 + i % 7, (rn & 7) | ((nr & 7) << 3), 2, 0, 0, 1, 1000.0 + i)
            if fmt == 4: body += struct.pack("<BQIffff", 1, 0, 0, 0.0, 0.0, 0.0, 0.0)
        else:
            body += struct.pack("<iiiHBBBBhHd", x, y, z, 100 + i % 7, (rn & 15) | ((nr & 15) << 4), 0, 2, 0, 0, 1, 1000.0 + i)
    off = hs + len(vlrs)
    h = bytearray(hs)
    h[0:4] = b"LASF"; h[24] = 1; h[25] = vmin
    h[26:32] = b"MYSYS\0"; h[58:64] = b"MYSOF\0"
    struct.pack_into("<HHH", h, 90, 1, 2024, hs)
    struct.pack_into("<IIBHI", h, 96, off, nvlr, fmt, rl, n if legacy_n is None else legacy_n)
    rns = [p[3] for p in pts]
    br = [sum(1 for r in rns if r == k) for k in range(1, 6)] if by_ret is None else by_ret
    struct.pack_into("<5I", h, 111, *br)
    struct.pack_into("<3d", h, 131, 0.01, 0.01, 0.01)
    xs = [p[0] * 0.01 for p in pts]; ys = [p[1] * 0.01 for p in pts]; zs = [p[2] * 0.01 for p in pts]
    struct.pack_into("<6d", h, 179, max(xs), min(xs), max(ys), min(ys), max(zs), min(zs))
    ev = b""
    if evlr is not None:
        ev = struct.pack("<H16sHQ32s", 0, b"LASF_Spec", 65535, len(evlr), b"waveform") + evlr
    if vmin >= 3:
        struct.pack_into("<Q", h, 227, (off + len(body)) if (evlr is not None and wave is None) else (wave or 0))
    if vmin >= 4:
        if evlr is not None: struct.pack_into("<QI", h, 235, off + len(body), 1)
        struct.pack_into("<Q", h, 247, n if ext_n is None else ext_n)
        eb = [sum(1 for r in rns if r == k) for k in range(1, 16)] if ext_by_ret is None else ext_by_ret
        struct.pack_into("<15Q", h, 255, *eb)
    open(path, "wb").write(bytes(h) + vlrs + bytes(body) + ev)
rng9 = np.random.default_rng(9)
P = [(int(a), int(b), int(c), int(r), 3) for (a, b, c), r in zip(rng9.integers(0, 50000, (400, 3)), rng9.integers(1, 4, 400))]
cases = {
    "14_f1_ext0":     dict(vmin=4, fmt=1, ext_by_ret=[0] * 15),                    # 旧形式だけ埋まっている
    "14_f1_extn0":    dict(vmin=4, fmt=1, ext_n=0),                                # 拡張の点数が 0
    "14_f1_legacy0":  dict(vmin=4, fmt=1, legacy_n=0, by_ret=[0] * 5),             # 旧形式が 0（規格どおり）
    "14_f6_ok":       dict(vmin=4, fmt=6, legacy_n=0, by_ret=[0] * 5),             # 規格どおりの点形式 6
    "14_f6_legacy":   dict(vmin=4, fmt=6),                                         # 点形式 6 で旧形式が埋まっている
    "13_f4_wave":     dict(vmin=3, fmt=4, evlr=bytes(range(256)) * 4),             # 1.3 の内部の波形データ
    "14_f4_wave":     dict(vmin=4, fmt=4, evlr=bytes(range(200))),                 # 1.4 の波形 EVLR
    "12_emptyEB":     dict(vmin=2, fmt=1, vlrs=struct.pack("<H16sHH32s", 0, b"LASF_Spec", 4, 0, b"empty eb"), nvlr=1),
}
for nm, kw in cases.items():
    src = os.path.join(T, nm + ".las"); las_raw(src, pts=P, **kw)
    for ext in (".las", ".laz"):
        c = os.path.join(T, nm + ".pcc2"); o = os.path.join(T, nm + ".out" + ext)
        r = run("pack", src, c, "--no-fallback")
        u = run("unpack", c, "--las", o) if r.returncode == 0 else r
        same = False
        if u.returncode == 0:
            a_, b_ = open(src, "rb").read(), open(o, "rb").read()
            hs_ = struct.unpack_from("<H", a_, 94)[0]
            # LAZ では LASzip 自身の VLR が 1 本増えるので、点データの位置・VLR の本数・
            # 点形式の圧縮ビットが違って当然（.las ではどれも一致を求める）
            skip = {104, 96, 97, 98, 99, 100, 101, 102, 103} if ext == ".laz" else set()
            if a_[25] >= 4: skip |= set(range(235, 247))    # EVLR の位置（点データの長さで変わる）
            if a_[25] >= 3 and kw.get("evlr") is not None: skip |= set(range(227, 235))
            same = all(a_[k] == b_[k] for k in range(hs_) if k not in skip)
        ok = r.returncode == 0 and "LAS に書き戻して元と一致 = true" in r.stdout and same
        check(f"ヘッダ {nm} {ext}", ok, "" if ok else (r.stdout[-200:] + r.stderr[-200:] + u.stderr[-200:]))
# 10. 以前の査読が作った反例 3 件（2026-09-22 の二値でしか確かめていなかった）
import laspy
def lp_case(name, fmt, fill):
    h = laspy.LasHeader(point_format=fmt, version="1.4")
    h.scales = [0.01, 0.01, 0.01]; h.offsets = [0, 0, 0]
    rng10 = np.random.default_rng(10)
    n = 3000
    if fill == "time":
        h.add_extra_dims([laspy.ExtraBytesParams(name="Time", type=np.float64)])
    d = laspy.LasData(h)
    d.X = rng10.integers(0, 100000, n); d.Y = rng10.integers(0, 100000, n); d.Z = rng10.integers(0, 5000, n)
    d.intensity = rng10.integers(0, 4000, n)
    d.return_number = np.ones(n, np.uint8); d.number_of_returns = np.ones(n, np.uint8)
    if fill == "flags":
        d.classification = rng10.integers(0, 10, n)
        d.synthetic = rng10.integers(0, 2, n).astype(bool)
        d.key_point = rng10.integers(0, 2, n).astype(bool)
        d.withheld = rng10.integers(0, 2, n).astype(bool)
        d.overlap = rng10.integers(0, 2, n).astype(bool)
    if fill == "time":
        d["Time"] = 1.7e9 + np.cumsum(rng10.uniform(0, 1e-3, n))   # 大きな値（エポック秒）
        d.gps_time = rng10.uniform(0, 1e6, n)
    if fill == "dupcolor":
        c = rng10.integers(0, 65535, n)
        d.red = c; d.green = c; d.blue = c                          # 3 列が同じ値
    path = os.path.join(T, name + ".las"); d.write(path)
    return path
for name, fmt, fill in [("pf6_flags", 6, "flags"), ("big_time", 6, "time"), ("dup_color", 7, "dupcolor")]:
    src = lp_case(name, fmt, fill)
    c = os.path.join(T, name + ".pcc2"); o = os.path.join(T, name + ".out.las")
    r = run("pack", src, c, "--no-fallback")
    u = run("unpack", c, "--las", o) if r.returncode == 0 else r
    same = False
    if u.returncode == 0:
        a_, b_ = laspy.read(src), laspy.read(o)
        same = all(np.array_equal(np.asarray(a_[d]), np.asarray(b_[d])) for d in a_.point_format.dimension_names)
    ok = r.returncode == 0 and "LAS に書き戻して元と一致 = true" in r.stdout and same
    check(f"反例 {name}", ok, "" if ok else (r.stdout[-200:] + r.stderr[-200:]))
# 11. 旗つきの版を名前で強制して往復させる（2026-09-23 の査読の指摘）。
#     verify_suite.py の force は基底の名前だけを強制し、固定した候補には旗を重ねないので、
#     光・符・速の付いた版は勝ったファイルでしか復号されていなかった。多重戻りのある USGS NY で通す。
for g in ["幾何v0光", "幾何v1光", "向き光", "幾何v0符1", "幾何v1符2", "向き符4",
          "幾何v1符1光", "走査v1符1", "走査変換符2", "幾何v1速", "幾何v4W16記光面8"]:
    c = os.path.join(T, "ff.pcc2")
    r = run("pack", "data/raw/usgs/NY_ClintonEssex_2014.laz", c, "--force-geom", g,
            "--no-fallback", "--max-points", "100000")
    m = [l for l in r.stdout.splitlines() if l.startswith("  X+Y+Z")]
    got = m[0].split()[1] if m else "?"
    ok = r.returncode == 0 and "全列一致 = true" in r.stdout and got == g
    check(f"旗つきの強制 {g}", ok, "" if ok else f"選ばれた {got} / rc={r.returncode} " + r.stderr[-150:])
# 12. combine（幾何を G-PCC に任せる構成）に定数の列がある入力を渡すと落ちていた（2026-09-23）。
#     G-PCC の参照ソフト（環境変数 TMC3）があるときだけ回す。
TMC3 = os.environ.get("TMC3")
if not (TMC3 and os.path.exists(TMC3)):
    print("（combine の項目は飛ばした: 環境変数 TMC3 に G-PCC の参照ソフトが無い）")
if TMC3 and os.path.exists(TMC3):
    rng12 = np.random.default_rng(12)
    n12 = 5000
    h = laspy.LasHeader(point_format=2, version="1.2"); h.scales = [0.01] * 3; h.offsets = [0] * 3
    d = laspy.LasData(h)
    d.X = rng12.integers(0, 20000, n12); d.Y = rng12.integers(0, 20000, n12); d.Z = rng12.integers(0, 2000, n12)
    d.red = rng12.integers(0, 65535, n12); d.green = rng12.integers(0, 65535, n12); d.blue = rng12.integers(0, 65535, n12)
    src = os.path.join(T, "rgb_only.las"); d.write(src)          # 強度・分類などは定数
    from bench_pcc2 import write_ply
    xyz = np.stack([np.asarray(d.X), np.asarray(d.Y), np.asarray(d.Z)], 1)
    write_ply(os.path.join(T, "g.ply"), xyz)
    te = subprocess.run([TMC3, "--mode=0", f"--uncompressedDataPath={T}/g.ply", f"--compressedStreamPath={T}/g.bin",
                         "--trisoupNodeSizeLog2=0", "--mergeDuplicatedPoints=0"], capture_output=True)
    td = subprocess.run([TMC3, "--mode=1", f"--compressedStreamPath={T}/g.bin", f"--reconstructedDataPath={T}/gd.ply"],
                        capture_output=True)
    if te.returncode or td.returncode:
        check("combine に定数の列", False, "G-PCC の参照ソフトが失敗した（combine の不具合ではない）")
    else:
        r = run("combine", src, os.path.join(T, "gd.ply"), os.path.join(T, "g.bin"))
        check("combine に定数の列", r.returncode == 0 and "属性 完全一致=true" in r.stdout, f"rc={r.returncode} " + r.stderr[-150:])
# 13. 決定性の検査が符号器を走らせ直していなかった（2026-09-23 の査読の指摘）。
#     流れを器に書き直して比べるだけで、食い違っても止めていなかった。いまは選ばれた流れを
#     新しい文脈で符号化し直して比べ、食い違えば戻り値 2 で何も残さない。
#     試験用のつまみで流れの 1 byte を反転させ、捕まえることを確かめる。
out13 = os.path.join(T, "det.pcc2")
r = run("pack", "data/raw/small/autzen_trim.laz", out13, "--no-fallback",
        env={"PCC_TEST_DET_FLIP": "1"})
check("決定性の検査が食い違いを捕まえる",
      r.returncode == 2 and "決定性に落ちた" in r.stderr and not os.path.exists(out13)
      and not os.path.exists(out13 + ".part"), f"rc={r.returncode} " + r.stderr.strip()[:150])
r = run("pack", "data/raw/small/autzen_trim.laz", out13, "--no-fallback")
check("決定性の検査が通常は通る", r.returncode == 0 and "流れを符号化し直してバイト一致" in r.stdout,
      f"rc={r.returncode} " + r.stderr.strip()[:150])
# 14. 器の版 4 で、非可逆の極座標を戻す sin / cos を libm から自前の関数（dettrig.hpp）に替えた。
#     版 3 の器は libm で戻さないと、座標（と、それで組む空間予測の順序）が変わりうる。
#     版 3 の二値で作った作り物（回転 LiDAR 風 6000 点、--eps 0.1 で極座標、強度は sp(P=5)）を
#     読み、版 3 の二値の復号と同じバイト列が出ることを確かめる。
fx = "cpp/tests/fixtures/polar_v3.pcc2"
out14 = os.path.join(T, "polar_v3.bin")
r = run("unpack", fx, "--bin", out14)
got = hashlib.md5(open(out14, "rb").read()).hexdigest() if os.path.exists(out14) else "-"
check("版 3 の極座標の器を版 3 と同じに戻す", r.returncode == 0 and got == "4d57c3ea1305cf4cb0abb7c7341721dd",
      f"rc={r.returncode} md5={got} " + r.stderr.strip()[:120])
# 15. 器の版 5（旗「類」を足した）の復号器が、版 4 の器を版 4 の二値と同じに戻す。
#     版 4 の二値（c05db51〜7eb1ee7）で作り物の l1_undoc.las を符号化した器と、その二値の unpack --las の md5。
fx4 = "cpp/tests/fixtures/synth_v4.pcc2"
out15 = os.path.join(T, "s4.las")
r = run("unpack", fx4, "--las", out15)
got = hashlib.md5(open(out15, "rb").read()).hexdigest() if os.path.exists(out15) else "-"
check("版 4 の器を版 4 と同じに戻す", r.returncode == 0 and got == "59ff7eeef04193259a406b8343783b39",
      f"rc={r.returncode} md5={got} " + r.stderr.strip()[:120])
# 16. 旗「類」（既に復号済みの列の類を文脈に足す）が選ばれた流れの往復。fullwave では波形の列などで選ばれる。
#     色の 3 列は鎖（blue → green → red）でスキーマと逆順に依存しうるので、互いを類の相手にしない
#     （しないと復号の依存が循環する。autzen_trim で「ストリームの依存が解けない」になった）。
for f16 in ["data/raw/small/fullwave.laz", "data/raw/small/autzen_trim.laz"]:
    o16 = os.path.join(T, "cls.pcc2")
    r = run("pack", f16, o16, "--no-fallback")
    ok = r.returncode == 0 and "全列一致 = true" in r.stdout and "LAS に書き戻して元と一致 = true" in r.stdout
    n_cls = r.stdout.count("類<")
    check(f"旗「類」の往復 {os.path.basename(f16)}", ok and n_cls > 0, f"類 {n_cls} 本 rc={r.returncode} " + r.stderr.strip()[:120])
print(f"{sum(o for _, o in res)}/{len(res)} 通過")
sys.exit(0 if all(o for _, o in res) else 1)
