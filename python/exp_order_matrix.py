"""順序感度の本測定。入力順を統制して、符号器ごとの費用を測る。

問い: 可逆符号器の費用は入力順にどれだけ依存するか。その依存は
      符号器の性質か、それとも順序を再構成する鍵の性質か。

符号器 4 つ:
  G-PCC/TMC13   多重集合を復元する。順序不変が期待値（帰無対照）
  LAZ 全列      系列を復元する。順序依存
  PCC2 幾何v3   系列を復元する。格納順の逐次予測
  PCC2 走査v1   系列を復元する。(psid, gps_time) で順序を作り直す

条件 8 つ:
  恒等 / 逆順 / Morton / 局所シャッフル w=100, 1000, 10000 / 完全ランダム 2 種
  逆順は健全性検査（対称な差分符号器なら Δ≈0）。

ブロックは 100 万点。TMC13 の sliceMaxPoints 既定 1,100,000 を下回るので
G-PCC が単一スライスになり、厳密に順序不変な対照になる。

PCC2 の値は --force-geom で候補を固定し、--no-fallback で退避路を切って測る。
退避路に落ちると検証の対象が埋め込んだ器になり、符号器を検証しなくなるため。

使い方:
    $PCCPY python/exp_order_matrix.py <出力先>
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import tmc13_bits           # noqa: E402

PCC = "./cpp/build/pccnorm"
BLOCK = int(os.environ.get("PCC_BLOCK", "1000000"))
# 大きいファイルから取る互いに素なブロックの数。ファイル内の分散を測るため。
# 1 だと Δ がすべて標本数 1 になり、分散推定が付かない。
NBLOCK = int(os.environ.get("PCC_NBLOCK", "3"))
# 動作確認や再開のために、対象のファイルを名前で絞る（カンマ区切り）
ONLY = [x for x in os.environ.get("PCC_ONLY", "").split(",") if x]
# ブロックの一覧だけ出して止める（設計の確認用）
DRY = os.environ.get("PCC_DRY", "") == "1"

# 名前 | 入力 | 開始点（負なら中央付近を自動）
# 4 つ目は種別。las 以外は点ごとの時刻も飛行線番号も持たないので、
# 走査モデルは候補から外す（鍵を合成すると、まさに交絡させたい量を自分で作る）。
FILES = [
    ("red-rocks",   "data/raw/extrabytes/entwine_data_red-rocks.laz", 0, "las"),
    ("simple1_4",   "data/raw/small/simple1_4.las", 0, "las"),
    ("plane",       "data/raw/small/plane.laz", 0, "las"),
    ("vegetation",  "data/raw/small/vegetation_1_3.las", 0, "las"),
    ("fullwave",    "data/raw/small/fullwave.laz", 0, "las"),
    ("autzen_trim", "data/raw/small/autzen_trim.laz", 0, "las"),
    ("workshop",    "data/raw/extrabytes/workshop_TM_551_101.laz", -1, "las"),
    ("autzen-2023", "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz", -1, "las"),
    ("AHN3 _20",    "data/raw/ahn3/31HZ1_20.LAZ", -1, "las"),
    ("AHN4 _20",    "data/raw/ahn4/31HZ1_20.LAZ", -1, "las"),
    ("AHN4 _21",    "data/raw/ahn4/31HZ1_21.LAZ", -1, "las"),
    ("AHN5 _20",    "data/raw/ahn5/31HZ1_20.LAZ", -1, "las"),
    ("USGS AK",     "data/raw/usgs/AK_Kenai_2008_000001.laz", -1, "las"),
    ("USGS NY",     "data/raw/usgs/NY_ClintonEssex_2014.laz", -1, "las"),
    ("KITTI",       "data/raw/kitti", 0, "kitti"),
    ("TLS p1",      "data/raw/tls/lecturehall/lecturehall1.pose1.object1.label.csv", 0, "tls"),
    ("TLS p2",      "data/raw/tls/lecturehall/lecturehall1.pose2.object2.label.csv", 0, "tls"),
]
FORCE = ("幾何v3", "走査v1")
FORCE_NOKEY = ("幾何v3",)          # 鍵を持たない入力では走査モデルを候補から外す
QSCALE = 0.001                      # 実数座標を持つ入力の量子化幅 [m]
GPCC_ONLY = os.environ.get("PCC_GPCC_ONLY", "") == "1"
TMC13_ARGS = [a for a in os.environ.get("PCC_TMC13_ARGS", "").split() if a]


def morton3(a: np.ndarray) -> np.ndarray:
    """21 bit ずつ 3 軸を交互に並べた Morton 符号。

    座標を各軸の範囲で 21 bit に正規化してから詰める。生の値をマスクすると
    範囲が 2^21 を超える軸（例: simple1_4 は X 2.25 億 / Y 9.9 億）で
    上位が黙って捨てられ、空間局所性と無関係な並びになる。
    """
    b = a - a.min(0)
    r = np.maximum(b.max(0), 1)
    a = ((b.astype(np.float64) * ((1 << 21) - 1)) / r).astype(np.int64)

    def spread(v):
        v = v.astype(np.uint64) & np.uint64((1 << 21) - 1)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
        return v
    return spread(a[:, 0]) | (spread(a[:, 1]) << np.uint64(1)) | (spread(a[:, 2]) << np.uint64(2))


def block_shuffle(n: int, w: int, rng) -> np.ndarray:
    """窓幅 w の中だけを置換する。w=n で完全ランダムに一致する。"""
    idx = np.arange(n)
    for a in range(0, n, w):
        b = min(a + w, n)
        idx[a:b] = rng.permutation(idx[a:b])
    return idx


# ---------------------------------------------------------------- 入力
# 種別ごとに「ブロックの総点数」と「ブロックの読み出し」を与える。
# las 以外は点ごとの時刻も飛行線番号も無いので、鍵は作らない（走査モデルを外す）。

def kitti_frames(root: str) -> list[Path]:
    return sorted(Path(root).glob("**/velodyne_points/data/*.bin"))


def total_points(kind: str, path: str) -> int:
    if kind == "las":
        with laspy.open(path) as fh:
            return fh.header.point_count
    if kind == "kitti":
        return sum(f.stat().st_size // 16 for f in kitti_frames(path))
    if kind == "tls":
        # 平均行長からの見積もりは 1M 点ずれ、末尾ブロックが短くなってブロック長が
        # 揃わなくなった。数え上げは 831MB でも 2 秒なので正確に数える。
        cnt = 0
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                cnt += b.count(b"\n")
        return cnt
    raise ValueError(kind)


def read_block(kind: str, path: str, start: int, count: int):
    """(xyz int64, gps or None, sid or None, scales, offsets, laspy の点列 or None)。"""
    if kind == "las":
        with laspy.open(path) as fh:
            hdr = fh.header
            pts = fh.read_points(start + count)
        pts = pts[start:start + count]
        xyz = np.stack([np.asarray(pts["X"]), np.asarray(pts["Y"]),
                        np.asarray(pts["Z"])], 1).astype(np.int64)
        return (xyz, np.asarray(pts["gps_time"]).astype(np.float64),
                np.asarray(pts["point_source_id"]).astype(np.int64),
                list(hdr.scales), list(hdr.offsets), pts, hdr)
    if kind == "kitti":
        # 取得順はフレームの順、フレーム内はセンサの発射順。連結して 1 本の系列にする。
        acc, got, skipped = [], 0, 0
        for f in kitti_frames(path):
            k = f.stat().st_size // 16
            if skipped + k <= start:
                skipped += k
                continue
            a = np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3]
            if skipped < start:
                a = a[start - skipped:]
                skipped = start
            acc.append(a)
            got += len(a)
            if got >= count:
                break
        w = np.concatenate(acc)[:count].astype(np.float64)
    elif kind == "tls":
        w = np.loadtxt(path, delimiter=",", usecols=(0, 1, 2), dtype=np.float64,
                       skiprows=start, max_rows=count)
    else:
        raise ValueError(kind)
    off = np.floor(w.min(0) / QSCALE) * QSCALE
    xyz = np.rint((w - off) / QSCALE).astype(np.int64)
    return xyz, None, None, [QSCALE] * 3, list(off), None, None


def write_las(dst: Path, xyz: np.ndarray, sc, of, pts=None, hdr_src=None) -> None:
    """PCC2 に渡す入力。las 由来なら元の点列と VLR を保つ。"""
    if pts is not None and hdr_src is not None:
        h2 = laspy.LasHeader(version=hdr_src.version, point_format=hdr_src.point_format)
        for v in hdr_src.vlrs:
            tn = type(v).__name__
            # ExtraBytes の記述子は laspy が点形式から自前で書き出すので、
            # ここで複製すると VLR が二重になり追加バイトのオフセットが狂う。
            # COPC の VLR は laspy が書き戻せない（空間索引で中身に無関係）。
            if tn.startswith("ExtraBytes") or tn.startswith("Copc"):
                continue
            try:
                v.record_data_bytes()
            except Exception:
                continue
            h2.vlrs.append(v)
        h2.scales, h2.offsets = hdr_src.scales, hdr_src.offsets
        las = laspy.LasData(h2)
        las.points = pts
    else:
        h2 = laspy.LasHeader(version="1.4", point_format=6)
        h2.scales, h2.offsets = sc, of
        las = laspy.LasData(h2)
        las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(str(dst))


def laz_geom_bpp(xyz: np.ndarray, scale, offset) -> tuple[float, bool, float, float]:
    """幾何のみの LAZ。属性を含めると他の 3 列（幾何のみ）と比較できない。

    返すのは**点データだけ**の bpp（LAS の頭を引く）。相手（G-PCC の bitstream、
    PCC2 のストリーム）が容器の頭を含まないので、揃えないと比較にならない。
    """
    n = len(xyz)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.laz"
        hdr = laspy.LasHeader(version="1.4", point_format=6)
        hdr.scales, hdr.offsets = scale, offset
        las = laspy.LasData(hdr)
        las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        t0 = time.perf_counter(); las.write(str(p)); enc = time.perf_counter() - t0
        # **点データだけを数える。**ファイル全体だと LAS の頭（469 byte）が入り、
        # G-PCC の bitstream や PCC2 のストリームと揃わない。1 万点のファイルでは
        # 0.35 bpp の下駄になり、比較が LAZ に不利な側へ片寄る。
        with laspy.open(str(p)) as fh0:
            hdr_bytes = fh0.header.offset_to_point_data
        b = p.stat().st_size - hdr_bytes
        t0 = time.perf_counter()
        with laspy.open(str(p)) as fh:
            back = fh.read()
        dec = time.perf_counter() - t0
        ok = bool(np.array_equal(np.asarray(back.X), xyz[:, 0]) and
                  np.array_equal(np.asarray(back.Y), xyz[:, 1]) and
                  np.array_equal(np.asarray(back.Z), xyz[:, 2]))
    return b * 8.0 / n, ok, enc, dec


def run_pcc(path: str, force: str) -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser("~/tools/laszip-install/lib") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([PCC, "pack", path, os.path.join(d, "o.pcc2"),
                            "--force-geom", force, "--fast-attr", "--no-fallback"],
                           capture_output=True, text=True, env=env)
    out = {"bpp": float("nan"), "ok": None, "det": None, "avail": True,
           "sel": None, "emb": False, "enc": float("nan"), "dec": float("nan"),
           "peak": float("nan"), "total": float("nan")}
    for ln in r.stdout.splitlines():
        if "候補にならない" in ln:
            out["avail"] = False
        m = re.match(r"^  X\+Y\+Z\s+(\S+)\s+([\d.]+) bpp", ln)
        if m:
            out["sel"] = m.group(1); out["bpp"] = float(m.group(2))
        if "全列一致" in ln:
            out["ok"] = "true" in ln
        if "決定性" in ln:
            out["det"] = "バイト一致" in ln
        if ln.startswith("中身") and "包んだ" in ln:
            out["emb"] = True
        m = re.match(r"^PCC2\s+[\d.]+ MB\s+([\d.]+) bpp", ln)
        if m:
            out["total"] = float(m.group(1))       # 容器込みの全列合計（幾何ストリームとの差が固定費）
        m = re.search(r"enc ([\d.]+)s / dec ([\d.]+)s .*ピーク ([\d.]+) GB", ln)
        if m:
            out["enc"] = float(m.group(1)); out["dec"] = float(m.group(2))
            out["peak"] = float(m.group(3)) * 1024.0
    if out["sel"] != force:
        out["avail"] = False
    return out


def main() -> None:
    dest = open(sys.argv[1], "w", encoding="utf-8") if len(sys.argv) > 1 else sys.stdout

    def say(s=""):
        print(s, file=dest, flush=True)

    say("順序感度の測定 — 入力順を統制して符号器ごとの費用を測る")
    say(f"大きいファイルはブロック {BLOCK} 点、小さいファイルは全点。")
    say("PCC2 の値は --force-geom で候補を固定し、--no-fallback で退避路を切って測る。")
    say("退避路に落ちた行は検証の対象が符号器でないので NG にする。")
    say()
    hdr = (f"{'データ':<13}{'条件':<10}{'点数':>9}{'鍵重複':>7}{'重複点':>7}"
           f"{'G-PCC':>9}{'LAZ幾何':>9}{'幾何v3':>9}{'走査v1':>9}"
           f"{'全列v3':>9}{'全列sc':>9}"
           f"{'Genc':>7}{'Gdec':>7}{'GpkMB':>8}{'Lenc':>7}{'Ldec':>7}"
           f"{'Penc':>7}{'Pdec':>7}{'PpkMB':>8}{'検証':>5}")
    say(hdr)
    say("-" * len(hdr))

    # 途中で落ちたときの再開。出力の 1 列目（13 文字）がブロックの名前なので、
    # そこを見る。ファイル名で照合するとブロック名（"AHN4 _20#0"）と
    # 一致せず、全部やり直しになる。
    # 1 行でも出ていれば完了と見なすと、8 条件のうち 3 条件で落ちたブロックが
    # 再開時に丸ごと飛ばされ、残りは二度と生成されない。条件数で判定する。
    seen = {}
    if len(sys.argv) > 2 and Path(sys.argv[2]).is_file():
        names = {nm for nm, _, _, _ in FILES}
        for ln in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
            lab = ln[:13].strip()
            if lab and (lab in names or lab.split("#")[0] in names):
                seen[lab] = seen.get(lab, 0) + 1

    # ブロックの一覧を先に作る。1 ファイルから互いに素なブロックを等間隔に取り、
    # ファイル内の分散を測れるようにする。取れないファイルは 1 ブロックのまま。
    blocks = []
    for name, path, off, kind in FILES:
        if ONLY and name not in ONLY:
            continue
        if not (Path(path).is_file() or (kind == "kitti" and Path(path).is_dir())):
            say(f"{name:<13}  入力が無い: {path}")
            continue
        total = total_points(kind, path)
        nb = NBLOCK if total >= BLOCK * NBLOCK else 1
        if nb == 1:
            st = [max(0, total // 2 - BLOCK // 2) if off < 0 and total > BLOCK else 0]
        else:
            step = (total - BLOCK) // (nb - 1)
            st = [i * step for i in range(nb)]
        for bi, x in enumerate(st):
            nb_pts = min(BLOCK, total - x)
            # 恒等・逆順・Morton・ランダム×2 に、作れる窓を足したものが期待条件数
            nexp = 5 + sum(1 for w in (100, 1000, 10000) if w * 10 <= nb_pts)
            blocks.append((name if nb == 1 else f"{name}#{bi}", path, x, kind, nexp))
    say(f"ブロック {len(blocks)} 個（{sum(1 for b in blocks if '#' in b[0])} 個は"
        f"大きいファイルから {NBLOCK} 分割）")
    say("Genc/Gdec/GpkMB は G-PCC、Lenc/Ldec は幾何のみの LAZ、"
        "Penc/Pdec/PpkMB は PCC2（las は 2 候補の最大、鍵の無い入力は 1 候補）。")
    say("鍵の無い入力（KITTI/TLS）は走査列が nan。全列は属性が全部定数なので"
        "幾何＋容器の固定費しか含まず、las 行と横断比較できない。")
    say("LAZ は laspy がこのプロセス内で動くのでピークを分離できない。")
    say()

    done = {nm for nm, _, _, _, nexp in blocks if seen.get(nm, 0) >= nexp}
    if seen and len(done) < len(seen):
        say(f"途中で切れたブロックを測り直す: "
            f"{sorted(set(seen) - done)}")
    if DRY:
        for nm, pa, st, kd, nexp in blocks:
            say(f"  {nm:<14} {kd:<6} 開始 {st:>10}  条件 {nexp}  {pa}")
        if dest is not sys.stdout:
            dest.close()
        return

    tmp = Path(tempfile.mkdtemp())
    for name, path, start, kind, nexp in blocks:
        if name in done:
            continue
        xyz, g, sid, sc, of, pts, hdr_src = read_block(kind, path, start, BLOCK)
        n = len(xyz)
        if sid is None:
            # 点ごとの時刻も飛行線番号も無い。鍵が作れないので走査モデルは候補外。
            forces, dup_key = FORCE_NOKEY, float("nan")
        else:
            # 走査モデルが実際に見るのは鍵の重複率である。隣接同値率ではない。
            key = np.stack([sid.astype(np.float64), g], 1)
            dup_key = 100.0 * (1.0 - len(np.unique(key, axis=0)) / n)
            forces = FORCE
        dup_pt = int(n - len(np.unique(xyz, axis=0)))   # 重複点の数（可逆判定の前提）

        rng = np.random.default_rng(20260920)
        conds = [("恒等", np.arange(n)), ("逆順", np.arange(n)[::-1].copy()),
                 ("Morton", np.argsort(morton3(xyz), kind="stable"))]
        for w in (100, 1000, 10000):
            # w が n に近いと完全ランダムと区別できず、単調性を見かけ上強める。
            if w * 10 <= n:
                conds.append((f"窓{w}", block_shuffle(n, w, rng)))
        conds.append(("ランダム1", rng.permutation(n)))
        conds.append(("ランダム2", rng.permutation(n)))

        for cname, idx in conds:
            p = tmp / "x.laz"
            lz, lzok, lz_enc, lz_dec = laz_geom_bpp(xyz[idx], sc, of)
            gp = tmc13_bits(xyz[idx], extra=TMC13_ARGS) if TMC13_ARGS else tmc13_bits(xyz[idx])
            gb = gp.bytes * 8.0 / n if gp.bytes else float("nan")
            if GPCC_ONLY:
                rs = {}
            else:
                write_las(p, xyz[idx], sc, of,
                          pts[idx].copy() if pts is not None else None, hdr_src)
                rs = {f: run_pcc(str(p), f) for f in forces}
                p.unlink()
            # 退避路に落ちた行は、検証の対象が符号器ではないので失格にする
            # is not False だと、復号できず None のままの行が合格になる。
            ver = (gp.lossless is True) and lzok
            if not GPCC_ONLY:
                # avail な候補が 1 つも無いと all() が空集合で真になるので、
                # 「少なくとも 1 つが検証を通った」ことを明示的に要求する。
                av = [r for r in rs.values() if r["avail"]]
                ver = ver and bool(av) and all(
                    r["ok"] and r["det"] and not r["emb"] for r in av)
            nanv = float("nan")
            get = lambda f, k: (rs[f][k] if f in rs and rs[f]["avail"] else nanv)
            vals = [get(f, "bpp") for f in FORCE]
            tots = [get(f, "total") for f in FORCE]
            allr = list(rs.values()) or [{"enc": nanv, "dec": nanv, "peak": nanv}]
            say(f"{name:<13}{cname:<10}{n:>9}{dup_key:>6.1f}%{dup_pt:>7}"
                f"{gb:>9.3f}{lz:>9.3f}{vals[0]:>9.3f}{vals[1]:>9.3f}"
                f"{tots[0]:>9.3f}{tots[1]:>9.3f}"
                f"{gp.enc_s:>7.1f}{gp.dec_s:>7.1f}{gp.peak_mb:>8.0f}"
                f"{lz_enc:>7.1f}{lz_dec:>7.1f}"
                f"{max(r['enc'] for r in allr):>7.1f}"
                f"{max(r['dec'] for r in allr):>7.1f}"
                f"{max(r['peak'] for r in allr):>8.0f}"
                f"{'ok' if ver else 'NG':>5}")
        say()
    if dest is not sys.stdout:
        dest.close()


if __name__ == "__main__":
    main()
