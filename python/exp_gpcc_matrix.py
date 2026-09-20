"""10 ファイル全部で G-PCC と PCC2 の幾何を比べる。

PCC2 は回さない。`scripts/run_scan_batch.sh` が残したログから読む。
新たに走らせるのは TMC13 と、幾何のみの LAZ（production の基準）だけである。

土俵と会計:
  * 幾何のみの比較。G-PCC は点の順序を保存せず、PCC2 と LASzip は保存する。
    この非対称は片側に効くものではない（PCC2 の予測子は格納順を利用する）ので、
    「差は下限」とは言わない。順序の寄与の分離は別途行う。
  * 走査モデルは gps_time / point_source_id / bit_fields が先に復号されている
    ことを前提とするので、その 3 列の符号長を走査側に計上する。
  * G-PCC はヘッダ込みの全ビットストリーム、PCC2 は幾何ストリームのみ。
    差は 0.01 bpp 未満だが対称ではない。

使い方:
    $PCCPY python/exp_gpcc_matrix.py [出力先]
"""
from __future__ import annotations
import os
import re
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import tmc13_bits           # noqa: E402

# 名前 | 入力 | バッチのログ名
JOBS = [
    ("autzen_trim",  "data/raw/small/autzen_trim.laz",                      "small_autzen_trim"),
    ("plane",        "data/raw/small/plane.laz",                            "small_plane"),
    ("fullwave",     "data/raw/small/fullwave.laz",                         "small_fullwave"),
    ("vegetation",   "data/raw/small/vegetation_1_3.las",                   "small_vegetation"),
    ("workshop",     "data/raw/extrabytes/workshop_TM_551_101.laz",         "workshop_TerraScan"),
    ("autzen-2023",  "data/raw/extrabytes/autzen_2023_autzen-2023.copc.laz","autzen2023_LasMonkey"),
    ("AHN5 _20",     "data/raw/ahn5/31HZ1_20.LAZ",                          "ahn5_31HZ1_20"),
    ("AHN3 _20",     "data/raw/ahn3/31HZ1_20.LAZ",                          "ahn3_31HZ1_20"),
    ("AHN4 _20",     "data/raw/ahn4/31HZ1_20.LAZ",                          "ahn4_tile"),
    ("AHN4 _21",     "data/raw/ahn4/31HZ1_21.LAZ",                          "ahn4_tile21"),
]
PRE = ("gps_time", "point_source_id", "bit_fields")
BATCH = Path("data/work/batch")


def read_xyz(path: str) -> np.ndarray:
    """XYZ の整数だけを塊ごとに読む。属性を持つと 5000 万点で数十 GB になる。"""
    out = []
    with laspy.open(path) as fh:
        for chunk in fh.chunk_iterator(4_000_000):
            out.append(np.stack([np.asarray(chunk.X), np.asarray(chunk.Y),
                                 np.asarray(chunk.Z)], 1).astype(np.int64))
    return np.concatenate(out) if len(out) > 1 else out[0]


def laz_geom_bits(xyz: np.ndarray, scale, offset) -> tuple[float, bool]:
    """幾何のみの LAZ。production の基準。順序は保存される。"""
    n = len(xyz)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.laz"
        hdr = laspy.LasHeader(version="1.4", point_format=6)
        hdr.scales, hdr.offsets = scale, offset
        las = laspy.LasData(hdr)
        las.X, las.Y, las.Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.write(str(p))
        b = p.stat().st_size
        with laspy.open(str(p)) as fh:
            back = fh.read()
        ok = bool(np.array_equal(np.asarray(back.X), xyz[:, 0]) and
                  np.array_equal(np.asarray(back.Y), xyz[:, 1]) and
                  np.array_equal(np.asarray(back.Z), xyz[:, 2]))
    return b * 8.0 / n, ok


def from_batch(log: str) -> dict:
    """バッチのログから幾何候補と副列の bpp を読む。"""
    f = BATCH / (log + ".log")
    if not f.is_file():
        return {}
    cand, pre, sel, n = {}, {}, None, None
    for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^      (\S+)\s+([\d.]+) bpp", ln)
        if m:
            cand[m.group(1)] = float(m.group(2))
        m = re.match(r"^  (\S+)\s+\S+\s+([\d.]+) bpp", ln)
        if m and m.group(1) in PRE:
            pre[m.group(1)] = float(m.group(2))
        m = re.match(r"^  X\+Y\+Z\s+(\S+)\s", ln)
        if m:
            sel = m.group(1)
        m = re.match(r"^\s+(\d+) 点 / 出所", ln)
        if m:
            n = int(m.group(1))
    return {"cand": cand, "pre": sum(pre.values()), "sel": sel, "n": n}


def main() -> None:
    out = open(sys.argv[1], "w", encoding="utf-8") if len(sys.argv) > 1 else sys.stdout

    def say(s=""):
        print(s, file=out, flush=True)

    say("G-PCC（TMC13）と PCC2 の幾何の比較 — 10 ファイル")
    say("PCC2 の値は scripts/run_scan_batch.sh のログから読んだもので、再実行していない。")
    say()
    hdr = (f"{'データ':<13}{'点数':>11}{'G-PCC':>9}{'G-PCC 2設定':>11}{'LAZ幾何':>9}"
           f"{'幾何v3':>8}{'走査+副列':>10}{'PCC2最良':>9}{'対G-PCC':>9}")
    say(hdr)
    say("-" * len(hdr))

    for name, path, log in JOBS:
        if not Path(path).is_file():
            say(f"{name:<13}  入力が無い: {path}")
            continue
        b = from_batch(log)
        if not b or not b["cand"]:
            say(f"{name:<13}  バッチのログが無い: {log}")
            continue
        try:
            xyz = read_xyz(path)
        except Exception as e:
            say(f"{name:<13}  読み込み失敗: {e}")
            continue
        n = len(xyz)
        # ログの点数と実際に読めた点数を照合する。同じ入力に対する別構成のログ
        # （標本選択など）を掴むと、別の点数の bpp が黙って並ぶ。
        if b["n"] is not None and b["n"] != n:
            say(f"{name:<13}{n:>11}  ログ {log} の点数 {b['n']} と一致しない。除外")
            del xyz
            continue
        a = xyz - xyz.min(0)
        if a.max() >= 2 ** 24:
            say(f"{name:<13}{n:>11}  座標が float32 の整数域を超える（最大 {a.max()}）。除外")
            del xyz, a
            continue
        del a

        g1 = tmc13_bits(xyz)
        g2 = tmc13_bits(xyz, extra=["--inferredDirectCodingMode=2"])
        gb1 = g1.bytes * 8.0 / n if g1.bytes else float("nan")
        gb2 = g2.bytes * 8.0 / n if g2.bytes else float("nan")
        gbest = np.nanmin([gb1, gb2])
        if not (g1.lossless and (g2.lossless or not g2.bytes)):
            say(f"{name:<13}  G-PCC が可逆にならなかった。除外")
            del xyz
            continue

        with laspy.open(path) as fh:
            hd = fh.header
            sc, of = list(hd.scales), list(hd.offsets)
        lz, lzok = laz_geom_bits(xyz, sc, of)
        del xyz

        c = b["cand"]
        g3 = c.get("幾何v3", float("nan"))
        scan = min([c[k] for k in ("走査v1", "走査変換") if k in c], default=float("nan"))
        scan_tot = scan + b["pre"] if scan == scan else float("nan")
        pbest = np.nanmin([g3, scan_tot])
        rel = 100 * (pbest / gbest - 1) if gbest == gbest else float("nan")
        say(f"{name:<13}{n:>11}{gb1:>9.3f}{gbest:>10.3f}{lz:>9.3f}"
            f"{g3:>8.3f}{scan_tot:>10.3f}{pbest:>9.3f}{rel:>8.1f}%"
            + ("" if lzok else "  ※LAZ 照合失敗"))

    say()
    say("G-PCC      調整済み設定（data/work/gpcc_config.txt の「比較に使う設定」）")
    say("G-PCC 2設定  上と IDCM=2 の短い方。全設定の掃引ではない")
    say("LAZ幾何    幾何のみの LAZ。順序を保存する production の基準")
    say("走査+副列  走査v1 と走査変換の短い方に gps_time/psid/bit_fields を加えたもの")
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
