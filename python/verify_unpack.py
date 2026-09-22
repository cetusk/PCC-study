"""pack → unpack --las で戻したファイルを、元のファイルと欄ごとに比べる。

pack の「全列一致」は PCC2 が列として持ったものしか見ず、`bench_cover.py` は
次元と VLR の本数までしか見ない。ここでは**元のファイルそのもの**と比べる:
  ・点の全次元（配列ごと一致）
  ・VLR のバイト列（user_id / record_id / 中身）
  ・EVLR のバイト列
  ・ヘッダの欄（global_encoding / file_source_id / system_identifier /
    generating_software / 作成日 / 点形式 / 版 / scale / offset）
比べる相手は元のファイル全体（標本ではない）。大きいファイルは時間が掛かる。
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
from bench_size import INPUTS

PCC = "./cpp/build/pccnorm"
MAXP = os.environ.get("VERIFY_MAXP")          # 指定すれば先頭 N 点だけ


def raw_vlrs(path):
    """LAS/LAZ のバイト列から VLR と EVLR を直接読む（laspy に頼らない）。

    laspy は COPC の VLR（copc/1, copc/1000）の中身を出せず例外を投げるので、
    それを空として比べると**確かめていないのに一致**と出てしまう。
    返すのは [(user_id, record_id, bytes)] の 2 つ（VLR, EVLR）。
    """
    import struct
    with open(path, "rb") as f:
        h = f.read(375)
        hsize = struct.unpack_from("<H", h, 94)[0]
        nvlr = struct.unpack_from("<I", h, 100)[0]
        vmin = h[25]
        f.seek(hsize)
        out = []
        for _ in range(nvlr):
            vh = f.read(54)
            uid = vh[2:18].split(b"\0", 1)[0].decode("latin-1")
            rid, rlen = struct.unpack_from("<HH", vh, 18)
            out.append((uid, rid, f.read(rlen)))
        ev = []
        if vmin >= 4 and len(h) >= 375:
            start, nev = struct.unpack_from("<QI", h, 235)
            if start and nev:
                f.seek(start)
                for _ in range(nev):
                    eh = f.read(60)
                    uid = eh[2:18].split(b"\0", 1)[0].decode("latin-1")
                    rid = struct.unpack_from("<H", eh, 18)[0]
                    rlen = struct.unpack_from("<Q", eh, 20)[0]
                    ev.append((uid, rid, f.read(rlen)))
    # LASzip 自身の VLR は書くときに作り直されるので比べない
    out = [v for v in out if v[0] != "laszip encoded"]
    return out, ev


def raw_points(path):
    """非圧縮の LAS から点レコードのバイト列を直接読む。圧縮なら None。

    laspy は ExtraBytes の名前が標準の次元と重なると開けない（numpy の型で
    名前が重複する）。点レコードをバイトで比べれば名前に依らず確かめられる。
    """
    import struct
    with open(path, "rb") as f:
        h = f.read(375)
        off = struct.unpack_from("<I", h, 96)[0]
        pf = h[104]
        rl = struct.unpack_from("<H", h, 105)[0]
        n = struct.unpack_from("<I", h, 107)[0]
        if h[25] >= 4:
            n = struct.unpack_from("<Q", h, 247)[0] or n
        if pf & 0xC0:
            return None                     # 圧縮（LAZ）
        f.seek(off)
        return rl, n, f.read(rl * n)


def vlrs(v):
    out = []
    for r in v:
        try:
            body = bytes(r.record_data_bytes())
        except Exception:
            body = getattr(r, "record_data", b"")
            body = bytes(body) if body is not None else b""
        out.append((r.user_id, r.record_id, body))
    return out


def run(lab, path):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "a.pcc2"); back = os.path.join(d, "b.las")   # 非圧縮で書き戻す
        cmd = [PCC, "pack", path, out, "--no-fallback", "--no-verify"]
        if MAXP: cmd += ["--max-points", MAXP]      # 元ファイルの先頭 N 点（格納順）
        subprocess.run(cmd, capture_output=True, text=True)
        r2 = subprocess.run([PCC, "unpack", out, "--las", back], capture_output=True, text=True)
        if not os.path.exists(back):
            return "書き出せない: " + (r2.stdout + r2.stderr)[-200:]
        bad = []
        # 元が非圧縮なら、点レコードをバイト列で比べる（名前に依らない一番強い確かめ）
        ra, rb = raw_points(path), raw_points(back)
        if ra is not None and rb is not None:
            nb_ = rb[1]
            if ra[0] != rb[0]:
                bad.append("点レコード長 %d != %d" % (ra[0], rb[0]))
            elif ra[2][: ra[0] * nb_] != rb[2]:
                bad.append("点レコードのバイト列")
        try:
            a, b = laspy.read(path), laspy.read(back)
        except Exception as e:
            # laspy が開けない（名前の重複など）。バイト比較と VLR の比較だけで判断する
            va_, ea_ = raw_vlrs(path); vb_, eb_ = raw_vlrs(back)
            if va_ != vb_: bad.append("VLR")
            if ea_ != eb_: bad.append("EVLR")
            if ra is None: bad.append("laspy が開けず、バイトでも比べられない（%s）" % type(e).__name__)
            return "一致（バイト比較）" if not bad else "違う: " + ", ".join(bad)
        n = len(b.points)
        ha, hb = a.header, b.header
        # 点数と範囲（min/max）は先頭 N 点に切り出したので違って当然。比べない。
        for k in ["global_encoding", "file_source_id", "system_identifier",
                  "generating_software", "creation_date", "version"]:
            va, vb = getattr(ha, k, None), getattr(hb, k, None)
            va = getattr(va, "value", va); vb = getattr(vb, "value", vb)
            if str(va) != str(vb): bad.append("ヘッダ %s" % k)
        if ha.point_format.id != hb.point_format.id: bad.append("点形式")
        if not np.allclose(ha.scales, hb.scales) or not np.allclose(ha.offsets, hb.offsets):
            bad.append("scale/offset")
        want = n if MAXP else len(a.points)
        if n != want: bad.append("点数 %d != %d" % (n, want))
        else:
            for dim in a.point_format.dimension_names:
                try:
                    if not np.array_equal(np.asarray(a[dim])[:n], np.asarray(b[dim])):
                        bad.append("次元 %s" % dim)
                except Exception:
                    # 名前で引けない次元（空の名前など）。バイト比較で見ているので、
                    # バイト比較ができないときだけ落とす。
                    if ra is None: bad.append("次元 %r は名前で比べられない" % dim)
        va_, ea_ = raw_vlrs(path)
        vb_, eb_ = raw_vlrs(back)
        for nm, xa, xb in (("VLR", va_, vb_), ("EVLR", ea_, eb_)):
            if xa != xb:
                diff = [x[0] + "/" + str(x[1]) for x, y in zip(xa, xb) if x != y]
                if len(xa) != len(xb): diff.append("本数 %d != %d" % (len(xa), len(xb)))
                bad.append("%s（%s）" % (nm, ", ".join(diff)))
        return "一致" if not bad else "違う: " + ", ".join(bad)


if len(sys.argv) > 1:
    # 単体で確かめる: python verify_unpack.py <file.las|laz> ...
    bad = 0
    for path in sys.argv[1:]:
        r = run(os.path.basename(path), path)
        bad += (not r.startswith("一致"))
        print("%-28s %s" % (os.path.basename(path), r))
    sys.exit(1 if bad else 0)

LAS = [i for i in INPUTS if i[2] == "las"]
print("pack → unpack --las を元のファイルと欄ごとに比べる" +
      ("（先頭 %s 点）" % MAXP if MAXP else "（全点）"))
ok = 0
for lab, path, kind in LAS:
    r = run(lab, path)
    ok += r.startswith("一致")
    print("%-14s %s" % (lab, r))
print("\n  全欄が一致したもの %d 件 / %d 件" % (ok, len(LAS)))
