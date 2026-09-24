"""ベースライン符号化器。常に 5 軸 (bytes / enc / dec / peak mem / 決定性) を返す。

bpp だけで評価すると「実は使えない」になる。最初から一緒に測る。
"""
from __future__ import annotations
import io, os, time, lzma, subprocess, tempfile, hashlib, resource
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

TMC3 = os.environ.get("TMC3", os.path.expanduser("~/tools/tmc13/build/tmc3/tmc3"))


@dataclass
class Result:
    name: str
    n_points: int
    bytes: int
    enc_s: float
    dec_s: float
    peak_mb: float
    lossless: bool | None = None
    note: str = ""

    @property
    def bpp(self) -> float:
        return self.bytes * 8.0 / self.n_points

    def row(self) -> str:
        return (f"{self.name:<26} {self.bytes/1e6:9.3f} MB  {self.bpp:8.3f} bpp  "
                f"enc {self.enc_s:7.2f}s  dec {self.dec_s:7.2f}s  peak {self.peak_mb:7.0f}MB  "
                f"{'lossless' if self.lossless else ('LOSSY' if self.lossless is False else '?'):>8}  {self.note}")


def _peak_mb() -> float:
    """この python 自身のピーク。子プロセスの測定には使えない（run_peak を使う）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_peak(cmd, env=None):
    """子プロセスを走らせ、その子だけのピーク常駐メモリ［MB］と所要［秒］を返す。

    `getrusage(RUSAGE_SELF)` は呼び出し側の python を測るので外部符号器の
    ピークにはならない。`RUSAGE_CHILDREN` は終了した全ての子の最大値が
    累積するので 1 回分を取り出せない。fork して `os.wait4` を使うと
    その子の rusage だけが得られる。

    返り値: (終了コード, 標準出力, 標準エラー, 所要秒, ピーク MB)
    """
    import select
    ro, wo = os.pipe()
    re_, we = os.pipe()
    t = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        os.close(ro); os.close(re_)
        os.dup2(wo, 1); os.dup2(we, 2)
        os.close(wo); os.close(we)
        try:
            os.execvpe(cmd[0], cmd, env or os.environ)
        except Exception:
            pass
        os._exit(127)
    os.close(wo); os.close(we)
    out, err, fds = b"", b"", [ro, re_]
    while fds:
        r, _, _ = select.select(fds, [], [])
        for fd in r:
            b = os.read(fd, 65536)
            if not b:
                fds.remove(fd); os.close(fd); continue
            if fd == ro:
                out += b
            else:
                err += b
    _, status, ru = os.wait4(pid, 0)
    dt = time.perf_counter() - t
    return (os.waitstatus_to_exitcode(status),
            out.decode("utf-8", "replace"), err.decode("utf-8", "replace"),
            dt, ru.ru_maxrss / 1024.0)


# ---------------------------------------------------------------- 汎用圧縮器

def _zstd(buf: bytes, level: int = 19) -> tuple[bytes, float, float]:
    import zstandard as zstd
    t = time.perf_counter(); out = zstd.ZstdCompressor(level=level).compress(buf); enc = time.perf_counter() - t
    t = time.perf_counter(); back = zstd.ZstdDecompressor().decompress(out, max_output_size=len(buf)*2+1024); dec = time.perf_counter() - t
    assert back == buf
    return out, enc, dec


def _xz(buf: bytes, preset: int = 6) -> tuple[bytes, float, float]:
    t = time.perf_counter(); out = lzma.compress(buf, preset=preset); enc = time.perf_counter() - t
    t = time.perf_counter(); back = lzma.decompress(out); dec = time.perf_counter() - t
    assert back == buf
    return out, enc, dec


CODECS = {"zstd19": _zstd, "xz6": _xz}


# ---------------------------------------------- 整数座標に対する参照パイプライン

def morton3(xyz: np.ndarray, bits: int = 21) -> np.ndarray:
    """(N,3) 非負 int64 -> Morton code int64。bits<=21。"""
    def spread(v):
        v = v.astype(np.uint64) & np.uint64((1 << bits) - 1)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
        return v
    return (spread(xyz[:, 0]) | (spread(xyz[:, 1]) << np.uint64(1))
            | (spread(xyz[:, 2]) << np.uint64(2)))


def byte_split(a: np.ndarray) -> bytes:
    """列ごとにバイト平面へ分離（同種バイトを隣接させると汎用圧縮が効く）。"""
    b = np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
    return b.T.tobytes()


def ref_geometry_bits(xyz_int: np.ndarray, order: str = "morton",
                      codec: str = "zstd19", delta: bool = True) -> Result:
    """整数座標のみの参照符号化: 並べ替え -> 差分 -> バイト平面分離 -> 汎用圧縮。

    実コーデックではないが「この並び・この差分で汎用圧縮がどこまで効くか」の
    仮定の少ない目安になる。LSB スイープの相対比較に使う。
    """
    n = len(xyz_int)
    t0 = time.perf_counter()
    a = xyz_int - xyz_int.min(0)
    if order == "morton":
        a = a[np.argsort(morton3(a))]
    elif order == "original":
        pass
    else:
        raise ValueError(order)
    if delta:
        a = np.diff(a, axis=0, prepend=a[:1])
    a = a.astype(np.int64)
    # zigzag して符号を潰す
    z = ((a << 1) ^ (a >> 63)).astype(np.uint64)
    buf = byte_split(z)
    prep = time.perf_counter() - t0
    out, enc, dec = CODECS[codec](buf)
    return Result(f"ref/{order}/{codec}{'/delta' if delta else ''}", n, len(out),
                  prep + enc, dec, _peak_mb(), lossless=True)


# ------------------------------------------------------------------- LASzip

def laszip_bits(pc, tmpdir: str | None = None, keep_attrs: bool = True) -> Result:
    """laspy+lazrs で実際に LAZ を書き、production ベースラインを測る。

    keep_attrs=True のときは元ファイルのヘッダ（point format / extra bytes VLR）を
    そのまま使う。そうしないと nir や Amplitude のような拡張次元が黙って落ち、
    「属性ぶん」を過小評価してしまう。
    """
    import laspy
    n = pc.n
    src_hdr = pc.meta.get("las_header")
    dropped = []
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        p = Path(d) / "t.laz"
        if keep_attrs and src_hdr is not None:
            hdr = laspy.LasHeader(version=src_hdr.version,
                                  point_format=src_hdr.point_format)
            for vlr in src_hdr.vlrs:        # ExtraBytes VLR を引き継ぐ
                hdr.vlrs.append(vlr)
        else:
            pf = 6 if not keep_attrs else (7 if "red" in pc.attrs else 6)
            hdr = laspy.LasHeader(version="1.4", point_format=pf)
        hdr.scales = pc.scale
        hdr.offsets = pc.offset
        las = laspy.LasData(hdr)
        las.X, las.Y, las.Z = pc.xyz_int[:, 0], pc.xyz_int[:, 1], pc.xyz_int[:, 2]
        if keep_attrs:
            names = set(las.point_format.dimension_names)
            for k, v in pc.attrs.items():
                if k not in names:
                    dropped.append(k); continue
                try:
                    las[k] = v
                except Exception:
                    dropped.append(k)
        t = time.perf_counter(); las.write(str(p)); enc = time.perf_counter() - t
        b = p.stat().st_size
        t = time.perf_counter()
        with laspy.open(str(p)) as fh:
            back = fh.read()
        dec = time.perf_counter() - t
        ok = bool(np.array_equal(np.asarray(back.X), pc.xyz_int[:, 0])
                  and np.array_equal(np.asarray(back.Y), pc.xyz_int[:, 1])
                  and np.array_equal(np.asarray(back.Z), pc.xyz_int[:, 2]))
    note = f"落ちた属性: {','.join(dropped)}" if dropped else ""
    return Result(f"LASzip{'' if keep_attrs else '/geom-only'}", n, b, enc, dec,
                  _peak_mb(), lossless=ok, note=note)


# -------------------------------------------------------------------- TMC13

def tmc13_bits(xyz_int: np.ndarray, tmpdir: str | None = None,
               extra: list[str] | None = None) -> Result:
    """G-PCC (TMC13) 可逆・幾何のみ。入力は非負整数格子。"""
    n = len(xyz_int)
    a = (xyz_int - xyz_int.min(0)).astype(np.int64)
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        d = Path(d)
        ply, bs, rec = d / "in.ply", d / "o.bin", d / "rec.ply"
        with open(ply, "wb") as f:
            f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "end_header\n".encode())
            f.write(np.ascontiguousarray(a.astype(np.float32)).tobytes())
        base = [TMC3, "--mode=0", f"--uncompressedDataPath={ply}",
                f"--compressedStreamPath={bs}", "--trisoupNodeSizeLog2=0",
                "--mergeDuplicatedPoints=0",
                # --positionQuantizationScale は v23 で非推奨。既定 1 と同じなので外した
                # （実測で出力は 1 バイトも変わらない）。
                "--inferredDirectCodingMode=1", "--neighbourAvailBoundaryLog2=8",
                "--intra_pred_max_node_size_log2=6", "--planarEnabled=1",
                "--maxNumQtBtBeforeOt=4", "--minQtbtSizeLog2=0"] + (extra or [])
        rc, so, se, enc, peak_e = run_peak(base)
        if rc != 0 or not bs.exists():
            return Result("G-PCC/TMC13", n, 0, 0, 0, 0, None,
                          note="ENC FAIL: " + (se or so)[-200:].replace("\n", " "))
        dcmd = [TMC3, "--mode=1", f"--compressedStreamPath={bs}",
                f"--reconstructedDataPath={rec}", "--outputBinaryPly=1"]
        drc, _, _, dec, peak_d = run_peak(dcmd)
        peak = max(peak_e, peak_d)
        b = bs.stat().st_size
        ok = None
        if drc == 0 and rec.exists():
            ok = _ply_matches(rec, a)
    return Result("G-PCC/TMC13 (geom, lossless)", n, b, enc, dec, peak, lossless=ok)


def tmc13_decode_order(xyz_int: np.ndarray, tmpdir: str | None = None,
                       extra: list[str] | None = None) -> np.ndarray | None:
    """G-PCC が復号時に点を出す順序を返す。

    返り値 `r` は入力添字ごとの出力位置（`r[i]` = 入力の i 番目が出力の何番目か）。
    多重集合として一致することは検査済みなので、両方を lexsort して突き合わせる。
    同一座標の点は区別できないが、区別しても置換の費用は変わらない。
    """
    n = len(xyz_int)
    a = (xyz_int - xyz_int.min(0)).astype(np.int64)
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        d = Path(d)
        ply, bs, rec = d / "in.ply", d / "o.bin", d / "rec.ply"
        with open(ply, "wb") as f:
            f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "end_header\n".encode())
            f.write(np.ascontiguousarray(a.astype(np.float32)).tobytes())
        base = [TMC3, "--mode=0", f"--uncompressedDataPath={ply}",
                f"--compressedStreamPath={bs}", "--trisoupNodeSizeLog2=0",
                "--mergeDuplicatedPoints=0",
                "--inferredDirectCodingMode=1", "--neighbourAvailBoundaryLog2=8",
                "--intra_pred_max_node_size_log2=6", "--planarEnabled=1",
                "--maxNumQtBtBeforeOt=4", "--minQtbtSizeLog2=0"] + (extra or [])
        if run_peak(base)[0] != 0 or not bs.exists():
            return None
        dcmd = [TMC3, "--mode=1", f"--compressedStreamPath={bs}",
                f"--reconstructedDataPath={rec}", "--outputBinaryPly=1"]
        if run_peak(dcmd)[0] != 0 or not rec.exists():
            return None
        got = _ply_points(rec)
    if got is None or len(got) != n:
        return None
    si = np.lexsort(a.T)            # 入力を整列させる添字
    sd = np.lexsort(got.T)          # 出力を整列させる添字
    out_of_in = np.empty(n, dtype=np.int64)
    out_of_in[si] = sd              # 整列後に同じ位置に来るもの同士が対応する
    return out_of_in


def _ply_points(path: Path) -> np.ndarray | None:
    """binary PLY の頂点を出力順のまま整数で返す。"""
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.readline()
        cnt = int([l for l in head.split(b"\n")
                   if l.startswith(b"element vertex")][0].split()[-1])
        raw = f.read()
    itemsize = len(raw) // cnt if cnt else 0
    dt = {12: np.float32, 24: np.float64}.get(itemsize)
    if dt is None:
        return None
    v = np.frombuffer(raw, dtype=dt).reshape(-1, 3)
    if not np.all(v == np.rint(v)):
        return None
    return np.rint(v).astype(np.int64)


def _ply_matches(path: Path, ref: np.ndarray) -> bool:
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.readline()
        cnt = int([l for l in head.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        raw = f.read()
    if cnt != len(ref):
        return False
    itemsize = len(raw) // cnt
    dt = {12: np.float32, 24: np.float64}.get(itemsize)
    if dt is None:
        return False
    raw_v = np.frombuffer(raw, dtype=dt).reshape(-1, 3)
    # 無条件に丸めると ±0.5 未満のずれを「可逆」と判定してしまう。
    # 復号値が整数そのものであることを先に検査する。
    if not np.all(raw_v == np.rint(raw_v)):
        return False
    got = np.rint(raw_v).astype(np.int64)
    a = got[np.lexsort(got.T)]
    b = ref[np.lexsort(ref.T)]
    return np.array_equal(a, b)
