"""ベースライン符号化器。常に 5 軸 (bytes / enc / dec / peak mem / 決定性) を返す。

survey 攻め筋⑧: bpp だけで評価すると「実は使えない」になる。最初から一緒に測る。
"""
from __future__ import annotations
import io, os, time, lzma, subprocess, tempfile, hashlib, resource
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

TMC3 = os.environ.get("TMC3", "/home/agent/tools/tmc13/build/tmc3/tmc3")


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
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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
                "--mergeDuplicatedPoints=0", "--positionQuantizationScale=1",
                "--inferredDirectCodingMode=1", "--neighbourAvailBoundaryLog2=8",
                "--intra_pred_max_node_size_log2=6", "--planarEnabled=1",
                "--maxNumQtBtBeforeOt=4", "--minQtbtSizeLog2=0"] + (extra or [])
        t = time.perf_counter(); e = subprocess.run(base, capture_output=True, text=True); enc = time.perf_counter() - t
        if e.returncode != 0 or not bs.exists():
            return Result("G-PCC/TMC13", n, 0, 0, 0, 0, None,
                          note="ENC FAIL: " + (e.stderr or e.stdout)[-200:].replace("\n", " "))
        t = time.perf_counter()
        dcmd = [TMC3, "--mode=1", f"--compressedStreamPath={bs}",
                f"--reconstructedDataPath={rec}", "--outputBinaryPly=1"]
        dr = subprocess.run(dcmd, capture_output=True, text=True); dec = time.perf_counter() - t
        b = bs.stat().st_size
        ok = None
        if dr.returncode == 0 and rec.exists():
            ok = _ply_matches(rec, a)
    return Result("G-PCC/TMC13 (geom, lossless)", n, b, enc, dec, _peak_mb(), lossless=ok)


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
    got = np.rint(np.frombuffer(raw, dtype=dt).reshape(-1, 3)).astype(np.int64)
    a = got[np.lexsort(got.T)]
    b = ref[np.lexsort(ref.T)]
    return np.array_equal(a, b)
