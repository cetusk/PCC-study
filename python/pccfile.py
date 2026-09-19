"""正規化済み点群のための最小コンテナ。

目的は「副情報を含めた本当のバイト数」を誤魔化さずに測ること。
仕様（復元に必要な全情報）もストリームの一部として必ず書き込む。

  magic 'PCC1' | u32 仕様長 | 仕様(zlib圧縮JSON) | ストリーム群
  ストリーム: u16 名前長 | 名前 | u8 符号化方式 | u32 長さ | データ
"""
from __future__ import annotations
import json, struct, zlib
import numpy as np
import zstandard as zstd
from rangecoder import encode_ints, decode_ints

MAGIC = b"PCC1"
M_ZSTD_DELTA = 0      # 差分 -> zigzag -> バイト平面分離 -> zstd
M_RANGE = 1           # 適応レンジコーダ（決定論的・整数のみ）


def _byte_split(a: np.ndarray) -> bytes:
    b = np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
    return b.T.tobytes()


def _byte_join(buf: bytes, n: int, itemsize: int, dtype) -> np.ndarray:
    b = np.frombuffer(buf, dtype=np.uint8).reshape(itemsize, n).T
    return np.ascontiguousarray(b).view(dtype).ravel()


def _enc_zstd_delta(v: np.ndarray) -> bytes:
    d = np.diff(v.astype(np.int64), prepend=np.int64(0))
    z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
    return zstd.ZstdCompressor(level=19).compress(_byte_split(z))


def _dec_zstd_delta(buf: bytes, n: int) -> np.ndarray:
    raw = zstd.ZstdDecompressor().decompress(buf, max_output_size=n * 8 + 1024)
    z = _byte_join(raw, n, 8, np.uint64).astype(np.int64)
    d = (z >> 1) ^ -(z & 1)
    return np.cumsum(d)


def encode_stream(v: np.ndarray, method: int | None = None) -> tuple[int, bytes]:
    """method=None なら両方試して小さい方を選ぶ。"""
    if method == M_ZSTD_DELTA:
        return M_ZSTD_DELTA, _enc_zstd_delta(v)
    if method == M_RANGE:
        return M_RANGE, encode_ints(np.diff(v.astype(np.int64), prepend=np.int64(0)))
    a = _enc_zstd_delta(v)
    b = encode_ints(np.diff(v.astype(np.int64), prepend=np.int64(0)))
    return (M_ZSTD_DELTA, a) if len(a) <= len(b) else (M_RANGE, b)


def decode_stream(method: int, buf: bytes, n: int) -> np.ndarray:
    if method == M_ZSTD_DELTA:
        return _dec_zstd_delta(buf, n)
    d = decode_ints(buf, n)
    return np.cumsum(d)


def write(path, spec: dict, streams: dict[str, np.ndarray],
          method: int | None = None) -> int:
    sp = zlib.compress(json.dumps(spec, separators=(",", ":"), sort_keys=True).encode(), 9)
    out = bytearray(MAGIC)
    out += struct.pack("<I", len(sp)) + sp
    for name, v in streams.items():
        m, blob = encode_stream(np.asarray(v), method)
        nb = name.encode()
        out += struct.pack("<H", len(nb)) + nb + struct.pack("<BI", m, len(blob)) + blob
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


def read(path, n: int) -> tuple[dict, dict[str, np.ndarray]]:
    buf = open(path, "rb").read()
    assert buf[:4] == MAGIC, "not a PCC1 file"
    p = 4
    (sl,) = struct.unpack_from("<I", buf, p); p += 4
    spec = json.loads(zlib.decompress(buf[p:p + sl]).decode()); p += sl
    streams = {}
    while p < len(buf):
        (nl,) = struct.unpack_from("<H", buf, p); p += 2
        name = buf[p:p + nl].decode(); p += nl
        m, bl = struct.unpack_from("<BI", buf, p); p += 5
        streams[name] = decode_stream(m, buf[p:p + bl], n); p += bl
    return spec, streams
