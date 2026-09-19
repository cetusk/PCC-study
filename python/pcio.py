"""点群の統一内部表現とローダ。

方針:
  LAS/LAZ の座標は規格上 int32 + scale/offset である（float32 生値ではない）。
  KITTI .bin は本物の float32。両者を区別したまま扱えるようにする。
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PointCloud:
    # 整数格子座標（LAS の X/Y/Z 相当）。None なら float のみ。
    xyz_int: np.ndarray | None = None          # (N,3) int64
    xyz_float: np.ndarray | None = None        # (N,3) float32/float64（原本が float の場合）
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attrs: dict[str, np.ndarray] = field(default_factory=dict)
    # 実際にファイル中に格納されている生の値。ExtraBytes は scale/offset が
    # 適用される前の整数なので、ビットを数えるときは必ずこちらを使う。
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        a = self.xyz_int if self.xyz_int is not None else self.xyz_float
        return len(a)

    def world(self) -> np.ndarray:
        """物理単位の float64 座標。"""
        if self.xyz_int is not None:
            return self.xyz_int * self.scale + self.offset
        return self.xyz_float.astype(np.float64)

    def summary(self) -> str:
        L = [f"N = {self.n:,}  src = {self.meta.get('path','?')}"]
        if self.xyz_int is not None:
            rng = self.xyz_int.max(0) - self.xyz_int.min(0)
            L.append(f"  int grid : scale={self.scale}  span(int)={rng}  "
                     f"needed bits/axis={[int(np.ceil(np.log2(r+1))) for r in rng]}")
        if self.xyz_float is not None:
            L.append(f"  float    : dtype={self.xyz_float.dtype} "
                     f"min={self.xyz_float.min(0)} max={self.xyz_float.max(0)}")
        for k, v in self.attrs.items():
            L.append(f"  attr {k:<18} dtype={str(v.dtype):<8} "
                     f"unique={len(np.unique(v)) if v.size < 5_000_000 else '>'} "
                     f"min={v.min()} max={v.max()}")
        return "\n".join(L)


def load_las(path: str | Path, max_points: int | None = None) -> PointCloud:
    import laspy
    path = Path(path)
    with laspy.open(str(path)) as fh:
        if max_points and fh.header.point_count > max_points:
            # 巨大タイルは先頭 max_points 点だけをチャンク読みする（取得順を保つ）
            las = next(fh.chunk_iterator(max_points))
            las = laspy.LasData(fh.header, points=las)
            max_points = None
        else:
            las = fh.read()
    hdr = las.header
    # laspy: las.X/Y/Z は scale/offset 適用前の生 int
    xyz_int = np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], 1).astype(np.int64)
    attrs = {}
    skip = {"X", "Y", "Z", "x", "y", "z"}
    for dim in las.point_format.dimension_names:
        if dim in skip:
            continue
        try:
            v = np.asarray(las[dim])
        except Exception:
            continue
        if v.ndim != 1 or v.dtype == object:
            continue
        attrs[dim] = v
    raw = {}
    arr = las.points.array           # ScaleAwarePointRecord -> 生の構造化配列
    for nm in arr.dtype.names:
        if nm in ("X", "Y", "Z"):
            continue
        raw[nm] = np.asarray(arr[nm])
    pc = PointCloud(
        raw=raw,
        xyz_int=xyz_int,
        scale=np.asarray(hdr.scales, float),
        offset=np.asarray(hdr.offsets, float),
        attrs=attrs,
        meta=dict(path=str(path), kind="las",
                  file_bytes=path.stat().st_size,
                  point_format=str(las.point_format.id),
                  version=str(hdr.version),
                  # LASzip 実測を元ファイルと同じ土俵で取るためヘッダを保持する
                  # (point format / extra bytes VLR がここに入っている)
                  las_header=hdr),
    )
    if max_points and pc.n > max_points:
        pc = subsample_prefix(pc, max_points)
    return pc


def load_kitti_bin(path: str | Path, max_points: int | None = None) -> PointCloud:
    """KITTI velodyne: float32 x,y,z,intensity のフラットバイナリ。"""
    path = Path(path)
    a = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    pc = PointCloud(
        xyz_float=a[:, :3].copy(),
        attrs={"intensity_f32": a[:, 3].copy()},
        meta=dict(path=str(path), kind="kitti_bin", file_bytes=path.stat().st_size),
    )
    if max_points and pc.n > max_points:
        pc = subsample_prefix(pc, max_points)
    return pc


def load_ply(path: str | Path, max_points: int | None = None) -> PointCloud:
    """PLY の頂点を読む（面は無視）。Stanford スキャンなど float32 座標が主。

    レンジスキャナ出力は confidence / intensity を持つことがあるので、
    頂点プロパティは全部属性として拾っておく。
    """
    from plyfile import PlyData
    path = Path(path)
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = list(v.dtype.names)
    xs = [n for n in ("x", "y", "z") if n in names]
    if len(xs) != 3:
        raise ValueError(f"PLY に x/y/z がない: {names}")
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1)
    attrs = {n: np.asarray(v[n]) for n in names if n not in ("x", "y", "z")}
    pc = PointCloud(
        xyz_float=xyz,
        attrs=attrs,
        raw=dict(attrs),
        meta=dict(path=str(path), kind="ply", file_bytes=path.stat().st_size,
                  ply_format=str(ply.header.splitlines()[1] if ply.header else ""),
                  n_elements={e.name: len(e.data) for e in ply.elements},
                  xyz_dtype=str(xyz.dtype)),
    )
    if max_points and pc.n > max_points:
        pc = subsample_prefix(pc, max_points)
    return pc


def subsample_prefix(pc: PointCloud, k: int) -> PointCloud:
    """先頭 k 点を取る。点の並び順そのものを解析対象にするので、ランダム抽出はしない。"""
    return PointCloud(
        xyz_int=None if pc.xyz_int is None else pc.xyz_int[:k],
        xyz_float=None if pc.xyz_float is None else pc.xyz_float[:k],
        scale=pc.scale, offset=pc.offset,
        attrs={k_: v[:k] for k_, v in pc.attrs.items()},
        raw={k_: v[:k] for k_, v in pc.raw.items()},
        meta={**pc.meta, "subsampled_to": k},
    )


def load(path: str | Path, **kw) -> PointCloud:
    p = Path(path)
    if p.suffix.lower() in (".las", ".laz"):
        return load_las(p, **kw)
    if p.suffix.lower() == ".bin":
        return load_kitti_bin(p, **kw)
    if p.suffix.lower() == ".ply":
        return load_ply(p, **kw)
    raise ValueError(f"unsupported: {p.suffix}")


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        print("=" * 70)
        print(load(f).summary())
