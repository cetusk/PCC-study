"""表現の正規化レイヤー。

実運用データのビットの多くは「モデル化が足りない」のではなく
「表現が冗長」であることに使われている。この層は、点群を
**センサが本来持っている量と分解能**に戻してから既存コーデックに渡す。

設計方針:
  1. 自動検出   フィールド名をハードコードしない。関係はデータから見つける。
  2. 完全検証   見つけた関係は全点で厳密に検証してからでないと採用しない。
  3. 小さな仕様 復元に必要な情報は数百バイトの仕様に収め、ヘッダに載せる。
  4. 決定論的   外部符号化は整数のみのレンジコーダ（浮動小数を使わない）。

各操作は exact（ビット完全に戻せる）か bounded（誤差上限つき）かを必ず申告する。
"""
from __future__ import annotations
import json, zlib
import numpy as np
from dataclasses import dataclass, asdict, field


# ============================================================ 仕様（Plan / Op）

@dataclass
class Op:
    kind: str                 # drop_constant | drop_duplicate | drop_affine | residual_code
    target: str               # 正規化で容器から外すフィールド
    params: dict = field(default_factory=dict)
    exact: bool = True        # ビット完全に戻せるか
    saved_note: str = ""

    def describe(self) -> str:
        p = self.params
        if self.kind == "drop_constant":
            return f"{self.target:<14} 定数 {p['value']} → 送らない"
        if self.kind == "drop_duplicate":
            return f"{self.target:<14} {p['source']} と完全に同一 → 送らない"
        if self.kind == "drop_affine":
            return f"{self.target:<14} = {p['a']}*{p['source']} + {p['b']} → 送らない"
        if self.kind == "residual_code":
            return f"{self.target:<14} {p['source']} との残差を外部符号化"
        return f"{self.target}: {self.kind}"


@dataclass
class Plan:
    ops: list[Op] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def dropped(self) -> set[str]:
        return {o.target for o in self.ops}

    def spec_bytes(self) -> bytes:
        """復元に必要な副情報。これもビット勘定に入れる。"""
        j = json.dumps({"ops": [asdict(o) for o in self.ops], "meta": self.meta},
                       separators=(",", ":"), sort_keys=True).encode()
        return zlib.compress(j, 9)

    @staticmethod
    def from_spec(b: bytes) -> "Plan":
        d = json.loads(zlib.decompress(b).decode())
        return Plan(ops=[Op(**o) for o in d["ops"]], meta=d.get("meta", {}))

    def report(self) -> str:
        if not self.ops:
            return "（正規化できる冗長は見つからなかった）"
        L = [f"検出した冗長 {len(self.ops)} 件   仕様サイズ {len(self.spec_bytes())} バイト"]
        for o in self.ops:
            mark = "可逆" if o.exact else "誤差有"
            L.append(f"  [{mark}] {o.describe()}")
        return "\n".join(L)


# ==================================================================== 検出器

def _as_int(a: np.ndarray) -> np.ndarray | None:
    if a.dtype.kind in "iub":
        return a.astype(np.int64)
    return None


def detect_constant(name: str, a: np.ndarray) -> Op | None:
    if a.size == 0:
        return None
    first = a.flat[0]
    if bool(np.all(a == first)):
        v = first.item()
        return Op("drop_constant", name, {"value": v, "dtype": str(a.dtype)},
                  exact=True)
    return None


def detect_duplicate(name: str, a: np.ndarray, others: dict[str, np.ndarray]) -> Op | None:
    for src, b in others.items():
        if src == name or b.shape != a.shape:
            continue
        if a.dtype.kind in "iub" and b.dtype.kind in "iub" and np.array_equal(a, b):
            return Op("drop_duplicate", name, {"source": src, "dtype": str(a.dtype)},
                      exact=True)
    return None


def detect_affine(name: str, a: np.ndarray, others: dict[str, np.ndarray],
                  sample: int = 50_000) -> Op | None:
    """a == A*src + B を整数で厳密に満たす src を探す。"""
    ai = _as_int(a)
    if ai is None:
        return None
    n = len(ai)
    idx = np.arange(0, n, max(1, n // sample))
    for src, b in others.items():
        if src == name or b.shape != a.shape:
            continue
        bi = _as_int(b)
        if bi is None:
            continue
        x, y = bi[idx], ai[idx]
        if x.max() == x.min():
            continue
        # 標本から傾きの候補を作り、単純な有理数に丸めてから全点検証
        A0 = float(np.polyfit(x, y, 1)[0])
        for den in (1, 2, 4, 5, 8, 10, 100):
            num = round(A0 * den)
            if num == 0 or den == 0:
                continue
            if abs(num / den - A0) > 1e-6:
                continue
            # B は整数でなければならない
            rem = ai * den - num * bi
            if rem.min() != rem.max():
                continue
            B, r = divmod(int(rem[0]), den)
            if r != 0:
                continue
            if np.array_equal(ai, (num * bi + B * den) // den):
                return Op("drop_affine", name,
                          {"source": src, "a": num / den, "b": B,
                           "num": num, "den": den, "dtype": str(a.dtype)}, exact=True)
    return None


def detect_residual(name: str, a: np.ndarray, others: dict[str, np.ndarray],
                    min_gain_bits: float = 1.0) -> Op | None:
    """他フィールドとの単純差分でエントロピーが十分下がるなら残差符号化に回す。"""
    ai = _as_int(a)
    if ai is None:
        return None
    h0 = _entropy(ai)
    best = None
    for src, b in others.items():
        if src == name or b.shape != a.shape:
            continue
        bi = _as_int(b)
        if bi is None or bi.max() == bi.min():
            continue
        h = _entropy(ai - bi)
        if best is None or h < best[1]:
            best = (src, h)
    if best and h0 - best[1] >= min_gain_bits:
        return Op("residual_code", name,
                  {"source": best[0], "dtype": str(a.dtype)}, exact=True,
                  saved_note=f"H {h0:.2f} -> {best[1]:.2f} bit")
    return None


def _entropy(a: np.ndarray) -> float:
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


# ==================================================================== 解析

# 位置決定に関わる、勝手に落としてはいけないフィールド
PROTECTED = {"X", "Y", "Z", "bit_fields", "classification_flags"}

# 属性フィールドに対する操作の種類（幾何側の操作と区別する）
ATTR_KINDS = {"drop_constant", "drop_duplicate", "drop_affine",
              "residual_code", "palette", "float_grid"}


def analyze(raw: dict[str, np.ndarray], enable_residual: bool = True) -> Plan:
    """生の格納値の辞書から、適用できる正規化を洗い出す。

    循環参照を作らないための規則:
      ・重複／アフィン関係の参照元は、ファイル中でその前に現れるフィールドに限る。
        LAS では ExtraBytes が後ろに来るので、自然と「派生側を落とす」になる。
      ・残差符号化の参照元は、落とされないフィールドに限る。
    """
    plan = Plan()
    names = [n for n in raw if n not in PROTECTED]
    pos = {n: i for i, n in enumerate(names)}
    dropped: set[str] = set()

    # 1) 定数
    for n in names:
        op = detect_constant(n, raw[n])
        if op:
            plan.ops.append(op); dropped.add(n)

    # 2) 完全重複 / アフィン関係（参照元は「自分より前」かつ落とさないもの）
    for n in names:
        if n in dropped:
            continue
        others = {k: raw[k] for k in names
                  if k not in dropped and pos[k] < pos[n]}
        op = detect_duplicate(n, raw[n], others) or detect_affine(n, raw[n], others)
        if op:
            plan.ops.append(op); dropped.add(n)

    # 3) 残差符号化（参照元は容器に残るフィールドに限る）
    if enable_residual:
        residual_targets: set[str] = set()
        for n in names:
            if n in dropped:
                continue
            others = {k: raw[k] for k in names
                      if k not in dropped and k not in residual_targets and k != n}
            op = detect_residual(n, raw[n], others)
            if op:
                plan.ops.append(op); residual_targets.add(n)

    return plan


# ============================================================ 適用 / 逆適用

def apply(raw: dict[str, np.ndarray], plan: Plan) -> tuple[dict[str, np.ndarray],
                                                           dict[str, np.ndarray]]:
    """容器に残すフィールドと、外部符号化に回す残差を返す。"""
    keep = {k: v for k, v in raw.items()}
    external: dict[str, np.ndarray] = {}
    for o in plan.ops:
        if o.kind in ("drop_constant", "drop_duplicate", "drop_affine"):
            keep.pop(o.target, None)
        elif o.kind == "residual_code":
            src = raw[o.params["source"]].astype(np.int64)
            external[o.target] = raw[o.target].astype(np.int64) - src
            keep.pop(o.target, None)
    return keep, external


def invert(keep: dict[str, np.ndarray], external: dict[str, np.ndarray],
           plan: Plan, n: int) -> dict[str, np.ndarray]:
    """正規化を逆回しして元のフィールド一式を復元する。

    参照元がまだ復元できていない操作は後回しにし、解けるものから順に適用する
    （依存グラフの反復解決）。循環していれば例外を出す。
    """
    out = dict(keep)
    pending = list(plan.ops)
    while pending:
        progressed = False
        still = []
        for o in pending:
            if o.kind not in ATTR_KINDS:      # 幾何側の操作はここでは扱わない
                progressed = True
                continue
            p, dt = o.params, np.dtype(o.params["dtype"])
            src_name = p.get("source")
            if src_name is not None and src_name not in out:
                still.append(o); continue
            if o.kind == "drop_constant":
                out[o.target] = np.full(n, p["value"], dtype=dt)
            elif o.kind == "drop_duplicate":
                out[o.target] = out[src_name].astype(dt)
            elif o.kind == "drop_affine":
                src = out[src_name].astype(np.int64)
                out[o.target] = ((p["num"] * src + p["b"] * p["den"]) // p["den"]).astype(dt)
            elif o.kind == "residual_code":
                src = out[src_name].astype(np.int64)
                out[o.target] = (external[o.target] + src).astype(dt)
            progressed = True
        if not progressed:
            raise RuntimeError(f"正規化仕様の依存が解決できない: "
                               f"{[o.target for o in still]}")
        pending = still
    return out


# ============================================================ 幾何の正規化

def detect_spinning_lidar(xyz: np.ndarray, min_monotonic: float = 0.95) -> dict:
    """回転式 LiDAR の1スイープかどうかを、取得順の方位角の単調性で判定する。

    走査順が方位角に沿って並んでいれば、点は原点まわりの極座標格子から
    来ている可能性が高い。デカルト float で保存されていても元は極座標である。
    """
    p = xyz.astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    if len(p) < 100 or r.max() <= 0:
        return {"is_spinning": False}
    az = np.arctan2(p[:, 1], p[:, 0])
    d = np.diff(az)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    frac = float((d > 0).mean())
    med = float(np.median(np.abs(d)))
    return {"is_spinning": frac >= min_monotonic and med > 0,
            "monotonic_frac": frac,
            "median_az_step_deg": float(np.degrees(med)),
            "origin_at_zero": bool(r.min() < 0.5 * np.median(r))}


def polar_forward(xyz: np.ndarray, r_step: float, ang_step: float):
    p = xyz.astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    az = np.arctan2(p[:, 1], p[:, 0])
    el = np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-12), -1.0, 1.0))
    return (np.rint(r / r_step).astype(np.int64),
            np.rint(az / ang_step).astype(np.int64),
            np.rint(el / ang_step).astype(np.int64))


def polar_inverse(qr, qa, qe, r_step: float, ang_step: float) -> np.ndarray:
    r = qr * r_step
    az = qa * ang_step
    el = qe * ang_step
    c = np.cos(el)
    return np.stack([r * c * np.cos(az), r * c * np.sin(az), r * np.sin(el)], 1)


def make_polar_op(xyz: np.ndarray, r_step: float, ang_step_deg: float) -> Op:
    """極座標復元の操作を作り、実際の再構成誤差を測って仕様に刻む。

    誤差上限は推定ではなく全点での実測値を書く。後から「実は超えていた」が
    起きないようにするため。
    """
    ang = np.radians(ang_step_deg)
    qr, qa, qe = polar_forward(xyz, r_step, ang)
    rec = polar_inverse(qr, qa, qe, r_step, ang)
    e = np.linalg.norm(rec - xyz.astype(np.float64), axis=1)
    return Op("polar_restore", "xyz",
              {"r_step": r_step, "ang_step_deg": ang_step_deg,
               "max_err_m": float(e.max()), "rms_err_m": float(e.std())},
              exact=False,
              saved_note=f"最大誤差 {e.max()*1000:.2f}mm / RMS {e.std()*1000:.2f}mm")


def detect_float_grid(name: str, a: np.ndarray, max_levels: int = 65536) -> Op | None:
    """float 配列が「小さな整数 × 一定刻み」でしかない場合に整数に戻す。

    復元後がビット単位で元の float と一致することを全点で確認したときだけ
    exact=True で採用する。一致しなければ採用しない（黙って劣化させない）。
    """
    if a.dtype.kind != "f":
        return None
    u = np.unique(a)
    if u.size < 2 or u.size > max_levels:
        return None
    d = np.diff(u)
    step = float(np.min(d))
    if step <= 0:
        return None
    k = np.rint(a.astype(np.float64) / step)
    if np.abs(a.astype(np.float64) / step - k).max() > 1e-4:
        return None
    ki = k.astype(np.int64)
    if ki.max() - ki.min() >= max_levels:
        return None
    rec = (ki * np.float64(step)).astype(a.dtype)
    if not np.array_equal(rec.view(np.uint32 if a.itemsize == 4 else np.uint64),
                          a.view(np.uint32 if a.itemsize == 4 else np.uint64)):
        return None
    return Op("float_grid", name,
              {"step": step, "dtype": str(a.dtype),
               "bits_needed": int(max(1, (int(ki.max() - ki.min()) + 1).bit_length()))},
              exact=True,
              saved_note=f"{a.itemsize*8} bit float -> {int(ki.max()-ki.min())+1} 段階")


def float_grid_forward(a: np.ndarray, step: float) -> np.ndarray:
    return np.rint(a.astype(np.float64) / step).astype(np.int64)


def float_grid_inverse(k: np.ndarray, step: float, dtype) -> np.ndarray:
    return (k.astype(np.float64) * np.float64(step)).astype(dtype)


def detect_palette(name: str, a: np.ndarray, max_levels: int = 4096) -> Op | None:
    """取りうる値が少ないフィールドを「辞書 + 索引」に置き換える。

    float でも必ずビット完全に戻せるのが利点（値をそのまま保持するため）。
    KITTI の intensity は float32 32bit だが実際には 97 通りしかない、
    といったケースを拾う。
    """
    u = np.unique(a)
    if u.size < 2 or u.size > max_levels:
        return None
    stored_bits = a.dtype.itemsize * 8
    idx_bits = max(1, (u.size - 1).bit_length())
    if idx_bits >= stored_bits:
        return None
    return Op("palette", name,
              {"values": [v.item() for v in u], "dtype": str(a.dtype)},
              exact=True,
              saved_note=f"{stored_bits} bit -> {u.size} 通り ({idx_bits} bit)")


def palette_forward(a: np.ndarray, values) -> np.ndarray:
    u = np.asarray(values, dtype=a.dtype)
    return np.searchsorted(u, a).astype(np.int64)


def palette_inverse(idx: np.ndarray, values, dtype) -> np.ndarray:
    return np.asarray(values, dtype=dtype)[idx]


def make_grid_op(scale: np.ndarray, drop_bits: int) -> Op:
    """整数格子座標の下位ビットを落とす（ノイズ床基準への移行）。

    LSB スイープで「その帯域は純ノイズ」と分かったビットを落とすための操作。
    誤差上限は格子から決まるので解析的に書ける（最悪 対角の半分）。
    """
    step = np.asarray(scale, float) * (1 << drop_bits)
    max_err = float(np.linalg.norm(step / 2.0))
    return Op("grid_quantize", "xyz",
              {"drop_bits": int(drop_bits), "step_m": step.tolist(),
               "max_err_m": max_err},
              exact=False,
              saved_note=f"格子 {step[0]*1000:.1f}mm / 最大誤差 {max_err*1000:.2f}mm")


def grid_forward(xyz_int: np.ndarray, drop_bits: int) -> np.ndarray:
    if drop_bits == 0:
        return xyz_int.copy()
    step = 1 << drop_bits
    return (xyz_int + (step >> 1)) >> drop_bits


def grid_inverse_scale(scale: np.ndarray, drop_bits: int) -> np.ndarray:
    return np.asarray(scale, float) * (1 << drop_bits)


# ================================================== 幾何表現の「選択」

"""幾何の表現は判定せず、実際に符号化して選ぶ。

判定を誤ったときの損失（直交格子に対して +25〜174%）が、
当たったときの利得（−13%）より遥かに大きいため、分類に賭けてはいけない。
候補を全部試して短い方を採り、選んだ結果を仕様に書く。復号器は推測しない。
"""

import zstandard as _zstd


def _rate_of(streams, n: int) -> int:
    """候補の符号長（バイト）。比較用なので符号化方式は候補間で揃える。"""
    tot = 0
    for v in streams:
        d = np.diff(np.asarray(v, np.int64), prepend=np.int64(0))
        z = ((d << 1) ^ (d >> 63)).astype(np.uint64)
        b = np.ascontiguousarray(z).view(np.uint8).reshape(len(z), -1)
        tot += len(_zstd.ZstdCompressor(level=12).compress(b.T.tobytes()))
    return tot


def _fit_budget(make, eps: float, m0: float = 1.0, iters: int = 14):
    """刻みを誤差上限いっぱいまで広げる。

    候補どうしを比べるとき、片方が上限を使い切っていないと不当に不利になる
    （レートを余分な精度に使ってしまう）。二分探索で上限直下に合わせる。
    """
    lo, hi = 0.0, m0
    res = make(m0)
    if res["err"] <= eps:                     # まだ余裕があるなら上へ広げる
        lo = m0
        for _ in range(8):
            hi = lo * 2
            r2 = make(hi)
            if r2["err"] > eps:
                break
            lo, res = hi, r2
        else:
            return res
    else:
        hi = m0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r2 = make(mid)
        if r2["err"] <= eps:
            lo, res = mid, r2
        else:
            hi = mid
    return res


def geometry_candidates(w: np.ndarray, eps: float,
                        origin_max_extent: float = 10.0) -> list[dict]:
    """誤差上限 eps を「使い切る」ように調整した候補表現を列挙する。

    極座標の原点候補は「センサがそこにある」と解釈できるものに限る。
    LAS の offset のように任意に選ばれた座標原点が数百 km 先にある場合、
    極座標は実質「滑らかな曲線座標への貼り替え」になり、たまたま数 % 得をする
    ことがある。それは走査幾何とは無関係で、格納時の offset の選び方に依存する
    脆い利得なので、物体の広がりに対して遠すぎる原点は候補から外す。
    """
    out = []
    extent = float(np.linalg.norm(w.max(0) - w.min(0)))

    def mk_grid(m):
        v = m * 2 * eps / np.sqrt(3)
        q = np.rint(w / v).astype(np.int64)
        return dict(kind="grid", params={"step_m": float(v)},
                    streams=[q[:, 0], q[:, 1], q[:, 2]],
                    err=float(np.linalg.norm(q * v - w, axis=1).max()))
    out.append(_fit_budget(mk_grid, eps))

    anchors = [("centroid", w.mean(0))]
    d_origin = float(np.linalg.norm(w.mean(0)))
    if d_origin <= origin_max_extent * extent:
        anchors.insert(0, ("origin", np.zeros(3)))
    for name, o in anchors:
        p0 = w - o
        r99 = float(np.percentile(np.linalg.norm(p0, axis=1), 99))
        if r99 <= 0:
            continue

        def mk_polar(m, _p=p0, _o=o, _r=r99, _n=name):
            dr = m * eps / np.sqrt(3)
            da = m * eps / (np.sqrt(3) * _r)
            qr, qa, qe = polar_forward(_p, dr, da)
            rec = polar_inverse(qr, qa, qe, dr, da) + _o
            return dict(kind="polar",
                        params={"r_step": float(dr), "ang_step_rad": float(da),
                                "origin": _o.tolist(), "anchor": _n},
                        streams=[qr, qa, qe],
                        err=float(np.linalg.norm(rec - w, axis=1).max()))
        out.append(_fit_budget(mk_polar, eps))
    return out


def choose_geometry(w: np.ndarray, eps: float, sample: int = 200_000,
                    allow_polar: bool | None = None,
                    verbose: bool = False) -> tuple[Op, list]:
    """候補を実際に符号化して選ぶ。誤差上限を超える候補は最初に捨てる。

    allow_polar=False を渡すと極座標の候補を最初から作らない（試行時間の節約）。
    取得構造スコアの推奨をここに渡す想定だが、**スコアは候補を減らすだけ**で、
    残った候補のどれを採るかは必ず実測で決める。
    """
    n = len(w)
    idx = slice(0, min(sample, n))          # 取得順の並びを壊さないよう先頭を使う
    cands = geometry_candidates(w, eps)
    if allow_polar is False:
        cands = [c for c in cands if c["kind"] != "polar"]
    scored = []
    for c in cands:
        if c["err"] > eps * 1.02:           # 申告した上限を守れない候補は不採用
            continue
        bytes_ = _rate_of([s[idx] for s in c["streams"]], len(range(*idx.indices(n))))
        scored.append((bytes_, c))
    if not scored:
        raise RuntimeError("誤差上限を満たす幾何表現の候補がない")
    scored.sort(key=lambda x: x[0])
    best_bytes, best = scored[0]
    if verbose:
        for b, c in scored:
            tag = c["kind"] + ("/" + c["params"]["anchor"] if c["kind"] == "polar" else "")
            print(f"    候補 {tag:<16} {b*8/len(range(*idx.indices(n))):7.3f} bpp"
                  f"  誤差 {c['err']*1000:7.3f}mm"
                  f"{'  ← 採用' if c is best else ''}")
    runner = scored[1][0] if len(scored) > 1 else best_bytes
    op = Op("geometry_choice", "xyz",
            {**best["params"], "kind": best["kind"],
             "max_err_m": best["err"], "eps_budget_m": float(eps)},
            exact=False,
            saved_note=f"{best['kind']} を選択（次点比 {best_bytes/runner*100:.0f}%）")
    return op, best["streams"]
