"""波形パケットの 29 byte を、byte の列ではなく元の型の列として見たらどうなるか。

LAS の波形パケット（点形式 4/5/9/10）は
  0      記述子の添字            u8
  1-8    波形データへの位置       u64
  9-12   パケットの大きさ         u32
  13-16  戻り点の波形上の位置     f32
  17-20  X(t)                    f32
  21-24  Y(t)                    f32
  25-28  Z(t)                    f32
という構造を持つ。いまは 29 本の 1 byte 列として運んでいるので、
f32 の下位仮数バイトが単独で 8.000 bpp になっている。
"""
import sys, numpy as np, laspy

def H(a):
    if len(a) == 0: return 0.0
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def H1(a):
    """直前の値を文脈にした条件付きエントロピー（標本内）。"""
    if len(a) < 2: return H(a)
    prev = a[:-1].astype(np.int64); cur = a[1:].astype(np.int64)
    key = prev * (int(cur.max()) + 1) + cur
    _, c = np.unique(key, return_counts=True)
    _, cp = np.unique(prev, return_counts=True)
    n = len(cur)
    return float((c / n * np.log2(c.sum() / c)).sum()) - \
           float((cp / n * np.log2(cp.sum() / cp)).sum()) + \
           float((cp / n * np.log2(cp.sum() / cp)).sum()) * 0  # H(prev,cur)-H(prev)

las = laspy.read(sys.argv[1] if len(sys.argv) > 1 else
                 "data/raw/small/fullwave.laz")
n = len(las.points)
print("点 %d  点形式 %d" % (n, las.header.point_format.id))

fields = [("記述子", np.asarray(las.wavepacket_index), 1),
          ("位置",   np.asarray(las.wavepacket_offset), 8),
          ("大きさ", np.asarray(las.wavepacket_size), 4),
          ("戻り位置", np.asarray(las.return_point_wave_location), 4),
          ("X(t)",  np.asarray(las.x_t), 4),
          ("Y(t)",  np.asarray(las.y_t), 4),
          ("Z(t)",  np.asarray(las.z_t), 4)]

print("\n%-10s %5s %10s %10s %10s %10s" %
      ("欄", "byte", "byte 列計", "値 H0", "値 差分H0", "ビット型差分"))
tot_b = tot_v = 0.0
for name, v, w in fields:
    raw = v.view(np.uint32) if v.dtype == np.float32 else \
          (v.view(np.uint64) if v.dtype == np.float64 else v.astype(np.uint64))
    raw = np.ascontiguousarray(raw)
    b = raw.view(np.uint8).reshape(n, -1)[:, :w]
    hb = sum(H1(b[:, i]) for i in range(w))
    h0 = H(raw)
    d = (raw.astype(np.int64)[1:] - raw.astype(np.int64)[:-1])
    hd = H(d)
    # 値そのものの差分（float は実数として）
    if v.dtype == np.float32:
        fv = v.astype(np.float64)
        fd = np.diff(fv)
        hv = H(np.round(fd, 12))
    else:
        hv = hd
    print("%-10s %5d %10.3f %10.3f %10.3f %10.3f" % (name, w, hb, h0, hv, hd))
    tot_b += hb; tot_v += min(h0, hd)
print("\nbyte の列として %.3f bpp / 値として（H0 と差分の小さい方）%.3f bpp" %
      (tot_b, tot_v))

# f32 が「粗い格子」に乗っていないか（整数化できるか）
print("\n== f32 が格子に乗っているか ==")
for name, v, w in fields[3:]:
    fv = np.asarray(v, dtype=np.float64)
    fin = fv[np.isfinite(fv)]
    print("%-10s 最小 %g 最大 %g 異なる値 %d/%d" %
          (name, fin.min() if len(fin) else 0, fin.max() if len(fin) else 0,
           len(np.unique(fv)), n))

# ---- 共通の最小刻みで整数化して、差分の桁数を見る ----
# float32 の値は m*2^e（m は 24 bit）。列の中で一番小さい ulp を単位に取れば、
# どの値も整数の倍数になる。これは**可逆**（戻すときに同じ倍数を掛ける）。
print("\n== 共通の最小刻みで整数化 ==")
def ulp_int(v):
    f = np.asarray(v, dtype=np.float32)
    u = f.view(np.uint32).astype(np.int64)
    e = (u >> 23) & 0xFF                     # 指数部
    fin = e[(e > 0) & (e < 255)]
    if len(fin) == 0: return None, None
    emin = int(fin.min())
    # 単位 = 2^(emin-127-23)。値 / 単位 = m * 2^(e-emin)（整数）
    m = (u & 0x7FFFFF) | np.where(e > 0, 1 << 23, 0)
    sh = e - emin
    if sh.max() > 40: return None, None      # 桁が離れすぎるなら諦める
    q = (m << np.maximum(sh, 0)).astype(object)
    q = np.array([int(x) for x in q], dtype=object)
    sign = np.where((u >> 31) & 1, -1, 1)
    return np.array([int(s) * int(x) for s, x in zip(sign, q)], dtype=object), emin

def bits_of(a):
    """桁数の合計（零は 1 bit）。符号器が払う量の目安。"""
    return sum(max(1, int(abs(int(x))).bit_length() + 1) for x in a)

for name, v, w in fields[3:]:
    q, emin = ulp_int(v)
    if q is None:
        print("%-10s 桁が離れすぎ" % name); continue
    d = np.array([q[i] - q[i - 1] for i in range(1, len(q))], dtype=object)
    print("%-10s 整数の桁数 %5.2f  差分の桁数 %5.2f  中央値|差分| %s"
          % (name, bits_of(q) / len(q), bits_of(d) / len(d),
             int(np.median([abs(int(x)) for x in d]))))
