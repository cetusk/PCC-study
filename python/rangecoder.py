"""決定論的な二値算術符号化器（整数演算のみ）。

survey 攻め筋⑥:
  可逆圧縮で学習モデルを使うと、符号化側と復号側で浮動小数の丸めが
  1ビットでも違えば算術復号が破綻する。GPU 世代が違うだけで復元できない、
  は実運用では致命的。FastPCC(TIP 2026) がこれに正面から答えた。
  → この制約は後から足せない。最初から整数のみで組む。

ここでは確率状態を 16bit 整数で持ち、更新もシフトのみで行う。
浮動小数は一切使わないので、どの環境でもビット一致する。
"""
from __future__ import annotations
import numpy as np

PROB_BITS = 16
PROB_ONE = 1 << PROB_BITS
TOP = 1 << 24
BOT = 1 << 16


class BitModel:
    """適応二値確率モデル。p0 を 16bit 固定小数で保持する。"""
    __slots__ = ("p0", "rate")

    def __init__(self, rate: int = 5):
        self.p0 = PROB_ONE >> 1
        self.rate = rate

    def update(self, bit: int) -> None:
        if bit:
            self.p0 -= self.p0 >> self.rate
        else:
            self.p0 += (PROB_ONE - self.p0) >> self.rate


class Encoder:
    """LZMA 方式のレンジ符号化器。キャリーは cache/cache_size で伝播させる。"""

    def __init__(self):
        self.low = 0            # 64bit（キャリー検出のため 32bit を超えさせる）
        self.range = 0xFFFFFFFF
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > 0xFFFFFFFF:
            carry = self.low >> 32
            t = self.cache
            while True:
                self.out.append((t + carry) & 0xFF)
                t = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low << 8) & 0xFFFFFFFF

    def encode(self, m: BitModel, bit: int) -> None:
        bound = (self.range >> PROB_BITS) * m.p0
        if not bit:
            self.range = bound
        else:
            self.low += bound
            self.range -= bound
        m.update(bit)
        while self.range < TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class Decoder:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 1            # 先頭バイトは cache の初期値なので読み飛ばす
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF

    def _byte(self) -> int:
        if self.pos < len(self.buf):
            b = self.buf[self.pos]
            self.pos += 1
            return b
        return 0

    def decode(self, m: BitModel) -> int:
        bound = (self.range >> PROB_BITS) * m.p0
        if self.code < bound:
            self.range = bound
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            bit = 1
        m.update(bit)
        while self.range < TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
        return bit


# ---------------------------------------------------- 整数の二値化（Exp-Golomb 風）

class UIntCoder:
    """非負整数を単項＋固定ビットで符号化する。文脈ごとにモデルを持つ。

    値 v を   (単位: プレフィックス長 k = bit_length(v+1)-1)
      prefix : k 個の 1 と 1 個の 0
      suffix : k ビットの生値（ただし上位ビットにも文脈を付ける）
    で表す。小さい値ほど短い。
    """

    def __init__(self, n_ctx: int = 1, max_k: int = 64):
        # max_k は 64 必須。int64 の zigzag は 64 bit になりうる。
        # 32 のままだと C++ 実装で配列外アクセス→無限ループになる
        # （Python は IndexError で止まるが、どちらにせよ壊れる）
        self.max_k = max_k
        self.prefix = [[BitModel() for _ in range(max_k + 1)] for _ in range(n_ctx)]
        self.suffix = [[BitModel() for _ in range(max_k + 1)] for _ in range(n_ctx)]

    def encode(self, enc: Encoder, v: int, ctx: int = 0) -> None:
        assert v >= 0
        # v = 2^64-1 のときだけ k=64 とし、サフィックスに v をそのまま書く
        # （C++ 実装と同じ規約。v+1 が 64bit で桁溢れするため）
        if v == (1 << 64) - 1:
            k, rem = 64, v
        else:
            k = (v + 1).bit_length() - 1
            rem = v + 1 - (1 << k)
        pm, sm = self.prefix[ctx], self.suffix[ctx]
        for i in range(k):
            enc.encode(pm[i], 1)
        if k < self.max_k:
            enc.encode(pm[k], 0)
        for i in range(k - 1, -1, -1):
            enc.encode(sm[i], (rem >> i) & 1)

    def decode(self, dec: Decoder, ctx: int = 0) -> int:
        pm, sm = self.prefix[ctx], self.suffix[ctx]
        k = 0
        while k < self.max_k and dec.decode(pm[k]):
            k += 1
        rem = 0
        for i in range(k - 1, -1, -1):
            rem = (rem << 1) | dec.decode(sm[i])
        if k == 64:
            return rem
        return (1 << k) + rem - 1


def zigzag(v):
    """符号を潰して非負にする。

    int64 のまま (v << 1) を計算すると、|v| が 2^62 を超えたときに桁が溢れて
    負になる（float64 のビットパターンを整数として運ぶ場合に実際に起きる）。
    符号なしで計算しなければならない。
    """
    v = np.asarray(v, dtype=np.int64)
    u = v.astype(np.uint64)
    return ((u << np.uint64(1)) ^ (v >> np.int64(63)).astype(np.uint64))


def unzigzag(v):
    u = np.asarray(v, dtype=np.uint64)
    return ((u >> np.uint64(1)).astype(np.int64)
            ^ -(u & np.uint64(1)).astype(np.int64))


def encode_ints(vals: np.ndarray, ctx: np.ndarray | None = None,
                n_ctx: int = 1, max_k: int = 64) -> bytes:
    """整数列を符号化。ctx は各値に対する文脈番号（0..n_ctx-1）。"""
    enc = Encoder()
    uc = UIntCoder(n_ctx, max_k)
    z = zigzag(vals)
    if ctx is None:
        for v in z.tolist():
            uc.encode(enc, int(v), 0)
    else:
        for v, c in zip(z.tolist(), ctx.tolist()):
            uc.encode(enc, int(v), int(c))
    return enc.finish()


def decode_ints(buf: bytes, n: int, ctx: np.ndarray | None = None,
                n_ctx: int = 1, max_k: int = 64) -> np.ndarray:
    dec = Decoder(buf)
    uc = UIntCoder(n_ctx, max_k)
    # 復号される値は zigzag 後の符号なし。int64 の配列に入れると
    # 2^63 以上で OverflowError になるので uint64 で受ける。
    out = np.empty(n, dtype=np.uint64)
    if ctx is None:
        for i in range(n):
            out[i] = uc.decode(dec, 0)
    else:
        cl = ctx.tolist()
        for i in range(n):
            out[i] = uc.decode(dec, int(cl[i]))
    return unzigzag(out)
