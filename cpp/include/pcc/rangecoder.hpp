// 決定論的な二値算術符号化器（整数演算のみ）。
//
// Python 実装 (python/rangecoder.py) と **バイト単位で一致する** ことを要件とする。
// 可逆圧縮では符号化側と復号側で丸めが 1 ビットでも違えば復号が破綻するため、
// 浮動小数は一切使わず、確率状態も 16bit 整数で持つ。
#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace pcc {

inline constexpr int PROB_BITS = 16;
inline constexpr uint32_t PROB_ONE = 1u << PROB_BITS;
inline constexpr uint32_t TOP = 1u << 24;

struct BitModel {
    uint32_t p0 = PROB_ONE >> 1;
    int rate = 5;
    inline void update(int bit) {
        if (bit) p0 -= p0 >> rate;
        else     p0 += (PROB_ONE - p0) >> rate;
        // 0 や PROB_ONE に張り付くと bound が 0 または range と等しくなり、
        // 正規化ループが進まなくなる。念のため内側に寄せる。
        if (p0 == 0) p0 = 1;
        else if (p0 >= PROB_ONE) p0 = PROB_ONE - 1;
    }
};

// LZMA 方式。キャリーは cache / cache_size で伝播させる。
class Encoder {
public:
    Encoder() { out_.reserve(1 << 16); }
    inline void encode(BitModel& m, int bit) {
        uint32_t bound = (range_ >> PROB_BITS) * m.p0;
        if (!bit) range_ = bound;
        else { low_ += bound; range_ -= bound; }
        m.update(bit);
        while (range_ < TOP) { range_ <<= 8; shiftLow(); }
    }
    std::vector<uint8_t> finish() {
        for (int i = 0; i < 5; ++i) shiftLow();
        return std::move(out_);
    }
private:
    inline void shiftLow() {
        if (low_ < 0xFF000000ull || low_ > 0xFFFFFFFFull) {
            uint8_t carry = static_cast<uint8_t>(low_ >> 32);
            uint8_t t = cache_;
            do { out_.push_back(static_cast<uint8_t>(t + carry)); t = 0xFF; }
            while (--cache_size_);
            cache_ = static_cast<uint8_t>((low_ >> 24) & 0xFF);
        }
        ++cache_size_;
        low_ = (low_ << 8) & 0xFFFFFFFFull;
    }
    uint64_t low_ = 0;
    uint32_t range_ = 0xFFFFFFFFu;
    uint8_t  cache_ = 0;
    uint32_t cache_size_ = 1;
    std::vector<uint8_t> out_;
};

class Decoder {
public:
    explicit Decoder(const uint8_t* buf, size_t n) : buf_(buf), n_(n) {
        pos_ = 1;                       // 先頭バイトは cache の初期値
        for (int i = 0; i < 4; ++i) code_ = (code_ << 8) | byte();
    }
    inline int decode(BitModel& m) {
        uint32_t bound = (range_ >> PROB_BITS) * m.p0;
        int bit;
        if (code_ < bound) { range_ = bound; bit = 0; }
        else { code_ -= bound; range_ -= bound; bit = 1; }
        m.update(bit);
        while (range_ < TOP) { range_ <<= 8; code_ = (code_ << 8) | byte(); }
        return bit;
    }
private:
    inline uint8_t byte() { return pos_ < n_ ? buf_[pos_++] : 0; }
    const uint8_t* buf_; size_t n_; size_t pos_;
    uint32_t range_ = 0xFFFFFFFFu, code_ = 0;
};

// 非負整数の二値化: 単項プレフィックス + 固定ビットサフィックス。
// ビット位置ごとに別モデルを持ち、文脈ごとにモデル一式を分ける。
// max_k は 64 でなければならない。int64 の zigzag は最大 64 bit になるので、
// 32 のままだと pm[k] が配列外に出て BitModel が壊れ、p0=0 → bound=0 → range=0
// となって renormalize の while が永久に抜けなくなる（実測: 35 分以上スピン）。
class UIntCoder {
public:
    explicit UIntCoder(int n_ctx = 1, int max_k = 64)
        : max_k_(max_k), stride_(max_k + 1),
          prefix_(n_ctx * (max_k + 1)), suffix_(n_ctx * (max_k + 1)) {}
    inline void encode(Encoder& e, uint64_t v, int ctx = 0) {
        // k = floor(log2(v+1))。v = 2^64-1 のとき v+1 が桁溢れするので
        // その場合だけ k=64 とし、サフィックスに v をそのまま 64 bit 書く。
        int k; uint64_t rem;
        if (v == ~0ull) { k = 64; rem = v; }
        else {
            // t >> 64 はシフト量が幅以上で未定義動作（x86 では t>>0 になり
            // ループが抜けなくなる）。k は 63 で必ず頭打ちにする。
            k = 0; uint64_t t = v + 1;
            while (k < 63 && (t >> (k + 1))) ++k;
            rem = v + 1 - (1ull << k);
        }
        BitModel* pm = &prefix_[ctx * stride_];
        BitModel* sm = &suffix_[ctx * stride_];
        for (int i = 0; i < k; ++i) e.encode(pm[i], 1);
        if (k < max_k_) e.encode(pm[k], 0);
        for (int i = k - 1; i >= 0; --i) e.encode(sm[i], (rem >> i) & 1);
    }
    inline uint64_t decode(Decoder& d, int ctx = 0) {
        BitModel* pm = &prefix_[ctx * stride_];
        BitModel* sm = &suffix_[ctx * stride_];
        int k = 0;
        while (k < max_k_ && d.decode(pm[k])) ++k;
        uint64_t rem = 0;
        for (int i = k - 1; i >= 0; --i) rem = (rem << 1) | static_cast<uint64_t>(d.decode(sm[i]));
        if (k == 64) return rem;                       // 符号化側の特例と対にする
        return (1ull << k) + rem - 1;
    }
private:
    int max_k_, stride_;
    std::vector<BitModel> prefix_, suffix_;   // 宣言順と初期化順を合わせる
};

inline uint64_t zigzag(int64_t v)   { return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63); }
inline int64_t  unzigzag(uint64_t z){ return static_cast<int64_t>(z >> 1) ^ -static_cast<int64_t>(z & 1); }

std::vector<uint8_t> encode_ints(const int64_t* v, size_t n,
                                 const int32_t* ctx = nullptr, int n_ctx = 1);
void decode_ints(const uint8_t* buf, size_t nbytes, int64_t* out, size_t n,
                 const int32_t* ctx = nullptr, int n_ctx = 1);

} // namespace pcc
