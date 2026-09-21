// 決定論的な二値算術符号化器（整数演算のみ）。
//
// Python 実装 (python/rangecoder.py) と **バイト単位で一致する** ことを要件とする。
// 可逆圧縮では符号化側と復号側で丸めが 1 ビットでも違えば復号が破綻するため、
// 浮動小数は一切使わず、確率状態も 16bit 整数で持つ。
#pragma once
#include <atomic>
#include <cstdint>
#include <vector>
#include <cstddef>

namespace pcc {

inline constexpr int PROB_BITS = 16;
inline constexpr uint32_t PROB_ONE = 1u << PROB_BITS;
inline constexpr uint32_t TOP = 1u << 24;

// p0 は [2^rate - 1, PROB_ONE - (2^rate - 1)] にしか入らないので 16 bit で足りる
// （rate=5 での実際の範囲は [31, 65505]、総当たりで確認した）。rate を持たせると
// 1 模型 8 byte になり、文脈が増えたときに当たりが悪くなる。定数にすると 2 byte。
// 算術は同じなので出力は 1 bit も変わらない。
inline constexpr int PROB_RATE = 5;
struct BitModel {
    uint16_t p0 = PROB_ONE >> 1;
    static constexpr int rate = PROB_RATE;
    // 0 や PROB_ONE に張り付くと正規化ループが進まなくなるが、rate >= 1 なら
    // そこには届かない。bit=1 を繰り返しても p0 >> rate が 0 になる 2^rate - 1 で
    // 止まり、bit=0 を繰り返しても PROB_ONE - (2^rate - 1) で止まる。
    // 分岐を使わない。下位ビットはほぼ一様なので分岐予測が当たらず、
    // 1 ビットあたり十数サイクルを取られる。mask は bit が 1 なら全ビット 1。
    inline void update_m(uint32_t mask) {
        uint32_t up = (PROB_ONE - p0) >> rate, dn = p0 >> rate;
        p0 = (uint16_t)(p0 + (up & ~mask) - (dn & mask));
    }
    inline void update(int bit) { update_m((uint32_t)-(uint32_t)(bit != 0)); }
};

// 候補を符号化している途中で、既に見つかっている最短を超えたら打ち切る。
// 超えた候補は最短にはなり得ないので、打ち切っても選ばれる符号器は変わらない。
// スレッドごとに「いまの最短」への参照を置く（候補は並列に符号化される）。
struct EncAbort {};
inline thread_local const std::atomic<size_t>* enc_best = nullptr;

// 多値の記号モデル。累積度数は「一定回数ごとに作り直す」方式で、符号化側は
// 表引きだけになる。作り直す時点は両側で同じなので、古い表を使っていても
// 復号は一致する（LASzip の ArithmeticModel と同じ考え方）。
struct FreqModel {
    int n = 0;
    uint32_t upd = 0, cyc = 0;
    std::vector<uint16_t> f;
    std::vector<uint32_t> c;               // c[i] = f[0..i-1] の和、c[n] が合計
    void init(int n_) {
        n = n_;
        f.assign((size_t)n, 1);
        c.assign((size_t)n + 1, 0);
        cyc = (uint32_t)(n < 32 ? 32 : n);
        rebuild();
    }
    void rebuild() {
        uint32_t s = 0;
        for (int i = 0; i < n; ++i) { c[(size_t)i] = s; s += f[(size_t)i]; }
        c[(size_t)n] = s;
        if (s >= (1u << 15)) {             // 合計が大きくなりすぎたら半分にする
            s = 0;
            for (int i = 0; i < n; ++i) {
                f[(size_t)i] = (uint16_t)((f[(size_t)i] + 1) >> 1);
                c[(size_t)i] = s; s += f[(size_t)i];
            }
            c[(size_t)n] = s;
        }
        upd = 0;
    }
    inline void bump(int s) {
        f[(size_t)s] = (uint16_t)(f[(size_t)s] + 32);
        if (++upd >= cyc) rebuild();
    }
};

// LZMA 方式。キャリーは cache / cache_size で伝播させる。
class Encoder {
public:
    Encoder() { out_.reserve(1 << 16); }
    inline void encode(BitModel& m, int bit) {
        uint32_t bound = (range_ >> PROB_BITS) * m.p0;
        uint32_t mask = (uint32_t)-(uint32_t)(bit != 0);
        low_ += bound & mask;
        range_ = (bound & ~mask) | ((range_ - bound) & mask);
        m.update_m(mask);
        while (range_ < TOP) { range_ <<= 8; shiftLow(); }
    }
    inline void encode_freq(FreqModel& m, int s) {
        uint32_t tot = m.c[(size_t)m.n];
        uint32_t r = range_ / tot;
        low_ += (uint64_t)r * m.c[(size_t)s];
        range_ = r * (m.c[(size_t)s + 1] - m.c[(size_t)s]);
        while (range_ < TOP) { range_ <<= 8; shiftLow(); }
        m.bump(s);
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
        if (enc_best && out_.size() > enc_best->load(std::memory_order_relaxed))
            throw EncAbort{};
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
        uint32_t mask = (uint32_t)-(uint32_t)(code_ >= bound);
        code_ -= bound & mask;
        range_ = (bound & ~mask) | ((range_ - bound) & mask);
        m.update_m(mask);
        while (range_ < TOP) { range_ <<= 8; code_ = (code_ << 8) | byte(); }
        return (int)(mask & 1u);
    }
    inline int decode_freq(FreqModel& m) {
        uint32_t tot = m.c[(size_t)m.n];
        uint32_t r = range_ / tot;
        uint32_t t = code_ / r;
        if (t >= tot) t = tot - 1;
        int lo = 0, hi = m.n;              // c[lo] <= t < c[hi] を保つ
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (m.c[(size_t)mid] <= t) lo = mid; else hi = mid;
        }
        code_ -= r * m.c[(size_t)lo];
        range_ = r * (m.c[(size_t)lo + 1] - m.c[(size_t)lo]);
        while (range_ < TOP) { range_ <<= 8; code_ = (code_ << 8) | byte(); }
        m.bump(lo);
        return lo;
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
// となって renormalize の while が永久に抜けなくなる。
class UIntCoder {
public:
    // FSYM=1 で、ビット長 k と下位 k ビットをそれぞれ 1 つの記号として送る。
    // k <= FKMAX のときだけ下位を記号にし、それより大きい k は従来どおり
    // ビットごとに送る。1 値あたりの二値符号化が 2k+1 回から 2 回になる。
    // 流れごとに切り替える。codec_encode / codec_decode が符号器 id の旗から設定する。
    static thread_local int FSYM;
    static constexpr int FKMAX = 8;
    explicit UIntCoder(int n_ctx = 1, int max_k = 64)
        : max_k_(max_k), stride_(max_k + 1), nctx_(n_ctx),
          prefix_(FSYM ? 0 : (size_t)n_ctx * (max_k + 1)),
          suffix_((size_t)n_ctx * (max_k + 1)) {
        if (FSYM) {
            kmod_.resize((size_t)n_ctx);
            for (auto& m : kmod_) m.init(max_k + 1);
            cmod_.resize((size_t)n_ctx * (FKMAX + 1));
            for (int c = 0; c < n_ctx; ++c)
                for (int k = 1; k <= FKMAX; ++k)
                    cmod_[(size_t)c * (FKMAX + 1) + k].init(1 << k);
        }
    }
    inline void encode(Encoder& e, uint64_t v, int ctx = 0) {
        // k = floor(log2(v+1))。v = 2^64-1 のとき v+1 が桁溢れするので
        // その場合だけ k=64 とし、サフィックスに v をそのまま 64 bit 書く。
        int k; uint64_t rem;
        if (v == ~0ull) { k = 64; rem = v; }
        else {
            // k = 最上位ビットの位置。1 つずつ数えると 1 値あたり k 回まわるので、
            // 前置ゼロの数から直に出す（t >= 1 なので clz は 63 以下）。
            uint64_t t = v + 1;
            k = 63 - __builtin_clzll(t);
            rem = t - (1ull << k);
        }
        if (FSYM) {
            e.encode_freq(kmod_[(size_t)ctx], k);
            if (k == 0) return;
            if (k <= FKMAX) {
                e.encode_freq(cmod_[(size_t)ctx * (FKMAX + 1) + k], (int)rem);
                return;
            }
            BitModel* sm2 = &suffix_[ctx * stride_];
            for (int i = k - 1; i >= 0; --i) e.encode(sm2[i], (rem >> i) & 1);
            return;
        }
        BitModel* pm = &prefix_[ctx * stride_];
        BitModel* sm = &suffix_[ctx * stride_];
        for (int i = 0; i < k; ++i) e.encode(pm[i], 1);
        if (k < max_k_) e.encode(pm[k], 0);
        for (int i = k - 1; i >= 0; --i) e.encode(sm[i], (rem >> i) & 1);
    }
    inline uint64_t decode(Decoder& d, int ctx = 0) {
        if (FSYM) {
            int k2 = d.decode_freq(kmod_[(size_t)ctx]);
            if (k2 == 0) return 0;
            uint64_t rem2;
            if (k2 <= FKMAX) {
                rem2 = (uint64_t)d.decode_freq(cmod_[(size_t)ctx * (FKMAX + 1) + k2]);
            } else {
                BitModel* sm2 = &suffix_[ctx * stride_];
                rem2 = 0;
                for (int i = k2 - 1; i >= 0; --i)
                    rem2 = (rem2 << 1) | (uint64_t)d.decode(sm2[i]);
            }
            if (k2 == 64) return rem2;
            return (1ull << k2) + rem2 - 1;
        }
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
    int max_k_, stride_, nctx_;
    std::vector<BitModel> prefix_, suffix_;   // 宣言順と初期化順を合わせる
    std::vector<FreqModel> kmod_, cmod_;
};

inline uint64_t zigzag(int64_t v)   { return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63); }
inline int64_t  unzigzag(uint64_t z){ return static_cast<int64_t>(z >> 1) ^ -static_cast<int64_t>(z & 1); }

std::vector<uint8_t> encode_ints(const int64_t* v, size_t n,
                                 const int32_t* ctx = nullptr, int n_ctx = 1);
void decode_ints(const uint8_t* buf, size_t nbytes, int64_t* out, size_t n,
                 const int32_t* ctx = nullptr, int n_ctx = 1);

} // namespace pcc
