// 決定論的な二値算術符号化器（整数演算のみ）。
//
// 既定の速さの表（BM_RATE_FIXED）では Python 実装 (python/rangecoder.py) と
// **バイト単位で一致する** ことを要件とする（旧形式の器 PCC1 が使う）。PCC2 の符号器は
// 出現回数で速さを変える表（BM_RATE_ADAPT）に切り替えるので、Python とは一致しない。
// 可逆圧縮では符号化側と復号側で丸めが 1 ビットでも違えば復号が破綻するため、
// 浮動小数は一切使わず、確率状態も 16bit 整数で持つ。
#pragma once
#include <array>
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
// **出現回数ごとの適応の速さ。**回数（255 で止める）から速さを引く表を糸ごとに指す。
//   従来（PCC1・既定）: 全部 PROB_RATE。Python 実装（python/rangecoder.py）とバイト一致する。
//   PCC2 の符号器（BM_RATE_ADAPT）: 回数の区間 0,1,2,3,4〜7,8〜15,16〜31,32〜63,64〜127,
//     128〜 に 1,2,2,3,3,4,5,6,6,7。出現の少ない文脈を速く寄せ、長く続いた文脈は遅く
//     寄せて雑音に振られないようにする。LAS の 15 件で全件が縮んだ。
//   速い後半（BM_RATE_FAST、旗 C_FAST_BIT）: 同じ区間に 1,2,2,3,3,4,5,5,5,5（後半は従来の
//     1/32）。場面の変わる KITTI・物体のスキャンでは既定の表より縮むので、流れごとに実測で
//     選ぶ（results/losses.md）。codec_encode / codec_decode が流れごとに切り替える。
constexpr std::array<uint8_t, 256> make_bm_rate(const int (&sched)[10]) {
    std::array<uint8_t, 256> t{};
    for (int n = 0; n < 256; ++n) {
        const int slot = n < 4 ? n : (n < 8 ? 4 : (n < 16 ? 5 : (n < 32 ? 6 :
                         (n < 64 ? 7 : (n < 128 ? 8 : 9)))));
        t[(size_t)n] = (uint8_t)sched[slot];
    }
    return t;
}
inline constexpr int BM_SCHED_FIXED[10] = {PROB_RATE, PROB_RATE, PROB_RATE, PROB_RATE, PROB_RATE,
                                           PROB_RATE, PROB_RATE, PROB_RATE, PROB_RATE, PROB_RATE};
inline constexpr int BM_SCHED_ADAPT[10] = {1, 2, 2, 3, 3, 4, 5, 6, 6, 7};
inline constexpr int BM_SCHED_FAST[10]  = {1, 2, 2, 3, 3, 4, 5, 5, 5, 5};
inline constexpr std::array<uint8_t, 256> BM_RATE_FIXED = make_bm_rate(BM_SCHED_FIXED);
inline constexpr std::array<uint8_t, 256> BM_RATE_ADAPT = make_bm_rate(BM_SCHED_ADAPT);
inline constexpr std::array<uint8_t, 256> BM_RATE_FAST  = make_bm_rate(BM_SCHED_FAST);
inline thread_local const uint8_t* g_bm_rate = BM_RATE_FIXED.data();
struct BitModel {
    uint16_t p0 = PROB_ONE >> 1;
    uint8_t n = 0;                       // 出現回数（255 で止める）
    static constexpr int rate = PROB_RATE;
    // 0 や PROB_ONE に張り付くと正規化ループが進まなくなるが、rate >= 1 なら
    // そこには届かない。bit=1 を繰り返しても p0 >> rate が 0 になる 2^rate - 1 で
    // 止まり、bit=0 を繰り返しても PROB_ONE - (2^rate - 1) で止まる。
    // 分岐を使わない。下位ビットはほぼ一様なので分岐予測が当たらず、
    // 1 ビットあたり十数サイクルを取られる。mask は bit が 1 なら全ビット 1。
    inline void update_m(uint32_t mask) {
        const int r = g_bm_rate[n];
        n = (uint8_t)(n + (n < 255));
        uint32_t up = (PROB_ONE - p0) >> r, dn = p0 >> r;
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
    uint32_t upd = 0, cyc = 0, mx = 0;
    std::vector<uint16_t> f;
    std::vector<uint32_t> c;               // c[i] = f[0..i-1] の和、c[n] が合計
    // 作り直す間隔を**加わった重みの相対量**で決める。
    // 間隔を字母の数で固定すると、**序盤が古い表のまま流れる**。
    // 1065 点の extra では、最初の 65 個を一様な表で符号化することになり、
    // 幾何が 0.25 bpp（33 byte）伸びていた。総和の 1/REFRESH ごとに作り直すと、
    // 序盤は毎回・終盤は稀になる。定常では総和が 32768 で頭打ちなので
    // 間隔は 64 に落ち着き、**作り直す回数は元とほとんど変わらない**。
    static constexpr uint32_t REFRESH = 32;
    void init(int n_) {
        n = n_;
        f.assign((size_t)n, 1);
        c.assign((size_t)n + 1, 0);
        mx = (uint32_t)(n < 32 ? 32 : n);
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
        cyc = c[(size_t)n] / (32u * REFRESH);
        if (cyc < 1) cyc = 1;
        if (cyc > mx) cyc = mx;
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
    // 模型を通さずに等確率で書く。**一様に近いビットに適応模型は無力どころか
    // 害になる**（p0 が雑音に振られるので、1 ビットが 1.0119 ビットに付く）。
    // 仮数部の下位はほぼ一様なので、ここを素通しにすると 1.19% ぶん返ってくる。
    inline void encode_raw(uint64_t v, int nbits) {
        for (int i = nbits - 1; i >= 0; --i) {
            range_ >>= 1;
            if ((v >> i) & 1) low_ += range_;
            while (range_ < TOP) { range_ <<= 8; shiftLow(); }
        }
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
    inline uint64_t decode_raw(int nbits) {
        uint64_t v = 0;
        for (int i = 0; i < nbits; ++i) {
            range_ >>= 1;
            uint32_t bit = (code_ >= range_) ? 1u : 0u;
            if (bit) code_ -= range_;
            v = (v << 1) | bit;
            while (range_ < TOP) { range_ <<= 8; code_ = (code_ << 8) | byte(); }
        }
        return v;
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
    // 仮数部のうち**上から何ビットを模型に通すか**。負なら全部通す（従来どおり）。
    // 残りは encode_raw で素通しにする。流れごとに切り替える。
    static thread_local int RAWKEEP;
    // 桁長の並びに照合模型を掛けるか。FSYM のときだけ効く。
    static thread_local int MATCH;
    static constexpr int MKEY_BITS = 21;      // 桁長 3 つ分（7 bit × 3）
    static constexpr int MTAB_BITS = 16;
    static constexpr int MRUN_MAX  = 3;
    // 文脈を束ねるか。FSYM のときだけ効く。
    static thread_local int BUNDLE;
    static constexpr uint32_t BND_T = 24;     // 自分の表に移るまでに見る数
    static inline int mdl_bits(int k) {
        if (RAWKEEP < 0 || k <= RAWKEEP) return k;
        return RAWKEEP;
    }
    static inline uint64_t low_mask(int nb) {
        return nb >= 64 ? ~0ull : ((1ull << nb) - 1);
    }
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
            if (MATCH) {
                // 旗が立った流れでしか確保しない。1 本あたり 64 KB ある。
                mtab_.assign((size_t)1 << MTAB_BITS, 0);
                mhit_.resize((size_t)n_ctx * (MRUN_MAX + 1));
            }
            if (BUNDLE) {
                gmod_.init(max_k + 1);
                gcnt_.assign((size_t)n_ctx, 0);
                // **下位ビットの表は束ねない。**束ねると extra が
                // 138.659 → 138.734 bpp と伸びた。桁数が決まったあとの
                // 下位の分布は文脈にほとんど依らないので、既に共通に近い。
            }
        }
    }
    // 下位ビットの模型を桁数の文脈と分ける。桁数 k が決まったあとの下位ビットの
    // 分布は、k を当てるのに使った文脈にはほとんど依らない。点数の少ない列では、
    // 文脈ごとに下位の模型を持つと適応が薄まるだけになる。
    inline void encode(Encoder& e, uint64_t v, int ctx = 0) { encode2(e, v, ctx, ctx); }
    inline uint64_t decode(Decoder& d, int ctx = 0) { return decode2(d, ctx, ctx); }
    inline void encode2(Encoder& e, uint64_t v, int ctx, int sctx) {
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
            if (MATCH)       match_enc(e, k, ctx);
            else if (BUNDLE) bnd_enc(e, k, ctx);
            else             e.encode_freq(kmod_[(size_t)ctx], k);
            if (k == 0) return;
            if (k <= FKMAX) {
                e.encode_freq(cmod_[(size_t)sctx * (FKMAX + 1) + k], (int)rem);
                return;
            }
            BitModel* sm2 = &suffix_[sctx * stride_];
            const int md2 = mdl_bits(k);
            for (int i = k - 1; i >= k - md2; --i) e.encode(sm2[i], (rem >> i) & 1);
            if (k > md2) e.encode_raw(rem & low_mask(k - md2), k - md2);
            return;
        }
        BitModel* pm = &prefix_[ctx * stride_];
        BitModel* sm = &suffix_[sctx * stride_];
        for (int i = 0; i < k; ++i) e.encode(pm[i], 1);
        if (k < max_k_) e.encode(pm[k], 0);
        const int md = mdl_bits(k);
        for (int i = k - 1; i >= k - md; --i) e.encode(sm[i], (rem >> i) & 1);
        if (k > md) e.encode_raw(rem & low_mask(k - md), k - md);
    }
    inline uint64_t decode2(Decoder& d, int ctx, int sctx) {
        if (FSYM) {
            int k2 = MATCH  ? match_dec(d, ctx)
                   : BUNDLE ? bnd_dec(d, ctx)
                            : d.decode_freq(kmod_[(size_t)ctx]);
            if (k2 == 0) return 0;
            uint64_t rem2;
            if (k2 <= FKMAX) {
                rem2 = (uint64_t)d.decode_freq(cmod_[(size_t)sctx * (FKMAX + 1) + k2]);
            } else {
                BitModel* sm2 = &suffix_[sctx * stride_];
                const int md2 = mdl_bits(k2);
                rem2 = 0;
                for (int i = k2 - 1; i >= k2 - md2; --i)
                    rem2 = (rem2 << 1) | (uint64_t)d.decode(sm2[i]);
                if (k2 > md2) {
                    // **64 ビットのシフトは未定義**。md2=0 かつ k2=64 で到達する。
                    // 符号化側の low_mask は場合分けしてあるので、こちらも合わせる。
                    const int nb = k2 - md2;
                    rem2 = (nb >= 64 ? 0ull : (rem2 << nb)) | d.decode_raw(nb);
                }
            }
            if (k2 == 64) return rem2;
            return (1ull << k2) + rem2 - 1;
        }
        BitModel* pm = &prefix_[ctx * stride_];
        BitModel* sm = &suffix_[sctx * stride_];
        int k = 0;
        while (k < max_k_ && d.decode(pm[k])) ++k;
        uint64_t rem = 0;
        const int md = mdl_bits(k);
        for (int i = k - 1; i >= k - md; --i)
            rem = (rem << 1) | static_cast<uint64_t>(d.decode(sm[i]));
        if (k > md) {
            const int nb = k - md;
            rem = (nb >= 64 ? 0ull : (rem << nb)) | d.decode_raw(nb);
        }
        if (k == 64) return rem;                       // 符号化側の特例と対にする
        return (1ull << k) + rem - 1;
    }
private:
    // 直前 3 つの桁長と文脈から合図を作る。表は「その合図のあと最後に来た桁長」を
    // 1 byte で覚える（0 は空）。衝突はそのまま外れになるだけなので確認は要らない。
    inline uint32_t mhash(int ctx) const {
        uint32_t h = mkey_ * 2654435761u + (uint32_t)ctx * 40503u;
        h ^= h >> 15;
        return h & (((uint32_t)1 << MTAB_BITS) - 1);
    }
    inline void mstep(int k, uint32_t h) {
        mtab_[h] = (uint8_t)(k + 1);
        mkey_ = ((mkey_ << 7) | (uint32_t)k) & (((uint32_t)1 << MKEY_BITS) - 1);
    }
    inline void match_enc(Encoder& e, int k, int ctx) {
        const uint32_t h = mhash(ctx);
        const int pred = (int)mtab_[h] - 1;
        if (pred >= 0) {
            const bool hit = (k == pred);
            e.encode(mhit_[(size_t)ctx * (MRUN_MAX + 1) + mrun_], hit ? 1 : 0);
            if (hit) { if (mrun_ < MRUN_MAX) ++mrun_; mstep(k, h); return; }
            mrun_ = 0;
        } else {
            mrun_ = 0;
        }
        e.encode_freq(kmod_[(size_t)ctx], k);
        mstep(k, h);
    }
    inline int match_dec(Decoder& d, int ctx) {
        const uint32_t h = mhash(ctx);
        const int pred = (int)mtab_[h] - 1;
        int k;
        if (pred >= 0) {
            if (d.decode(mhit_[(size_t)ctx * (MRUN_MAX + 1) + mrun_])) {
                if (mrun_ < MRUN_MAX) ++mrun_;
                mstep(pred, h);
                return pred;
            }
            mrun_ = 0;
        } else {
            mrun_ = 0;
        }
        k = d.decode_freq(kmod_[(size_t)ctx]);
        mstep(k, h);
        return k;
    }
    // 束ね: まだ見た数が足りない文脈は共通の表に乗せる。どちらの表も毎回
    // 更新するので、移った時点で自分の表は温まっている。
    inline void bnd_enc(Encoder& e, int k, int ctx) {
        FreqModel& own = kmod_[(size_t)ctx];
        if (gcnt_[(size_t)ctx] < BND_T) { e.encode_freq(gmod_, k); own.bump(k); }
        else                            { e.encode_freq(own, k); gmod_.bump(k); }
        ++gcnt_[(size_t)ctx];
    }
    inline int bnd_dec(Decoder& d, int ctx) {
        FreqModel& own = kmod_[(size_t)ctx];
        int k;
        if (gcnt_[(size_t)ctx] < BND_T) { k = d.decode_freq(gmod_); own.bump(k); }
        else                            { k = d.decode_freq(own); gmod_.bump(k); }
        ++gcnt_[(size_t)ctx];
        return k;
    }
    int max_k_, stride_, nctx_;
    std::vector<BitModel> prefix_, suffix_;   // 宣言順と初期化順を合わせる
    std::vector<FreqModel> kmod_, cmod_;
    std::vector<uint8_t> mtab_;               // 照合の表（MATCH のときだけ確保）
    std::vector<BitModel> mhit_;              // 当たり外れ（文脈 × 連続当たり数）
    uint32_t mkey_ = 0;
    int mrun_ = 0;
    FreqModel gmod_;                          // 全文脈に共通の表（BUNDLE のとき）
    std::vector<uint32_t> gcnt_;              // 文脈ごとに見た数
};

inline uint64_t zigzag(int64_t v)   { return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63); }
inline int64_t  unzigzag(uint64_t z){ return static_cast<int64_t>(z >> 1) ^ -static_cast<int64_t>(z & 1); }

std::vector<uint8_t> encode_ints(const int64_t* v, size_t n,
                                 const int32_t* ctx = nullptr, int n_ctx = 1);
void decode_ints(const uint8_t* buf, size_t nbytes, int64_t* out, size_t n,
                 const int32_t* ctx = nullptr, int n_ctx = 1);

} // namespace pcc
