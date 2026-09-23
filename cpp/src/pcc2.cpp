#include "pcc/normalize.hpp"
#include "pcc/pcc2.hpp"
#include "pcc/rangecoder.hpp"
#include "pcc/attr.hpp"
#include "pcc/scanmodel.hpp"
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <array>
#include <atomic>
#include <thread>
#include <memory>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <deque>

namespace pcc {

// 列ごとの所要時間を測るための時計。
static double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}


// ---------------------------------------------------------------- crc64 (ECMA-182, 反転)
static uint64_t crc_tab[256];
static bool crc_ready = false;
static void crc_init() {
    for (int i = 0; i < 256; ++i) {
        uint64_t c = (uint64_t)i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ 0xC96C5795D7870F42ull : c >> 1;
        crc_tab[i] = c;
    }
    crc_ready = true;
}
uint64_t crc64(const uint8_t* p, size_t n) {
    if (!crc_ready) crc_init();
    uint64_t c = ~0ull;
    for (size_t i = 0; i < n; ++i) c = crc_tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return ~c;
}

// ---------------------------------------------------------------- 直列化の小道具
namespace {
template <class T> void put(std::vector<uint8_t>& o, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    o.insert(o.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t>& o, const std::string& s) {
    put<uint16_t>(o, (uint16_t)s.size());
    o.insert(o.end(), s.begin(), s.end());
}
void put_blob(std::vector<uint8_t>& o, const std::vector<uint8_t>& b) {
    put<uint32_t>(o, (uint32_t)b.size());
    o.insert(o.end(), b.begin(), b.end());
}
struct Rd {
    const uint8_t* b; size_t n, p = 0; bool ok = true;
    template <class T> T get() {
        T v{};
        if (p + sizeof(T) > n) { ok = false; return v; }
        memcpy(&v, b + p, sizeof(T)); p += sizeof(T); return v;
    }
    std::string str() {
        uint16_t l = get<uint16_t>(); std::string s;
        if (!ok || p + l > n) { ok = false; return s; }
        s.assign((const char*)b + p, l); p += l; return s;
    }
    std::vector<uint8_t> blob() {
        uint32_t l = get<uint32_t>(); std::vector<uint8_t> v;
        if (!ok || p + l > n) { ok = false; return v; }
        v.assign(b + p, b + p + l); p += l; return v;
    }
    // 可変長整数。小さな値が多い場所（流れの長さ・列の添字）で 8 byte を払わない。
    uint64_t varint() {
        uint64_t v = 0; int sh = 0;
        while (p < n) {
            uint8_t c = b[p++];
            v |= (uint64_t)(c & 0x7F) << sh;
            if (!(c & 0x80)) return v;
            sh += 7;
            if (sh > 63) { ok = false; return 0; }
        }
        ok = false; return 0;
    }
};

inline int bitlen(uint64_t z) { int k = 0; while (z) { ++k; z >>= 1; } return k; }
// 文脈の数。zigzag は最大 64 bit なので 65 段を 24 段に丸めて使う。
inline constexpr int NCTX = 24;
inline int ctx_of(uint64_t z) { int k = bitlen(z); return k >= NCTX ? NCTX - 1 : k; }

// 予測子。いずれも復号済みの点だけから計算できるので副情報は要らない。
// mode 0 は MedPred そのもの（直近 3 つの差分の中央値）。
inline int64_t fdiv(int64_t a, int64_t b) {         // 負でも切り捨て方向を揃える
    int64_t q = a / b;
    if ((a % b) && ((a < 0) != (b < 0))) --q;
    return q;
}
static inline int64_t med5(const int64_t* r) {
    int64_t v[5] = {r[0], r[1], r[2], r[3], r[4]};
    for (int i = 1; i < 5; ++i) {
        int64_t t = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > t) { v[j + 1] = v[j]; --j; }
        v[j + 1] = t;
    }
    return v[2];
}
struct Pred {
    MedPred m;
    int64_t q[4] = {0, 0, 0, 0};
    int64_t r[5] = {0, 0, 0, 0, 0};   // 直近 5 つの差分（mode 3 用）
    uint64_t k = 0;                    // 2^31 点を超えても負にならないよう 64 bit
    int mode = 0;
    inline int64_t predict() const {
        if (mode == 0) return m.predict();
        if (mode == 2) return m.prev + fdiv(m.predict() - m.prev, 2);
        // r[0] / q[0] には最初の点の絶対座標が入る（prev の初期値が 0 のため）。
        // 1 巡して上書きされるまでは差分でないので使わない。
        if (mode == 3) return k < 6 ? m.predict() : m.prev + med5(r);
        // 傾きの効かせ方を変えた版。葉群のように隣の点が跳ねる入力では、
        // 中央値の傾きをそのまま足すと外すので、弱めるか足さないほうが当たる。
        if (mode >= 4) {
            if (mode == 6) return m.prev;                 // 傾きを足さない
            int64_t d = (mode == 7 ? (k < 6 ? m.predict() : m.prev + med5(r))
                                   : m.predict()) - m.prev;
            if (mode == 4) return m.prev + fdiv(d, 4);
            if (mode == 5) return m.prev + d - fdiv(d, 4); // 3/4（掛け算を避ける）
            return m.prev + fdiv(d, 2);                    // mode 7
        }
        if (k < 5) return m.predict();
        return m.prev + fdiv(q[0] + q[1] + q[2] + q[3], 4);
    }
    inline void push(int64_t x) {
        int64_t d = x - m.prev;
        q[k & 3] = d; r[k % 5] = d; ++k; m.push(x);
    }
};
} // namespace

// ---------------------------------------------------------------- 符号器
// 符号つきの文脈（旗 符1/2/4）。定義は旗の変数の後ろにある。
static inline bool sgn_ctx_on();
static inline int sgn_mul();
static inline int sgn_push(int prev, int64_t d);
// 戻りの種類の文脈（属性に光線の旗が立っているとき）。定義は旗の変数の後ろ。
//   0 単独の戻り / 1 複数の最初 / 2 複数の最後 / 3 中間。旗が無ければ常に 0。
static inline int ret_class(size_t i);
static inline int ret_mul();

// 共通形: 1 点ごとに列を横断して符号化する（幾何 3 軸の同時符号化に必要）。
// mode: 0 そのまま / 1 1次差分 / 2 1次差分＋ビット数文脈 / 3 2次差分＋文脈
// 3 は「一定の刻みで増える列」に効く。gps_time は float64 のビットパターンを
// 整数と見なして差を取ると刻みがほぼ一定になるので、2 次差分はほぼ 0 になる。
static void enc_cols(const std::vector<const Col*>& cols, int mode,
                     std::vector<uint8_t>& out) {
    size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    Encoder e;
    // 符号つきの文脈（旗が立っているときだけ。差分を送る mode 1〜3 に効く）
    const bool sgnctx = mode >= 1 && sgn_ctx_on();
    const int SM = sgnctx ? sgn_mul() : 1;
    const int RM = ret_mul();
    UIntCoder uc((int)nc * NCTX * SM * RM, 64);
    std::vector<int64_t> prev(nc, 0), prev2(nc, 0);
    std::vector<int> ctx(nc, 0), sg(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            int64_t x = (*cols[c])[i];
            int64_t d = (mode == 0) ? x : x - prev[c];
            if (mode == 3) { int64_t t = d; d = d - prev2[c]; prev2[c] = t; }
            uint64_t z = zigzag(d);
            uc.encode(e, z, (((int)c * NCTX + (mode >= 2 ? ctx[c] : 0)) * SM + sg[c]) * RM + ret_class(i));
            prev[c] = x;
            if (mode >= 2) ctx[c] = ctx_of(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], d);
        }
    out = e.finish();
}

// 値そのものを、直前の値の下位 8 bit を文脈にして送る。
// 戻り番号のように系列に型がある列で、桁数の文脈より当たる。
inline constexpr int PREVCTX = 256;

static void enc_cols_prev(const std::vector<const Col*>& cols,
                          std::vector<uint8_t>& out) {
    size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    Encoder e;
    UIntCoder uc((int)nc * PREVCTX, 64);
    std::vector<int64_t> prev(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            int64_t x = (*cols[c])[i];
            uc.encode(e, zigzag(x), (int)c * PREVCTX + (int)((uint64_t)prev[c] & 0xFF));
            prev[c] = x;
        }
    out = e.finish();
}

static void dec_cols_prev(const uint8_t* data, size_t len, size_t n, size_t ncol,
                          std::vector<std::vector<int64_t>>& out) {
    Decoder d(data, len);
    UIntCoder uc((int)ncol * PREVCTX, 64);
    out.assign(ncol, std::vector<int64_t>(n));
    std::vector<int64_t> prev(ncol, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < ncol; ++c) {
            int64_t x = unzigzag(uc.decode(d, (int)c * PREVCTX
                                              + (int)((uint64_t)prev[c] & 0xFF)));
            out[c][i] = x; prev[c] = x;
        }
}

// 値の種類が少ない列を、字母の添字として直接送る。文脈は直前の記号。
// 字母は先頭に差分＋可変長で書く（列ごと）。
inline constexpr size_t SYM_MAX = 512;      // これを超える字母は扱わない

// 列ごとの字母。候補によらないので使い回す。
struct SymAlpha {
    bool ok = false;
    std::vector<int64_t> a;
    std::vector<int32_t> flat;     // 値域が狭いときの平坦な添字表
    int64_t base = 0;
    std::map<int64_t, int> idx;    // 広いときの木
    // 字母に無い値は -1 を返す（呼ぶ側は ok と列の中身が一致する前提だが、
    // 前提が崩れたときに範囲外を読まないようにしておく）。
    inline int at(int64_t v) const {
        if (flat.empty()) { auto it = idx.find(v); return it == idx.end() ? -1 : it->second; }
        const uint64_t o = (uint64_t)v - (uint64_t)base;
        return o < flat.size() ? (int)flat[(size_t)o] : -1;
    }
    // 使い回し表の鍵の確かめに使う、列の中身の指紋。
    size_t n = 0;
    uint64_t fp = 0;
};
// 列の中身の指紋（点数と、等間隔に選んだ最大 64 点の値）。**使い回し表の鍵を
// `Col*` だけにしてはいけない。**予備選別の標本は一時の列で、消えたあとの番地に
// 別の列が作られると、別の列の字母が当たって範囲外を読む（PCC_PRESELECT_ATTR=1 で
// 15 回中 5 回 segfault した）。番地が一致しても指紋が違えば作り直す。
static uint64_t col_fingerprint(const Col* col) {
    const size_t n = col->size();
    uint64_t h = 1469598103934665603ull ^ (uint64_t)n;
    const size_t step = n > 64 ? n / 64 : 1;
    for (size_t i = 0; i < n; i += step) {
        h ^= (uint64_t)(*col)[i];
        h *= 1099511628211ull;
    }
    return h;
}
struct SymCache {
    std::mutex mu;
    std::map<const Col*, std::shared_ptr<const SymAlpha>> by_col;
};

static std::shared_ptr<const SymAlpha> sym_alpha(const Col* col, const CodecCtx* ctx) {
    std::shared_ptr<SymCache> cache;
    if (ctx) {
        static std::mutex mk;
        std::lock_guard<std::mutex> lk(mk);
        if (!ctx->sym_cache) const_cast<CodecCtx*>(ctx)->sym_cache =
            std::make_shared<SymCache>();
        cache = std::static_pointer_cast<SymCache>(ctx->sym_cache);
    }
    const uint64_t fp = col_fingerprint(col);
    if (cache) {
        std::lock_guard<std::mutex> lk(cache->mu);
        auto it = cache->by_col.find(col);
        if (it != cache->by_col.end() && it->second->n == col->size() && it->second->fp == fp)
            return it->second;
    }
    auto r = std::make_shared<SymAlpha>();
    const size_t n = col->size();
    r->n = n; r->fp = fp;
    // 先に間引いた標本で見切る。字母が大きい列では全点を走る意味がない。
    {
        std::set<int64_t> probe;
        const size_t step = n > 8192 ? n / 8192 : 1;
        for (size_t i = 0; i < n; i += step) {
            probe.insert((*col)[i]);
            if (probe.size() > SYM_MAX) break;
        }
        if (probe.size() > SYM_MAX) {
            if (cache) { std::lock_guard<std::mutex> lk(cache->mu);
                         cache->by_col[col] = r; }
            return r;
        }
    }
    std::set<int64_t> s;
    for (size_t i = 0; i < n; ++i) {
        s.insert((*col)[i]);
        if (s.size() > SYM_MAX) {
            if (cache) { std::lock_guard<std::mutex> lk(cache->mu);
                         cache->by_col[col] = r; }
            return r;
        }
    }
    if (s.empty()) {
        if (cache) { std::lock_guard<std::mutex> lk(cache->mu); cache->by_col[col] = r; }
        return r;
    }
    r->a.assign(s.begin(), s.end());
    const uint64_t span = (uint64_t)r->a.back() - (uint64_t)r->a.front();
    if (span < (1u << 20)) {
        r->base = r->a.front();
        r->flat.assign((size_t)span + 1, -1);
        for (size_t k = 0; k < r->a.size(); ++k)
            r->flat[(size_t)((uint64_t)r->a[k] - (uint64_t)r->base)] = (int32_t)k;
    } else {
        for (size_t k = 0; k < r->a.size(); ++k) r->idx[r->a[k]] = (int)k;
    }
    r->ok = true;
    if (cache) { std::lock_guard<std::mutex> lk(cache->mu); cache->by_col[col] = r; }
    return r;
}
inline constexpr int SYM_REF = 16;          // 参照列から取る文脈の段数
// 旗「類」: 既に復号済みの列の値を 4 つに分けた類（点ごと）。残差の符号器では光と同じ差し込み口を使い、
// 字母では文脈（直前の記号 × 参照列）に掛け合わせる。
// codec_encode / codec_decode の外側（類の旗を剥がす段）が g_cls_next に置き、内側が g_cls に移す。
static thread_local const std::vector<uint8_t>* g_cls = nullptr;
static thread_local const std::vector<uint8_t>* g_cls_next = nullptr;

// param から参照列と、その列のどのビット位置を文脈に使うかを取り出す。
//   param = [shift u8][名前長 u16][名前]
static bool sym_ref(const std::vector<uint8_t>& p, const CodecCtx* ctx,
                    const Col*& out, int& shift) {
    if (p.size() < 3 || !ctx || !ctx->fr) return false;
    shift = p[0];
    uint16_t l; memcpy(&l, p.data() + 1, 2);
    if (p.size() < (size_t)3 + l) return false;
    std::string nm((const char*)p.data() + 3, l);
    out = ctx->fr->get(nm);
    return out != nullptr;
}

static bool enc_cols_sym(const std::vector<const Col*>& cols,
                         const Col* ref, int shift, const CodecCtx* ctx,
                         std::vector<uint8_t>& out) {
    const size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    if (!n) return false;
    std::vector<std::shared_ptr<const SymAlpha>> al(nc);
    for (size_t c = 0; c < nc; ++c) {
        al[c] = sym_alpha(cols[c], ctx);
        if (!al[c]->ok) return false;
    }
    std::vector<uint8_t> hdr;
    auto pv = [&](uint64_t v) {
        while (v >= 0x80) { hdr.push_back((uint8_t)(v | 0x80)); v >>= 7; }
        hdr.push_back((uint8_t)v);
    };
    for (size_t c = 0; c < nc; ++c) {
        pv(al[c]->a.size());
        int64_t prev = 0;
        for (int64_t v : al[c]->a) { pv(zigzag((int64_t)((uint64_t)v - (uint64_t)prev))); prev = v; }
    }
    const int R = ref ? SYM_REF : 1;
    const int K = g_cls ? 4 : 1;               // 旗「類」の 4 通り
    Encoder e;
    // **頻度表は、その文脈が初めて出たときに作る。**類を掛けると表の数が 4 倍になり、
    // 字母 512 種で 1 列 100 MB を超える。作る時期が違っても表の中身は同じなので出力は変わらない。
    std::vector<std::vector<FreqModel>> mod(nc);
    for (size_t c = 0; c < nc; ++c) {
        const int m = (int)al[c]->a.size();
        mod[c].resize(((size_t)m + 1) * (size_t)R * (size_t)K);
    }
    std::vector<int> prev(nc, 0), first(nc, 1);
    for (size_t i = 0; i < n; ++i) {
        const int rc = ref ? (int)((((uint64_t)(*ref)[i]) >> shift) & (SYM_REF - 1)) : 0;
        const int kc = g_cls ? (*g_cls)[i] : 0;
        for (size_t c = 0; c < nc; ++c) {
            const int sy = al[c]->at((*cols[c])[i]);
            if (sy < 0) return false;          // 字母に無い値（表と列が食い違った）
            const int cx = ((first[c] ? (int)al[c]->a.size() : prev[c]) * R + rc) * K + kc;
            FreqModel& q = mod[c][(size_t)cx];
            if (q.n == 0) q.init((int)al[c]->a.size());
            e.encode_freq(q, sy);
            prev[c] = sy; first[c] = 0;
        }
    }
    std::vector<uint8_t> body = e.finish();
    out = std::move(hdr);
    out.insert(out.end(), body.begin(), body.end());
    return true;
}

static bool dec_cols_sym(const uint8_t* data, size_t len, size_t n, size_t ncol,
                         const Col* ref, int shift,
                         std::vector<std::vector<int64_t>>& out, std::string& err) {
    size_t p = 0;
    auto gv = [&]() -> uint64_t {
        uint64_t v = 0;
        for (int sh = 0; p < len && sh < 64; sh += 7) {   // 64 bit 以上のシフトはしない
            const uint8_t c = data[p++];
            v |= (uint64_t)(c & 0x7F) << sh;
            if (!(c & 0x80)) return v;
        }
        return v;
    };
    // **壊れた器で落ちないように**、字母の大きさと文脈のずらしを確かめる。
    // m が 0 だと頻度表の合計が 0 になって割り算が落ち、巨大だと確保で落ちる。
    if (shift < 0 || shift >= 64) { err = "字母: 文脈のずらしが範囲外"; return false; }
    std::vector<std::vector<int64_t>> alpha(ncol);
    for (size_t c = 0; c < ncol; ++c) {
        const uint64_t m = gv();
        if (m > SYM_MAX || (m == 0 && n > 0)) { err = "字母: 大きさが範囲外"; return false; }
        alpha[c].resize((size_t)m);
        int64_t prev = 0;
        for (size_t k = 0; k < m; ++k) {
            prev = (int64_t)((uint64_t)prev + (uint64_t)unzigzag(gv()));   // 符号化側と対
            alpha[c][k] = prev;
        }
    }
    if (p > len) { err = "字母: 頭が長さを超える"; return false; }
    const int R = ref ? SYM_REF : 1;
    const int K = g_cls ? 4 : 1;               // 旗「類」の 4 通り
    Decoder d(data + p, len - p);
    std::vector<std::vector<FreqModel>> mod(ncol);   // 頻度表は初めて出た文脈で作る（符号化側と同じ）
    for (size_t c = 0; c < ncol; ++c) {
        const int m = (int)alpha[c].size();
        mod[c].resize(((size_t)m + 1) * (size_t)R * (size_t)K);
    }
    out.assign(ncol, std::vector<int64_t>(n));
    std::vector<int> prev(ncol, 0), first(ncol, 1);
    for (size_t i = 0; i < n; ++i) {
        const int rc = ref ? (int)((((uint64_t)(*ref)[i]) >> shift) & (SYM_REF - 1)) : 0;
        const int kc = g_cls ? (*g_cls)[i] : 0;
        for (size_t c = 0; c < ncol; ++c) {
            const int cx = ((first[c] ? (int)alpha[c].size() : prev[c]) * R + rc) * K + kc;
            FreqModel& q = mod[c][(size_t)cx];
            if (q.n == 0) q.init((int)alpha[c].size());
            const int s = d.decode_freq(q);
            if (s < 0 || (size_t)s >= alpha[c].size()) { err = "字母: 添字が範囲外"; return false; }
            out[c][i] = alpha[c][(size_t)s];
            prev[c] = s; first[c] = 0;
        }
    }
    return true;
}

// 3 軸を順に送り、後の軸の文脈に前の軸の差の桁数を使う。
//   dx … 直前の dx の桁
//   dy … いまの dx の桁 × 直前の dy の桁
//   dz … いまの dx の桁 × いまの dy の桁
// 軸ごとに閉じた文脈では見えない相関を掴む（点が動く向きは 3 軸で揃うため）。
static void enc_geom_cross(const std::vector<const Col*>& cols,
                           std::vector<uint8_t>& out) {
    const size_t n = cols[0]->size();
    Encoder e;
    UIntCoder uc(NCTX * NCTX * 3, 64);
    int64_t pv[3] = {0, 0, 0};
    int pk[3] = {0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        int k[3];
        for (int c = 0; c < 3; ++c) {
            const int64_t x = (*cols[c])[i];
            const uint64_t z = zigzag(x - pv[c]);
            int cx;
            if (c == 0)      cx = pk[0];
            else if (c == 1) cx = NCTX * NCTX + k[0] * NCTX + pk[1];
            else             cx = 2 * NCTX * NCTX + k[0] * NCTX + k[1];
            uc.encode(e, z, cx);
            k[c] = ctx_of(z);
            pv[c] = x;
        }
        for (int c = 0; c < 3; ++c) pk[c] = k[c];
    }
    out = e.finish();
}

static void dec_geom_cross(const uint8_t* data, size_t len, size_t n,
                           std::vector<std::vector<int64_t>>& out) {
    Decoder d(data, len);
    UIntCoder uc(NCTX * NCTX * 3, 64);
    out.assign(3, std::vector<int64_t>(n));
    int64_t pv[3] = {0, 0, 0};
    int pk[3] = {0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        int k[3];
        for (int c = 0; c < 3; ++c) {
            int cx;
            if (c == 0)      cx = pk[0];
            else if (c == 1) cx = NCTX * NCTX + k[0] * NCTX + pk[1];
            else             cx = 2 * NCTX * NCTX + k[0] * NCTX + k[1];
            const uint64_t z = uc.decode(d, cx);
            const int64_t x = pv[c] + unzigzag(z);
            out[c][i] = x;
            k[c] = ctx_of(z);
            pv[c] = x;
        }
        for (int c = 0; c < 3; ++c) pk[c] = k[c];
    }
}

// 差分の値が繰り返す列に効く。gps_time の差分は値の種類が少ないのに桁が大きく、
// bit 長を送る方式では同じ値が何度出ても毎回その桁ぶん払う。直近 K 個の相異なる
// 差分を表に持ち、当たれば添字だけを送る。外れたら逃げ道として従来どおり送る。
// 表は移動前置（当たった値を先頭へ）で更新するので副情報は要らない。
inline constexpr int DCACHE = 1024;

static void enc_cols_cache(const std::vector<const Col*>& cols,
                           std::vector<uint8_t>& out) {
    size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    Encoder e;
    UIntCoder ui((int)nc, 16);                 // 添字（0..DCACHE、DCACHE は逃げ道）
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<std::vector<int64_t>> cache(nc);
    std::vector<int64_t> prev(nc, 0);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            int64_t x = (*cols[c])[i], d = x - prev[c];
            prev[c] = x;
            auto& ca = cache[c];
            int pos = DCACHE;
            for (size_t t = 0; t < ca.size(); ++t)
                if (ca[t] == d) { pos = (int)t; break; }
            ui.encode(e, (uint64_t)pos, (int)c);
            if (pos == DCACHE) {
                uint64_t z = zigzag(d);
                uc.encode(e, z, (int)c * NCTX + ctx[c]);
                ctx[c] = ctx_of(z);
                if ((int)ca.size() == DCACHE) ca.pop_back();
            } else {
                ca.erase(ca.begin() + pos);
            }
            ca.insert(ca.begin(), d);
        }
    out = e.finish();
}

static void dec_cols_cache(const uint8_t* data, size_t len, size_t n, size_t nc,
                           std::vector<std::vector<int64_t>>& out) {
    out.assign(nc, std::vector<int64_t>(n));
    Decoder dd(data, len);
    UIntCoder ui((int)nc, 16);
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<std::vector<int64_t>> cache(nc);
    std::vector<int64_t> prev(nc, 0);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            auto& ca = cache[c];
            int pos = (int)ui.decode(dd, (int)c);
            int64_t d;
            if (pos == DCACHE) {
                uint64_t z = uc.decode(dd, (int)c * NCTX + ctx[c]);
                d = unzigzag(z);
                ctx[c] = ctx_of(z);
                if ((int)ca.size() == DCACHE) ca.pop_back();
            } else {
                d = ca[pos];
                ca.erase(ca.begin() + pos);
            }
            ca.insert(ca.begin(), d);
            int64_t x = prev[c] + d;
            out[c][i] = x;
            prev[c] = x;
        }
}

// 幾何 v1: LASzip と同じ考え方で、直近 3 つの差分の中央値を次の差分の予測に使う。
// 走査線に沿った逐次予測であり、取得順が最も得意な入力になる。
// 文脈は直前の残差のビット数（大きさごとに別の確率モデルを持つ）。
// med3 / MedPred は scanmodel.hpp にある（走査モデルの退避路と共有）。

static void enc_geom_med(const std::vector<const Col*>& cols,
                         std::vector<uint8_t>& out) {
    size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    Encoder e;
    const bool sgnctx = sgn_ctx_on();          // 符号つきの文脈（旗 符1/2/4）
    const int SM = sgnctx ? sgn_mul() : 1;
    const int RM = ret_mul();                  // 戻りの種類の文脈（旗 光）
    UIntCoder uc((int)nc * NCTX * SM * RM, 64);
    std::vector<MedPred> mp(nc);
    std::vector<int> ctx(nc, 0), sg(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            int64_t x = (*cols[c])[i];
            const int64_t r = x - mp[c].predict();
            uint64_t z = zigzag(r);
            uc.encode(e, z, (((int)c * NCTX + ctx[c]) * SM + sg[c]) * RM + ret_class(i));
            mp[c].push(x);
            ctx[c] = ctx_of(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], r);
        }
    out = e.finish();
}
static void dec_geom_med(const uint8_t* data, size_t len, size_t n, size_t nc,
                         std::vector<std::vector<int64_t>>& out) {
    out.assign(nc, std::vector<int64_t>(n));
    Decoder d(data, len);
    const bool sgnctx = sgn_ctx_on();
    const int SM = sgnctx ? sgn_mul() : 1;
    const int RM = ret_mul();
    UIntCoder uc((int)nc * NCTX * SM * RM, 64);
    std::vector<MedPred> mp(nc);
    std::vector<int> ctx(nc, 0), sg(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = uc.decode(d, (((int)c * NCTX + ctx[c]) * SM + sg[c]) * RM + ret_class(i));
            const int64_t r = unzigzag(z);
            int64_t x = mp[c].predict() + r;
            out[c][i] = x;
            mp[c].push(x);
            ctx[c] = ctx_of(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], r);
        }
}

// 幾何 v2: 補助列（復帰番号などの状態）ごとに別の予測器状態を持つ。
// 航空 LiDAR では 1 発から複数の点が返るので、格納順の隣は走査線上の隣とは限らない。
// 同じ状態の点だけを繋ぐと逐次予測が成立する。LASzip が点レコードに対して
// しているのと同じ考え方で、復号器は補助列を先に復号していれば副情報を要しない。
static void enc_geom_aux(const std::vector<const Col*>& cols,
                         const Col& aux, std::vector<uint8_t>& out) {
    size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    Encoder e;
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<std::vector<MedPred>> mp(256, std::vector<MedPred>(nc));
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i) {
        int a = (int)(aux[i] & 0xFF);
        for (size_t c = 0; c < nc; ++c) {
            int64_t x = (*cols[c])[i];
            uint64_t z = zigzag(x - mp[a][c].predict());
            uc.encode(e, z, (int)c * NCTX + ctx[c]);
            mp[a][c].push(x);
            ctx[c] = ctx_of(z);
        }
    }
    out = e.finish();
}
static void dec_geom_aux(const uint8_t* data, size_t len, size_t n, size_t nc,
                         const Col& aux,
                         std::vector<std::vector<int64_t>>& out) {
    out.assign(nc, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<std::vector<MedPred>> mp(256, std::vector<MedPred>(nc));
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i) {
        int a = (int)(aux[i] & 0xFF);
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = uc.decode(d, (int)c * NCTX + ctx[c]);
            int64_t x = mp[a][c].predict() + unzigzag(z);
            out[c][i] = x;
            mp[a][c].push(x);
            ctx[c] = ctx_of(z);
        }
    }
}

// 幾何 v3: 軸をまたぐ文脈。LASzip は Y を符号化するとき X の残差が
// 何ビットだったかを文脈に使い、Z には X と Y の平均を使う。
// 走査線上では 3 軸の残差の大きさが連動するので、これが効く。
// cross=true のとき、z の文脈を (k0+k1)/2 の平均ではなく **2 次元の組**にする。
// 平均は k0=2,k1=6 と k0=6,k1=2 を同じ文脈にしてしまう。
static inline int64_t wsub(int64_t a, int64_t b);
static inline int64_t wadd(int64_t a, int64_t b);

// 適応フィルタ（MPEG-4 ALS / Monkey's Audio と同じ構え）。
// 予測子が出した値に「直近 LMS_ORD 個の残差から作る補正」を足す。
// 重みは**符号×符号**で 1 ずつ動かす。乗算も割り算も要らず、桁も溢れない。
// 履歴の末尾は定数 1 に固定してあるので、JPEG-LS の偏り補正も兼ねる。
//
// **両側が同じ復号済みの残差から同じ重みを作る**ので、副情報は 1 ビットも増えない。
// 履歴は int32 に切り詰める（f64bits の幾何は残差が int64 の端まで使う）。
struct LmsPred {
    // **重みには歯止めが要る。**符号×符号の更新は歩幅が入力の大きさに依らないので、
    // 縛らないと重みが乱歩で大きくなり、補正が残差より大きくなって暴れる。
    // 全タップの合計が「直近の残差 1 個ぶん」を超えないところで止める。
    static const int32_t WMAX = (int32_t)((1 << LMS_SH) / LMS_ORD);
    int32_t w[LMS_ORD] = {0};
    int32_t h[LMS_ORD] = {0};
    int ord = LMS_ORD;
    static inline int32_t clip32(int64_t v) {
        const int64_t M = (int64_t)1 << 30;
        return (int32_t)(v > M ? M : (v < -M ? -M : v));
    }
    static inline int sgn(int64_t v) { return (v > 0) - (v < 0); }
    inline int64_t corr() const {
        int64_t dot = 0;
        for (int j = 0; j < ord; ++j) dot += (int64_t)w[j] * (int64_t)h[j];
        const int64_t c = dot >> LMS_SH;
        const int64_t M = (int64_t)1 << 30;
        return c > M ? M : (c < -M ? -M : c);
    }
    inline void push(int64_t d) {
        const int sd = sgn(d);
        if (sd)
            for (int j = 0; j < ord; ++j) {
                int32_t v = w[j] + sd * sgn(h[j]);
                if (v > WMAX) v = WMAX;
                if (v < -WMAX) v = -WMAX;
                w[j] = v;
            }
        for (int j = ord - 1; j > 0; --j) h[j] = h[j - 1];
        h[0] = clip32(d);
        h[ord - 1] = 1;                     // 定数項（偏りの補正）
    }
};
// 次数を外から変えて測るための口。既定は LMS_ORD。1 なら定数項だけ。
// **復号は環境変数を読まない。**以下の実験用の口（PCC_LMS_KIND / PCC_LMS_ORD /
// PCC_RAY_PRED / PCC_RAY_HIST / PCC_RAWKEEP）は符号化にも復号にも効くのに、
// 器に記録されない。以前は復号も同じ値を読んだので、別の値で復号すると
// 誤りを出さずに別の点群が戻った（査読が再現した）。いまは復号中は常に既定値を使う。
// 符号化で既定以外を使うと、pack の自己検証（既定値の復号）で必ず食い違うので分かる。
// pack は既定以外の値を PCC_ALLOW_EXPERIMENT=1 が無ければ拒む（main.cpp）。
static thread_local int g_decoding = 0;
static const int LMS_ORD_ENV = [] {
    if (const char* e = getenv("PCC_LMS_ORD")) {
        int v = atoi(e);
        if (v >= 1 && v <= LMS_ORD) return v;
    }
    return (int)LMS_ORD;
}();
static inline int lms_ord() { return g_decoding ? (int)LMS_ORD : LMS_ORD_ENV; }

// **文脈ごとの偏り補正**（JPEG-LS / CALIC と同じ考え）。
// 予測子は平均して当たっている前提で作ってあるが、文脈ごとに見ると偏る。
// 文脈ごとに残差の和と個数を持ち、その平均を予測に足す。個数が上限に達したら
// 両方を半分にして、直近を重く見る。**両側が同じ復号済みの残差から作る**ので
// 副情報は増えない。和は f64bits の幾何で桁が跳ねるので必ず縛る。
struct BiasTab {
    static constexpr int32_t NMAX = 64;
    static constexpr int64_t BMAX = (int64_t)1 << 40;
    std::vector<int64_t> B;
    std::vector<int32_t> N;
    explicit BiasTab(size_t nctx) : B(nctx, 0), N(nctx, 0) {}
    inline int64_t at(int cx) const {
        const int32_t n = N[(size_t)cx];
        return n ? B[(size_t)cx] / n : 0;
    }
    inline void push(int cx, int64_t d) {
        int64_t b = B[(size_t)cx] + d;
        if (b > BMAX) b = BMAX;
        if (b < -BMAX) b = -BMAX;
        int32_t n = N[(size_t)cx] + 1;
        if (n >= NMAX) { b >>= 1; n >>= 1; }
        B[(size_t)cx] = b;
        N[(size_t)cx] = n;
    }
};
// 0 = 文脈ごとの偏り補正、1 = 適応フィルタ、2 = **符号つきの文脈**。
// 2 が本命。我々の文脈は残差の桁数だけで**符号を捨てている**ので、
// 条件付き分布が対称になり、偏りも向きも掴めない。JPEG-LS が条件にしているのは
// 符号つきの勾配である。直前の残差の符号を文脈に 1 ビット足す。
// 無傾（傾きを足さない）が勝つファイルでは残差が生の差分なので、向きが続く。
static const int LMS_KIND_ENV = [] {
    if (const char* e = getenv("PCC_LMS_KIND")) return atoi(e);
    return 2;
}();
static inline int lms_kind() { return g_decoding ? 2 : LMS_KIND_ENV; }
// 符号の履歴を何個取るか。流れごとに旗で決まる（1 / 2 / 4）。
static thread_local int g_sgnbits = 1;
static inline int sgn_mul() { return 1 << g_sgnbits; }
static inline int sgn_push(int prev, int64_t d) {
    return ((prev << 1) | ((d > 0) ? 1 : 0)) & (sgn_mul() - 1);
}

// 光線モデル（C_RAY_BIT）。bit_fields は幾何より先に復号されるので、
// 符号化・復号の両側が同じ列を引ける。codec_encode / codec_decode が設定する。
static thread_local int g_ray = 0;
static thread_local const Col* g_ray_bf = nullptr;
// **効いているのは予測ではなく、履歴と文脈を分けることである。**
// 続きの点を「前の戻り＋戻り間ベクトル」で予測する部分は、実際の勝者（幾何v4W4）が
// 続きの点を既にほぼ同じ精度で送っているので効かない（USGS NY で残差 21.25 →
// 20.82 bit/点、符号長ではかえって伸びた）。効くのは次の 2 つ:
//   1. 続きの点を**予測子の履歴に入れない**。傾きを足す予測子（幾何v4W4傾 など）は
//      履歴の差分から傾きを作るので、戻り間の 10〜27 m の跳びで傾きが汚れる。
//      AHN5 で 24.591 → 23.916 bpp。
//   2. 続きの点に**別の文脈**を割り当てる（分布が違う）。
// 予測部分は PCC_RAY_PRED=1 で戻せる。
//   PCC_RAY_HIST=1 … 続きの点も予測子の履歴に入れる（切り分け用）
static const int RAY_PRED_ENV = [] { const char* e = getenv("PCC_RAY_PRED"); return e ? atoi(e) : 0; }();
static const int RAY_HIST_ENV = [] { const char* e = getenv("PCC_RAY_HIST"); return e ? atoi(e) : 0; }();
static inline int ray_pred() { return g_decoding ? 0 : RAY_PRED_ENV; }
static inline int ray_hist() { return g_decoding ? 0 : RAY_HIST_ENV; }
struct RayPred {
    int64_t gap[16][3];
    RayPred() { for (auto& g : gap) g[0] = g[1] = g[2] = 0; }
    // i 番目の点が、直前の点と**同じパルスの続き**か。戻り番号が 1 増え、
    // 戻り総数が同じなら同じパルスとみなす（LAS は 1 パルスの戻りを続けて書く）。
    static inline bool cont(size_t i, int& rn) {
        rn = 0;
        if (!g_ray || !g_ray_bf || i == 0) return false;
        const uint64_t a = (uint64_t)(*g_ray_bf)[i], b = (uint64_t)(*g_ray_bf)[i - 1];
        const int r1 = (int)(a & 15), n1 = (int)((a >> 4) & 15);
        const int r0 = (int)(b & 15), n0 = (int)((b >> 4) & 15);
        rn = r1;
        return r1 >= 2 && r1 == r0 + 1 && n1 == n0;
    }
};

// 曲面による z の予測（C_SURF_A / C_SURF_B）。0 なら切。それ以外は格子の桁数。
static thread_local int g_surf = 0;
// __int128 の丸め付き割り算（0 から遠いほうへ半分を丸める）。
static inline __int128 rdiv128(__int128 a, __int128 b) {
    if (b < 0) { a = -a; b = -b; }
    return a >= 0 ? (a + b / 2) / b : -((-a + b / 2) / b);
}
struct SurfPred {
    int S;
    std::unordered_map<uint64_t, std::array<int64_t, 3>> g;
    int64_t se[3] = {0, 0, 0};        // 減衰誤差（桁数 ×16 の固定小数）
    explicit SurfPred(int s) : S(s) {}
    static inline uint64_t key(int64_t gx, int64_t gy) {
        return ((uint64_t)(uint32_t)(int32_t)gx << 32) | (uint64_t)(uint32_t)(int32_t)gy;
    }
    static inline int64_t zbits(int64_t d) {
        const uint64_t z = zigzag(d);
        return (int64_t)(64 - __builtin_clzll(z | 1));
    }
    // 3 つの z の予測を出す: [0] いまの予測子 / [1] (x,y) の最近傍 / [2] 近傍に当てた平面。
    // 近傍が無ければ全部いまの予測子。**整数だけで計算する**（両側で厳密に一致させるため）。
    void preds(int64_t x, int64_t y, int64_t cur, int64_t pr[3]) const {
        pr[0] = pr[1] = pr[2] = cur;
        const int64_t gx = x >> S, gy = y >> S;
        std::array<int64_t, 3> nb[9]; int m = 0;
        // **近傍として使うのは、本当に隣のマスにある点だけ。**鍵は 32 bit に
        // 切り詰めているので、座標が広い範囲に散る入力（double のビット列を持つ
        // 幾何など）では遠い点が同じ鍵に入る。そのまま差を 2 乗すると __int128 でも
        // 溢れうる（符号つきの溢れは未定義動作）。隣の 3×3 マスに入る点なら
        // |dx|, |dy| < 3·2^S なので、2^(S+2) を超えるものは捨てる。
        const int64_t nlim = (int64_t)1 << (S + 2);
        for (int ax = -1; ax <= 1; ++ax)
            for (int ay = -1; ay <= 1; ++ay) {
                auto it = g.find(key(gx + ax, gy + ay));
                if (it == g.end()) continue;
                const int64_t dx = wsub(it->second[0], x), dy = wsub(it->second[1], y);
                if (dx > nlim || dx < -nlim || dy > nlim || dy < -nlim) continue;
                nb[m++] = it->second;
            }
        if (m == 0) return;
        int bi = 0; __int128 bd = -1;
        for (int t = 0; t < m; ++t) {
            const __int128 dx = (__int128)nb[t][0] - x, dy = (__int128)nb[t][1] - y;
            const __int128 q = dx * dx + dy * dy;
            if (bd < 0 || q < bd) { bd = q; bi = t; }
        }
        pr[1] = pr[2] = nb[bi][2];
        if (m < 3) return;
        // 平面 z = a·X + b·Y + c を、原点を (x, y)、z を最近傍の z からの差で解く。
        // 予測は c（＋最近傍の z）。値はどれも小さいので __int128 で溢れない。
        const int64_t z0 = nb[bi][2];
        // z の差も縛る。|dz| が 2^40 を超える近傍は平面に使わない（最近傍で済ませる）。
        for (int t = 0; t < m; ++t) {
            const int64_t dz = wsub(nb[t][2], z0);
            if (dz > ((int64_t)1 << 40) || dz < -((int64_t)1 << 40)) return;
        }
        __int128 Sx = 0, Sy = 0, Sz = 0, Sxx = 0, Sxy = 0, Syy = 0, Sxz = 0, Syz = 0;
        for (int t = 0; t < m; ++t) {
            const __int128 X_ = (__int128)nb[t][0] - x, Y_ = (__int128)nb[t][1] - y;
            const __int128 Z_ = (__int128)nb[t][2] - z0;
            Sx += X_; Sy += Y_; Sz += Z_;
            Sxx += X_ * X_; Sxy += X_ * Y_; Syy += Y_ * Y_; Sxz += X_ * Z_; Syz += Y_ * Z_;
        }
        const __int128 M = m;
        const __int128 det = Sxx * (Syy * M - Sy * Sy) - Sxy * (Sxy * M - Sy * Sx)
                           + Sx * (Sxy * Sy - Syy * Sx);
        if (det == 0) return;
        const __int128 dc = Sxx * (Syy * Sz - Syz * Sy) - Sxy * (Sxy * Sz - Syz * Sx)
                          + Sxz * (Sxy * Sy - Syy * Sx);
        const __int128 c = rdiv128(dc, det);
        const __int128 lim = (__int128)1 << 40;
        if (c > lim || c < -lim) return;      // 退化した当てはめは使わない
        pr[2] = wadd(z0, (int64_t)c);
    }
    // 直近の減衰誤差が最も小さい予測子（同点なら番号の小さいほう）。
    int pick() const {
        int k = 0;
        for (int t = 1; t < 3; ++t) if (se[t] < se[k]) k = t;
        return k;
    }
    void update(int64_t z, const int64_t pr[3]) {
        for (int t = 0; t < 3; ++t) se[t] = se[t] - (se[t] >> 4) + zbits(wsub(z, pr[t])) * 16;
    }
    void insert(int64_t x, int64_t y, int64_t z) { g[key(x >> S, y >> S)] = {x, y, z}; }
};
static thread_local int g_lms = 0;
// 符号つきの文脈を使うか（旗が立ち、かつ既定の種類のとき）
static inline bool sgn_ctx_on() { return g_lms && lms_kind() == 2; }
static inline int ret_mul() { return ((g_ray && g_ray_bf) || g_cls) ? 4 : 1; }
// 残差が符号化順（空間予測の順）に並んでいるとき、その順の表（位置 → 点の番号）。
// 空なら残差は点の並び順。codec_encode / codec_decode が抜けるときに戻す。
static thread_local const std::vector<int32_t>* g_resid_ord = nullptr;
static inline int ret_class(size_t i) {
    if (g_cls) {
        if (g_resid_ord) i = (size_t)(*g_resid_ord)[i];
        return (*g_cls)[i];
    }
    if (!g_ray || !g_ray_bf) return 0;
    if (g_resid_ord) i = (size_t)(*g_resid_ord)[i];
    const uint64_t a = (uint64_t)(*g_ray_bf)[i];
    const int r = (int)(a & 15), n = (int)((a >> 4) & 15);
    return n <= 1 ? 0 : (r <= 1 ? 1 : (r >= n ? 2 : 3));
}

static void enc_geom_x(const std::vector<const Col*>& cols,
                       int pmode, std::vector<uint8_t>& out, bool cross = false,
                       const Col* aux = nullptr, int amask = 0, int cshift = 0,
                       bool sshare = false) {
    size_t n = cols[0]->size();
    Encoder e;
    const int NC0 = cross ? (2 * NCTX + NCTX * NCTX) : (3 * NCTX);
    const bool sgnctx = (g_lms && lms_kind() == 2);
    const int RAYM = g_ray ? 2 : 1;
    UIntCoder uc((sgnctx ? NC0 * sgn_mul() : NC0) * RAYM, 64);
    const size_t ns = (aux && amask) ? (size_t)amask + 1 : 1;
    std::vector<Pred> mps(ns * 3);
    for (auto& q : mps) q.mode = pmode;
    int psg[3] = {0, 0, 0};
    LmsPred lms[3];
    for (auto& q : lms) q.ord = lms_ord();
    BiasTab bias((size_t)NC0);
    RayPred ray;
    int prev_kx = 0;
    for (size_t i = 0; i < n; ++i) {
        Pred* mp = &mps[(ns == 1 ? 0 : (size_t)((uint64_t)(*aux)[i] & (uint64_t)amask)) * 3];
        int rn = 0;
        const bool ct = RayPred::cont(i, rn);
        int k[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            int cx;
            if (cross && c == 2) cx = 2 * NCTX + (k[0] >> cshift) * NCTX + (k[1] >> cshift);
            else {
                int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
                ctxc >>= cshift;
                if (ctxc >= NCTX) ctxc = NCTX - 1;
                cx = c * NCTX + ctxc;
            }
            const int64_t x = (*cols[c])[i];
            int64_t p;
            if (ct && ray_pred()) p = wadd((*cols[c])[i - 1], ray.gap[rn][c]);
            else {
                p = mp[c].predict();
                if (g_lms && lms_kind() == 1) p = wadd(p, lms[c].corr());
                if (g_lms && lms_kind() == 0) p = wadd(p, bias.at(cx));
            }
            const int64_t dd = wsub(x, p);
            const uint64_t z = zigzag(dd);
            int cx2 = sgnctx ? cx * sgn_mul() + psg[c] : cx;
            if (g_ray) cx2 = cx2 * 2 + (ct ? 1 : 0);
            uc.encode2(e, z, cx2, sshare ? c : cx2);
            if (g_lms) {
                if (lms_kind() == 1) lms[c].push(dd);
                else if (lms_kind() == 0) bias.push(cx, dd);
                else psg[c] = sgn_push(psg[c], dd);
            }
            if (ct) ray.gap[rn][c] = wsub(x, (*cols[c])[i - 1]);
            if (!ct || ray_hist()) mp[c].push(x);
            k[c] = ctx_of(z);
        }
        prev_kx = k[0];
    }
    out = e.finish();
}
// 幾何の列は `f64bits` のとき **double のビット列を int64 として持つ**ので、
// 負の値が約 -4.6e18 になり、正の値との差が int64 をはみ出す。
// 符号つきの溢れは未定義動作で、**往復検証は同じ二値の同じ最適化で走るため
// すり抜ける**。符号なしで回してから戻す（2 の補数の巻き戻しは規定済みで、
// zigzag / unzigzag はその巻き戻しに対して完全な逆写像になっている）。
static inline int64_t wsub(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a - (uint64_t)b);
}
static inline int64_t wadd(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a + (uint64_t)b);
}

// 向き追従 — y の予測に、その点の dx と直前の向き (前の dx, 前の dy) を使う。
// 予測は整数だけで作る。積は int64 で溢れうるので 128 bit を経由する。
// 割り算は fdiv（床）に合わせる。このファイルの他の予測子が全部 fdiv なので、
// ここだけ 0 方向に切り捨てると負の傾きの区間で予測が偏る。
static inline int64_t dir_pred_y(int64_t prev_y, int64_t dx, int64_t pdx, int64_t pdy) {
    if (pdx == 0) return wadd(prev_y, pdy);    // 向きが分からない回は直前の dy
    __int128 num = (__int128)dx * (__int128)pdy;
    __int128 den = (__int128)pdx;
    __int128 t = num / den;
    if ((num % den != 0) && ((num < 0) != (den < 0))) --t;   // 床に合わせる
    if (t > INT64_MAX / 4) t = INT64_MAX / 4;
    if (t < INT64_MIN / 4) t = INT64_MIN / 4;
    return wadd(prev_y, (int64_t)t);
}
static void enc_geom_dir(const std::vector<const Col*>& cols, int pmode,
                         std::vector<uint8_t>& out) {
    size_t n = cols[0]->size();
    Encoder e;
    const bool sgnctx = sgn_ctx_on();          // 符号つきの文脈（旗 符1/2/4）
    const int SM = sgnctx ? sgn_mul() : 1;
    const int RM = ret_mul();                  // 戻りの種類の文脈（旗 光）
    UIntCoder uc(3 * NCTX * SM * RM, 64);
    Pred mp[3];
    for (int c = 0; c < 3; ++c) mp[c].mode = pmode;
    int prev_kx = 0, sg[3] = {0, 0, 0};
    int64_t px = 0, py = 0, pdx = 0, pdy = 0;
    for (size_t i = 0; i < n; ++i) {
        int k[3] = {0, 0, 0};
        int64_t x = (*cols[0])[i], y = (*cols[1])[i];
        const int64_t dx = wsub(x, px), dy = wsub(y, py);
        for (int c = 0; c < 3; ++c) {
            int64_t v = (*cols[c])[i];
            int64_t pr = (c == 1) ? dir_pred_y(py, dx, pdx, pdy) : mp[c].predict();
            const int64_t r = wsub(v, pr);
            uint64_t z = zigzag(r);
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            uc.encode(e, z, ((c * NCTX + ctxc) * SM + sg[c]) * RM + ret_class(i));
            mp[c].push(v);
            k[c] = ctx_of(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], r);
        }
        prev_kx = k[0];
        px = x; py = y; pdx = dx; pdy = dy;
    }
    out = e.finish();
}
static void dec_geom_dir(const uint8_t* data, size_t len, size_t n, int pmode,
                         std::vector<std::vector<int64_t>>& out) {
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    const bool sgnctx = sgn_ctx_on();
    const int SM = sgnctx ? sgn_mul() : 1;
    const int RM = ret_mul();
    UIntCoder uc(3 * NCTX * SM * RM, 64);
    Pred mp[3];
    for (int c = 0; c < 3; ++c) mp[c].mode = pmode;
    int prev_kx = 0, sg[3] = {0, 0, 0};
    int64_t px = 0, py = 0, pdx = 0, pdy = 0;
    for (size_t i = 0; i < n; ++i) {
        int k[3] = {0, 0, 0};
        int64_t x = 0, dx = 0;
        for (int c = 0; c < 3; ++c) {
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            uint64_t z = uc.decode(d, ((c * NCTX + ctxc) * SM + sg[c]) * RM + ret_class(i));
            int64_t pr = (c == 1) ? dir_pred_y(py, dx, pdx, pdy) : mp[c].predict();
            const int64_t r = unzigzag(z);
            int64_t v = wadd(pr, r);
            out[c][i] = v;
            mp[c].push(v);
            k[c] = ctx_of(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], r);
            if (c == 0) { x = v; dx = wsub(x, px); }
        }
        prev_kx = k[0];
        const int64_t y = out[1][i];
        pdx = dx; pdy = wsub(y, py);
        px = x; py = y;
    }
}

// 区間ごとに予測子を切り替える。模型は通しで持つ。
// 選択は符号化側が区間ごとに全部の予測子の桁数を数えて決め、流れの先頭に書く。
static void seg_choose(const std::vector<const Col*>& cols, size_t seg,
                       std::vector<uint8_t>& pick) {
    const size_t n = cols[0]->size();
    const size_t ns = (n + seg - 1) / seg;
    pick.assign(ns, 0);
    Pred mp[8][3];
    for (int m = 0; m < 8; ++m) for (int c = 0; c < 3; ++c) mp[m][c].mode = m;
    std::vector<uint64_t> acc(8, 0);
    size_t s0 = 0;
    for (size_t i = 0; i < n; ++i) {
        for (int m = 0; m < 8; ++m)
            for (int c = 0; c < 3; ++c) {
                const int64_t v = (*cols[c])[i];
                acc[m] += (uint64_t)ctx_of(zigzag(wsub(v, mp[m][c].predict())));
            }
        for (int m = 0; m < 8; ++m) for (int c = 0; c < 3; ++c) mp[m][c].push((*cols[c])[i]);
        if ((i + 1) % seg == 0 || i + 1 == n) {
            int bm = 0;
            for (int m = 1; m < 8; ++m) if (acc[m] < acc[bm]) bm = m;
            pick[s0++] = (uint8_t)bm;
            std::fill(acc.begin(), acc.end(), 0ull);
        }
    }
}
// 選んだ予測子で通しに符号化する（模型は区間で切らない）。
// **8 本すべての予測子の状態を進める。**切り替えた先がその時点の履歴を
// 持っていなければ、切り替えた直後だけ予測が外れる。
template <class STEP>
static void seg_walk(size_t n, size_t seg, const std::vector<uint8_t>& pick, STEP step) {
    Pred mp[8][3];
    for (int m = 0; m < 8; ++m) for (int c = 0; c < 3; ++c) mp[m][c].mode = m;
    int prev_kx = 0;
    for (size_t i = 0; i < n; ++i) {
        const int m = pick[i / seg] & 7;
        int k[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            const int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            const int cx = c * NCTX + (ctxc >= NCTX ? NCTX - 1 : ctxc);
            const int64_t pr = mp[m][c].predict();
            const int64_t v = step(i, c, pr, cx);
            k[c] = ctx_of(zigzag(wsub(v, pr)));      // 実際に送った残差の桁数
            for (int q = 0; q < 8; ++q) mp[q][c].push(v);
        }
        prev_kx = k[0];
    }
}
static bool enc_geom_seg(const std::vector<const Col*>& cols, int lg,
                         std::vector<uint8_t>& out, std::string& err) {
    const size_t n = cols[0]->size();
    const size_t seg = (size_t)1 << lg;
    if (n < seg * 4) { err = "点数が区間に足りない"; return false; }
    std::vector<uint8_t> pick;
    seg_choose(cols, seg, pick);
    Encoder e;
    UIntCoder uc(3 * NCTX, 64);
    seg_walk(n, seg, pick, [&](size_t i, int c, int64_t pr, int cx) {
        const int64_t v = (*cols[c])[i];
        uc.encode(e, zigzag(wsub(v, pr)), cx);
        return v;
    });
    std::vector<uint8_t> body = e.finish();
    // 選択表は 1 区間 1 byte（3 bit しか使わないが、範囲符号にかけない素の表）
    out.clear();
    out.reserve(pick.size() + body.size());
    out.insert(out.end(), pick.begin(), pick.end());
    out.insert(out.end(), body.begin(), body.end());
    return true;
}
static bool dec_geom_seg(const uint8_t* data, size_t len, size_t n, int lg,
                         std::vector<std::vector<int64_t>>& out, std::string& err) {
    const size_t seg = (size_t)1 << lg;
    const size_t ns = (n + seg - 1) / seg;
    if (len < ns) { err = "流れが短い"; return false; }
    std::vector<uint8_t> pick(data, data + ns);
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data + ns, len - ns);
    UIntCoder uc(3 * NCTX, 64);
    seg_walk(n, seg, pick, [&](size_t i, int c, int64_t pr, int cx) {
        const int64_t v = wadd(pr, unzigzag(uc.decode(d, cx)));
        out[c][i] = v;
        return v;
    });
    return true;
}

// 座標系の回転 — せん断 3 回で 1 平面を回す。整数のまま完全に可逆。
//
// 順: x -= (t*y)>>SH ; y += (s*x)>>SH ; x -= (t*y)>>SH
// 逆: x += (t*y)>>SH ; y -= (s*x)>>SH ; x += (t*y)>>SH
// 各段は相手の座標をそのときの値で使うので、順に戻せば厳密に元へ戻る。
// flip=1 は「この平面を π 回す」＝両軸の符号反転。誤差ゼロで行える。
// 角が ±π に近いと tan(ang/2) が発散して係数が格納できないので、
// π ぶんを符号反転に出して、残りの小さい角だけをせん断にする。
// π の回転は同じ平面の回転と可換なので、せん断の前後どちらに置いてもよい。
struct RotPlane { int a, b; int32_t t, s; int8_t flip; };
static inline void rot_apply(std::vector<int64_t>* col, const RotPlane& p, bool inv) {
    std::vector<int64_t>& X = col[p.a];
    std::vector<int64_t>& Y = col[p.b];
    const int64_t t = p.t, s = p.s;
    const size_t n = X.size();
    // **順は「符号反転 → せん断」、逆は「せん断の逆 → 符号反転」。**
    // 整数のせん断は符号反転と厳密には可換でない（算術右シフトは −∞ 方向に
    // 丸めるので、-(t*y>>SH) と (t*(-y))>>SH が 1 だけ食い違う）。
    auto flip = [&]() {
        if (!p.flip) return;
        for (size_t i = 0; i < n; ++i) {
            X[i] = (int64_t)(0ull - (uint64_t)X[i]);
            Y[i] = (int64_t)(0ull - (uint64_t)Y[i]);
        }
    };
    if (!inv) {
        flip();
        for (size_t i = 0; i < n; ++i) {
            X[i] = wsub(X[i], (t * Y[i]) >> ROT_SH);
            Y[i] = wadd(Y[i], (s * X[i]) >> ROT_SH);
            X[i] = wsub(X[i], (t * Y[i]) >> ROT_SH);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            X[i] = wadd(X[i], (t * Y[i]) >> ROT_SH);
            Y[i] = wsub(Y[i], (s * X[i]) >> ROT_SH);
            X[i] = wadd(X[i], (t * Y[i]) >> ROT_SH);
        }
        flip();
    }
}
// 3x3 の対称行列の固有分解（ヤコビ法）。**符号化側でしか使わない**ので
// 浮動小数でよい。結果から作った係数を整数に丸めて格納する。
static void jacobi3(double A[3][3], double V[3][3]) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) V[i][j] = (i == j);
    for (int sweep = 0; sweep < 32; ++sweep) {
        double off = 0;
        for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j) off += A[i][j] * A[i][j];
        if (off < 1e-24) break;
        for (int p = 0; p < 3; ++p) for (int q = p + 1; q < 3; ++q) {
            if (std::fabs(A[p][q]) < 1e-30) continue;
            double th = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
            double tt = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1.0));
            double c = 1.0 / std::sqrt(tt * tt + 1.0), sn = tt * c;
            for (int k = 0; k < 3; ++k) {
                double akp = A[k][p], akq = A[k][q];
                A[k][p] = c * akp - sn * akq; A[k][q] = sn * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k) {
                double apk = A[p][k], aqk = A[q][k];
                A[p][k] = c * apk - sn * aqk; A[q][k] = sn * apk + c * aqk;
            }
            for (int k = 0; k < 3; ++k) {
                double vkp = V[k][p], vkq = V[k][q];
                V[k][p] = c * vkp - sn * vkq; V[k][q] = sn * vkp + c * vkq;
            }
        }
    }
}
// 差分の共分散から主軸を求め、せん断の係数 3 組を作る。
static bool rot_fit(const std::vector<const Col*>& cols, RotPlane out[3]) {
    const size_t n = cols[0]->size();
    if (n < 64) return false;
    double m[3] = {0, 0, 0}, C[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    const size_t step = n > 200000 ? n / 200000 : 1;
    size_t cnt = 0;
    for (size_t i = step; i < n; i += step) {
        double d[3];
        for (int c = 0; c < 3; ++c) d[c] = (double)((*cols[c])[i] - (*cols[c])[i - step]);
        for (int a = 0; a < 3; ++a) { m[a] += d[a]; for (int b = 0; b < 3; ++b) C[a][b] += d[a] * d[b]; }
        ++cnt;
    }
    if (cnt < 32) return false;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C[a][b] = C[a][b] / cnt - m[a] * m[b] / (double)cnt / (double)cnt;
    double V[3][3];
    jacobi3(C, V);
    // 固有値は A の対角に残る。大きい順に並べ替える
    double ev[3] = {C[0][0], C[1][1], C[2][2]};
    int ord[3] = {0, 1, 2};
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j)
        if (ev[ord[j]] > ev[ord[i]]) std::swap(ord[i], ord[j]);
    double R[3][3];                       // 行ベクトルに掛ける行列の転置
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) R[i][j] = V[j][ord[i]];
    if (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
      - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
      + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]) < 0)
        for (int j = 0; j < 3; ++j) R[2][j] = -R[2][j];
    // オイラー角（ZYX）に分ける
    double bb = -std::asin(std::max(-1.0, std::min(1.0, R[2][0])));
    double aa, gg;
    if (std::fabs(std::cos(bb)) > 1e-8) {
        aa = std::atan2(R[1][0], R[0][0]); gg = std::atan2(R[2][1], R[2][2]);
    } else { aa = 0; gg = std::atan2(-R[1][2], R[1][1]); }
    double ang[3] = {gg, bb, aa};
    const int pa[3] = {1, 2, 0}, pb[3] = {2, 0, 1};
    const double LIM = (double)(1ll << (ROT_SH + 2));
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < 3; ++i) {
        // ±π に近い角は tan(ang/2) が発散する。π ぶんを符号反転に出す。
        int8_t fl = 0;
        if (ang[i] > PI / 2) { ang[i] -= PI; fl = 1; }
        else if (ang[i] < -PI / 2) { ang[i] += PI; fl = 1; }
        double t = std::tan(ang[i] / 2.0) * (double)(1ll << ROT_SH);
        double sv = std::sin(ang[i]) * (double)(1ll << ROT_SH);
        if (!(std::fabs(t) < LIM) || !(std::fabs(sv) < LIM)) return false;
        out[i] = {pa[i], pb[i], (int32_t)std::llround(t), (int32_t)std::llround(sv), fl};
    }
    return true;
}

// 持ち上げ変換 — 粗い層を先に送り、細かい層は前後で挟んで内挿する。
//
// 段 l（刻み step = 2^l）で送るのは「step の奇数倍の位置」であり、その前後
// （p ± step）は 1 つ粗い段で既に確定している。だから内挿してよい。
// 最も粗い層（2^L の倍数の位置）だけは従来どおり直前からの差で送る。
// 文脈は「挟んだ幅」の桁数。復号側も両端を持っているので副情報は生じない。
static inline int64_t lift_pred(int64_t a, int64_t b) {
    // 床で割る。両側で同じ式にすること（fdiv はこのファイルの他の予測子と同じ）。
    return a + fdiv(b - a, 2);
}
template <class GET, class PUT>
static void lift_walk(size_t n, int levels, GET get, PUT put) {
    // 最も粗い層: 2^L の倍数の位置を、直前の粗い点からの差で送る
    const size_t step0 = (size_t)1 << levels;
    int64_t prev[3] = {0, 0, 0};
    int pk[3] = {0, 0, 0};
    for (size_t p = 0; p < n; p += step0) {
        for (int c = 0; c < 3; ++c) {
            const int cx = c * NCTX + (pk[c] < NCTX ? pk[c] : NCTX - 1);
            const int64_t v = put(p, c, prev[c], cx, get(p, c));
            pk[c] = ctx_of(zigzag(wsub(v, prev[c])));
            prev[c] = v;
        }
    }
    // 細かい段へ。刻みを半分にしながら、step の奇数倍の位置を内挿で送る。
    for (int l = levels - 1; l >= 0; --l) {
        const size_t step = (size_t)1 << l;
        for (size_t p = step; p < n; p += 2 * step) {
            const size_t lo = p - step;
            const size_t hi = (p + step < n) ? (p + step) : lo;
            for (int c = 0; c < 3; ++c) {
                const int64_t a = get(lo, c), b = get(hi, c);
                const int64_t pr = (hi == lo) ? a : lift_pred(a, b);
                // 挟んだ幅を文脈にする（両端は復号側も持っている）
                int sp = ctx_of(zigzag(wsub(b, a)));
                if (sp >= NCTX) sp = NCTX - 1;
                put(p, c, pr, 3 * NCTX + c * NCTX + sp, get(p, c));
            }
        }
    }
}
static void enc_geom_lift(const std::vector<const Col*>& cols, int levels,
                          std::vector<uint8_t>& out) {
    const size_t n = cols[0]->size();
    Encoder e;
    UIntCoder uc(6 * NCTX, 64);
    auto get = [&](size_t i, int c) { return (*cols[c])[i]; };
    auto put = [&](size_t, int, int64_t pr, int cx, int64_t v) {
        uc.encode(e, zigzag(wsub(v, pr)), cx);
        return v;
    };
    lift_walk(n, levels, get, put);
    out = e.finish();
}
static void dec_geom_lift(const uint8_t* data, size_t len, size_t n, int levels,
                          std::vector<std::vector<int64_t>>& out) {
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc(6 * NCTX, 64);
    auto get = [&](size_t i, int c) { return out[c][i]; };
    auto put = [&](size_t i, int c, int64_t pr, int cx, int64_t) {
        const int64_t v = wadd(pr, unzigzag(uc.decode(d, cx)));
        out[c][i] = v;
        return v;
    };
    lift_walk(n, levels, get, put);
}

// dec_geom_x は下で定義する（回転の復号がそれを使う）。
static void dec_geom_x(const uint8_t* data, size_t len, size_t n, int pmode,
                       std::vector<std::vector<int64_t>>& out, bool cross = false,
                       const Col* aux = nullptr, int amask = 0, int cshift = 0,
                       bool sshare = false);

// 回した座標を、既存の幾何v3 の仕組みで符号化する。
// 係数は流れの先頭に 24 byte 置く（復号側で三角関数を計算しない）。
static bool enc_geom_rot(const std::vector<const Col*>& cols, int pmode,
                         std::vector<uint8_t>& out, std::string& err) {
    RotPlane rp[3];
    if (!rot_fit(cols, rp)) { err = "回転を当てはめられない"; return false; }
    const size_t n = cols[0]->size();
    // せん断の掛け算が int64 を溢れない範囲か。f64bits の幾何はここで落ちる。
    const int64_t LIM = 1ll << 40;
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < n; ++i) {
            const int64_t v = (*cols[c])[i];
            if (v > LIM || v < -LIM) { err = "値が大きすぎる"; return false; }
        }
    std::vector<int64_t> q[3];
    for (int c = 0; c < 3; ++c) q[c] = cols[c]->to_vector();
    for (int i = 0; i < 3; ++i) rot_apply(q, rp[i], false);
    Col cc[3];
    std::vector<const Col*> cv(3);
    // **素の並びは詰めた直後に手放す。**両方抱えると 200 万点で 48 MB 余分に要る。
    for (int c = 0; c < 3; ++c) {
        cc[c] = std::move(q[c]);
        std::vector<int64_t>().swap(q[c]);
        cv[c] = &cc[c];
    }
    std::vector<uint8_t> body;
    enc_geom_x(cv, pmode, body);
    out.clear();
    out.reserve(27 + body.size());
    for (int i = 0; i < 3; ++i) {
        const uint32_t t = (uint32_t)rp[i].t, sv = (uint32_t)rp[i].s;
        for (int k = 0; k < 4; ++k) out.push_back((uint8_t)(t >> (8 * k)));
        for (int k = 0; k < 4; ++k) out.push_back((uint8_t)(sv >> (8 * k)));
        out.push_back((uint8_t)rp[i].flip);
    }
    out.insert(out.end(), body.begin(), body.end());
    return true;
}
static bool dec_geom_rot(const uint8_t* data, size_t len, size_t n, int pmode,
                         std::vector<std::vector<int64_t>>& out, std::string& err) {
    if (len < 27) { err = "流れが短い"; return false; }
    const int pa[3] = {1, 2, 0}, pb[3] = {2, 0, 1};
    RotPlane rp[3];
    for (int i = 0; i < 3; ++i) {
        uint32_t t = 0, sv = 0;
        for (int k = 0; k < 4; ++k) t |= (uint32_t)data[i * 9 + k] << (8 * k);
        for (int k = 0; k < 4; ++k) sv |= (uint32_t)data[i * 9 + 4 + k] << (8 * k);
        rp[i] = {pa[i], pb[i], (int32_t)t, (int32_t)sv, (int8_t)data[i * 9 + 8]};
    }
    dec_geom_x(data + 27, len - 27, n, pmode, out);
    for (int i = 2; i >= 0; --i) rot_apply(out.data(), rp[i], true);
    return true;
}

// 幾何v4 — 直近 W 点のうち最も近い点から予測し、選んだ点を明示的に送る。
//
// v0〜v3 の予測子はどれも「格納順で直前の点」に基づく。格納順が空間的に
// 連続していない入力（取得順でない配布、複数の戻り・チャネルの交互配置）では
// 直前の点が遠く、差分が大きくなる。実測では格納順の連続性（隣接点の距離 ÷
// 点間隔）が 1.6 を超えると逐次予測が G-PCC に負けた。
//
// **選んだ点は送らなければならない。** 最初に「復号側も直近 W 点を持っている
// のだから最近傍を選び直せる」と考えて実装したが、最近傍の判定にはこれから
// 復号する点そのものが要るので成立しない（往復検証が即座に落ちた）。
// 添字（直前から何点戻るか）を符号化する。残差の減りから添字の費用を引いた
// 正味は、実測で AHN4 _21 が −7.5、USGS NY が −2.2 bit/点。効かない入力も
// あるので候補の 1 つとして出し、選択原理に任せる。
template <class T>
static inline int nn_back(const T& X, const T& Y, const T& Z,
                          size_t i, size_t w) {
    size_t a = i > w ? i - w : 0;
    int64_t bd = INT64_MAX; size_t bj = i - 1;
    for (size_t j = i; j-- > a; ) {
        int64_t dx = X[j] - X[i], dy = Y[j] - Y[i], dz = Z[j] - Z[i];
        int64_t ad = (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy) + (dz < 0 ? -dz : dz);
        if (ad < bd) { bd = ad; bj = j; }      // 同値なら直前に近い方（j が大きい方）
    }
    return (int)(i - 1 - bj);                  // 0 が直前の点
}

// tmode 0 は選んだ点そのもの、1 は直近 3 つの差分の中央値を足す、2 はその半分。
// 差分は格納順で連続する 2 点の差なので、復号側も同じ値を作れる。
static inline int64_t trend_of(const MedPred& t, int tmode) {
    // 差分が 3 つ揃うまでは傾きを足さない。MedPred は k<3 のとき d[0] を返すが、
    // その d[0] は最初の点の絶対座標である（prev の初期値が 0 のため）。
    if (!tmode || t.k < 3) return 0;
    int64_t v = t.predict() - t.prev;
    return tmode == 2 ? fdiv(v, 2) : v;
}

static void enc_geom_w(const std::vector<const Col*>& cols,
                       size_t w, int tmode, std::vector<uint8_t>& out) {
    size_t n = cols[0]->size();
    // **ここで列を実体化してはいけない。**候補ごとに 3 列ぶんの場所を取ることに
    // なり、並列に走る本数だけ積み上がる（200 万点・16 並列で 768 MB）。
    // 幅つきの列をそのまま引く。探す窓は直近 w 点だけなので、引く回数は増えない。
    const Col &X = *cols[0], &Y = *cols[1], &Z = *cols[2];
    Encoder e;
    const bool sgnctx = (g_lms && lms_kind() == 2);
    const int RAYM = g_ray ? 2 : 1;
    const int SRFM = g_surf ? 3 : 1;           // z の文脈に「選んだ予測子」を足す
    UIntCoder uc((sgnctx ? 3 * NCTX * sgn_mul() : 3 * NCTX) * RAYM * SRFM, 64);
    UIntCoder ui(NCTX, 8);                     // 添字。文脈は直前の添字
    MedPred tp[3];
    RayPred ray;
    SurfPred surf(g_surf);
    int prev_kx = 0, prev_idx = 0, sg[3] = {0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        int rn = 0;
        const bool ct = RayPred::cont(i, rn);
        const int64_t v[3] = {X[i], Y[i], Z[i]};
        int64_t p[3];
        int back = 0;
        const bool use_ray = ct && ray_pred();
        if (use_ray) {
            // 同じパルスの続き。最近傍の添字は送らない（復号側も ct を知っている）。
            const int64_t q[3] = {X[i - 1], Y[i - 1], Z[i - 1]};
            for (int c = 0; c < 3; ++c) p[c] = wadd(q[c], ray.gap[rn][c]);
        } else {
            back = i ? nn_back(X, Y, Z, i, w) : 0;
            if (i) ui.encode(e, (uint64_t)back, prev_idx < NCTX ? prev_idx : NCTX - 1);
            size_t j = i ? i - 1 - (size_t)back : 0;
            p[0] = i ? X[j] + trend_of(tp[0], tmode) : 0;
            p[1] = i ? Y[j] + trend_of(tp[1], tmode) : 0;
            p[2] = i ? Z[j] + trend_of(tp[2], tmode) : 0;
        }
        int k[3] = {0, 0, 0};
        int64_t zp[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            int sk = 0;
            if (c == 2 && surf.S) {
                // (x, y) は送り終えている。z の予測子を 3 つから直近の成績で選ぶ。
                surf.preds(v[0], v[1], p[2], zp);
                sk = surf.pick();
                p[2] = zp[sk];
            }
            const int64_t dd = wsub(v[c], p[c]);
            uint64_t z = zigzag(dd);
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            int cc = sgnctx ? (c * NCTX + ctxc) * sgn_mul() + sg[c] : c * NCTX + ctxc;
            if (g_ray) cc = cc * 2 + (ct ? 1 : 0);
            if (g_surf) cc = cc * 3 + sk;
            uc.encode(e, z, cc);
            if (sgnctx) sg[c] = sgn_push(sg[c], dd);
            k[c] = ctx_of(z);
        }
        if (surf.S) { surf.update(v[2], zp); surf.insert(v[0], v[1], v[2]); }
        prev_kx = k[0];
        if (ct)
            for (int c = 0; c < 3; ++c)
                ray.gap[rn][c] = wsub(v[c], c == 0 ? X[i - 1] : (c == 1 ? Y[i - 1] : Z[i - 1]));
        if (!use_ray) prev_idx = back;
        if (!ct || ray_hist()) for (int c = 0; c < 3; ++c) tp[c].push(v[c]);
    }
    out = e.finish();
}

static void dec_geom_w(const uint8_t* data, size_t len, size_t n, size_t w, int tmode,
                       std::vector<std::vector<int64_t>>& out) {
    (void)w;
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    const bool sgnctx = (g_lms && lms_kind() == 2);
    const int RAYM = g_ray ? 2 : 1;
    const int SRFM = g_surf ? 3 : 1;
    UIntCoder uc((sgnctx ? 3 * NCTX * sgn_mul() : 3 * NCTX) * RAYM * SRFM, 64);
    UIntCoder ui(NCTX, 8);
    MedPred tp[3];
    RayPred ray;
    SurfPred surf(g_surf);
    int prev_kx = 0, prev_idx = 0, sg[3] = {0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        int rn = 0;
        const bool ct = RayPred::cont(i, rn);
        int64_t p[3];
        int back = 0;
        const bool use_ray = ct && ray_pred();
        if (use_ray) {
            for (int c = 0; c < 3; ++c) p[c] = wadd(out[c][i - 1], ray.gap[rn][c]);
        } else {
            if (i) back = (int)ui.decode(d, prev_idx < NCTX ? prev_idx : NCTX - 1);
            // 壊れた入力で先頭より前を指さないように縛る（正しい流れでは起きない）
            if (i && (size_t)back > i - 1) back = (int)(i - 1);
            size_t j = i ? i - 1 - (size_t)back : 0;
            p[0] = i ? out[0][j] + trend_of(tp[0], tmode) : 0;
            p[1] = i ? out[1][j] + trend_of(tp[1], tmode) : 0;
            p[2] = i ? out[2][j] + trend_of(tp[2], tmode) : 0;
        }
        int k[3] = {0, 0, 0};
        int64_t zp[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            int sk = 0;
            if (c == 2 && surf.S) {
                surf.preds(out[0][i], out[1][i], p[2], zp);
                sk = surf.pick();
                p[2] = zp[sk];
            }
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            int cc = sgnctx ? (c * NCTX + ctxc) * sgn_mul() + sg[c] : c * NCTX + ctxc;
            if (g_ray) cc = cc * 2 + (ct ? 1 : 0);
            if (g_surf) cc = cc * 3 + sk;
            uint64_t z = uc.decode(d, cc);
            const int64_t dd = unzigzag(z);
            out[c][i] = wadd(p[c], dd);
            if (sgnctx) sg[c] = sgn_push(sg[c], dd);
            k[c] = ctx_of(z);
        }
        if (surf.S) { surf.update(out[2][i], zp); surf.insert(out[0][i], out[1][i], out[2][i]); }
        prev_kx = k[0];
        if (ct)
            for (int c = 0; c < 3; ++c) ray.gap[rn][c] = wsub(out[c][i], out[c][i - 1]);
        if (!use_ray) prev_idx = back;
        if (!ct || ray_hist()) for (int c = 0; c < 3; ++c) tp[c].push(out[c][i]);
    }
}

static void dec_geom_x(const uint8_t* data, size_t len, size_t n, int pmode,
                       std::vector<std::vector<int64_t>>& out, bool cross,
                       const Col* aux, int amask, int cshift,
                       bool sshare) {
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    const int NC0 = cross ? (2 * NCTX + NCTX * NCTX) : (3 * NCTX);
    const bool sgnctx = (g_lms && lms_kind() == 2);
    const int RAYM = g_ray ? 2 : 1;
    UIntCoder uc((sgnctx ? NC0 * sgn_mul() : NC0) * RAYM, 64);
    const size_t ns = (aux && amask) ? (size_t)amask + 1 : 1;
    std::vector<Pred> mps(ns * 3);
    for (auto& q : mps) q.mode = pmode;
    int psg[3] = {0, 0, 0};
    LmsPred lms[3];
    for (auto& q : lms) q.ord = lms_ord();
    BiasTab bias((size_t)NC0);
    RayPred ray;
    int prev_kx = 0;
    for (size_t i = 0; i < n; ++i) {
        Pred* mp = &mps[(ns == 1 ? 0 : (size_t)((uint64_t)(*aux)[i] & (uint64_t)amask)) * 3];
        int rn = 0;
        const bool ct = RayPred::cont(i, rn);
        int k[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            int cx;
            if (cross && c == 2) cx = 2 * NCTX + (k[0] >> cshift) * NCTX + (k[1] >> cshift);
            else {
                int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
                ctxc >>= cshift;
                if (ctxc >= NCTX) ctxc = NCTX - 1;
                cx = c * NCTX + ctxc;
            }
            int cx2 = sgnctx ? cx * sgn_mul() + psg[c] : cx;
            if (g_ray) cx2 = cx2 * 2 + (ct ? 1 : 0);
            const uint64_t z = uc.decode2(d, cx2, sshare ? c : cx2);
            int64_t p;
            if (ct && ray_pred()) p = wadd(out[c][i - 1], ray.gap[rn][c]);
            else {
                p = mp[c].predict();
                if (g_lms && lms_kind() == 1) p = wadd(p, lms[c].corr());
                if (g_lms && lms_kind() == 0) p = wadd(p, bias.at(cx));
            }
            const int64_t rr = unzigzag(z);
            const int64_t x = wadd(p, rr);
            out[c][i] = x;
            if (g_lms) {
                if (lms_kind() == 1) lms[c].push(rr);
                else if (lms_kind() == 0) bias.push(cx, rr);
                else psg[c] = sgn_push(psg[c], rr);
            }
            if (ct) ray.gap[rn][c] = wsub(x, out[c][i - 1]);
            if (!ct || ray_hist()) mp[c].push(x);
            k[c] = ctx_of(z);
        }
        prev_kx = k[0];
    }
}

static void dec_cols(const uint8_t* data, size_t len, size_t n, size_t nc, int mode,
                     std::vector<std::vector<int64_t>>& out) {
    out.assign(nc, std::vector<int64_t>(n));
    Decoder d(data, len);
    const bool sgnctx = mode >= 1 && sgn_ctx_on();
    const int SM = sgnctx ? sgn_mul() : 1;
    const int RM = ret_mul();
    UIntCoder uc((int)nc * NCTX * SM * RM, 64);
    std::vector<int64_t> prev(nc, 0), prev2(nc, 0);
    std::vector<int> ctx(nc, 0), sg(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = uc.decode(d, (((int)c * NCTX + (mode >= 2 ? ctx[c] : 0)) * SM + sg[c]) * RM + ret_class(i));
            int64_t v = unzigzag(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], v);
            if (mode == 3) { v += prev2[c]; prev2[c] = v; }
            int64_t x = (mode == 0) ? v : prev[c] + v;
            out[c][i] = x;
            prev[c] = x;
            if (mode >= 2) ctx[c] = ctx_of(z);
        }
}


// ---------------------------------------------------------------- 副次情報
const std::vector<double>* CodecCtx::world_ptr() const {
    if (world) return world;
    if (!want_world || !fr) return nullptr;
    if (world_own.empty()) frame_world(*fr, world_own);
    return world_own.empty() ? nullptr : &world_own;
}

const std::vector<int32_t>* CodecCtx::ensure(size_t n, int P,
                                             const std::vector<int32_t>** perm_out) const {
    std::lock_guard<std::mutex> lk(mu);
    const std::vector<double>* w = world_ptr();
    if (!w || w->size() < n * 3) return nullptr;
    // 点数が変わったら作り直す。標本で順位を付けるときに同じ P で呼ばれると、
    // 標本ぶんの表を全点の符号化に使ってしまう。
    auto pit = perm_by_n.find(n);
    if (pit == perm_by_n.end()) {
        double t0 = now_sec();
        pit = perm_by_n.emplace(n, coding_order(*w, n, "morton")).first;
        if (getenv("PCC_DPROF"))
            fprintf(stderr, "        [順序表] %.3fs\n", now_sec() - t0);
    }
    auto it = pred_by_np.find({n, P});
    if (it == pred_by_np.end()) {
        // 候補が使う P はいつも 1 / 3 / 5 なので、最初の 1 回でまとめて作る。
        // 近傍探索も KdTree の構築も 1 回で済む（P ごとだと 3 回になる）。
        std::vector<int> Ps = want_Ps.empty() ? std::vector<int>{1, 3, 5} : want_Ps;
        if (std::find(Ps.begin(), Ps.end(), P) == Ps.end()) Ps.push_back(P);
        std::vector<std::vector<int32_t>> pds;
        double t0 = now_sec();
        build_causal_predictors_multi(*w, n, pit->second, Ps, pds);
        if (getenv("PCC_DPROF"))
            fprintf(stderr, "      [近傍表 n=%zu] %.3fs\n", n, now_sec() - t0);
        for (size_t a = 0; a < Ps.size(); ++a)
            pred_by_np.emplace(std::make_pair(n, Ps[a]), std::move(pds[a]));
        it = pred_by_np.find({n, P});
    }
    if (perm_out) *perm_out = &pit->second;
    return &it->second;
}

// 可逆な輝度・色差変換 YCoCg-R。各段は u <- u ± floor(v/2) の形で、
// v が不変なので丸め方によらず Z^2 上の全単射である。
static inline void ycocg_fwd(int64_t R, int64_t G, int64_t B,
                             int64_t& Y, int64_t& Co, int64_t& Cg) {
    Co = R - B;
    int64_t t = B + (Co >> 1);
    Cg = G - t;
    Y = t + (Cg >> 1);
}
static inline void ycocg_inv(int64_t Y, int64_t Co, int64_t Cg,
                             int64_t& R, int64_t& G, int64_t& B) {
    int64_t t = Y - (Cg >> 1);
    G = Cg + t;
    B = t - (Co >> 1);
    R = Co + B;
}

// 残差列（すでに残差なので差分は掛けない）をビット数文脈で符号化する
// 残差の要素型を選べるようにする。走査モデルの残差は座標の差なので
// 実際には 32 bit に収まる（収まらない点が 1 つでもあれば int64 に落とす）。
// 400 万点では 3 列 × 4 バイトの節約が 48 MB になる。符号化する値は同じなので
// 出力は 1 バイトも変わらない。
template <class T>
static void enc_resid(const std::vector<std::vector<T>>& res, std::vector<uint8_t>& out) {
    size_t nc = res.size(), n = nc ? res[0].size() : 0;
    Encoder e;
    // 符号つきの文脈（旗 符1/2/4 が立っているときだけ）。桁数の文脈は符号を捨てるので、
    // 直前の残差の符号の履歴を足す。幾何で効いたものを属性の残差にも使う。
    const bool sgnctx = sgn_ctx_on();
    const int SM = sgnctx ? sgn_mul() : 1;
    // 戻りの種類（光線の旗）。残差が点の並び順に並んでいる符号器でだけ意味がある。
    const int RM = ret_mul();
    UIntCoder uc((int)nc * NCTX * SM * RM, 64);
    std::vector<int> ctx(nc, 0), sg(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            const int64_t r = (int64_t)res[c][i];
            uint64_t z = zigzag(r);
            uc.encode(e, z, (((int)c * NCTX + ctx[c]) * SM + sg[c]) * RM + ret_class(i));
            ctx[c] = ctx_of(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], r);
        }
    out = e.finish();
}
// 走査モデル専用。3 列のうち z（列 2）の文脈を、同じ点の面内・面外の残差の
// ビット長から作る。幾何v3 が Z に X と Y の平均を使うのと同じ考え方。
template <class T>
static void enc_resid_x(const std::vector<std::vector<T>>& res, std::vector<uint8_t>& out) {
    size_t n = res[0].size();
    Encoder e;
    // 符号つきの文脈（lms_kind()==2）。文脈は残差の桁数だけで符号を捨てているので、
    // 直前の残差の符号を 1 ビット足す。enc_geom_x と同じ考え。
    const bool sgnctx = (g_lms && lms_kind() == 2);
    UIntCoder uc(sgnctx ? 3 * NCTX * sgn_mul() : 3 * NCTX, 64);
    int p0 = 0, p1 = 0, sg[3] = {0, 0, 0};
    auto CX = [&](int c, int k) { return sgnctx ? (c * NCTX + k) * sgn_mul() + sg[c] : c * NCTX + k; };
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = zigzag((int64_t)res[0][i]); uc.encode(e, a, CX(0, p0)); int k0 = ctx_of(a);
        uint64_t b = zigzag((int64_t)res[1][i]); uc.encode(e, b, CX(1, p1)); int k1 = ctx_of(b);
        uint64_t c = zigzag((int64_t)res[2][i]); uc.encode(e, c, CX(2, (k0 + k1) / 2));
        if (sgnctx) for (int t = 0; t < 3; ++t) sg[t] = sgn_push(sg[t], (int64_t)res[t][i]);
        p0 = k0; p1 = k1;
    }
    out = e.finish();
}
static void dec_resid_x(const uint8_t* data, size_t len, size_t n,
                        std::vector<std::vector<int64_t>>& res) {
    res.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    const bool sgnctx = (g_lms && lms_kind() == 2);
    UIntCoder uc(sgnctx ? 3 * NCTX * sgn_mul() : 3 * NCTX, 64);
    int p0 = 0, p1 = 0, sg[3] = {0, 0, 0};
    auto CX = [&](int c, int k) { return sgnctx ? (c * NCTX + k) * sgn_mul() + sg[c] : c * NCTX + k; };
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = uc.decode(d, CX(0, p0)); res[0][i] = unzigzag(a); int k0 = ctx_of(a);
        uint64_t b = uc.decode(d, CX(1, p1)); res[1][i] = unzigzag(b); int k1 = ctx_of(b);
        uint64_t c = uc.decode(d, CX(2, (k0 + k1) / 2)); res[2][i] = unzigzag(c);
        if (sgnctx) for (int t = 0; t < 3; ++t) sg[t] = sgn_push(sg[t], res[t][i]);
        p0 = k0; p1 = k1;
    }
}

static void dec_resid(const uint8_t* data, size_t len, size_t n, size_t nc,
                      std::vector<std::vector<int64_t>>& res) {
    res.assign(nc, std::vector<int64_t>(n));
    Decoder d(data, len);
    const bool sgnctx = sgn_ctx_on();
    const int SM = sgnctx ? sgn_mul() : 1;
    const int RM = ret_mul();
    UIntCoder uc((int)nc * NCTX * SM * RM, 64);
    std::vector<int> ctx(nc, 0), sg(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = uc.decode(d, (((int)c * NCTX + ctx[c]) * SM + sg[c]) * RM + ret_class(i));
            const int64_t r = unzigzag(z);
            res[c][i] = r;
            ctx[c] = ctx_of(z);
            if (sgnctx) sg[c] = sgn_push(sg[c], r);
        }
}


// ============================================================ 走査モデル経由の幾何
// 裏付けは results/als_scan_structure.md、設計は notes/03_scan_model_codec.md。
//
// 復号器が先に持っているもの: point_source_id, gps_time, bit_fields（戻り番号）。
// そこから走査順（psid と gps の安定ソート）と掃引の切れ目（時刻の隙間。しきい値は
// 発射間隔の中央値の 20 倍で、gps_time から測る）が
// 再現できるので、順序と区切りには副情報が要らない。
// 1 掃引に 2 台のスキャナが混ざる場合だけ、どちらの走査線かの標識を送る。

namespace {

struct ScanCtx {
    std::vector<int32_t> ord;        // 走査順 → 格納順
    // 時刻は元の列を ord 経由で引く。並べ替えた複製を持つと 400 万点で 32 MB
    // 余計に要る（値は同じなので、指す先を変えるだけ）。
    const Col* gcol = nullptr;
    inline int64_t gps(int32_t t) const { return (*gcol)[ord[t]]; }
    std::vector<uint8_t> ret;        // 走査順の戻り番号（4 ビットしか使わない）
    std::vector<int32_t> swp;        // 掃引の先頭位置（末尾に n）
    int64_t gap_thr = 200;           // 掃引を切る時刻の隙間（データから測る）
};

static bool scan_names(const std::vector<uint8_t>& p, std::string& g,
                       std::string& s, std::string& r) {
    if (p.empty()) return false;
    size_t q = 1;
    auto rd = [&](std::string& out) {
        if (q + 2 > p.size()) return false;
        uint16_t l; memcpy(&l, p.data() + q, 2); q += 2;
        if (q + l > p.size()) return false;
        out.assign((const char*)p.data() + q, l); q += l;
        return true;
    };
    return rd(g) && rd(s) && rd(r);
}

struct ScanCache {
    std::mutex mu;
    std::map<std::string, std::shared_ptr<const ScanCtx>> by_key;
};

// 走査モデルの文脈を作る。**同じ列名・同じ点数なら中身は同じ**なので、
// 変種ごとに作り直さずに共有する。走査順の安定ソートが 1 回で済み、
// 同じ配列を候補の本数だけ抱えることもなくなる。
static std::shared_ptr<const ScanCtx> build_scan_ctx(
        const CodecCtx* ctx, const std::vector<uint8_t>& param,
        size_t n, std::string& err) {
    std::string ng, np, nr;
    if (!scan_names(param, ng, np, nr)) { err = "走査モデルの列名が壊れている"; return nullptr; }
    if (!ctx || !ctx->fr) { err = "副次情報がない"; return nullptr; }
    const std::string key = ng + "\x1f" + np + "\x1f" + nr + "\x1f" + std::to_string(n);
    std::shared_ptr<ScanCache> cache;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        if (!ctx->scan_cache) ctx->scan_cache = std::make_shared<ScanCache>();
        cache = std::static_pointer_cast<ScanCache>(ctx->scan_cache);
    }
    {
        std::lock_guard<std::mutex> lk(cache->mu);
        auto it = cache->by_key.find(key);
        if (it != cache->by_key.end()) return it->second;
    }
    auto made = std::make_shared<ScanCtx>();
    ScanCtx& sc = *made;
    const auto* g = ctx->fr->get(ng);
    const auto* s = ctx->fr->get(np);
    const auto* r = nr.empty() ? nullptr : ctx->fr->get(nr);
    if (!g || !s || g->size() < n || s->size() < n) {
        err = "gps_time / point_source_id が先に復号されていない"; return nullptr;
    }
    sc.ord.resize(n);
    for (size_t i = 0; i < n; ++i) sc.ord[i] = (int32_t)i;
    std::stable_sort(sc.ord.begin(), sc.ord.end(), [&](int32_t a, int32_t b) {
        if ((*s)[a] != (*s)[b]) return (*s)[a] < (*s)[b];
        return (*g)[a] < (*g)[b];
    });
    sc.gcol = g;
    sc.ret.assign(n, (uint8_t)1);
    if (r && r->size() >= n)
        for (size_t t = 0; t < n; ++t) sc.ret[t] = (uint8_t)((*r)[sc.ord[t]] & 0x0F);
    // 掃引の切れ目は時刻の隙間で決める。しきい値は発射間隔の中央値の 20 倍とし、
    // データから測る。gps_time の刻みは記録の仕方で桁が変わる（AHN4 は ulp 単位の
    // 10 刻み、AHN3 は GPS 週秒で約 43000 ulp）ので、定数では片方が必ず壊れる。
    // 復号器も同じ gps_time から同じ値を得るため、副情報は要らない。
    {
        std::vector<int64_t> gap;
        size_t stride = n > (1u << 21) ? n / (1u << 21) : 1;
        for (size_t t = 1; t < n; t += stride) {
            if ((*s)[sc.ord[t - 1]] != (*s)[sc.ord[t]]) continue;
            int64_t d = sc.gps((int32_t)t) - sc.gps((int32_t)(t - 1));
            if (d > 0) gap.push_back(d);
        }
        int64_t med = 10;
        if (!gap.empty()) {
            size_t h = gap.size() / 2;
            std::nth_element(gap.begin(), gap.begin() + h, gap.end());
            med = gap[h];
        }
        sc.gap_thr = med > 0 ? med * 20 : 200;
    }
    sc.swp.clear();
    for (size_t t = 0; t < n; ++t) {
        bool brk = (t == 0);
        if (!brk) {
            int32_t a = sc.ord[t - 1], b = sc.ord[t];
            brk = ((*s)[a] != (*s)[b]) || (sc.gps(t) - sc.gps(t - 1) > sc.gap_thr);
        }
        if (brk) sc.swp.push_back((int32_t)t);
    }
    sc.swp.push_back((int32_t)n);
    {
        std::lock_guard<std::mutex> lk(cache->mu);
        auto it = cache->by_key.find(key);
        if (it != cache->by_key.end()) return it->second;   // 競って作った。先着を使う
        cache->by_key[key] = made;
    }
    return made;
}

// 走査線 1 本を符号化したときの面内・面外の費用［bit］。
// 分割するかどうかを推定ではなく実測で決めるために使う。
// 予測の手順は下の符号化ループと同じで、z は分割の影響を受けないので数えない。
static double line_cost(const SweepParam& sp, const std::vector<int64_t>& bx,
                        const std::vector<int64_t>& by, const std::vector<int64_t>& bz,
                        const std::vector<int64_t>& bg) {
    size_t m = bx.size();
    if (!m) return 0.0;
    MedPred mps, mpo;
    mps.prev = sp.s0; mpo.prev = sp.off0;
    double bits = 0;
    for (size_t i = 0; i < m; ++i) {
        int64_t s, off;
        shear_fwd(bx[i], by[i], sp.t2, sp.sn, s, off);
        int64_t ps = sp.ok ? scan_predict(sp, bg[i] - bg[0], bz[i]) : mps.predict();
        bits += bitlen(zigzag(s - ps)) + bitlen(zigzag(off - mpo.predict()));
        mps.push(s); mpo.push(off);
    }
    return bits;
}

// 掃引ごとに 1 本か 2 本の走査線を割り当てた結果
struct LineSet {
    std::vector<uint8_t> split;              // 掃引ごと 0/1
    std::vector<uint8_t> label;              // 走査順の点ごと 0/1（分割した掃引のみ意味を持つ）
    std::vector<SweepParam> line;            // 掃引順・ラベル順に並ぶ
    std::vector<int32_t> line_of_sweep;      // 掃引 k の line 配列での開始位置
};

// 走査順の位置 t が属する走査線と、その線の先頭位置
static inline int line_index(const LineSet& L, size_t k, size_t t) {
    return L.line_of_sweep[k] + (L.split[k] ? L.label[t] : 0);
}

static void put_u64(std::vector<uint8_t>& o, uint64_t v) {
    const uint8_t* p = (const uint8_t*)&v;
    o.insert(o.end(), p, p + 8);
}

static void delta_ip(std::vector<int64_t>& v) {
    int64_t prev = 0;
    for (auto& x : v) { int64_t t = x; x = t - prev; prev = t; }
}
static void undelta_ip(std::vector<int64_t>& v) {
    int64_t acc = 0;
    for (auto& x : v) { acc += x; x = acc; }
}

} // namespace


// ---- 符号化: 走査線を当てはめ、副情報・標識・残差の 3 つを並べて書く
static bool enc_geom_scan(const std::vector<const Col*>& cols,
                          const std::vector<uint8_t>& param, std::vector<uint8_t>& out,
                          std::string& err, const CodecCtx* ctx) {
    if (cols.size() != 3) { err = "走査モデルは 3 軸"; return false; }
    size_t n = cols[0]->size();
    const bool prof = getenv("PCC_SCAN_PROF") != nullptr;
    double tp = now_sec();
    auto mark = [&](const char* what) {
        if (prof) { fprintf(stderr, "    [走査] %-22s %6.2fs\n", what, now_sec() - tp); tp = now_sec(); }
    };
    auto scp = build_scan_ctx(ctx, param, n, err);
    if (!scp) return false;
    const ScanCtx& sc = *scp;
    mark("並べ替えと掃引の切り出し");
    const auto &CX = *cols[0], &CY = *cols[1], &CZ = *cols[2];
    const int var = param.empty() ? 1 : param[0];
    const bool use_med = (var == 2 || var == 4);     // z を中央値予測にする
    const bool use_xctx = (var == 3 || var == 4);    // z の文脈を面内・面外から作る

    size_t nsw = sc.swp.size() - 1;
    LineSet L;
    L.split.assign(nsw, 0);
    L.label.assign(n, 0);
    L.line_of_sweep.assign(nsw, 0);

    std::vector<int64_t> bx, by, bz, bg, bs;
    const auto* sa_col = ctx->fr->get("scan_angle");
    // 走査線を 2 本に分ける試みは、掃引ごとに EM と当てはめ 2 回を回すので重い。
    // 分けるべき掃引は主軸から見た散らばり（thinness）が大きい。先頭 200 本の
    // 中央値を基準にして、その C 倍を超える掃引だけ試す。C はファイルの座標の
    // 刻みに依らない。**この足切りは符号化側だけの判断**で、分けた結果は
    // 流れに書かれるので復号側には影響しない。
    static const double SPLIT_THIN = [] {
        const char* e = getenv("PCC_SPLIT_THIN");
        // 1.5 は掃引で決めた。17 件すべてで bpp が変わらない上限であり、
        // 走査変換 1 本が AHN3 _20 で 0.09 → 0.08 s、AHN4 _20 で 0.10 → 0.08 s になる。
        return e ? atof(e) : 1.5;              // 0 なら足切りしない
    }();
    std::vector<double> thin_hist;
    double thin_gate = -1.0;
    // 鍵が定数の入力では掃引が 1 本しかできない。ファイルの大半を占める「掃引」は
    // 走査線ではないので、2 本に分ける試み（EM と当てはめ 2 回）は無駄が大きい。
    static const double SPLIT_MAXFRAC = [] {
        const char* e = getenv("PCC_SPLIT_MAXFRAC");
        // 0.25 は掃引で決めた。17 件すべてで bpp が変わらない。
        return e ? atof(e) : 0.25;             // 0 なら上限なし
    }();
    const size_t split_max = SPLIT_MAXFRAC > 0 ? (size_t)(SPLIT_MAXFRAC * (double)n) : 0;
    if (prof) fprintf(stderr, "    [走査] 掃引 %zu 本\n", nsw);
    // PCC_SCAN_DUMP=<path> を付けたときだけ、走査線ごとの当てはめの様子を書き出す。
    // 符号化の結果には影響しない。
    FILE* dump = nullptr; FILE* dump_raw = nullptr;
    int n_fat_dumped = 0;
    if (const char* dp = getenv("PCC_SCAN_DUMP")) {
        dump = fopen(dp, "w");
        if (dump) {
            fprintf(dump, "id\tn\tspan_deg\tthin_mm\theight_m\tomega_deg_s"
                          "\tmdl_bits\talt_bits\tok\tswp_r0_mm\tswp_w_mm\tsplit_code");
            for (int i = 0; i < FIT_ITERS; ++i) fprintf(dump, "\ti%d", i);
            fprintf(dump, "\n");
        }
        std::string rp = std::string(dp) + ".raw";
        dump_raw = fopen(rp.c_str(), "w");
        if (dump_raw) fprintf(dump_raw, "# x\ty\tz\tshot_ulp\tscan_angle\n");
    }
    // 掃引ごとの当てはめは互いに独立である（触るのは自分の区間の L.label と
    // L.split[k] だけ）。結果を一度配列に貯め、L.line への追加だけを後で順に
    // 行えば、並列にしても出力は 1 本のときと同じになる。
    std::vector<std::array<SweepParam, 2>> sp_out(nsw);
    std::vector<int> nl_out(nsw, 1);
    // warm が真のときだけ thin_hist に積む。並列に回す間は基準が決まっており、
    // 共有の vector に push_back すると壊れる（実際に double free を出した）。
    auto do_sweep = [&](size_t k, bool warm, std::vector<int64_t>& bx, std::vector<int64_t>& by,
                        std::vector<int64_t>& bz, std::vector<int64_t>& bg,
                        std::vector<int64_t>& bs) {
        int32_t a = sc.swp[k], b = sc.swp[k + 1];
        size_t m = (size_t)(b - a);
        auto gather = [&](uint8_t want, bool use_label) {
            bx.clear(); by.clear(); bz.clear(); bg.clear(); bs.clear();
            for (int32_t t = a; t < b; ++t) {
                if (use_label && L.label[t] != want) continue;
                int32_t i = sc.ord[t];
                bx.push_back(CX[i]); by.push_back(CY[i]); bz.push_back(CZ[i]);
                bg.push_back(sc.gps(t));
                bs.push_back(sa_col && sa_col->size() >= n ? (*sa_col)[i] : 0);
            }
        };
        double sp_r0 = 0, sp_w = 0;
        int sp_code = 0;                 // 0 試みず / 1 EM が偏った / 2 費用で棄却 / 3 分割した
        FitDiag fd1;
        SweepParam one;
        gather(0, false);
        if (!bx.empty()) {
            sp_r0 = line_thinness(bx.data(), by.data(), m);
            one = fit_scan_line(bx.data(), by.data(), bz.data(), bg.data(), bs.data(),
                                bx.size(), &fd1);
        }
        double cost1 = line_cost(one, bx, by, bz, bg);

        // 2 本に分けたほうが短いかを、推定ではなく実際の符号長で決める。
        // 標識は 1 点 1 bit を上限として見込む（実際は文脈符号化でこれより安い）。
        SweepParam two[2];
        FitDiag fd2[2];
        if (warm && SPLIT_THIN > 0 && sp_r0 < 1e17) thin_hist.push_back(sp_r0);
        // 基準を作るのに使った先頭 200 本には足切りを掛けない。順に回していた
        // ときと同じ判断にするため（掛けると AHN4 _20 が +0.04% 伸びた）。
        const bool gated = (k >= 200) && thin_gate >= 0;
        if (m >= 40 && (!split_max || m <= split_max) && (!gated || sp_r0 > thin_gate)) {
            std::vector<uint8_t> lab;
            split_two_lines(bx.data(), by.data(), m, lab);
            size_t c1 = 0; for (auto v : lab) c1 += v;
            if (c1 < 10 || m - c1 < 10) {
                sp_code = 1;
            } else {
                for (size_t i = 0; i < m; ++i) L.label[a + i] = lab[i];
                double cost2 = 0;
                for (int g = 0; g < 2; ++g) {
                    gather((uint8_t)g, true);
                    two[g] = bx.empty() ? SweepParam()
                        : fit_scan_line(bx.data(), by.data(), bz.data(), bg.data(),
                                        bs.data(), bx.size(), &fd2[g]);
                    cost2 += line_cost(two[g], bx, by, bz, bg);
                }
                sp_w = cost2 / (double)m;
                if (cost2 + (double)m < cost1) {
                    sp_code = 3;
                    L.split[k] = 1;
                } else {
                    sp_code = 2;
                    for (size_t i = 0; i < m; ++i) L.label[a + i] = 0;
                }
            }
        }

        int nl = L.split[k] ? 2 : 1;
        nl_out[k] = nl;
        for (int g = 0; g < nl; ++g) {
            SweepParam sp = L.split[k] ? two[g] : one;
            FitDiag fd = L.split[k] ? fd2[g] : fd1;
            sp_out[k][g] = sp;
            if (dump) gather((uint8_t)g, L.split[k] != 0);
            if (dump && !bx.empty()) {
                size_t id = L.line.size();
                fprintf(dump, "%zu\t%zu\t%.3f\t%.4f\t%.1f\t%.1f\t%.4f\t%.4f\t%d"
                              "\t%.4f\t%.4f\t%d",
                        id, bx.size(), fd.span_deg, fd.thin_mm, fd.height_m,
                        fd.omega_deg_s, fd.mdl_bits, fd.alt_bits, (int)sp.ok,
                        sp_r0, sp_w, sp_code);
                for (int it = 0; it < FIT_ITERS; ++it) fprintf(dump, "\t%.4f", fd.iter_bits[it]);
                fprintf(dump, "\n");
                // 50 本に 1 本は生データも出す。Python 側で同じ走査線に
                // scipy の当てはめを掛け、C++ の結果と直接比べるために使う。
                // PCC_SCAN_DUMP_ALL を付けると全本出す。隣り合う走査線どうしを
                // 比べたいときに要る（50 本に 1 本では隣が手に入らない）。
                static const bool DUMP_ALL = getenv("PCC_SCAN_DUMP_ALL") != nullptr;
                bool fat_fail = (sp_code == 2 && sp_r0 > 1000.0 && n_fat_dumped < 40);
                if (dump_raw && (DUMP_ALL || (id % 50) == 0 || fat_fail)) {
                    if (fat_fail) ++n_fat_dumped;
                    fprintf(dump_raw, "# line %zu n %zu code %d r0 %.1f\n",
                            id, bx.size(), sp_code, sp_r0);
                    for (size_t i = 0; i < bx.size(); ++i)
                        fprintf(dump_raw, "%lld\t%lld\t%lld\t%lld\t%lld\n",
                                (long long)bx[i], (long long)by[i], (long long)bz[i],
                                (long long)(bg[i] - bg[0]), (long long)bs[i]);
                }
            }
        }
    };

    // 足切りの基準は先頭 200 本の散らばりから決める。散らばりは O(m) で
    // 当てはめを伴わないので、当てはめを 1 本も順に回さずに基準を作れる。
    if (SPLIT_THIN > 0) {
        size_t kw = nsw < 200 ? nsw : (size_t)200;
        for (size_t k = 0; k < kw; ++k) {
            int32_t a = sc.swp[k], b = sc.swp[k + 1];
            bx.clear(); by.clear();
            for (int32_t t = a; t < b; ++t) {
                int32_t i = sc.ord[t];
                bx.push_back(CX[i]); by.push_back(CY[i]);
            }
            if (bx.empty()) continue;
            double r0 = line_thinness(bx.data(), by.data(), (size_t)(b - a));
            if (r0 < 1e17) thin_hist.push_back(r0);
        }
        if (!thin_hist.empty()) {
            std::vector<double> v = thin_hist;
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            thin_gate = v[v.size() / 2] * SPLIT_THIN;
        }
    }
    size_t kwarm = dump ? nsw : (size_t)0;       // 書き出しは順序に依るので並列にしない
    for (size_t k = 0; k < kwarm; ++k) do_sweep(k, true, bx, by, bz, bg, bs);
    if (kwarm < nsw) {
        static const size_t NT = [] {
            if (const char* e = getenv("PCC_THREADS")) { long v = atol(e); if (v > 0) return (size_t)v; }
            unsigned hw = std::thread::hardware_concurrency();
            return (size_t)(hw ? hw : 1);
        }();
        size_t nt = NT;
        if (nt > nsw - kwarm) nt = nsw - kwarm;
        if (nt <= 1) {
            for (size_t k = kwarm; k < nsw; ++k) do_sweep(k, false, bx, by, bz, bg, bs);
        } else {
            std::atomic<size_t> next{kwarm};
            std::vector<std::thread> th;
            th.reserve(nt);
            for (size_t t = 0; t < nt; ++t)
                th.emplace_back([&] {
                    std::vector<int64_t> lx, ly, lz2, lg, ls;
                    // 掃引を 1 本ずつ取ると、隣り合う掃引が別のスレッドに散って
                    // L.label と sp_out の同じキャッシュ行を取り合う。16 本ずつ取る。
                    const size_t CH = 16;
                    for (size_t k0 = next.fetch_add(CH); k0 < nsw; k0 = next.fetch_add(CH)) {
                        size_t k1 = k0 + CH < nsw ? k0 + CH : nsw;
                        for (size_t k = k0; k < k1; ++k)
                            do_sweep(k, false, lx, ly, lz2, lg, ls);
                    }
                });
            for (auto& x : th) x.join();
        }
    }
    for (size_t k = 0; k < nsw; ++k) {
        L.line_of_sweep[k] = (int32_t)L.line.size();
        for (int g = 0; g < nl_out[k]; ++g) L.line.push_back(sp_out[k][g]);
    }

    mark("走査線の分割と当てはめ");
    if (dump) { fclose(dump); dump = nullptr; }
    if (dump_raw) { fclose(dump_raw); dump_raw = nullptr; }

    // 変種 5: 掃引の分割と走査線ごとの回転だけを使い、当てはめた曲線は使わない。
    // 使わないパラメタを 0 にすると、差分をとった副情報からほぼ消える。
    if (var == 5)
        for (auto& ln : L.line) { ln.ok = 0; ln.Sz = 0; ln.th0 = 0; ln.om = 0; }

    // 残差は座標の差なので実際には 32 bit に収まる。まず int32 で持ち、
    // 収まらない点が出たらその列だけ int64 に退避する（符号化する値は同じなので
    // 出力は 1 バイトも変わらない）。400 万点では 48 MB の節約になる。
    std::vector<std::vector<int32_t>> res32(3, std::vector<int32_t>(n, 0));
    std::vector<std::vector<int64_t>> res64;      // 溢れたときだけ使う
    bool wide = false;
    auto put_res = [&](int c, size_t t, int64_t v) {
        if (!wide) {
            if (v >= INT32_MIN && v <= INT32_MAX) { res32[c][t] = (int32_t)v; return; }
            // 溢れた。ここまでの分を 64 bit に写してから続ける。
            res64.assign(3, std::vector<int64_t>(n, 0));
            for (int cc = 0; cc < 3; ++cc)
                for (size_t k = 0; k < n; ++k) res64[cc][k] = res32[cc][k];
            res32.clear(); res32.shrink_to_fit();
            wide = true;
        }
        res64[c][t] = v;
    };
    // PCC_RESID_DUMP=<path> を付けたときだけ、符号化器が実際に払う残差を書き出す。
    // 当てはめを外で再現すると（Python で組み直すと）別のものを測ってしまうので、
    // 天井を測るときはここから出した値を使う。符号化の結果には影響しない。
    FILE* rd = nullptr;
    if (const char* rp = getenv("PCC_RESID_DUMP")) {
        rd = fopen(rp, "w");
        if (rd) fprintf(rd, "line\tok\tx\ty\tz\tr_in\tr_off\tr_z"
                            "\tr_mdl\tr_alt\n");
    }
    // z は「同じ掃引・同じ戻り番号の直前の点」から予測する。
    // その系列がまだ空なら、直前に符号化した任意の点の z を使う（掃引をまたぐ）。
    int64_t last_any = 0;
    for (size_t k = 0; k < nsw; ++k) {
        int32_t a = sc.swp[k], b = sc.swp[k + 1];
        int64_t lz[16]; bool hz[16]; MedPred mz[16];
        for (int i = 0; i < 16; ++i) { lz[i] = 0; hz[i] = false; mz[i] = MedPred(); }
        int64_t g0[2] = {0, 0};
        bool hg[2] = {false, false};
        // 面外と、モデルを採らない走査線の面内は中央値予測（幾何v1 と同じ）。
        // 走査線の先頭は副情報の s0 / off0 から始める。
        MedPred mps[2], mpo[2];
        for (int32_t t = a; t < b; ++t) {
            int li = line_index(L, k, (size_t)t);
            const SweepParam& sp = L.line[li];
            int gidx = (L.split[k] && L.label[t]) ? 1 : 0;
            if (!hg[gidx]) {
                g0[gidx] = sc.gps(t); hg[gidx] = true;
                mps[gidx] = MedPred(); mps[gidx].prev = sp.s0;
                mpo[gidx] = MedPred(); mpo[gidx].prev = sp.off0;
            }
            int32_t i = sc.ord[t];
            int64_t s, off;
            shear_fwd(CX[i], CY[i], sp.t2, sp.sn, s, off);
            int64_t po = mpo[gidx].predict();
            // 面内には予測子が 2 つある。当てはめ（走査モデル）と、素朴な
            // 中央値予測（退避路）。今は走査線ごとにどちらか一方を選んでいる。
            // 当てはめを採らない走査線では scan_predict を呼ばない。
            // 診断のときだけ、比べるために両方を出す。
            int64_t ps_alt = mps[gidx].predict();
            int64_t ps_mdl = (sp.ok || rd)
                ? scan_predict(sp, sc.gps(t) - g0[gidx], CZ[i]) : ps_alt;
            int64_t ps = sp.ok ? ps_mdl : ps_alt;
            put_res(0, (size_t)t, s - ps);
            put_res(1, (size_t)t, off - po);
            mps[gidx].push(s); mpo[gidx].push(off);
            int r = (int)sc.ret[t] & 15;
            if (!hz[r]) { mz[r] = MedPred(); mz[r].prev = last_any; }
            int64_t pz = use_med ? mz[r].predict() : (hz[r] ? lz[r] : last_any);
            put_res(2, (size_t)t, CZ[i] - pz);
            if (rd) fprintf(rd, "%d\t%d\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld"
                                "\t%lld\t%lld\n",
                            li, (int)sp.ok, (long long)CX[i], (long long)CY[i],
                            (long long)CZ[i], (long long)(s - ps),
                            (long long)(off - po), (long long)(CZ[i] - pz),
                            (long long)(s - ps_mdl), (long long)(s - ps_alt));
            mz[r].push(CZ[i]);
            lz[r] = CZ[i]; hz[r] = true; last_any = CZ[i];
        }
    }
    if (rd) fclose(rd);

    std::vector<int64_t> flags(nsw);
    for (size_t k = 0; k < nsw; ++k) flags[k] = L.split[k];
    size_t nl = L.line.size();
    std::vector<int64_t> pars;
    pars.reserve(nl * 8);
    auto push_series = [&](int64_t SweepParam::*f) {
        std::vector<int64_t> v(nl);
        for (size_t i = 0; i < nl; ++i) v[i] = L.line[i].*f;
        delta_ip(v);
        pars.insert(pars.end(), v.begin(), v.end());
    };
    push_series(&SweepParam::t2);  push_series(&SweepParam::sn);
    push_series(&SweepParam::s0);  push_series(&SweepParam::Sz);
    push_series(&SweepParam::th0); push_series(&SweepParam::om);
    push_series(&SweepParam::off0);
    { std::vector<int64_t> v(nl);
      for (size_t i = 0; i < nl; ++i) v[i] = L.line[i].ok;
      pars.insert(pars.end(), v.begin(), v.end()); }

    std::vector<int64_t> labs;
    for (size_t k = 0; k < nsw; ++k)
        if (L.split[k]) for (int32_t t = sc.swp[k]; t < sc.swp[k + 1]; ++t) labs.push_back(L.label[t]);

    auto bf = encode_ints(flags.data(), flags.size());
    auto bp = encode_ints(pars.data(), pars.size());
    std::vector<uint8_t> bl;
    if (!labs.empty()) {
        Encoder e; UIntCoder uc(2, 2);
        int prev = 0;
        for (int64_t v : labs) { uc.encode(e, (uint64_t)v, prev); prev = (int)v; }
        bl = e.finish();
    }
    std::vector<uint8_t> br;
    mark("残差を作る");
    // 走査文脈はここから先で参照しないが、**いまは候補どうしで共有しているので
    // 解放しない。**1 本ぶん（200 万点で ord 8 MB + ret 2 MB）を抱え続ける代わりに、
    // 候補の本数だけ作り直して同時に持つことがなくなる。
    // 走査線の標識は候補ごとの持ち物なので、これは手放す。
    L.label.clear(); L.label.shrink_to_fit();
    // 残差は走査の順に並ぶ。旗「類」の類は点の番号で引くので、走査の順 → 格納の順の表を渡す
    // （渡さないと別の点の類を当てることになり、往復は壊れないが文脈としてほとんど効かない）。
    if (g_cls) g_resid_ord = &sc.ord;
    if (wide) { if (use_xctx) enc_resid_x(res64, br); else enc_resid(res64, br); }
    else       { if (use_xctx) enc_resid_x(res32, br); else enc_resid(res32, br); }
    mark("残差を符号化");

    // PCC_SCAN_DEBUG=1 で内訳を出す。走査モデルが効く規模と効かない規模を
    // 切り分けるための診断で、符号化の結果には影響しない。
    if (getenv("PCC_SCAN_DEBUG")) {
        size_t n_ok = 0, n_split = 0;
        for (size_t k = 0; k < nsw; ++k) n_split += L.split[k];
        for (const auto& ln : L.line) n_ok += ln.ok ? (size_t)ln.count : 0;
        // 採用した走査線と退避した走査線で、残差の大きさがどう違うか。
        // 符号長そのものは 1 本のレンジ符号にまとめているので分けられない。
        // zigzag のビット長の平均を代理として使う（1 点 3 軸ぶんの合計）。
        double sok[3] = {0, 0, 0}, sng[3] = {0, 0, 0};
        size_t c_ok = 0, c_ng = 0;
        for (size_t k2 = 0; k2 < nsw; ++k2)
            for (int32_t t = sc.swp[k2]; t < sc.swp[k2 + 1]; ++t) {
                bool okl = L.line[line_index(L, k2, (size_t)t)].ok != 0;
                for (int c = 0; c < 3; ++c)
                    (okl ? sok : sng)[c] += bitlen(zigzag(
                        wide ? res64[c][t] : (int64_t)res32[c][t]));
                if (okl) ++c_ok; else ++c_ng;
            }
        fprintf(stderr,
            "       残差のビット長（成分別）\n"
            "         採用した線 (%zu 点): 面内 %.2f  面外 %.2f  z %.2f  計 %.2f\n"
            "         退避した線 (%zu 点): 面内 %.2f  面外 %.2f  z %.2f  計 %.2f\n",
            c_ok, c_ok ? sok[0]/c_ok : 0.0, c_ok ? sok[1]/c_ok : 0.0,
            c_ok ? sok[2]/c_ok : 0.0, c_ok ? (sok[0]+sok[1]+sok[2])/c_ok : 0.0,
            c_ng, c_ng ? sng[0]/c_ng : 0.0, c_ng ? sng[1]/c_ng : 0.0,
            c_ng ? sng[2]/c_ng : 0.0, c_ng ? (sng[0]+sng[1]+sng[2])/c_ng : 0.0);
        double bpp = 8.0 / (double)n;
        fprintf(stderr,
            "[走査] 点 %zu / 掃引 %zu / 走査線 %zu / 分割 %zu\n"
            "       走査線あたり %.0f 点  モデル採用 %.3f\n"
            "       副情報 %.4f bpp（フラグ %.4f + パラメタ %.4f）\n"
            "       標識 %.4f bpp  残差 %.4f bpp  合計 %.4f bpp\n",
            n, nsw, nl, n_split,
            (double)n / (double)std::max<size_t>(nl, 1),
            (double)n_ok / (double)n,
            (bf.size() + bp.size()) * bpp, bf.size() * bpp, bp.size() * bpp,
            bl.size() * bpp, br.size() * bpp,
            (bf.size() + bp.size() + bl.size() + br.size() + 56) * bpp);
    }

    out.clear();
    put_u64(out, (uint64_t)nsw);
    put_u64(out, (uint64_t)nl);
    put_u64(out, (uint64_t)labs.size());
    put_u64(out, bf.size()); out.insert(out.end(), bf.begin(), bf.end());
    put_u64(out, bp.size()); out.insert(out.end(), bp.begin(), bp.end());
    put_u64(out, bl.size()); out.insert(out.end(), bl.begin(), bl.end());
    put_u64(out, br.size()); out.insert(out.end(), br.begin(), br.end());
    return true;
}

// ---- 復号: 同じ手順で走査順と掃引を組み直し、残差から座標を戻す
static bool dec_geom_scan(const std::vector<uint8_t>& param, const uint8_t* data, size_t len,
                          size_t n, std::vector<std::vector<int64_t>>& out,
                          std::string& err, const CodecCtx* ctx) {
    auto scp = build_scan_ctx(ctx, param, n, err);
    if (!scp) return false;
    const ScanCtx& sc = *scp;
    const int var = param.empty() ? 1 : param[0];
    const bool use_med = (var == 2 || var == 4);
    const bool use_xctx = (var == 3 || var == 4);
    size_t p = 0;
    auto rd64 = [&](uint64_t& v) {
        if (p + 8 > len) return false;
        memcpy(&v, data + p, 8); p += 8; return true;
    };
    uint64_t nsw, nl, nlab, lf, lp, ll, lr;
    if (!rd64(nsw) || !rd64(nl) || !rd64(nlab)) { err = "走査ストリームが短い"; return false; }
    if (nsw != sc.swp.size() - 1) { err = "掃引数が合わない"; return false; }
    // 走査線は掃引ごとに 1 本か 2 本、標識は点ごとに高々 1 つ。
    // 壊れた数で巨大な配列を取らない・範囲外を書かないよう先に確かめる。
    if (nl > 2 * nsw || nlab > n) { err = "走査線・標識の数が掃引と合わない"; return false; }
    if (!rd64(lf) || lf > len - p) { err = "フラグが短い"; return false; }
    std::vector<int64_t> flags(nsw);
    decode_ints(data + p, lf, flags.data(), nsw); p += lf;
    if (!rd64(lp) || lp > len - p) { err = "パラメタが短い"; return false; }
    std::vector<int64_t> pars(nl * 8);
    decode_ints(data + p, lp, pars.data(), nl * 8); p += lp;
    if (!rd64(ll) || ll > len - p) { err = "標識が短い"; return false; }
    std::vector<int64_t> labs(nlab);
    if (nlab) {
        Decoder d(data + p, ll); UIntCoder uc(2, 2);
        int prev = 0;
        // 標識は 0/1。文脈は 2 つしかないので、壊れた値で範囲外を引かないよう 0/1 に縛る。
        for (size_t i = 0; i < nlab; ++i) { labs[i] = (int64_t)uc.decode(d, prev); prev = labs[i] ? 1 : 0; }
    }
    p += ll;
    if (!rd64(lr) || lr > len - p) { err = "残差が短い"; return false; }
    std::vector<std::vector<int64_t>> res;
    double t_a = now_sec();
    if (g_cls) g_resid_ord = &sc.ord;           // 符号化側と同じく、類は走査の順 → 格納の順で引く
    if (use_xctx) dec_resid_x(data + p, lr, n, res); else dec_resid(data + p, lr, n, 3, res);
    if (getenv("PCC_DPROF")) fprintf(stderr, "        [残差復号] %.3fs\n", now_sec() - t_a);
    t_a = now_sec();

    LineSet L;
    L.split.resize(nsw);
    for (size_t k = 0; k < nsw; ++k) L.split[k] = (uint8_t)flags[k];
    L.line.resize(nl);
    { size_t base = 0;
      auto pull = [&](int64_t SweepParam::*f, bool dl) {
          std::vector<int64_t> v(pars.begin() + base, pars.begin() + base + nl);
          if (dl) undelta_ip(v);
          for (size_t i = 0; i < nl; ++i) L.line[i].*f = v[i];
          base += nl;
      };
      pull(&SweepParam::t2, true);  pull(&SweepParam::sn, true);
      pull(&SweepParam::s0, true);  pull(&SweepParam::Sz, true);
      pull(&SweepParam::th0, true); pull(&SweepParam::om, true);
      pull(&SweepParam::off0, true);
      for (size_t i = 0; i < nl; ++i) L.line[i].ok = (uint8_t)pars[base + i];
    }
    L.line_of_sweep.resize(nsw);
    { int32_t c = 0;
      for (size_t k = 0; k < nsw; ++k) { L.line_of_sweep[k] = c; c += L.split[k] ? 2 : 1; }
      if ((size_t)c != nl) { err = "走査線の数が合わない"; return false; } }
    L.label.assign(n, 0);
    { size_t q = 0;
      for (size_t k = 0; k < nsw; ++k)
          if (L.split[k]) for (int32_t t = sc.swp[k]; t < sc.swp[k + 1]; ++t) {
              if (q >= nlab) { err = "標識が足りない"; return false; }
              L.label[t] = (uint8_t)labs[q++];
          } }

    out.assign(3, std::vector<int64_t>(n, 0));
    int64_t last_any = 0;
    for (size_t k = 0; k < nsw; ++k) {
        int32_t a = sc.swp[k], b = sc.swp[k + 1];
        int64_t lz[16]; bool hz[16]; MedPred mz[16];
        for (int i = 0; i < 16; ++i) { lz[i] = 0; hz[i] = false; mz[i] = MedPred(); }
        int64_t g0[2] = {0, 0};
        bool hg[2] = {false, false};
        MedPred mps[2], mpo[2];
        for (int32_t t = a; t < b; ++t) {
            int li = line_index(L, k, (size_t)t);
            const SweepParam& sp = L.line[li];
            int gidx = (L.split[k] && L.label[t]) ? 1 : 0;
            if (!hg[gidx]) {
                g0[gidx] = sc.gps(t); hg[gidx] = true;
                mps[gidx] = MedPred(); mps[gidx].prev = sp.s0;
                mpo[gidx] = MedPred(); mpo[gidx].prev = sp.off0;
            }
            int r = (int)sc.ret[t] & 15;
            if (!hz[r]) { mz[r] = MedPred(); mz[r].prev = last_any; }
            int64_t pz = use_med ? mz[r].predict() : (hz[r] ? lz[r] : last_any);
            int64_t z = pz + res[2][t];
            mz[r].push(z);
            lz[r] = z; hz[r] = true; last_any = z;
            int64_t po = mpo[gidx].predict();
            int64_t ps = sp.ok ? scan_predict(sp, sc.gps(t) - g0[gidx], z)
                               : mps[gidx].predict();
            int64_t s = res[0][t] + ps, off = res[1][t] + po;
            mps[gidx].push(s); mpo[gidx].push(off);
            int64_t X, Y;
            shear_inv(s, off, sp.t2, sp.sn, X, Y);
            int32_t i = sc.ord[t];
            out[0][i] = X; out[1][i] = Y; out[2][i] = z;
        }
    }
    if (getenv("PCC_DPROF")) fprintf(stderr, "        [復元] %.3fs\n", now_sec() - t_a);
    return true;
}

// param の 2 バイト目以降に入っている名前で、既に復号済みの列を引く
// 仮数部のうち上から何ビットを模型に通すか。-1 で従来どおり全部。
// **測ってから既定を決める**ための入口で、最終的には定数にする。
static const int RAWKEEP_ENV = [] {
    const char* e = getenv("PCC_RAWKEEP");
    return e ? atoi(e) : -1;
}();
static inline int rawkeep_env() { return g_decoding ? -1 : RAWKEEP_ENV; }
// 速さの表の差し替え（実験用。PCC_BM_SCHED="1,2,2,3,3,4,5,6,6,7" のように 10 個）。
// 他の実験用の変数と同じく、復号は常に既定の表を使い、pack は許可なしでは断る。
static const std::array<uint8_t, 256> BM_RATE_ENV = [] {
    int sch[10];
    for (int k = 0; k < 10; ++k) sch[k] = BM_SCHED_ADAPT[k];
    if (const char* e = getenv("PCC_BM_SCHED")) {
        int k = 0;
        for (const char* p = e; *p && k < 10; ) {
            char* end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            if (v >= 1 && v <= 12) sch[k++] = (int)v;
            p = (*end == ',') ? end + 1 : end;
        }
        for (; k > 0 && k < 10; ++k) sch[k] = sch[k - 1];
    }
    return make_bm_rate(sch);
}();
static inline const uint8_t* bm_rate_pcc2() {
    return g_decoding ? BM_RATE_ADAPT.data() : BM_RATE_ENV.data();
}
// 旗（C_FAST_BIT）の表の差し替え（実験用。PCC_BM_SCHED2、同じく 10 個）
static const std::array<uint8_t, 256> BM_RATE_ENV2 = [] {
    int sch[10];
    for (int k = 0; k < 10; ++k) sch[k] = BM_SCHED_FAST[k];
    if (const char* e = getenv("PCC_BM_SCHED2")) {
        int k = 0;
        for (const char* p = e; *p && k < 10; ) {
            char* end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            if (v >= 1 && v <= 12) sch[k++] = (int)v;
            p = (*end == ',') ? end + 1 : end;
        }
        for (; k > 0 && k < 10; ++k) sch[k] = sch[k - 1];
    }
    return make_bm_rate(sch);
}();
static inline const uint8_t* bm_rate_flag() {
    return g_decoding ? BM_RATE_FAST.data() : BM_RATE_ENV2.data();
}

static const Col* aux_col_at(const std::vector<uint8_t>& param,
                             const CodecCtx* ctx, size_t off) {
    if (!ctx || !ctx->fr || param.size() < off + 2) return nullptr;
    uint16_t l; memcpy(&l, param.data() + off, 2);
    if (param.size() < off + 2u + l) return nullptr;
    std::string nm((const char*)param.data() + off + 2, l);
    return ctx->fr->get(nm);
}
static const Col* aux_col(const std::vector<uint8_t>& param, const CodecCtx* ctx) {
    return aux_col_at(param, ctx, 1);
}

// 候補ごとに確保していた作業領域。糸ごとに使い回す。
// 候補の仕事は 1 本の糸で最後まで走るので入れ子にならない。
static thread_local std::vector<std::vector<int64_t>> g_res;
static thread_local std::vector<std::vector<int32_t>> g_res32;
// **使わないほうの幅は手放す。**両方抱えると糸ごとに 1.5 倍になり、
// 並列の本数だけ効いてしまう（200 万点・16 並列で 128 MB）。
static void free_scratch64() { for (auto& v : g_res) std::vector<int64_t>().swap(v); }
static void free_scratch32() { for (auto& v : g_res32) std::vector<int32_t>().swap(v); }

// 旗「類」の param は [元の param][列の名前][u16 名前の長さ]。末尾から剥がす。
static bool cls_split(const std::vector<uint8_t>& p, std::vector<uint8_t>& orig, std::string& name) {
    if (p.size() < 2) return false;
    uint16_t l; memcpy(&l, p.data() + p.size() - 2, 2);
    if (l == 0 || (size_t)l + 2 > p.size()) return false;
    name.assign((const char*)p.data() + p.size() - 2 - l, l);
    orig.assign(p.begin(), p.end() - 2 - l);
    return true;
}
static std::vector<uint8_t> cls_join(const std::vector<uint8_t>& orig, const std::string& name) {
    std::vector<uint8_t> p = orig;
    p.insert(p.end(), name.begin(), name.end());
    const uint16_t l = (uint16_t)name.size();
    p.insert(p.end(), (const uint8_t*)&l, (const uint8_t*)&l + 2);
    return p;
}
// 列の値を 4 つの類に分ける。列だけから決まるので、復号側も同じ類を作れる。
//   値の種類が 64 以下: 出現の多い順に上位 3 値を 1〜3、それ以外を 0（同数なら値の小さい順）
//   それより多い      : 値の 4 分位（下から 0〜3）
template <class C>
static void cls_build_t(const C& c, std::vector<uint8_t>& out);
static void cls_build(const Col& c, std::vector<uint8_t>& out) { cls_build_t(c, out); }
// 「Δ列名」は、その列の前の点との差の絶対値で類を分ける（gps_time の跳びなど）。
static bool cls_build_named(const CodecCtx* ctx, const std::string& nm, size_t n, std::vector<uint8_t>& out);
// 文脈にとっておいた類を返す（無ければ作る）。作れなければ空。
static std::shared_ptr<const std::vector<uint8_t>> cls_get(const CodecCtx* ctx, const std::string& nm, size_t n) {
    if (!ctx) return nullptr;
    const auto key = std::make_pair(nm, n);
    {
        std::lock_guard<std::mutex> lk(ctx->cls_mu);
        auto it = ctx->cls_cache.find(key);
        if (it != ctx->cls_cache.end()) return it->second;
    }
    auto v = std::make_shared<std::vector<uint8_t>>();
    if (!cls_build_named(ctx, nm, n, *v)) return nullptr;
    std::lock_guard<std::mutex> lk(ctx->cls_mu);
    auto ins = ctx->cls_cache.emplace(key, v);   // 同時に作った糸がいれば先に入ったほうを使う（中身は同じ）
    return ins.first->second;
}
static bool cls_build_named(const CodecCtx* ctx, const std::string& nm, size_t n, std::vector<uint8_t>& out) {
    static const std::string D = "\xCE\x94";
    const bool diff = nm.compare(0, D.size(), D) == 0;
    const std::string base = diff ? nm.substr(D.size()) : nm;
    const Col* rc = (ctx && ctx->fr) ? ctx->fr->get(base) : nullptr;
    if (!rc || rc->size() != n) return false;
    if (!diff) { cls_build(*rc, out); return true; }
    std::vector<int64_t> d(n, 0);
    for (size_t i = 1; i < n; ++i) {
        const int64_t a = (*rc)[i], b = (*rc)[i - 1];
        const uint64_t u = a >= b ? (uint64_t)a - (uint64_t)b : (uint64_t)b - (uint64_t)a;
        d[i] = u > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)u;
    }
    cls_build_t(d, out);
    return true;
}
template <class C>
static void cls_build_t(const C& c, std::vector<uint8_t>& out) {
    const size_t n = c.size();
    out.assign(n, 0);
    std::unordered_map<int64_t, uint32_t> cnt;
    bool many = false;
    for (size_t i = 0; i < n; ++i) {
        const int64_t v = c[i];
        auto it = cnt.find(v);
        if (it != cnt.end()) { ++it->second; continue; }
        if (cnt.size() >= 64) { many = true; break; }
        cnt.emplace(v, 1u);
    }
    if (!many) {
        std::vector<std::pair<uint32_t, int64_t>> fv;
        for (const auto& kv : cnt) fv.push_back({kv.second, kv.first});
        std::sort(fv.begin(), fv.end(), [](const std::pair<uint32_t, int64_t>& a,
                                           const std::pair<uint32_t, int64_t>& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second; });
        std::unordered_map<int64_t, uint8_t> cl;
        for (size_t k = 0; k < fv.size() && k < 3; ++k) cl[fv[k].second] = (uint8_t)(k + 1);
        for (size_t i = 0; i < n; ++i) { auto it = cl.find(c[i]); out[i] = it == cl.end() ? 0 : it->second; }
        return;
    }
    std::vector<int64_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = c[i];
    int64_t q[3];
    for (int k = 0; k < 3; ++k) {
        const size_t at = n * (size_t)(k + 1) / 4;
        std::nth_element(v.begin(), v.begin() + at, v.end());
        q[k] = v[at];
    }
    for (size_t i = 0; i < n; ++i) {
        const int64_t x = c[i];
        out[i] = (uint8_t)((x > q[0]) + (x > q[1]) + (x > q[2]));
    }
}

bool codec_encode(uint16_t id, const std::vector<const Col*>& cols,
                  const std::vector<uint8_t>& param, std::vector<uint8_t>& out,
                  std::string& err, const CodecCtx* ctx) {
    if (cols.empty()) { err = "列がない"; return false; }
    if (id & C_CLS_BIT) {
        if (id & C_RAY_BIT) { err = "類と光は併用しない"; return false; }
        std::vector<uint8_t> orig; std::string nm;
        if (!cls_split(param, orig, nm)) { err = "類の列名が壊れている"; return false; }
        const auto cl = cls_get(ctx, nm, cols[0]->size());
        if (!cl) { err = "類の列が無い: " + nm; return false; }
        g_cls_next = cl.get();
        const bool ok = codec_encode((uint16_t)(id & ~C_CLS_BIT), cols, orig, out, err, ctx);
        g_cls_next = nullptr;
        return ok;
    }
    // **抜けるときに戻す。**enc_best と同じで、設定しっぱなしにすると
    // この関数を通らない経路（encode_ints など）が直前の候補の旗を拾う。
    struct FlagGuard {
        int fsym, raw;
        int mtc, bnd, lms, sgb, ray, srf, dcd;
        const Col* rbf;
        const std::vector<int32_t>* ord;
        const uint8_t* bmr;
        const std::vector<uint8_t>* cls;
        FlagGuard() : fsym(UIntCoder::FSYM), raw(UIntCoder::RAWKEEP),
                      mtc(UIntCoder::MATCH), bnd(UIntCoder::BUNDLE),
                      lms(g_lms), sgb(g_sgnbits), ray(g_ray), srf(g_surf),
                      dcd(g_decoding), rbf(g_ray_bf), ord(g_resid_ord), bmr(g_bm_rate), cls(g_cls) {}
        ~FlagGuard() { UIntCoder::FSYM = fsym; UIntCoder::RAWKEEP = raw;
                       UIntCoder::MATCH = mtc; UIntCoder::BUNDLE = bnd;
                       g_lms = lms; g_sgnbits = sgb; g_ray = ray; g_ray_bf = rbf;
                       g_surf = srf; g_decoding = dcd; g_resid_ord = ord; g_bm_rate = bmr;
                       g_cls = cls; }
    } flag_guard;
    // 旗「類」の外側が置いた類を引き取る（置かれていなければ類は無い）
    g_cls = g_cls_next;
    g_cls_next = nullptr;
    g_resid_ord = nullptr;
    // PCC2 の符号器は出現回数で適応の速さを変える模型を使う（rangecoder.hpp）
    g_decoding = 0;
    g_bm_rate = (id & C_FAST_BIT) ? bm_rate_flag() : bm_rate_pcc2();   // g_decoding を決めてから
    UIntCoder::FSYM = (id & C_FSYM_BIT) ? 1 : 0;
    UIntCoder::RAWKEEP = (id & C_RAW_BIT) ? C_RAW_KEEP : rawkeep_env();
    UIntCoder::MATCH = (id & C_MTC_BIT) ? 1 : 0;
    UIntCoder::BUNDLE = (id & C_BND_BIT) ? 1 : 0;
    {
        const int sv = ((id & C_LMS_BIT) ? 1 : 0) | ((id & C_SGN2_BIT) ? 2 : 0);
        g_lms = sv ? 1 : 0;
        g_sgnbits = (sv == 1) ? 1 : (sv == 2 ? 2 : 4);
    }
    {
        const int sv = ((id & C_SURF_A) ? 1 : 0) | ((id & C_SURF_B) ? 2 : 0);
        g_surf = sv ? 5 + sv : 0;             // 1/2/3 → 格子 2^6 / 2^7 / 2^8
    }
    g_ray = (id & C_RAY_BIT) ? 1 : 0;
    g_ray_bf = nullptr;
    if (g_ray) {
        // 光線モデルは bit_fields（幾何より先に復号される）が要る。無ければ候補にならない。
        const Col* bf = (ctx && ctx->fr) ? ctx->fr->get("bit_fields") : nullptr;
        if (!bf || bf->size() != cols[0]->size()) {
            err = "光線モデルに bit_fields が無い"; return false;
        }
        g_ray_bf = bf;
    }
    id = (uint16_t)(id & ~C_FLAG_MASK);
    switch (id) {
    case C_RAW64: {
        size_t nc = cols.size(), n = cols[0]->size();
        out.resize(nc * n * 8);
        uint8_t* p = out.data();
        for (size_t c = 0; c < nc; ++c) {
            for (size_t i = 0; i < n; ++i) { int64_t v = (*cols[c])[i]; memcpy(p, &v, 8); p += 8; }
        }
        return true;
    }
    case C_RANGE:        enc_cols(cols, 0, out); return true;
    case C_RANGE_DELTA:  enc_cols(cols, 1, out); return true;
    case C_RANGE_CTX:    enc_cols(cols, 2, out); return true;
    case C_RANGE_CTX2:   enc_cols(cols, 3, out); return true;
    case C_RANGE_MED:    enc_geom_med(cols, out); return true;
    case C_RANGE_CACHE:  enc_cols_cache(cols, out); return true;
    case C_RANGE_PREV:   enc_cols_prev(cols, out); return true;
    case C_RANGE_SYM: {
        const Col* rf = nullptr; int sh = 0;
        if (!param.empty() && !sym_ref(param, ctx, rf, sh)) { err = "参照列がない"; return false; }
        if (rf && rf->size() < cols[0]->size()) { err = "参照列が短い"; return false; }
        if (!enc_cols_sym(cols, rf, sh, ctx, out)) { err = "字母が大きすぎる"; return false; }
        return true;
    }
    case C_GEOM_CROSS:
        if (cols.size() != 3) { err = "交差軸は 3 軸"; return false; }
        enc_geom_cross(cols, out); return true;
    case C_GEOM_DIR:
        if (cols.size() != 3) { err = "向き追従は 3 軸"; return false; }
        enc_geom_dir(cols, param.empty() ? 0 : (param[0] & 7), out); return true;
    case C_GEOM_ROT:
        if (cols.size() != 3) { err = "回転は 3 軸"; return false; }
        return enc_geom_rot(cols, param.empty() ? 0 : (param[0] & 7), out, err);
    case C_GEOM_SEG: {
        if (cols.size() != 3) { err = "区間ごとは 3 軸"; return false; }
        const int lg = param.empty() ? 12 : (int)param[0];
        if (lg < 4 || lg > 24) { err = "区間の大きさが範囲外"; return false; }
        return enc_geom_seg(cols, lg, out, err);
    }
    case C_GEOM_LIFT: {
        if (cols.size() != 3) { err = "持ち上げは 3 軸"; return false; }
        int L = param.empty() ? 8 : (int)param[0];
        if (L < 1 || L > 24) { err = "段数が範囲外"; return false; }
        if (cols[0]->size() < ((size_t)1 << L)) { err = "点数が段数に足りない"; return false; }
        enc_geom_lift(cols, L, out); return true;
    }
    case C_RAW_W: {
        const int w = param.empty() ? 8 : (int)param[0];
        const bool sg = param.size() > 1 && param[1];
        if (w < 1 || w > 8) { err = "幅が範囲外"; return false; }
        const size_t nc = cols.size(), n2 = nc ? cols[0]->size() : 0;
        out.clear(); out.reserve(nc * n2 * (size_t)w);
        for (size_t i = 0; i < n2; ++i)
            for (size_t c = 0; c < nc; ++c) {
                const int64_t v = (*cols[c])[i];
                // **入らない値があれば候補ごと外す。**切り詰めて黙って壊さない。
                const uint64_t u = (uint64_t)v;
                int64_t back;
                if (w == 8) back = v;
                else {
                    const uint64_t m = (1ull << (w * 8)) - 1;
                    const uint64_t t = u & m;
                    back = sg ? (int64_t)(t ^ (1ull << (w * 8 - 1))) -
                                (int64_t)(1ull << (w * 8 - 1))
                              : (int64_t)t;
                }
                if (back != v) { err = "幅に入らない値がある"; out.clear(); return false; }
                for (int b2 = 0; b2 < w; ++b2) out.push_back((uint8_t)(u >> (8 * b2)));
            }
        return true;
    }
    case C_GEOM_XYZ: {
        int var = param.empty() ? 0 : param[0];
        if (var == 4) {
            if (cols.size() != 3) { err = "幾何v4 は 3 軸"; return false; }
            size_t w = param.size() > 1 ? (size_t)param[1] : 16;
            enc_geom_w(cols, w ? w : 16, param.size() > 2 ? param[2] : 0, out);
        } else if (var == 2) {
            const Col* a = aux_col(param, ctx);
            if (!a || a->size() < cols[0]->size()) { err = "補助列がない"; return false; }
            enc_geom_aux(cols, *a, out);
        } else if (var == 3) {
            if (cols.size() != 3) { err = "幾何v3 は 3 軸"; return false; }
            const int pm = param.size() > 1 ? param[1] : 0;
            const Col* a = nullptr; int am = 0;
            if (pm & 8) {                       // param = [3][pm][覆い][名前長 u16][名前]
                am = param.size() > 2 ? param[2] : 0;
                a = aux_col_at(param, ctx, 3);
                if (!a || a->size() < cols[0]->size()) { err = "補助列がない"; return false; }
            }
            enc_geom_x(cols, (pm & 3) | ((pm >> 5) & 4), out, (pm & 4) != 0, a, am, (pm >> 4) & 3, (pm & 64) != 0);
        } else if (var == 1) enc_geom_med(cols, out);
        else enc_cols(cols, 2, out);
        return true;
    }
    case C_GEOM_SCAN:
        return enc_geom_scan(cols, param, out, err, ctx);
    case C_ATTR_XREF: {
        // 他の列との残差。その列が先に復号されていれば副情報は生じない。
        // PCC2 は列ごとに独立して符号化するので、周辺費用は加法的であり、
        // LASzip 経路で必要だった「抜いた容器を書いて測る」段は要らない。
        const Col* a = aux_col(param, ctx);
        if (!a || a->size() < cols[0]->size()) { err = "参照列がない"; return false; }
        int P = param[0];
        size_t n = cols[0]->size();
        // 作業領域は糸ごとに使い回す。候補ごとに取ると並列の本数だけ要る。
        // **まず int32 で試す。**列の差も空間予測の残差も普通は収まり、場所が半分で済む。
        // 符号化する値は同じなので出力は 1 バイトも変わらない。
        const std::vector<int32_t>* pm = nullptr;
        const std::vector<int32_t>* pdt = nullptr;
        if (P > 0 && P != 100) {
            pdt = ctx->ensure(n, P, &pm);
            if (!pdt) { err = "座標がない"; return false; }
            g_resid_ord = pm;                   // 残差は符号化順に並ぶ
        }
        {
            // **空間予測を掛けるときは、最初から符号化順に差を作る。**
            // 後から並べ替えると作業配列がもう 1 本要る。
            std::vector<std::vector<int32_t>>& d32 = g_res32;
            d32.resize(1); d32[0].resize(n);
            bool narrow = true;
            if (P > 0 && P != 100) {
                for (size_t t = 0; t < n && narrow; ++t) {
                    size_t i = (size_t)(*pm)[t];
                    int64_t x = (*cols[0])[i] - (*a)[i];
                    if (x < INT32_MIN || x > INT32_MAX) narrow = false;
                    else d32[0][t] = (int32_t)x;
                }
                if (narrow) narrow = residual_backward32(d32[0], *pdt, P, n);
            } else {
                for (size_t i = 0; i < n && narrow; ++i) {
                    int64_t x = (*cols[0])[i] - (*a)[i];
                    if (x < INT32_MIN || x > INT32_MAX) narrow = false;
                    else d32[0][i] = (int32_t)x;
                }
                if (narrow && P == 100) {
                    int32_t prev = 0;
                    for (size_t i = 0; i < n && narrow; ++i) {
                        int64_t r = (int64_t)d32[0][i] - prev;
                        if (r < INT32_MIN || r > INT32_MAX) narrow = false;
                        else { prev = d32[0][i]; d32[0][i] = (int32_t)r; }
                    }
                }
            }
            if (narrow) {
                free_scratch64();
                enc_resid(d32, out);
                return true;
            }
            free_scratch32();
        }
        std::vector<std::vector<int64_t>>& d = g_res;
        d.resize(1); d[0].resize(n);
        if (P > 0 && P != 100) {
            for (size_t t = 0; t < n; ++t) {
                size_t i = (size_t)(*pm)[t];
                d[0][t] = (*cols[0])[i] - (*a)[i];
            }
            residual_backward64(d[0], *pdt, P, n);
        } else {
            for (size_t i = 0; i < n; ++i) d[0][i] = (*cols[0])[i] - (*a)[i];
            if (P == 100) {                   // 残差にさらに格納順の 1 次差分を掛ける
                int64_t prev = 0;
                for (size_t i = 0; i < n; ++i) { int64_t v = d[0][i]; d[0][i] = v - prev; prev = v; }
            }
        }
        enc_resid(d, out);
        return true;
    }
    case C_ATTR_SPATIAL:
    case C_ATTR_COLOR: {
        if (!ctx || param.empty()) { err = "空間予測に必要な副次情報がない"; return false; }
        int P = param[0];
        size_t n = cols[0]->size();
        const std::vector<int32_t>* pm = nullptr;
        const std::vector<int32_t>* pdt = ctx->ensure(n, P, &pm);
        if (!pdt) { err = "座標がない"; return false; }
        g_resid_ord = pm;                       // 残差は符号化順に並ぶ（戻りの種類の文脈に要る）
        // **入力列を複製してはいけない。**候補ごとに n*8 byte を余分に取ることになり、
        // 並列に走る本数だけ積み上がる（200 万点・16 並列で 256 MB）。
        if (id == C_ATTR_COLOR) {
            if (cols.size() != 3) { err = "色は 3 列でなければならない"; return false; }
            std::vector<std::vector<int64_t>>& res = g_res;
            res.resize(3);
            for (auto& v : res) v.resize(n);
            for (size_t t = 0; t < n; ++t) {           // 最初から符号化順に作る
                size_t i = (size_t)(*pm)[t];
                ycocg_fwd((*cols[0])[i], (*cols[1])[i], (*cols[2])[i],
                          res[0][t], res[1][t], res[2][t]);
            }
            for (size_t c = 0; c < 3; ++c) residual_backward64(res[c], *pdt, P, n);
            enc_resid(res, out);
            return true;
        }
        // **残差は普通 int32 に収まる。**収まる間は半分の場所で足りる。
        // 符号化する値は同じなので出力は 1 バイトも変わらない。
        std::vector<std::vector<int32_t>>& r32 = g_res32;
        r32.resize(cols.size());
        bool narrow = true;
        for (size_t c = 0; c < cols.size() && narrow; ++c)
            narrow = spatial_residual32(*cols[c], *pm, *pdt, P, n, r32[c]);
        if (narrow) { free_scratch64(); enc_resid(r32, out); return true; }
        free_scratch32();
        std::vector<std::vector<int64_t>>& res = g_res;
        res.resize(cols.size());
        for (size_t c = 0; c < cols.size(); ++c)
            spatial_residual(*cols[c], *pm, *pdt, P, n, res[c]);
        enc_resid(res, out);
        return true;
    }
    }
    err = "未知の符号器";
    return false;
}

bool codec_decode(uint16_t id, const std::vector<uint8_t>& param,
                  const uint8_t* data, size_t len, size_t n, size_t ncol,
                  std::vector<std::vector<int64_t>>& out, std::string& err,
                  const CodecCtx* ctx) {
    if (id & C_CLS_BIT) {
        if (id & C_RAY_BIT) { err = "類と光は併用しない"; return false; }
        std::vector<uint8_t> orig; std::string nm;
        if (!cls_split(param, orig, nm)) { err = "類の列名が壊れている"; return false; }
        const auto cl = cls_get(ctx, nm, n);
        if (!cl) { err = "類の列が無い: " + nm; return false; }
        g_cls_next = cl.get();
        const bool ok = codec_decode((uint16_t)(id & ~C_CLS_BIT), orig, data, len, n, ncol, out, err, ctx);
        g_cls_next = nullptr;
        return ok;
    }
    // **抜けるときに戻す。**enc_best と同じで、設定しっぱなしにすると
    // この関数を通らない経路（encode_ints など）が直前の候補の旗を拾う。
    struct FlagGuard {
        int fsym, raw;
        int mtc, bnd, lms, sgb, ray, srf, dcd;
        const Col* rbf;
        const std::vector<int32_t>* ord;
        const uint8_t* bmr;
        const std::vector<uint8_t>* cls;
        FlagGuard() : fsym(UIntCoder::FSYM), raw(UIntCoder::RAWKEEP),
                      mtc(UIntCoder::MATCH), bnd(UIntCoder::BUNDLE),
                      lms(g_lms), sgb(g_sgnbits), ray(g_ray), srf(g_surf),
                      dcd(g_decoding), rbf(g_ray_bf), ord(g_resid_ord), bmr(g_bm_rate), cls(g_cls) {}
        ~FlagGuard() { UIntCoder::FSYM = fsym; UIntCoder::RAWKEEP = raw;
                       UIntCoder::MATCH = mtc; UIntCoder::BUNDLE = bnd;
                       g_lms = lms; g_sgnbits = sgb; g_ray = ray; g_ray_bf = rbf;
                       g_surf = srf; g_decoding = dcd; g_resid_ord = ord; g_bm_rate = bmr;
                       g_cls = cls; }
    } flag_guard;
    // 旗「類」の外側が置いた類を引き取る（置かれていなければ類は無い）
    g_cls = g_cls_next;
    g_cls_next = nullptr;
    g_resid_ord = nullptr;
    // PCC2 の符号器は出現回数で適応の速さを変える模型を使う（rangecoder.hpp）
    g_decoding = 1;                       // 実験用の環境変数を読まない
    g_bm_rate = (id & C_FAST_BIT) ? bm_rate_flag() : bm_rate_pcc2();   // g_decoding を決めてから
    UIntCoder::FSYM = (id & C_FSYM_BIT) ? 1 : 0;
    UIntCoder::RAWKEEP = (id & C_RAW_BIT) ? C_RAW_KEEP : rawkeep_env();
    UIntCoder::MATCH = (id & C_MTC_BIT) ? 1 : 0;
    UIntCoder::BUNDLE = (id & C_BND_BIT) ? 1 : 0;
    {
        const int sv = ((id & C_LMS_BIT) ? 1 : 0) | ((id & C_SGN2_BIT) ? 2 : 0);
        g_lms = sv ? 1 : 0;
        g_sgnbits = (sv == 1) ? 1 : (sv == 2 ? 2 : 4);
    }
    {
        const int sv = ((id & C_SURF_A) ? 1 : 0) | ((id & C_SURF_B) ? 2 : 0);
        g_surf = sv ? 5 + sv : 0;             // 1/2/3 → 格子 2^6 / 2^7 / 2^8
    }
    g_ray = (id & C_RAY_BIT) ? 1 : 0;
    g_ray_bf = nullptr;
    if (g_ray) {
        // 復号側でも bit_fields は幾何より前の流れから既に戻っている。
        const Col* bf = (ctx && ctx->fr) ? ctx->fr->get("bit_fields") : nullptr;
        if (!bf || bf->size() != n) { err = "光線モデルに bit_fields が無い"; return false; }
        g_ray_bf = bf;
    }
    id = (uint16_t)(id & ~C_FLAG_MASK);
    switch (id) {
    case C_RAW64: {
        if (len != ncol * n * 8) { err = "RAW64 の長さが合わない"; return false; }
        out.assign(ncol, std::vector<int64_t>(n));
        const uint8_t* p = data;
        for (size_t c = 0; c < ncol; ++c) { memcpy(out[c].data(), p, n * 8); p += n * 8; }
        return true;
    }
    case C_RANGE:        dec_cols(data, len, n, ncol, 0, out); return true;
    case C_RANGE_DELTA:  dec_cols(data, len, n, ncol, 1, out); return true;
    case C_RANGE_CTX:    dec_cols(data, len, n, ncol, 2, out); return true;
    case C_RANGE_CTX2:   dec_cols(data, len, n, ncol, 3, out); return true;
    case C_RANGE_MED:    dec_geom_med(data, len, n, ncol, out); return true;
    case C_RANGE_CACHE:  dec_cols_cache(data, len, n, ncol, out); return true;
    case C_RANGE_PREV:   dec_cols_prev(data, len, n, ncol, out); return true;
    case C_RANGE_SYM: {
        const Col* rf = nullptr; int sh = 0;
        if (!param.empty() && !sym_ref(param, ctx, rf, sh)) { err = "参照列がない"; return false; }
        if (rf && rf->size() < n) { err = "参照列が短い"; return false; }
        return dec_cols_sym(data, len, n, ncol, rf, sh, out, err);
    }
    case C_GEOM_CROSS:
        if (ncol != 3) { err = "交差軸は 3 軸"; return false; }
        dec_geom_cross(data, len, n, out); return true;
    case C_GEOM_DIR:
        if (ncol != 3) { err = "向き追従は 3 軸"; return false; }
        dec_geom_dir(data, len, n, param.empty() ? 0 : (param[0] & 7), out); return true;
    case C_GEOM_ROT:
        if (ncol != 3) { err = "回転は 3 軸"; return false; }
        return dec_geom_rot(data, len, n, param.empty() ? 0 : (param[0] & 7), out, err);
    case C_GEOM_SEG: {
        if (ncol != 3) { err = "区間ごとは 3 軸"; return false; }
        const int lg = param.empty() ? 12 : (int)param[0];
        if (lg < 4 || lg > 24) { err = "区間の大きさが範囲外"; return false; }
        return dec_geom_seg(data, len, n, lg, out, err);
    }
    case C_GEOM_LIFT: {
        if (ncol != 3) { err = "持ち上げは 3 軸"; return false; }
        int L = param.empty() ? 8 : (int)param[0];
        if (L < 1 || L > 24) { err = "段数が範囲外"; return false; }
        if (n < ((size_t)1 << L)) { err = "点数が段数に足りない"; return false; }
        dec_geom_lift(data, len, n, L, out); return true;
    }
    case C_RAW_W: {
        const int w = param.empty() ? 8 : (int)param[0];
        const bool sg = param.size() > 1 && param[1];
        if (w < 1 || w > 8) { err = "幅が範囲外"; return false; }
        if (len < ncol * n * (size_t)w) { err = "流れが短い"; return false; }
        out.assign(ncol, std::vector<int64_t>(n));
        size_t q = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t c = 0; c < ncol; ++c) {
                uint64_t u = 0;
                for (int b2 = 0; b2 < w; ++b2) u |= (uint64_t)data[q++] << (8 * b2);
                if (w == 8) out[c][i] = (int64_t)u;
                else if (sg) out[c][i] = (int64_t)(u ^ (1ull << (w * 8 - 1))) -
                                         (int64_t)(1ull << (w * 8 - 1));
                else out[c][i] = (int64_t)u;
            }
        return true;
    }
    case C_GEOM_XYZ: {
        int var = param.empty() ? 0 : param[0];
        if (var == 4) {
            if (ncol != 3) { err = "幾何v4 は 3 軸"; return false; }
            size_t w = param.size() > 1 ? (size_t)param[1] : 16;
            dec_geom_w(data, len, n, w ? w : 16, param.size() > 2 ? param[2] : 0, out);
        } else if (var == 2) {
            const Col* a = aux_col(param, ctx);
            if (!a || a->size() < n) { err = "補助列がない"; return false; }
            dec_geom_aux(data, len, n, ncol, *a, out);
        } else if (var == 3) {
            if (ncol != 3) { err = "幾何v3 は 3 軸"; return false; }
            const int pm = param.size() > 1 ? param[1] : 0;
            const Col* a = nullptr; int am = 0;
            if (pm & 8) {
                am = param.size() > 2 ? param[2] : 0;
                a = aux_col_at(param, ctx, 3);
                if (!a || a->size() < n) { err = "補助列がない"; return false; }
            }
            dec_geom_x(data, len, n, (pm & 3) | ((pm >> 5) & 4), out, (pm & 4) != 0, a, am, (pm >> 4) & 3, (pm & 64) != 0);
        } else if (var == 1) dec_geom_med(data, len, n, ncol, out);
        else dec_cols(data, len, n, ncol, 2, out);
        return true;
    }
    case C_GEOM_SCAN:
        return dec_geom_scan(param, data, len, n, out, err, ctx);
    case C_ATTR_XREF: {
        const Col* a = aux_col(param, ctx);
        if (!a || a->size() < n) { err = "参照列がない"; return false; }
        int P = param[0];
        // 配列は 1 本で足りる。残差 → 復元 → 参照列を足す、をその場で行う。
        std::vector<std::vector<int64_t>> d;
        // 戻りの種類の文脈を使う流れは、残差をほどく前に順序表が要る
        const std::vector<int32_t>* pm = nullptr;
        const std::vector<int32_t>* pdt = nullptr;
        if (P > 0 && P != 100 && (g_ray || g_cls)) {   // 光・類の文脈は順序表が先に要る
            pdt = ctx->ensure(n, P, &pm);
            if (!pdt) { err = "座標がない"; return false; }
            g_resid_ord = pm;
        }
        dec_resid(data, len, n, 1, d);
        if (P == 100) {
            int64_t acc = 0;
            for (size_t i = 0; i < n; ++i) { acc += d[0][i]; d[0][i] = acc; }
        } else if (P > 0) {
            if (!pdt) pdt = ctx->ensure(n, P, &pm);
            if (!pdt) { err = "座標がない"; return false; }
            spatial_restore(d[0], *pm, *pdt, P, n, d[0]);
        }
        for (size_t i = 0; i < n; ++i) d[0][i] += (*a)[i];
        out = std::move(d);
        return true;
    }
    case C_ATTR_SPATIAL:
    case C_ATTR_COLOR: {
        if (!ctx || param.empty()) { err = "空間予測に必要な副次情報がない"; return false; }
        int P = param[0];
        // **残差をほどくのが先。**近傍表は復元にしか要らない。表を先に作ると、
        // 同じ波で走る他の流れが、表が建つまで残差の復号を始められない。
        // 先にほどけば、表の構築と他の流れの復号が重なる（200 万点で 0.12 s）。
        std::vector<std::vector<int64_t>> res;
        const std::vector<int32_t>* pm = nullptr;
        const std::vector<int32_t>* pdt = nullptr;
        if (g_ray || g_cls) {                   // 戻りの種類・類の文脈は順序表が先に要る
            pdt = ctx->ensure(n, P, &pm);
            if (!pdt) { err = "座標がない"; return false; }
            g_resid_ord = pm;
        }
        dec_resid(data, len, n, ncol, res);
        if (!pdt) pdt = ctx->ensure(n, P, &pm);
        if (!pdt) { err = "座標がない"; return false; }
        for (size_t c = 0; c < ncol; ++c)
            spatial_restore(res[c], *pm, *pdt, P, n, res[c]);   // その場で復元する
        if (id == C_ATTR_COLOR) {
            if (ncol != 3) { err = "色は 3 列でなければならない"; return false; }
            // 3 つとも読んでから書く。同じ配列に書き戻すため。
            for (size_t i = 0; i < n; ++i) {
                int64_t Y = res[0][i], Co = res[1][i], Cg = res[2][i];
                ycocg_inv(Y, Co, Cg, res[0][i], res[1][i], res[2][i]);
            }
        }
        out = std::move(res);
        return true;
    }
    }
    err = "未知の符号器";
    return false;
}

// ---------------------------------------------------------------- 候補を実測して選ぶ

// 参照相手の絞り込み。全対を全点で測ると O(列^2 * 点数) になるので標本で選ぶ。
// 見積りは 25 万個の差を数えて乱雑さを出す。std::map に 25 万回入れると、
// これだけで符号化の 2 割を超えた。差は 0 の近くに集まるので、その範囲は
// 配列で数え、外れ値だけ木に入れる。
// **足す順は木だけのときと同じ（昇順）**にしてあるので、値は 1 bit も変わらない。
namespace {
struct DiffCounter {
    static constexpr int64_t K = 1 << 14;
    std::vector<uint32_t> lo;
    std::map<int64_t, uint32_t> hi;
    DiffCounter() : lo((size_t)(2 * K), 0) {}
    inline void reset() { std::fill(lo.begin(), lo.end(), 0u); hi.clear(); }
    inline void add(int64_t d) {
        if (d >= -K && d < K) ++lo[(size_t)(d + K)]; else ++hi[d];
    }
    double entropy(size_t n) const {
        double e = 0;
        auto take = [&](uint32_t c) { double q = (double)c / n; e -= q * std::log2(q); };
        auto it = hi.begin();
        for (; it != hi.end() && it->first < -K; ++it) take(it->second);
        for (size_t i = 0; i < lo.size(); ++i) if (lo[i]) take(lo[i]);
        for (; it != hi.end(); ++it) take(it->second);
        return e;
    }
};
}
static thread_local DiffCounter g_dc;

static double entropy_diff_sample(const Col& a, const Col& b, size_t cap) {
    size_t n = std::min({a.size(), b.size(), cap});
    if (!n) return 1e30;
    g_dc.reset();
    for (size_t i = 0; i < n; ++i) g_dc.add(a[i] - b[i]);
    return g_dc.entropy(n);
}

// 差にさらに空間予測を掛けたあとの乱雑さ。参照の順位付けに使う。
//
// 上の見積りは「差そのものの乱雑さ」を見ているが、候補は差に空間予測を掛ける。
// **差が乱雑でも空間的に滑らかなら短くなる。**AHN4 _20 の nir がそれで、
// green は差の乱雑さでは 5 位（自分自身の乱雑さより悪い）なのに、
// 空間予測を掛けると全点で 1 位になる（4.289 → 3.736 bpp、全列で −1.23%）。
static double entropy_spatial_diff(const Col& a, const Col& b,
                                   const std::vector<int32_t>& perm,
                                   const std::vector<int32_t>& pred, int P,
                                   size_t n, size_t cap) {
    if (!n || a.size() < n || b.size() < n) return 1e30;
    // 符号化順の先頭 m 個だけで足りる。予測子は自分より前の位置しか指さないので、
    // 先頭を切り出しても残差は全点で計算したものと同じ値になる。
    const size_t m = std::min(n, cap);
    std::vector<int64_t> w(m);
    for (size_t t = 0; t < m; ++t) { int32_t i = perm[t]; w[t] = a[(size_t)i] - b[(size_t)i]; }
    g_dc.reset();
    for (size_t t = 0; t < m; ++t) {
        int64_t p = 0;
        if (t) {
            int64_t sum = 0; int c = 0;
            for (int j = 0; j < P; ++j) {
                int32_t q = pred[t * (size_t)P + j];
                if (q < 0) break;
                sum += w[(size_t)q]; ++c;
            }
            p = (c == 0) ? w[t - 1]
                         : (sum >= 0 ? (sum + c / 2) / c : -((-sum + c / 2) / c));
        }
        g_dc.add(w[t] - p);
    }
    return g_dc.entropy(m);
}

// **書き込む static を使わないこと。**`--trace` のときは列ごとの best_stream が
// プールで並列に走り、ここを同時に呼ぶ。以前は名前の文字列を関数内の static に
// 作って返していたので、糸どうしがヒープ上の同じ文字列を書き換え、
// 「double free or corruption」で落ちることがあった（ThreadSanitizer で 101 件）。
// 読むだけの static（名前の表）は問題ない。
std::string cand_name(uint16_t c, const std::vector<uint8_t>& p) {
    if (c & C_FAST_BIT)
        return cand_name((uint16_t)(c & ~C_FAST_BIT), p) + "速";
    if (c & C_CLS_BIT) {
        std::vector<uint8_t> o; std::string nm;
        if (!cls_split(p, o, nm)) return cand_name((uint16_t)(c & ~C_CLS_BIT), p) + "類?";
        return cand_name((uint16_t)(c & ~C_CLS_BIT), o) + "類<" + nm + ">";
    }
    if (c & C_RAW_BIT)
        return cand_name((uint16_t)(c & ~C_RAW_BIT), p) + "生";
    if (c & C_MTC_BIT)
        return cand_name((uint16_t)(c & ~C_MTC_BIT), p) + "照";
    if (c & C_BND_BIT)
        return cand_name((uint16_t)(c & ~C_BND_BIT), p) + "束";
    if (c & (C_SURF_A | C_SURF_B)) {
        const int sv = ((c & C_SURF_A) ? 1 : 0) | ((c & C_SURF_B) ? 2 : 0);
        return cand_name((uint16_t)(c & ~(C_SURF_A | C_SURF_B)), p) +
               (sv == 1 ? "面6" : (sv == 2 ? "面7" : "面8"));
    }
    if (c & C_RAY_BIT)
        return cand_name((uint16_t)(c & ~C_RAY_BIT), p) + "光";
    if (c & (C_LMS_BIT | C_SGN2_BIT)) {
        const int sv = ((c & C_LMS_BIT) ? 1 : 0) | ((c & C_SGN2_BIT) ? 2 : 0);
        const char* sfx = (sv == 1) ? "符1" : (sv == 2 ? "符2" : "符4");
        return cand_name((uint16_t)(c & ~(C_LMS_BIT | C_SGN2_BIT)), p) + sfx;
    }
    if (c & C_FSYM_BIT)
        return cand_name((uint16_t)(c & ~C_FSYM_BIT), p) + "記";
    char b[32];
    switch (c) {
    case C_RAW64: return "raw64";
    case C_RANGE: return "range";
    case C_RANGE_DELTA: return "delta";
    case C_RANGE_CTX: return "ctx";
    case C_RANGE_CTX2: return "ctx2";
    case C_RANGE_MED: return "med3";
    case C_RANGE_CACHE: return "差分表";
    case C_RANGE_PREV:  return "直前値";
    case C_GEOM_CROSS:  return "交差軸";
    case C_RAW_W:
        snprintf(b, sizeof b, "素%dB", p.empty() ? 8 : (int)p[0]);
        return b;
    case C_GEOM_LIFT:
        snprintf(b, sizeof b, "内挿%d段", p.empty() ? 8 : (int)p[0]);
        return b;
    case C_GEOM_ROT: {
        static const char* MN[8] = {"", "平均4", "半", "中5", "四分", "三四", "無傾", "中5半"};
        snprintf(b, sizeof b, "回転%s", MN[p.empty() ? 0 : (p[0] & 7)]);
        return b;
    }
    case C_GEOM_SEG:
        snprintf(b, sizeof b, "区間2^%d", p.empty() ? 12 : (int)p[0]);
        return b;
    case C_GEOM_DIR: {
        static const char* DN[8] = {"", "平均4", "半", "中5", "四分", "三四", "無傾", "中5半"};
        snprintf(b, sizeof b, "向き%s", DN[p.empty() ? 0 : (p[0] & 7)]);
        return b;
    }
    case C_RANGE_SYM: {
        if (p.empty()) return "字母";
        std::string nm;
        if (p.size() >= 3) { uint16_t l; memcpy(&l, p.data() + 1, 2);
                             if (p.size() >= (size_t)3 + l)
                                 nm.assign((const char*)p.data() + 3, l); }
        // 文脈のずらし（param[0]）を名前に出す。0 と 4 の 2 本が同じ名前で並んでいた。
        std::string r = "字母<" + nm.substr(0, 6) + ">";
        if (p[0]) r += ">>" + std::to_string((int)p[0]);
        return r;
    }
    case C_GEOM_SCAN: {
        int v = p.empty() ? 1 : p[0];
        if (v == 5) return "走査変換";
        snprintf(b, sizeof b, "走査v%d", v);
        return b;
    }
    case C_GEOM_XYZ:
        if (p.empty() || p[0] == 0) return "幾何v0";
        if (p[0] == 4) {
            char b4[32];
            int tm = p.size() > 2 ? p[2] : 0;
            snprintf(b4, sizeof b4, "幾何v4W%d%s", p.size() > 1 ? p[1] : 16,
                     tm == 0 ? "" : (tm == 1 ? "傾" : "傾半"));
            return b4;
        }
        if (p[0] == 3 && p.size() > 1 && p[1]) {
            char b3[64];
            const int pm = p[1];
            char ax[12] = "";
            if (pm & 8) snprintf(ax, sizeof ax, "補%d", p.size() > 2 ? p[2] : 0);
            char cs[8] = "";
            if ((pm >> 4) & 3) snprintf(cs, sizeof cs, "粗%d", (pm >> 4) & 3);
            static const char* MN[8] = {"", "平均4", "半", "中5",
                                        "四分", "三四", "無傾", "中5半"};
            snprintf(b3, sizeof b3, "幾何v3%s%s%s%s%s",
                     MN[(pm & 3) | ((pm >> 5) & 4)],
                     (pm & 4) ? "十" : "", cs, (pm & 64) ? "共" : "", ax);
            return b3;
        }
        return p[0] == 1 ? "幾何v1" : (p[0] == 2 ? "幾何v2" : "幾何v3");
    case C_ATTR_SPATIAL: snprintf(b, sizeof b, "sp(P=%d)", p.empty() ? 0 : p[0]); return b;
    case C_ATTR_COLOR:   snprintf(b, sizeof b, "色(P=%d)", p.empty() ? 0 : p[0]); return b;
    case C_ATTR_XREF: {
        std::string nm;
        if (p.size() >= 3) { uint16_t l; memcpy(&l, p.data() + 1, 2);
            if (p.size() >= 3u + l) nm.assign((const char*)p.data() + 3, l); }
        int pp = p.empty() ? 0 : p[0];
        if (pp == 100) snprintf(b, sizeof b, "参照%.6s+差分", nm.c_str());
        else if (pp == 0) snprintf(b, sizeof b, "参照%.6s", nm.c_str());
        else snprintf(b, sizeof b, "参照%.6s+空間%d", nm.c_str(), pp);
        return b;
    }
    }
    return "?";
}

// 列の型から「素の幅」符号器の param を作る。入らない値があれば符号化側が外す。
static std::vector<uint8_t> raw_w_param(FType t) {
    const int w = ftype_size(t);
    const bool sg = (t == FType::I8 || t == FType::I16 ||
                     t == FType::I32 || t == FType::I64);
    return {(uint8_t)(w < 1 ? 8 : (w > 8 ? 8 : w)), (uint8_t)(sg ? 1 : 0)};
}

// 候補は互いに独立に符号化できる。幾何の候補だけを並列に回す。
// 属性の候補は CodecCtx の可変メンバ（順序表・予測子表・world）を作るので
// 同時に走らせられない。幾何の候補が触るのは復号済みの列だけである。
// 出力は候補の順に並べ直してから比べるので、選ばれる符号器は並列でも変わらない。
// abort が真のときだけ、最短を超えた候補を打ち切る。
// **事前選別の標本では打ち切ってはいけない。**打ち切った候補は順位表から
// 消えるので、族の保護が「その族が 1 本も無い」と見て守れなくなる。
// 17 件のうち 1 件で、走査変換が標本で打ち切られて選ばれなくなった。
// 符号化の機械の埋まり具合は 16 コアのうち 3.5 だった。候補は 16 並列に走るのに
// **列が逐次**で、列の中では遅い候補が 1 本だけ残って他のコアが空くからである。
// 列の仕事と候補の仕事を同じ待ち行列に入れれば、別の列の候補が空きを埋める。
//
// 待つ側も待ち行列から仕事を取る。こうしないと、列の仕事がその列の候補を
// 待って寝たときに、プールの糸が全部寝て詰まる。
class Pool {
public:
    explicit Pool(size_t n) {
        for (size_t i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    // **notify_one にしてはいけない。** 起こされた 1 本が「自分の left はもう 0 だ」と
    // 判って抜ける待ち手だと、その知らせはそこで消える。仕事は待ち行列に残り、
    // 他は全員寝たままになる。100 万点で 17 本ぜんぶが futex 待ちになって止まった。
    void add(std::function<void()> f, std::atomic<size_t>& left) {
        left.fetch_add(1, std::memory_order_relaxed);
        { std::lock_guard<std::mutex> lk(m_); q_.emplace_back(std::move(f), &left); }
        cv_.notify_all();
    }
    // 待つ側は空回りしない。仕事が無ければ寝て、誰かが 1 つ終えたら起こされる。
    // 空回りさせると、自分の列の候補を待つ糸が 10 本以上あるときに、その空回りが
    // 他の列の候補からコアを奪う（実測で CPU 時間だけが増えた）。
    void help_until(std::atomic<size_t>& left) {
        for (;;) {
            std::function<void()> f;
            std::atomic<size_t>* l = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                // 期限付きで待つ。知らせを取りこぼしても遅れで済み、止まらない。
                cv_.wait_for(lk, std::chrono::milliseconds(2),
                             [&] { return left.load(std::memory_order_acquire) == 0
                                          || !q_.empty(); });
                if (q_.empty() && left.load(std::memory_order_acquire) != 0) continue;
                if (left.load(std::memory_order_acquire) == 0) {
                    // 抜けるときに仕事が残っていたら、誰かを必ず起こしてから抜ける。
                    const bool more = !q_.empty();
                    lk.unlock();
                    if (more) cv_.notify_all();
                    return;
                }
                f = std::move(q_.front().first); l = q_.front().second; q_.pop_front();
            }
            run(f);
            done(l);
        }
    }
private:
    // 仕事の中の例外は糸の外へ出すと std::terminate になる。候補の符号化は
    // encode_many が、列の仕事は plan_streams が自分で受けて失敗として扱うので、
    // ここは最後の備えである。
    static void run(std::function<void()>& f) {
        try { f(); }
        catch (const std::exception& e) { fprintf(stderr, "符号化中の例外: %s\n", e.what()); }
    }
    // **減らすのは錠の中でなければならない。** left は待ち手の述語が読む。
    // 錠の外で減らして通知すると、待ち手が述語を偽と判じてから実際に眠るまでの
    // 隙間に通知が落ちる。待ち手はその眠りから覚めない。
    // 100 万点で 17 本ぜんぶが futex 待ちになって止まったのがこれである。
    void done(std::atomic<size_t>* l) {
        { std::lock_guard<std::mutex> lk(m_); l->fetch_sub(1, std::memory_order_release); }
        cv_.notify_all();
    }
    void loop() {
        for (;;) {
            std::function<void()> f;
            std::atomic<size_t>* l = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait_for(lk, std::chrono::milliseconds(2),
                             [this] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                if (q_.empty()) continue;
                f = std::move(q_.front().first); l = q_.front().second; q_.pop_front();
            }
            run(f);
            done(l);
        }
    }
    std::vector<std::thread> th_;
    std::deque<std::pair<std::function<void()>, std::atomic<size_t>*>> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
};

static size_t pool_threads() {
    static const size_t NT = [] {
        if (const char* e = getenv("PCC_THREADS")) { long v = atol(e); if (v > 0) return (size_t)v; }
        unsigned hw = std::thread::hardware_concurrency();
        return (size_t)(hw ? hw : 1);
    }();
    return NT;
}

// 計画を立てている間だけ生きるプール。plan_streams が建てて、その中から呼ばれる
// encode_many が使う。単独で符号化するときは無いので、そのときは今までどおり。
static Pool* g_pool = nullptr;

// 候補の出力は全部が同時に生きている。列も並列にしたので、列数 × 候補数ぶんが
// 一度にメモリに載る（200 万点・13 列・20 候補で 1 GB を超える）。
// **最短より長いと判った出力は、その場で捨ててよい。**最短は単調に縮むので、
// ある時点の最短より長いものが後から勝つことはない。大きさだけ控えておく。
// 同じ大きさのものは捨てない（選択は添字の早いほうを採るので、捨てると変わる）。
static void encode_many(const std::vector<Cand>& cs,
                        const std::vector<const Col*>& cv,
                        const CodecCtx* ctx, std::vector<std::vector<uint8_t>>& blobs,
                        std::vector<std::string>& errs, std::vector<char>& ok,
                        bool abort_long, std::vector<size_t>* sizes = nullptr) {
    const size_t m = cs.size();
    blobs.assign(m, {}); errs.assign(m, std::string()); ok.assign(m, 0);
    if (sizes) sizes->assign(m, 0);
    static const size_t NT = [] {
        if (const char* e = getenv("PCC_THREADS")) { long v = atol(e); if (v > 0) return (size_t)v; }
        unsigned hw = std::thread::hardware_concurrency();
        return (size_t)(hw ? hw : 1);
    }();
    size_t nt = NT < m ? NT : m;
    if (g_pool && nt > 1) {
        std::atomic<size_t> best{(size_t)-1};
        std::atomic<size_t> left{0};
        for (size_t i = 0; i < m; ++i) {

            g_pool->add([&, i] {
                const std::atomic<size_t>* save = enc_best;
                enc_best = abort_long ? &best : nullptr;
                try {
                    ok[i] = codec_encode(cs[i].codec, cv, cs[i].param, blobs[i], errs[i], ctx) ? 1 : 0;
                } catch (const EncAbort&) {
                    ok[i] = 0; errs[i] = "最短を超えたので打ち切り"; blobs[i].clear();
                } catch (const std::exception& e) {
                    // 糸の外へ出すと std::terminate。候補の失敗として扱う（どの候補も
                    // 失敗すれば恒等符号器で書くので、器は壊れない）。
                    ok[i] = 0; errs[i] = std::string("例外: ") + e.what(); blobs[i].clear();
                }
                enc_best = save;
                if (ok[i]) {
                    if (sizes) (*sizes)[i] = blobs[i].size();
                    size_t b = best.load(std::memory_order_relaxed);
                    while (blobs[i].size() < b &&
                           !best.compare_exchange_weak(b, blobs[i].size(),
                                                       std::memory_order_relaxed)) {}
                    if (sizes && blobs[i].size() > best.load(std::memory_order_relaxed)) {
                        std::vector<uint8_t>().swap(blobs[i]);   // 負けが確定。捨てる
                    }
                }
            }, left);
        }
        g_pool->help_until(left);
        return;
    }
    if (nt <= 1) {
        std::atomic<size_t> best{(size_t)-1};
        // 呼び手の打ち切り基準を戻せるよう覚えておく（糸の中から入れ子で呼ばれうる。
        // 以前は nullptr に戻していたので、呼び手の打ち切りが消えていた）。
        const std::atomic<size_t>* save = enc_best;
        enc_best = abort_long ? &best : nullptr;
        for (size_t i = 0; i < m; ++i) {
            try {
                ok[i] = codec_encode(cs[i].codec, cv, cs[i].param, blobs[i], errs[i], ctx) ? 1 : 0;
            } catch (const EncAbort&) {
                ok[i] = 0; errs[i] = "最短を超えたので打ち切り"; blobs[i].clear();
            } catch (const std::exception& e) {
                ok[i] = 0; errs[i] = std::string("例外: ") + e.what(); blobs[i].clear();
            }
            if (ok[i]) {
                if (sizes) (*sizes)[i] = blobs[i].size();
                if (blobs[i].size() < best.load(std::memory_order_relaxed))
                    best.store(blobs[i].size(), std::memory_order_relaxed);
            }
        }
        enc_best = save;
        return;
    }
    std::atomic<size_t> next{0};
    std::atomic<size_t> best{(size_t)-1};
    std::vector<std::thread> th;
    th.reserve(nt);
    for (size_t t = 0; t < nt; ++t)
        th.emplace_back([&] {
            enc_best = abort_long ? &best : nullptr;
            for (size_t i = next++; i < m; i = next++) {
                try {
                    ok[i] = codec_encode(cs[i].codec, cv, cs[i].param, blobs[i], errs[i], ctx) ? 1 : 0;
                } catch (const EncAbort&) {
                    ok[i] = 0; errs[i] = "最短を超えたので打ち切り"; blobs[i].clear();
                } catch (const std::exception& e) {
                    // 糸の外へ出すと std::terminate。候補の失敗として扱う（どの候補も
                    // 失敗すれば恒等符号器で書くので、器は壊れない）。
                    ok[i] = 0; errs[i] = std::string("例外: ") + e.what(); blobs[i].clear();
                }
                if (ok[i]) {
                    if (sizes) (*sizes)[i] = blobs[i].size();
                    size_t b = best.load(std::memory_order_relaxed);
                    while (blobs[i].size() < b &&
                           !best.compare_exchange_weak(b, blobs[i].size(),
                                                       std::memory_order_relaxed)) {}
                }
            }
            enc_best = nullptr;
        });
    for (auto& x : th) x.join();
}

// **勝者に旗を重ねた版を測る（後追い）。**best_stream の最後と、標本で選んで
// 全点で測り直す経路（pack --sample-select）の両方から呼ぶ。以前は後者が
// 素通し（生）しか試さず、符号・光線・束ね・曲面の旗が一度も付かなかった。
// `--force-geom` で名前ごと固定された勝者には何も足さない（固定した名前と
// 違う符号器で書かれてしまうため）。
static void post_flags(Stream& best, size_t& bestsz, const std::vector<const Col*>& cv,
                       const CodecCtx* ctx, std::string* trace, size_t npts,
                       size_t* before_fast = nullptr) {
    if (ctx && !ctx->force_geom.empty() && cand_name(best.codec, best.param) == ctx->force_geom)
        return;
    const uint16_t b0 = best.codec;
    std::vector<Cand> pc;
    // 素通しの旗は UIntCoder の下位ビットに効く。模型を通らない符号器
    // （恒等・素の幅）と、UIntCoder を使わない字母（FreqModel だけ）では
    // 同じバイト列になるので測らない。
    const bool no_model = ((b0 & ~C_FLAG_MASK) == C_RAW64 || (b0 & ~C_FLAG_MASK) == C_RAW_W ||
                           (b0 & ~C_FLAG_MASK) == C_RANGE_SYM);
    if (!(b0 & C_RAW_BIT) && !no_model) pc.push_back({(uint16_t)(b0 | C_RAW_BIT), best.param});
    // 符号つきの文脈は、それを読む符号器（下の sgn_ok）にだけ効く。長さの最適値は
    // ファイルで違うので、1 / 2 / 4 個の 3 通りを出して実測で選ばせる。
    //
    // **実装している経路に限る。**旗を読まない符号器に立てても、同じバイト列を
    // 3 回符号化するだけになる（以前は幾何v1 がそうだった。2026-09-23 に enc_cols・
    // enc_resid・enc_geom_med・enc_geom_dir にも符号つきの文脈を入れた。幾何v2 の
    // enc_geom_aux はまだ読まない）。
    const uint16_t bid0 = (uint16_t)(b0 & ~C_FLAG_MASK);
    const int gp0 = best.param.empty() ? 0 : best.param[0];
    // 属性の残差符号器（差分・文脈つき差分・参照・空間予測・色）も符号つきの文脈を持つ
    // （enc_resid と enc_cols の mode 1〜3）。PCC_ATTR_SGN=0 で外せる（測り比べ用）。
    static const bool ATTR_SGN = [] {
        const char* e = getenv("PCC_ATTR_SGN"); return !e || e[0] != '0'; }();
    const bool attr_sgn = ATTR_SGN &&
        (bid0 == C_RANGE_DELTA || bid0 == C_RANGE_CTX || bid0 == C_RANGE_CTX2 ||
         bid0 == C_ATTR_XREF || bid0 == C_ATTR_SPATIAL || bid0 == C_ATTR_COLOR);
    const bool sgn_ok =
        (bid0 == C_GEOM_XYZ && (gp0 == 3 || gp0 == 4)) ||
        bid0 == C_GEOM_ROT ||
        // 走査モデルは var 3/4 が enc_resid_x、それ以外（1・2 = 走査v1/v2、5 = 走査変換）が enc_resid を
        // 通る。どちらも符号つきの文脈を持つので全部の var で試す（2026-09-23 に広げた）。
        bid0 == C_GEOM_SCAN || attr_sgn ||
        // 幾何v0（enc_cols mode 2）・幾何v1（enc_geom_med）・向き追従（enc_geom_dir）。
        // med3（C_RANGE_MED）も enc_geom_med なので、属性の流れで勝ったときも試す
        (bid0 == C_GEOM_XYZ && (gp0 == 0 || gp0 == 1)) || bid0 == C_RANGE_MED ||
        bid0 == C_GEOM_DIR;
    // 属性の流れにも 3 通り（符1・符2・符4）を全部試す。
    // **2 段（まず符1、勝ったら符2・符4）にすると速いが、縮む量の大半を失う。**
    // 15 件の中央 200 万点で符号化 39.34 → 36.36 s（全部試す前は 34.71）になる代わりに、
    // AHN4 _21 は −0.244 → −0.030 bpp、fullwave は −0.231 → 0 bpp。符2・符4 は
    // **符1 だけでは無印に勝てない流れ**で勝っていた。サイズが第一なので既定は全部。
    // PCC_ATTR_SGN_STAGED=1 で 2 段にできる（符号化器だけの選択で、器は変わらない）。
    static const bool SGN_STAGED = [] {
        const char* e = getenv("PCC_ATTR_SGN_STAGED"); return e && e[0] == '1'; }();
    const bool sgn_staged = attr_sgn && SGN_STAGED;
    if (!(b0 & (C_LMS_BIT | C_SGN2_BIT)) && sgn_ok) {
        pc.push_back({(uint16_t)(b0 | C_LMS_BIT), best.param});
        if (!sgn_staged) {
            pc.push_back({(uint16_t)(b0 | C_SGN2_BIT), best.param});
            pc.push_back({(uint16_t)(b0 | C_LMS_BIT | C_SGN2_BIT), best.param});
        }
    }
    // **光線モデル**（同一パルスの戻りは 1 本の直線に乗る）。実装したのは
    // enc_geom_x（幾何v3・回転）と enc_geom_w（幾何v4）の 2 経路。
    // bit_fields が幾何より前に出ているファイルでしか使えない。
    // 符号つき文脈とは独立に効くので、符号の 4 通り（無・1・2・4）と組にして出す。
    // **続きの点が 1 つも無いファイルでは出さない。**多重戻りの無い入力
    // （plane・TLS・KITTI など）では続きの判定が一度も立たず、光線版は
    // 無印と同じ予測になる。文脈の数だけ倍になるが使われないので、
    // 同じ長さのバイト列を 4 回符号化するだけになる。
    const bool ray_geo =
        ((bid0 == C_GEOM_XYZ && (gp0 == 3 || gp0 == 4)) || bid0 == C_GEOM_ROT) &&
        !(b0 & C_RAY_BIT);
    // 続きの点を探すのは光線版を出しうる幾何の流れだけ（属性の流れごとに
    // bit_fields を全点なめていた）。
    bool has_cont = false;
    if (const Col* bf = (ray_geo && ctx && ctx->fr) ? ctx->fr->get("bit_fields") : nullptr) {
        const size_t nb = bf->size();
        for (size_t i = 1; i < nb && !has_cont; ++i) {
            const uint64_t a = (uint64_t)(*bf)[i], b = (uint64_t)(*bf)[i - 1];
            const int r1 = (int)(a & 15), n1 = (int)((a >> 4) & 15);
            const int r0 = (int)(b & 15), n0 = (int)((b >> 4) & 15);
            if (r1 >= 2 && r1 == r0 + 1 && n1 == n0) has_cont = true;
        }
    }
    const bool ray_ok = ray_geo && has_cont;
    // **属性の戻りの種類の文脈**（光線の旗を属性の符号器で使う）。空間予測は残差を
    // 符号化順に送るので、順序表（g_resid_ord）で点の番号に引き直す。
    // bit_fields が属性より先に出る並び（走査モデル用の前置き列）でなければ使わない。
    // 前置き列そのもの（point_source_id・gps_time・bit_fields）には付けない（循環する）。
    static const bool ATTR_RAY = [] {
        const char* e = getenv("PCC_ATTR_RAY"); return !e || e[0] != '0'; }();
    bool ray_attr = false;
    // 2026-09-23 から、点の並び順に残差を送る幾何の符号器（幾何v0 = enc_cols・
    // 幾何v1 と med3 = enc_geom_med・向き追従 = enc_geom_dir）にも同じ文脈を試す。
    if (ATTR_RAY && ctx && ctx->bitfields_first && !(b0 & C_RAY_BIT) &&
        (bid0 == C_RANGE || bid0 == C_RANGE_DELTA || bid0 == C_RANGE_CTX || bid0 == C_RANGE_CTX2 ||
         bid0 == C_ATTR_XREF || bid0 == C_ATTR_SPATIAL || bid0 == C_ATTR_COLOR ||
         bid0 == C_RANGE_MED || bid0 == C_GEOM_DIR ||
         (bid0 == C_GEOM_XYZ && (gp0 == 0 || gp0 == 1)))) {
        bool pre_col = false;
        for (const auto& c : best.cols)
            if (c == "point_source_id" || c == "gps_time" || c == "bit_fields") pre_col = true;
        const Col* bf = (!pre_col && ctx->fr) ? ctx->fr->get("bit_fields") : nullptr;
        if (bf) {                       // 複数の戻りが 1 つでもあるときだけ（無ければ文脈が 1 つ）
            const size_t nb = bf->size();
            for (size_t i = 0; i < nb && !ray_attr; ++i)
                if ((((uint64_t)(*bf)[i] >> 4) & 15) > 1) ray_attr = true;
        }
    }
    // 光は符号の 4 通りと組にして同時に試す。符号の勝者が決まってから 1 本だけ重ねる形は
    // 4 件で縮みを失った（autzen-2023 +0.058% など。losses.md §33）。
    if (ray_attr) {
        pc.push_back({(uint16_t)(b0 | C_RAY_BIT), best.param});
        if (!(b0 & (C_LMS_BIT | C_SGN2_BIT))) {
            pc.push_back({(uint16_t)(b0 | C_RAY_BIT | C_LMS_BIT), best.param});
            pc.push_back({(uint16_t)(b0 | C_RAY_BIT | C_SGN2_BIT), best.param});
            pc.push_back({(uint16_t)(b0 | C_RAY_BIT | C_LMS_BIT | C_SGN2_BIT), best.param});
        }
    }
    // **旗「類」。**既に復号済みの別の列の類を、残差の文脈に足す（属性の同時符号化の最初の一手）。
    // 光と同じ差し込み口を使うので、光を読む属性の符号器（戻りの種類の文脈と同じ組）にだけ試す。
    // 相手の列は ctx_cols（plan_streams が埋める。参照残差と同じ規則）から。符号の 4 通りと組にする。
    // 15 件合計 −0.074%（autzen-2023 −0.324%、fullwave −0.188%、AHN4 _20 −0.110%）、符号化 +27%
    // （s12_combined_v22.log）。PCC_CLS=0 で外せる（符号化器だけの選択で、器の読み方は変わらない）。
    static const bool CLS_ON = [] { const char* e = getenv("PCC_CLS"); return !e || e[0] != '0'; }();
    static const bool CLS_GEOM = [] { const char* e = getenv("PCC_CLS_GEOM"); return !e || e[0] != '0'; }();
    static const size_t CLS_MAX = [] {
        const char* e = getenv("PCC_CLS_MAX"); return e ? (size_t)atol(e) : (size_t)99; }();
    if (CLS_ON && ctx && !best.cols.empty() && !(b0 & (C_RAY_BIT | C_CLS_BIT)) &&
        (bid0 == C_RANGE || bid0 == C_RANGE_DELTA || bid0 == C_RANGE_CTX || bid0 == C_RANGE_CTX2 ||
         bid0 == C_ATTR_XREF || bid0 == C_ATTR_SPATIAL || bid0 == C_ATTR_COLOR || bid0 == C_RANGE_MED ||
         bid0 == C_RANGE_SYM ||
         // 幾何（前置きの列と gps_time の差を相手に）にも試す（PCC_CLS_GEOM=0 で外せる）。
         // autzen-2023 は 向き無傾記符2類<Δgps_time> が勝って全体 −1.10%（時刻の跳び＝走査線の変わり目を文脈に）。
         // 15 件で動いたのはこの 1 件で、符号化の時間は揺れの範囲（cls_geom_v22.log）。
         // 走査モデルは var 1・2・5 が enc_resid（類の差し込み口を読む）。var 3・4 は読まない
         (CLS_GEOM && (bid0 == C_GEOM_DIR || (bid0 == C_GEOM_XYZ && (gp0 == 0 || gp0 == 1)) ||
                       (bid0 == C_GEOM_SCAN && (gp0 == 1 || gp0 == 2 || gp0 == 5)))))) {
        auto it = ctx->ctx_cols.find(best.cols[0]);
        if (it != ctx->ctx_cols.end()) {
            size_t k = 0;
            for (const auto& nm : it->second) {
                if (k++ >= CLS_MAX) break;
                const std::vector<uint8_t> p2 = cls_join(best.param, nm);
                pc.push_back({(uint16_t)(b0 | C_CLS_BIT), p2});
                if (sgn_ok && !(b0 & (C_LMS_BIT | C_SGN2_BIT))) {
                    pc.push_back({(uint16_t)(b0 | C_CLS_BIT | C_LMS_BIT), p2});
                    pc.push_back({(uint16_t)(b0 | C_CLS_BIT | C_SGN2_BIT), p2});
                    pc.push_back({(uint16_t)(b0 | C_CLS_BIT | C_LMS_BIT | C_SGN2_BIT), p2});
                }
            }
        }
    }
    if (ray_ok) {
        pc.push_back({(uint16_t)(b0 | C_RAY_BIT), best.param});
        pc.push_back({(uint16_t)(b0 | C_RAY_BIT | C_LMS_BIT), best.param});
        pc.push_back({(uint16_t)(b0 | C_RAY_BIT | C_SGN2_BIT), best.param});
        pc.push_back({(uint16_t)(b0 | C_RAY_BIT | C_LMS_BIT | C_SGN2_BIT), best.param});
    }
    // **実験: 旗の全組合せ（PCC_POST_ALL=1）。**既定は 1 段目（生・符・光・束）→ 2 段目（面）→
    // 3 段目（速）と貪欲に重ねる。組合せの取りこぼしがどれだけあるかを測るため、勝者の基底の
    // 符号器に、それが読む旗のすべての組合せを重ねて測る。旗を読むかどうかの判定（循環の禁止を含む）は
    // 既定の段と同じものを使うので、作る流れはどれも既定でも作りうる形である（器の読み方は変わらない）。
    // 旗「類」の候補は含まない（類は相手の列ごとに増えるので、組合せに入れると数が爆発する）。
    static const bool POST_ALL = [] { const char* e = getenv("PCC_POST_ALL"); return e && e[0] == '1'; }();
    if (POST_ALL) {
        static const bool FAST_ON_A = [] {
            const char* e = getenv("PCC_FAST"); return !e || e[0] != '0'; }();
        const uint16_t base = (uint16_t)(b0 & ~C_FLAG_MASK);
        const bool modelled = !no_model;
        std::vector<uint16_t> fs{0};
        auto mul = [&](const std::vector<uint16_t>& opts) {
            std::vector<uint16_t> out;
            for (uint16_t a : fs) for (uint16_t o : opts) out.push_back((uint16_t)(a | o));
            fs.swap(out);
        };
        const bool pair_ok = modelled && !(ctx && !ctx->force_geom.empty());
        if (pair_ok) mul({0, C_FSYM_BIT});
        if (modelled) mul({0, C_RAW_BIT});
        if (sgn_ok) mul({0, C_LMS_BIT, C_SGN2_BIT, (uint16_t)(C_LMS_BIT | C_SGN2_BIT)});
        if (ray_attr || ray_ok) mul({0, C_RAY_BIT});
        if (bid0 == C_GEOM_XYZ && gp0 == 4) mul({0, C_SURF_A, C_SURF_B, (uint16_t)(C_SURF_A | C_SURF_B)});
        if (FAST_ON_A && modelled) mul({0, C_FAST_BIT});
        // 束は記号版にだけ効く。記号版でない組合せに束を立てても同じバイト列になるので省く。
        if (pair_ok) {
            std::vector<uint16_t> out;
            for (uint16_t a : fs) {
                out.push_back(a);
                if (a & C_FSYM_BIT) out.push_back((uint16_t)(a | C_BND_BIT));
            }
            fs.swap(out);
        }
        // 前置きの旗（記など）が勝者に既に立っている場合も、基底から組み直す
        std::vector<Cand> all;
        for (uint16_t f : fs) {
            const uint16_t c = (uint16_t)(base | f);
            if (c == best.codec) continue;
            all.push_back({c, best.param});
        }
        size_t best_nofast = (best.codec & C_FAST_BIT) ? (size_t)-1 : bestsz;
        for (size_t at = 0; at < all.size(); at += 16) {
            std::vector<Cand> chunk(all.begin() + at, all.begin() + std::min(all.size(), at + 16));
            std::vector<std::vector<uint8_t>> pb;
            std::vector<std::string> pe;
            std::vector<char> pok;
            encode_many(chunk, cv, ctx, pb, pe, pok, false);
            for (size_t i = 0; i < chunk.size(); ++i) {
                if (!pok[i]) continue;
                if (!(chunk[i].codec & C_FAST_BIT) && pb[i].size() < best_nofast) best_nofast = pb[i].size();
                if (pb[i].size() < bestsz) {
                    bestsz = pb[i].size();
                    best.codec = chunk[i].codec;
                    best.data = std::move(pb[i]);
                }
            }
        }
        if (trace) {
            char m[200];
            snprintf(m, sizeof m, "      （旗の全組合せ %zu 通りを測った: %s %.3f bpp）\n", all.size() + 1,
                     cand_name(best.codec, best.param).c_str(), npts ? bestsz * 8.0 / npts : 0.0);
            *trace += m;
        }
        if (before_fast) *before_fast = best_nofast == (size_t)-1 ? bestsz : best_nofast;
        return;
    }
    if ((b0 & C_FSYM_BIT) && !(b0 & C_BND_BIT)) {
        pc.push_back({(uint16_t)(b0 | C_BND_BIT), best.param});
        // **組合せ（素通し＋束ね）も要る。**外したら extra が
        // 138.659 → 138.682 bpp に伸びた。要約行（中央値と四分位）は
        // 動かないので、**ファイルごとの行を見ないと気づかない**。
        // 候補が同時に生きるぶんメモリーは 592 → 634 byte/点 になるが、
        // 指標の順はサイズが先である。
        if (!(b0 & C_RAW_BIT))
            pc.push_back({(uint16_t)(b0 | C_RAW_BIT | C_BND_BIT), best.param});
    }
    // 旗「類」で候補が相手の列の数 × 4 に増える。8 本ずつに分けて測ると符号化が +9.7% 遅くなり、
    // 1 点あたりのピークの中央値は 836 → 832 byte と変わらなかった（ab_v23_v23c.log）ので、まとめて測る。
    if (!pc.empty()) {
        std::vector<std::vector<uint8_t>> pb;
        std::vector<std::string> pe;
        std::vector<char> pok;
        encode_many(pc, cv, ctx, pb, pe, pok, false);
        for (size_t i = 0; i < pc.size(); ++i) {
            if (!pok[i]) continue;
            if (trace) {
                char m[160];
                snprintf(m, sizeof m, "      %-10s %8.3f bpp（勝者に旗を重ねた版）\n",
                         cand_name(pc[i].codec, pc[i].param).c_str(),
                         npts ? pb[i].size() * 8.0 / npts : 0.0);
                *trace += m;
            }
            // 旗「類」は param に列名を足すので、流れの表に書く param の増分まで含めて比べる
            // （本体が縮んでも、名前の byte で差し引き伸びることがある。workshop で +10 byte）。
            if (pb[i].size() + pc[i].param.size() < bestsz + best.param.size()) {
                bestsz = pb[i].size();
                best.codec = pc[i].codec;
                best.param = pc[i].param;
                best.data = std::move(pb[i]);
            }
        }
    }
    // 符1 が勝った属性の流れにだけ、符号の履歴を長くした版（符2・符4）を試す。
    if (sgn_staged && !(b0 & (C_LMS_BIT | C_SGN2_BIT)) && best.codec == (uint16_t)(b0 | C_LMS_BIT)) {
        std::vector<Cand> ps = {{(uint16_t)(b0 | C_SGN2_BIT), best.param},
                                {(uint16_t)(b0 | C_LMS_BIT | C_SGN2_BIT), best.param}};
        std::vector<std::vector<uint8_t>> pb;
        std::vector<std::string> pe;
        std::vector<char> pok;
        encode_many(ps, cv, ctx, pb, pe, pok, false);
        for (size_t i = 0; i < ps.size(); ++i) {
            if (!pok[i]) continue;
            if (trace) {
                char m[160];
                snprintf(m, sizeof m, "      %-10s %8.3f bpp（符1 が勝ったので符号の履歴を長くした版）\n",
                         cand_name(ps[i].codec, ps[i].param).c_str(),
                         npts ? pb[i].size() * 8.0 / npts : 0.0);
                *trace += m;
            }
            if (pb[i].size() < bestsz) {
                bestsz = pb[i].size();
                best.codec = ps[i].codec;
                best.data = std::move(pb[i]);
            }
        }
    }
    // **2 段目: 曲面による z の予測を、1 段目の勝者に重ねる。**
    // 旗の組み合わせを全部出すと数が爆発するので、1 段目（符号・光線・生・束）で
    // 決まった勝者にだけ、格子の大きさ 3 通りを足して測る。効くのは幾何v4 だけ。
    const uint16_t b1 = best.codec;
    const uint16_t bid1 = (uint16_t)(b1 & ~C_FLAG_MASK);
    const int gp1 = best.param.empty() ? 0 : best.param[0];
    if (bid1 == C_GEOM_XYZ && gp1 == 4 && !(b1 & (C_SURF_A | C_SURF_B))) {
        std::vector<Cand> pc2 = {
            {(uint16_t)(b1 | C_SURF_A), best.param},
            {(uint16_t)(b1 | C_SURF_B), best.param},
            {(uint16_t)(b1 | C_SURF_A | C_SURF_B), best.param}};
        std::vector<std::vector<uint8_t>> pb2;
        std::vector<std::string> pe2;
        std::vector<char> pok2;
        encode_many(pc2, cv, ctx, pb2, pe2, pok2, false);
        for (size_t i = 0; i < pc2.size(); ++i) {
            if (!pok2[i]) continue;
            if (trace) {
                char m[160];
                snprintf(m, sizeof m, "      %-10s %8.3f bpp（勝者に曲面を重ねた版）\n",
                         cand_name(pc2[i].codec, pc2[i].param).c_str(),
                         npts ? pb2[i].size() * 8.0 / npts : 0.0);
                *trace += m;
            }
            if (pb2[i].size() < bestsz) {
                bestsz = pb2[i].size();
                best.codec = pc2[i].codec;
                best.data = std::move(pb2[i]);
            }
        }
    }
    // **3 段目: 速い後半の適応（速）を、ここまでの勝者に重ねる。**1 本だけ符号化する。
    // 候補の比較は既定の表（遅い後半）で行い、最後の勝者にだけ速い表を試す
    // （逆に、既定を速い表にして最後に遅い表を試すと、候補の選び方が変わって
    // LAS の 6 件で縮みが減った）。二値の模型を通らない符号器（恒等・素の幅・字母）には
    // 効かないので試さない。
    if (before_fast) *before_fast = bestsz;     // 速 を試す前の長さ（選び直しの判断に使う）
    {
        static const bool FAST_ON = [] {
            const char* e = getenv("PCC_FAST"); return !e || e[0] != '0'; }();
        const uint16_t b2 = best.codec;
        const uint16_t bid2 = (uint16_t)(b2 & ~C_FLAG_MASK);
        if (FAST_ON && !(b2 & C_FAST_BIT) && bid2 != C_RAW64 && bid2 != C_RAW_W &&
            bid2 != C_RANGE_SYM) {
            std::vector<Cand> p3 = {{(uint16_t)(b2 | C_FAST_BIT), best.param}};
            std::vector<std::vector<uint8_t>> pb3;
            std::vector<std::string> pe3;
            std::vector<char> pok3;
            encode_many(p3, cv, ctx, pb3, pe3, pok3, false);
            if (pok3[0]) {
                if (trace) {
                    char m[160];
                    snprintf(m, sizeof m, "      %-10s %8.3f bpp（勝者を速い後半の適応にした版）\n",
                             cand_name(p3[0].codec, p3[0].param).c_str(),
                             npts ? pb3[0].size() * 8.0 / npts : 0.0);
                    *trace += m;
                }
                if (pb3[0].size() < bestsz) {
                    bestsz = pb3[0].size();
                    best.codec = p3[0].codec;
                    best.data = std::move(pb3[0]);
                }
            }
        }
    }
}

void apply_post_flags(Stream& s, const std::vector<const Col*>& cv, const CodecCtx* ctx,
                      std::string* trace) {
    size_t sz = s.data.size();
    post_flags(s, sz, cv, ctx, trace, cv.empty() ? 0 : cv[0]->size());
}

Stream best_stream(const Frame& f, const std::vector<std::string>& cols,
                   const std::vector<Cand>& candidates, const CodecCtx* ctx,
                   std::string* trace, bool preselect) {
    // 属性の列にも事前選別を掛けるか。49 節では全列 bpp が +3.7% 悪化したので
    // 幾何だけにしたが、候補も残す本数も変わったので測り直せるようにしておく。
    // **既定では掛けない。** 全列の符号化時間は 15 件の中央値で 34% 縮むが、
    // 回帰試験（先頭 30 万点・全列）では USGS NY が +4.5%、AHN4 _20 が +2.7%、
    // AHN3 _20 が +1.7% 伸びる。標本（中央 20 万点）では +0.12% しか出ないので、
    // どの標本で測るかで結論が変わる。サイズが第一なので掛けない。
    static const bool PRE_ATTR = [] {
        const char* e = getenv("PCC_PRESELECT_ATTR");
        return e && e[0] == '1';
    }();
    // 並列にしてよいのは幾何の候補だけである（属性の候補は CodecCtx の
    // 可変メンバを作る）。標本で順位を付けることと、並列に符号化することは
    // 別の話なので、旗を分ける。
    // 属性の候補も並列にしてよいか。CodecCtx の可変メンバ（順序表・予測子表・
    // world）は排他で守られ、P ごとに取ってあるので読むだけになった。
    static const bool PAR_ALL = [] {
        const char* e = getenv("PCC_PAR_ATTR");
        return !e || e[0] != '0';
    }();
    const bool par = preselect || PAR_ALL;
    if (PRE_ATTR) preselect = true;
    std::vector<const Col*> cv;
    for (const auto& c : cols) cv.push_back(f.get(c));
    // 下位ビットを 1 記号で送る版も候補に出す。どちらが短いかは列で変わる
    // （全列では中央値が縮むが、TLS p1 の幾何のように伸びる列もある）。
    // 流れに書かれる旗なので、選択原理に選ばせれば悪化はしない。
    static const bool FSYM_CAND = [] {
        const char* e = getenv("PCC_FSYM_CAND");
        return !e || e[0] != '0';
    }();
    std::vector<Cand> cand2;
    if (FSYM_CAND) {
        cand2.reserve(candidates.size() * 2);
        // どの符号器に対を作るか。1 は全部、2 は空間予測を除く（高い候補を
        // 倍にしないぶん速いが、その列で記号版が勝てなくなる）。
        static const long PAIR_MODE = [] {
            const char* e = getenv("PCC_FSYM_PAIR");
            return e ? atol(e) : 1L;
        }();
        for (const auto& c : candidates) {
            cand2.push_back(c);
            // `--force-geom` で名前ごと固定した候補には記号版の対を作らない
            // （記号版が勝つと固定した名前と違う符号器で書かれる）
            if (ctx && !ctx->force_geom.empty() && cand_name(c.codec, c.param) == ctx->force_geom)
                continue;
            // 模型を通らない符号器（恒等・素の幅）と UIntCoder を使わない字母には
            // 記の旗が効かない。対を作っても同じバイト列をもう一度符号化するだけになる
            // （字母の記号版は束・生束の後置検査も呼び込んでいた）。
            if (c.codec == C_RAW64 || c.codec == C_RAW_W || c.codec == C_RANGE_SYM) continue;
            if (PAIR_MODE == 2) {
                bool sp = (c.codec == C_ATTR_SPATIAL || c.codec == C_ATTR_COLOR) ||
                          (c.codec == C_ATTR_XREF && !c.param.empty() &&
                           c.param[0] != 0 && c.param[0] != 100);
                if (sp) continue;
            }
            cand2.push_back({(uint16_t)(c.codec | C_FSYM_BIT), c.param});
        }
    }
    // **素通し版を候補として出すのは、幾何でも割に合わなかった。**候補が倍になると
    // 標本での順位が崩れ、全点での最良が上位から押し出される（KITTI が
    // 16.830 → 17.010 に伸びた）。上位数本への後追い（下の RAW_TRY）で止める。
    const std::vector<Cand>& cands = FSYM_CAND ? cand2 : candidates;
    // 候補が多いときは、まず先頭の標本で順位を付け、上位だけを全点で測る。
    // 全候補を全点で符号化すると、幾何だけで十数候補 × 全点になる。
    // 標本の 1 位が全点でも 1 位とは限らないので上位 3 つを残す。
    const size_t ncols_n = (cv.empty() || !cv[0]) ? 0 : cv[0]->size();
    // 標本は点数に比例させる（固定だと小さい入力で効かず、大きい入力で重い）。
    // 0 を指定すると事前の順位付けをしない。
    static const long PRE_OPT = [] {
        const char* e = getenv("PCC_PRESELECT");
        return e ? atol(e) : -1;
    }();
    // 上位いくつを全点で測るか。**標本の 1 位が全点でも 1 位とは限らない。**
    // 候補が増えるほど標本での順位が競り合い、2 本では全点の最良が押し出される。
    // 実測（200 万点、幾何のみ）: 2 → 4 で KITTI 17.077 → 16.899（−1.0%）、
    // TLS p1 25.554 → 25.457（−0.4%）。4 より増やしても 1 件も動かない。
    // 代償は符号化の時間で +2〜8%（autzen-2023 の全列で 2.58 → 2.74 s）。
    // **サイズが第一**という順位なので 4 を採る。
    static const size_t PRE_KEEP = [] {
        const char* e = getenv("PCC_PREKEEP");
        return e ? (size_t)atol(e) : (size_t)4;
    }();
    size_t PRE_SAMP;
    if (PRE_OPT >= 0) PRE_SAMP = (size_t)PRE_OPT;
    else {
        // 標本は 2 万〜2.5 万点。上限を 20 万点から 2.5 万点に下げても、
        // 幾何の 17 件・全列の 15 件（20 万点と 100 万点）・回帰の 9 件の
        // どれも bpp が 1 つも動かず、100 万点の幾何の列が 0.27 s → 0.22 s になる。
        PRE_SAMP = ncols_n / 8;
        if (PRE_SAMP < 20000) PRE_SAMP = 20000;
        if (PRE_SAMP > 25000) PRE_SAMP = 25000;
    }
    std::vector<Cand> use = cands;
    // 標本での順位を、選択のあとでも使えるように外に出す。
    // **勝者と同じ族だけを後から詳しく測る**ために要る（全族を厚くすると
    // 符号化が 70% 伸びてサイズは 1 ビットも動かなかった）。
    std::vector<std::pair<size_t, Cand>> pre_rank;
    auto cand_family = [](const Cand& c) -> int {
        const uint16_t id = c.codec & ~C_FLAG_MASK;
        if (id == C_GEOM_SCAN) return 1;
        if (id == C_ATTR_SPATIAL || id == C_ATTR_COLOR) return 2;
        if (id == C_ATTR_XREF && !c.param.empty() &&
            c.param[0] != 0 && c.param[0] != 100) return 2;
        if (id == C_GEOM_XYZ && !c.param.empty() && c.param[0] == 4) return 3;
        if (id == C_GEOM_DIR) return 4;
        if (id == C_GEOM_ROT) return 5;
        return 0;
    };
    // 属性でも「どちらの版か」だけを標本で決める案を試したが、標本パスの費用が
    // 節約を上回った（全列の比が 14.6〜15.5 倍から 17 倍に伸びた）。入れない。
    if (preselect && PRE_SAMP && cands.size() > PRE_KEEP + 1 &&
        ncols_n > PRE_SAMP * 2) {
        // 標本は列の先頭から取る。**等間隔に置いた連続した塊に変える案は、
        // 測って落とした。**先頭 200 万点で測ると red-rocks が 0.090 bpp 縮んだが、
        // それは外側の標本（--max-points）が先頭だったことの副作用で、
        // 全点（400 万点）では塊 1 と 2 が同値だった。一方、中央の 200 万点では
        // autzen-2023 が 0.142 bpp 伸びる。3 以上では塊が掃引を切り、
        // AHN4 _21 が 0.056 bpp 伸びる。PCC_PRE_CHUNKS で再現できる。
        static const int PRE_CHUNKS = [] {
            const char* e = getenv("PCC_PRE_CHUNKS");
            return e ? atoi(e) : 1;
        }();
        std::vector<size_t> idx;
        idx.reserve(PRE_SAMP);
        {
            const int nc = PRE_CHUNKS > 0 ? PRE_CHUNKS : 1;
            const size_t per = PRE_SAMP / (size_t)nc;
            for (int b = 0; b < nc && idx.size() < PRE_SAMP; ++b) {
                // 塊の先頭を等間隔に置く。最後の塊が末尾からはみ出さないようにする。
                size_t start = (ncols_n - per) * (size_t)b / (size_t)(nc > 1 ? nc - 1 : 1);
                for (size_t i = 0; i < per && idx.size() < PRE_SAMP; ++i)
                    idx.push_back(start + i);
            }
            while (idx.size() < PRE_SAMP) idx.push_back(idx.size());
        }
        std::vector<Col> sub(cv.size());
        for (size_t k = 0; k < cv.size(); ++k) {
            std::vector<int64_t> t(PRE_SAMP);
            for (size_t i = 0; i < PRE_SAMP; ++i) t[i] = (*cv[k])[idx[i]];
            sub[k] = std::move(t);
        }
        std::vector<const Col*> scv;
        for (auto& v : sub) scv.push_back(&v);
        std::vector<std::vector<uint8_t>> sb; std::vector<std::string> se;
        std::vector<char> sok;
        if (par) {
            encode_many(cands, scv, ctx, sb, se, sok, false);
        } else {
            sb.assign(cands.size(), {}); se.assign(cands.size(), std::string());
            sok.assign(cands.size(), 0);
            for (size_t t = 0; t < cands.size(); ++t)
                sok[t] = codec_encode(cands[t].codec, scv, cands[t].param,
                                      sb[t], se[t], ctx) ? 1 : 0;
        }
        std::vector<std::pair<size_t, Cand>> pre;
        for (size_t t = 0; t < cands.size(); ++t)
            if (sok[t]) pre.push_back({sb[t].size(), cands[t]});
        if (pre.size() > PRE_KEEP) {
            std::sort(pre.begin(), pre.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            use.clear();
            for (size_t i = 0; i < PRE_KEEP; ++i) use.push_back(pre[i].second);
            pre_rank = pre;          // 勝者の族を後から測り直すために控える
            // 標本で系統的に不利になる族は、順位に入らなくても 1 つ残す。
            //   走査モデル … 掃引の数が減ると当てはめが効かない
            //                 （AHN3 の走査変換は標本では 6 位以内にも入らないのに
            //                   全点では 1 位だった）
            //   空間予測   … 点を間引くと近傍が遠くなる（30 節の nir と同じ理由）
            //   幾何v4     … 最近傍の当たり方が場所で変わる（AHN4 _21 は
            //                   先頭 12.5 万点だけ見ると 3 位以内に入らない）
            //   記号版   … 多値の模型は馴染むのに点数が要る（2.5 万点の標本では
            //                 実力より悪く見える。AHN4 _20 の走査v1記が実際にそう）
            // 記号版かどうかは族と**直交する軸**である。族だけで数えると、記号版が
            // 上位に入ったときに「その族はもう居る」と判じて非記号版の代表を
            // 足さなくなる（AHN3 _20 の先頭 30 万点で 幾何v4W4 が決勝に残らず
            // 21.315 → 21.328 bpp に伸びた）。逆に記号版だけで数えると、族ごとの
            // 記号版が守れない（autzen-2023 で 走査v3記 が残らず +0.065%）。
            // **族 × 記号版の組で数える。**
            // 族の分け方は外の cand_family と同じ。
            //   走査モデル … 掃引の数が減ると当てはめが効かない
            //   空間予測   … 点を間引くと近傍が遠くなる
            //   幾何v4     … 最近傍の当たり方が場所で変わる
            //   向き追従   … y の予測が他の族と全く違う
            //   回転       … 座標系ごと変える
            auto family = cand_family;
            auto is_fsym = [](const Cand& c) { return (c.codec & C_FSYM_BIT) != 0; };
            // 族の保護は無条件だと重い。走査モデルは全点で 1 本 0.09 s かかり、
            // 既定の符号化時間の半分近くを占める。標本での符号長が最良から
            // 離れすぎている族は、全点で測っても勝たないと見て落とす。
            static const double FAM_THR = [] {
                const char* e = getenv("PCC_FAMTHR");
                return e ? atof(e) : 1e9;
            }();
            const double best_pre = (double)pre[0].first;
            // **族の番号を足したら、この上限も上げること。**
            // 向き追従（族 4）を足したとき上限が 3 のままで、族の保護が一度も
            // 効いていなかった（TLS p1 が 25.457 → 26.765 に伸びた）。
            static constexpr int FAM_MAX = 5;
            // **族ごとに 3 本残す。**1 本だと、標本が族の中の順位を取り違えたときに
            // 全点での最良が守られない。TLS p1 は 向き が最良 25.457 なのに、
            // 1 本だと 向き無傾 25.606、2 本でも 向き半 25.554 になり、
            // 3 本で 25.457 に届く。4 本以上は 1 件も動かない。
            // 代償は符号化の時間で、TLS p1 が 1.10 → 1.61 秒。サイズが第一なので採る。
            // 族ごとに 1 本。**族の中の順位の取り違えは、勝者の族だけを
            // 後から詳しく測って直す**（下の「勝者の族を測り直す」）。
            // 全族を 3 本にすると符号化が 70% 伸びて、サイズは 1 ビットも動かなかった。
            static const size_t FAM_KEEP = [] {
                const char* e = getenv("PCC_FAMKEEP");
                return e ? (size_t)atoi(e) : (size_t)1;
            }();
            for (int fam = 1; fam <= FAM_MAX; ++fam)
                for (int fs = 0; fs < 2; ++fs) {
                    size_t have = 0;
                    for (const auto& c : use)
                        if (family(c) == fam && is_fsym(c) == (fs != 0)) ++have;
                    for (const auto& pr : pre) {
                        if (have >= FAM_KEEP) break;
                        if (family(pr.second) != fam || is_fsym(pr.second) != (fs != 0)) continue;
                        bool dup = false;
                        for (const auto& c : use)
                            if (c.codec == pr.second.codec && c.param == pr.second.param) dup = true;
                        if (dup) continue;
                        if ((double)pr.first <= best_pre * FAM_THR) { use.push_back(pr.second); ++have; }
                    }
                }
            // 族に属さない符号器（delta や range）の記号版も 1 本残す。
            bool have_fsym = false;
            for (const auto& c : use) if (family(c) == 0 && is_fsym(c)) have_fsym = true;
            if (!have_fsym)
                for (const auto& pr : pre)
                    if (family(pr.second) == 0 && is_fsym(pr.second)) {
                        if ((double)pr.first <= best_pre * FAM_THR) use.push_back(pr.second);
                        break;
                    }
            // **ビット版か記号版かは、標本に決めさせない。**
            // 族ごとに 1 本しか残さないと、標本が族の中の順位を取り違えたときに
            // 別の変種の記号版が守られてしまう（AHN4 _20 の 100 万点で、全点なら
            // 勝つ 走査v1記 が残らず 17.727 → 17.864 bpp に伸びた）。
            // 決勝に残った候補それぞれについて、対になる版を必ず一緒に測る。
            // 候補の集合に無い対（PCC_FSYM_PAIR で作らなかったもの）は足さない。
            {
                const size_t nsel = use.size();
                for (size_t i = 0; i < nsel; ++i) {
                    Cand tw{(uint16_t)(use[i].codec ^ C_FSYM_BIT), use[i].param};
                    bool in_univ = false;
                    for (const auto& c : cands)
                        if (c.codec == tw.codec && c.param == tw.param) { in_univ = true; break; }
                    if (!in_univ) continue;
                    bool dup = false;
                    for (const auto& c : use)
                        if (c.codec == tw.codec && c.param == tw.param) { dup = true; break; }
                    if (!dup) use.push_back(tw);
                }
            }
        }
    }

    Stream best; best.cols = cols; best.codec = C_RAW64;
    bool first = true;
    // 標本で順位を付けたとき、全点での 1 位が標本で 3 位まで落ちることがある
    // （AHN3 の gps_time と nir で実際に起きた）。上位を何本か残しておく。
    static const size_t KEEP_ALT = [] {
        const char* e = getenv("PCC_KEEP_ALT");
        return e ? (size_t)atoi(e) : (size_t)3;
    }();
    std::vector<std::pair<size_t, Cand>> rank;      // (符号長, 候補) を短い順に
    std::vector<std::vector<uint8_t>> fb; std::vector<std::string> fe;
    std::vector<char> fok;
    // **標本の上で計画を立てているときは打ち切らない。**打ち切った候補は
    // rank から消え、rank は best.alt（--sample-select が全点で測り直す控え）と
    // 「空間予測を必ず 1 本残す」保証に使われる。標本で負けた候補が全点で
    // 勝つことがあるので、控えが欠けると選択が変わる（autzen-2023 で最大
    // +4.4%、しかも実行ごとに変わった）。標本かどうかは ctx->full で判る。
    const bool on_sample = ctx && ctx->full;
    // 打ち切ると --trace の候補一覧が「不可: 最短を超えたので打ち切り」だらけに
    // なり、どれが落ちるかは実行ごとに変わる。候補を並べて見たいときは 0 にする。
    static const bool ABORT_ON = [] {
        const char* e = getenv("PCC_ABORT");
        return !e || e[0] != '0';
    }();
    std::vector<size_t> fsz;
    if (par) encode_many(use, cv, ctx, fb, fe, fok, ABORT_ON && !on_sample, &fsz);
    // 捨てた出力があるので、勝者の判定は blob の大きさではなく控えた値で行う。
    size_t bestsz = (size_t)-1;
    for (size_t ui = 0; ui < use.size(); ++ui) {
        const Cand& cd = use[ui];
        std::vector<uint8_t> blob; std::string err;
        bool got;
        size_t sz;
        if (par) {
            got = fok[ui] != 0; blob = std::move(fb[ui]); err = fe[ui];
            sz = fsz.empty() ? blob.size() : fsz[ui];
        } else {
            got = codec_encode(cd.codec, cv, cd.param, blob, err, ctx);
            sz = blob.size();
        }
        if (!got) {
            if (trace) *trace += "      " + cand_name(cd.codec, cd.param) + " 不可: " + err + "\n";
            continue;
        }
        if (trace) {
            char m[160];
            snprintf(m, sizeof m, "      %-10s %8.3f bpp\n", cand_name(cd.codec, cd.param).c_str(),
                     f.n ? sz * 8.0 / f.n : 0.0);
            *trace += m;
        }
        rank.push_back({sz, {cd.codec, cd.param}});
        if (sz < bestsz) {
            bestsz = sz;
            best.codec = cd.codec; best.param = cd.param;
            best.data = std::move(blob); first = false;
        }
    }
    // **勝者と同じ族の控えを、後から全点で測り直す。**
    // 標本は族の**中の**順位も取り違える（TLS p1 は 向き が最良 25.457 なのに
    // 標本は 向き無傾 を上位に置き 25.606 になった）。全族を 3 本ずつ厚くすると
    // 符号化が 70% 伸びてサイズは 1 ビットも動かなかったので、
    // **勝った族だけ**を詳しく測る。pre_rank は標本の走査（打ち切り無し）で
    // 作るので決定的である。
    // 何本測り直すかは PCC_FAMALT（既定 4。2026-09-24 に 2 → 4）。控えはまとめて並列に符号化し、
    // 標本の順位の順に比べる（全部を測ってから比べるので、結果は並列の具合に依らない）。
    // 予備選別を外したときの縮み（AHN4 _21 −0.265%、幾何v4W16 → W4）は 4 本で取れ、2 本では取れない。
    // 予備選別の残す本数（PCC_PREKEEP）を 6・8 にしても取れなかった（s123_v22.log）。
    // 3 本目以降の控えで勝者が入れ替わったときの、入れ替わる前の勝者（旗を重ねてから比べ直す）
    Stream fam_backup;
    size_t fam_backup_sz = 0;
    bool have_fam_backup = false;
    if (!first && !pre_rank.empty()) {
        static const size_t FAM_ALT = [] {
            const char* e = getenv("PCC_FAMALT"); return e ? (size_t)atol(e) : (size_t)4; }();
        const int fam = cand_family({best.codec, best.param});
        std::vector<Cand> alt;
        for (const auto& pr : pre_rank) {
            if (alt.size() >= FAM_ALT) break;
            if (cand_family(pr.second) != fam) continue;
            bool done = false;
            for (const auto& c : use)
                if (c.codec == pr.second.codec && c.param == pr.second.param) done = true;
            if (done) continue;
            alt.push_back(pr.second);
        }
        if (!alt.empty()) {
            std::vector<std::vector<uint8_t>> ab; std::vector<std::string> ae; std::vector<char> aok;
            encode_many(alt, cv, ctx, ab, ae, aok, false);
            for (size_t k = 0; k < alt.size(); ++k) {
                if (!aok[k]) continue;
                if (trace) {
                    char m[160];
                    snprintf(m, sizeof m, "      %-10s %8.3f bpp（勝った族の測り直し）\n",
                             cand_name(alt[k].codec, alt[k].param).c_str(),
                             f.n ? ab[k].size() * 8.0 / f.n : 0.0);
                    *trace += m;
                }
                if (ab[k].size() < bestsz) {
                    // **控えは旗を重ねる前の長さで選ぶので、控えを増やすと単調でない。**3 本目以降
                    // （既定 2 本だった頃には測らなかった控え）で入れ替わるときは、入れ替わる前の勝者を
                    // 取っておき、旗を重ねた後の長さで比べ直す（KITTI 20 mm の 20 frame で、控え 4 本が
                    // 2 本より 2,978 byte 長かった）。
                    if (k >= 2 && !have_fam_backup) {
                        fam_backup = best; fam_backup_sz = bestsz; have_fam_backup = true;
                    }
                    bestsz = ab[k].size();
                    best.codec = alt[k].codec; best.param = alt[k].param;
                    best.data = std::move(ab[k]);
                }
            }
        }
    }
    std::sort(rank.begin(), rank.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    // **勝者に旗を重ねた版を、まとめて測る。**
    // **`rank` の上位を使ってはいけない。**打ち切り（enc_best）は候補が終わる順で
    // 効くので、どの候補が rank に残るかはスレッド数で変わる。上位 3 本に広げたら
    // 17 件中 3 件で md5 がスレッド数によって動いた。**勝者は打ち切りに依らず
    // 決まる**（打ち切られた候補は必ず勝者より長い）ので、ここだけなら決定的である。
    // 標本の上では測らない（全点で測り直すので二度手間になる）。
    // 順に測ると勝者の符号化を 2 回ぶん待つので、**まとめて並列に回す**。
    // 組合せ（素通し＋束ね）もここで一緒に測れる。
    // **照合模型（C_MTC_BIT）はここに入れない。**22 列で測って採られたのは 0 列。
    // 桁長の文脈が既に効いているので、当たり外れの 1 ビットぶんだけ損になる。
    // 符号器は残してある。記録は losses.md §17。
    // **どの候補も失敗したら、恒等符号器で書く。**
    // 器の設計は「恒等候補 raw64 を必ず含むので長さは有限で閉じる」だが、
    // `--force-geom` は候補を 1 本に絞るので、その 1 本が失敗すると
    // 恒等候補ごと消える。すると `best` は codec=raw64・data 空のまま書かれ、
    // 復号が「RAW64 の長さが合わない」で落ちる（tls_scan1.ply を
    // `--force-geom 素4B` で強制すると再現した。f64bits の座標は 4 byte に
    // 収まらないので候補が失敗する）。**器を壊すより恒等で書く。**
    if (first) {
        std::vector<uint8_t> ib; std::string ie;
        if (!codec_encode(C_RAW64, cv, {}, ib, ie, ctx)) {
            // 恒等符号器まで失敗することは無いが、黙って壊れた器を書かない。
            if (trace) *trace += "      恒等符号器も失敗した: " + ie + "\n";
        } else {
            best.codec = C_RAW64;
            best.param.clear();
            best.data = std::move(ib);
            bestsz = best.data.size();
            first = false;
            if (trace)
                *trace += "      どの候補も通らなかったので恒等符号器で書いた\n";
        }
    }
    size_t before_fast = 0;
    if (!first && !on_sample) post_flags(best, bestsz, cv, ctx, trace, (size_t)f.n, &before_fast);
    if (have_fam_backup && !first && !on_sample) {
        size_t bf2 = 0;
        post_flags(fam_backup, fam_backup_sz, cv, ctx, trace, (size_t)f.n, &bf2);
        if (fam_backup.data.size() + fam_backup.param.size() < best.data.size() + best.param.size()) {
            if (trace) *trace += "      （3 本目以降の控えで入れ替わった勝者より、元の勝者のほうが旗を重ねた後で短かった）\n";
            best = std::move(fam_backup);
            bestsz = fam_backup_sz;
            before_fast = bf2;
        }
    }
    for (size_t i = 1; i < rank.size() && best.alt.size() < KEEP_ALT; ++i)
        best.alt.push_back(rank[i].second);
    // 標本で順位を付けると、空間予測を使う候補が系統的に不利になる。
    // 点を間引くと近傍が遠くなり、標本上では実力より悪く見えるためである
    // （AHN3 の nir は全点では参照＋空間予測が勝つのに、標本では 4 位にも入らない）。
    // 上位に入らなくても、空間予測を使う最良の候補は必ず 1 つ残す。
    auto uses_spatial = [](const Cand& c) {
        // **旗を剥がしてから比べる。**記号版・素通し版は同じ符号器である
        // （すぐ上の family は剥がしているのに、ここだけ漏れていた）。
        const uint16_t id = (uint16_t)(c.codec & ~C_FLAG_MASK);
        if (id == C_ATTR_SPATIAL || id == C_ATTR_COLOR) return true;
        return id == C_ATTR_XREF && !c.param.empty() &&
               c.param[0] != 0 && c.param[0] != 100;
    };
    auto same = [](const Cand& a, const Cand& b) {
        return a.codec == b.codec && a.param == b.param;
    };
    bool have = uses_spatial({best.codec, best.param});
    for (const auto& a : best.alt) if (uses_spatial(a)) have = true;
    if (!have)
        for (const auto& r : rank) {
            if (!uses_spatial(r.second)) continue;
            bool dup = same(r.second, {best.codec, best.param});
            for (const auto& a : best.alt) if (same(r.second, a)) dup = true;
            if (!dup) { best.alt.push_back(r.second); break; }
        }
    // **速い後半が最後に勝った流れは、候補の比較を速い表でやり直す。**
    // 候補の比較は既定の表（遅い後半）で行うので、場面の変わる流れでは選ばれる候補
    // そのものが変わる。最後の勝者に速い表を重ねても、比べ直すのはその 1 本だけで、
    // 本来の勝者には戻れない（KITTI 200 万点の幾何: 修理後 16.830 → 17.008 bpp に悪化）。
    // 速い表が勝ったこと自体を「場面の変わるデータ」の印に使い、その流れだけ全候補に
    // 旗「速」を付けて選び直す。短いほうを採る。速い表が勝たない流れには費用が無い。
    static const bool FAST_RERUN = [] {
        const char* e = getenv("PCC_FAST_RERUN"); return !e || e[0] != '0'; }();
    // 速い表の勝ち幅の下限（‰。PCC_FAST_RERUN_MIN。符号化器だけの選択で器は変わらない）。
    // 既定は 0（わずかでも勝てば選び直す）。1‰ で AHN5 +0.027%、3‰ で plane +0.117%、
    // 10‰ で符号化 −20% の代わりに 5 件で縮みを失った（losses.md §33）。サイズが第一なので 0。
    static const long FAST_RERUN_MIN = [] {
        const char* e = getenv("PCC_FAST_RERUN_MIN");
        const long v = e ? atol(e) : 0L;
        return v < 0 ? 0L : (v > 1000 ? 1000L : v); }();   // 負や 1000 超えは端に寄せる
    const bool fast_gain_ok = before_fast > 0 &&
        (before_fast - best.data.size()) * 1000 >= (size_t)FAST_RERUN_MIN * before_fast;
    if (FAST_RERUN && fast_gain_ok && !on_sample && !first && (best.codec & C_FAST_BIT) &&
        !candidates.empty() && !(candidates[0].codec & C_FAST_BIT)) {
        std::vector<Cand> fc;
        fc.reserve(candidates.size());
        for (const auto& c : candidates) fc.push_back({(uint16_t)(c.codec | C_FAST_BIT), c.param});
        std::string tr2;
        Stream again = best_stream(f, cols, fc, ctx, trace ? &tr2 : nullptr, preselect);
        if (trace) {
            char m[200];
            snprintf(m, sizeof m, "      （速い後半が勝ったので候補の比較を速い表でやり直した: %s %.3f bpp）\n",
                     cand_name(again.codec, again.param).c_str(),
                     f.n ? again.data.size() * 8.0 / f.n : 0.0);
            *trace += m + tr2;
        }
        if (!again.data.empty() &&
            again.data.size() + again.param.size() < best.data.size() + best.param.size())
            best = std::move(again);
    }
    return best;
}

std::vector<Stream> plan_streams(const Frame& f, bool joint_geom, std::string* log,
                                 const CodecCtx* ctx, bool trace_all) {
    std::vector<Stream> out;
    // 空間予測を使う列が 1 つでもあるか。**これを見ずに表を作ってはいけない。**
    // 幾何だけの入力（属性がすべて定数で落ちるもの）でも「幾何が揃っている」
    // という理由だけで順序表・KD木・P ごとの表を作っていた。100 万点で
    // 0.15 秒と約 80 MB を、一度も引かれない表に使っていたことになる。
    bool any_attr = false;
    for (const auto& c : f.schema) {
        if (joint_geom && c.role == Role::Geometry) continue;
        if (c.storage == Storage::Derived) continue;
        if (joint_geom && (c.name == "point_source_id" || c.name == "gps_time" ||
                           c.name == "bit_fields")) continue;
        if (f.get(c.name)) { any_attr = true; break; }
    }
    const bool want_spatial = ctx && (ctx->world || ctx->want_world) && any_attr;

    // 空間予測の近傍表は、最初にそれを呼ぶ列（たいてい intensity）が 1 人で
    // 背負っている。AHN3 _20 の 100 万点では intensity が 0.35 秒、同じ候補数の
    // red が 0.12 秒で、差の 0.23 秒がこの構築である。幾何の列を符号化している
    // 間に裏で作っておけば隠れる。表の中身は変わらないので出力は同じ。
    std::thread warm;
    if (want_spatial)
        warm = std::thread([ctx, &f] { ctx->ensure(f.n, 1, nullptr); });
    struct Joiner {
        std::thread& t;
        ~Joiner() { if (t.joinable()) t.join(); }
    } joiner{warm};
    std::vector<Cand> cand{{C_RAW64, {}}, {C_RANGE, {}}, {C_RANGE_DELTA, {}},
                           {C_RANGE_CTX, {}}, {C_RANGE_CTX2, {}}, {C_RANGE_MED, {}},
                           {C_RANGE_CACHE, {}}, {C_RANGE_PREV, {}}};
    // PCC_SYM=0 で字母符号器の候補を全部切る。入れた前後を比べるために置く。
    static const bool SYM_ON = [] {
        const char* e = getenv("PCC_SYM");
        return !e || atoi(e) != 0;
    }();
    if (SYM_ON) cand.push_back({C_RANGE_SYM, {}});
    // 幾何が揃っていれば、空間予測の候補（予測子 1 / 3 / 5 個）も加える
    // 予測子の数。既定は 1 / 3 / 5。上に余地が無いかは PCC_SPATIAL_PS で測れる。
    static const std::vector<uint8_t> SPS = [] {
        std::vector<uint8_t> v;
        if (const char* e = getenv("PCC_SPATIAL_PS")) {
            for (const char* q = e; *q;) {
                long x = strtol(q, (char**)&q, 10);
                if (x > 0 && x < 64) v.push_back((uint8_t)x);
                while (*q && (*q < '0' || *q > '9')) ++q;
            }
        }
        if (v.empty()) v = {1, 3, 5};
        return v;
    }();
    std::vector<Cand> cand_attr = cand;
    if (want_spatial)
        for (uint8_t P : SPS) cand_attr.push_back({C_ATTR_SPATIAL, {P}});
    // 順序の実験では幾何だけが関心で、属性の候補掃引が時間の大半を占める。
    // 絞っても往復検証は全列に掛かるので、検証の強さは落ちない。
    // 幾何より前に置く列（gps_time など）も cand から作るので、ここで一緒に絞られる。
    // 幾何の候補 gc は別に組むので影響しない。
    if (ctx && ctx->fast_attr) {
        cand = {{C_RANGE_DELTA, {}}, {C_RANGE_CTX, {}}};
        cand_attr = cand;
    }

    // 列を 1 本ずつ符号化すると、列の中の候補が終わりかけたところで機械が空く。
    // 仕事をいったん貯めてから、列も候補もまとめて 1 つの待ち行列に流す。
    // **どの列にどの候補を出すかは列の名前と並び順だけで決まる**ので、貯める側は
    // 結果を 1 つも要らない。唯一の例外は色の同時符号化と鎖の比較で、これは
    // 両方を仕事にしておいて、走らせたあとに短いほうを採る。
    struct Job {
        std::vector<std::string> cols;
        std::vector<Cand> cands;
        bool presel = false;
        Stream out;
        std::string tr;
        double sec = 0;
        std::string fail;              // 例外で終わった仕事の理由（空なら成功）
    };
    const double t_plan0 = now_sec();
    const size_t NT = pool_threads();
    std::unique_ptr<Pool> pool;
    Pool* const prev_pool = g_pool;
    if (NT > 1) { pool.reset(new Pool(NT)); g_pool = pool.get(); }
    // 入れ子で呼ばれても外側のプールを消さないように、元の値に戻す。
    struct PoolGuard { Pool* p; ~PoolGuard() { g_pool = p; } } pool_guard{prev_pool};
    std::vector<Job> jobs;
    std::vector<size_t> order;                 // 出す順。SIZE_MAX は色の判定の位置
    auto defer = [&](std::vector<std::string> cs, std::vector<Cand> cd,
                     bool presel = false) {
        jobs.push_back(Job{std::move(cs), std::move(cd), presel, {}, {}, 0, {}});
        return jobs.size() - 1;
    };

    // 幾何より前に置く列。復号器はこれらを使って走査順・掃引・戻り番号を
    // 組み直すので、幾何の符号器に副情報を持たせずに済む。
    // これらの列自身は空間予測を候補に持てない（座標がまだ無い）。
    std::vector<std::string> pre;
    if (joint_geom)
        // 走査モデルが必要とする 3 列に限る（前に置いた列は空間予測を使えなくなる）
        for (const char* nm : {"point_source_id", "gps_time", "bit_fields"}) {
            const ColSpec* sp = f.spec(nm);
            if (sp && sp->storage == Storage::Raw && f.get(nm)) pre.push_back(nm);
        }
    std::vector<std::string> pre_done;
    for (const auto& nm : pre) {
        std::vector<Cand> cs = cand;
        for (const auto& e : pre_done) {          // 既に出した列との残差も候補に
            uint16_t l = (uint16_t)e.size();
            for (uint8_t P : {0, 100}) {
                std::vector<uint8_t> pv{P};
                pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                pv.insert(pv.end(), e.begin(), e.end());
                cs.push_back({C_ATTR_XREF, pv});
            }
            // 既出の列を「文脈」に使う字母符号器。残差ではなく文脈なので、
            // 相関が単調でなくても効く（強度は戻り総数で分布が変わる）。
            if (SYM_ON) for (uint8_t sh : {0, 4}) {
                std::vector<uint8_t> pv{sh};
                pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                pv.insert(pv.end(), e.begin(), e.end());
                cs.push_back({C_RANGE_SYM, pv});
            }
        }
        order.push_back(defer({nm}, cs));
        pre_done.push_back(nm);
    }
    std::string aux = pre.empty() ? std::string() : pre.back();

    if (joint_geom) {
        std::vector<std::string> g{f.geom[0], f.geom[1], f.geom[2]};
        std::vector<Cand> gc{{C_RAW64, {}}, {C_RAW_W, raw_w_param(FType::I32)},
                             {C_RANGE, {}}, {C_RANGE_DELTA, {}},
                             {C_GEOM_CROSS, {}},
                             // y をその点の dx から予測する版（autzen-2023 の
                             // y 軸で −1.0 bit の見積り）。
                             // **param の予測子は x と z にしか効かない**
                             // （y は常に向き追従で、mp[1] は使われない）。
                             {C_GEOM_DIR, {0}}, {C_GEOM_DIR, {2}}, {C_GEOM_DIR, {6}},
                             // 主成分に合わせて座標を回してから符号化する。
                             // AHN3 −5.0%、TLS p1 −2.5% の実測。
                             {C_GEOM_ROT, {0}}, {C_GEOM_ROT, {2}}, {C_GEOM_ROT, {6}},
                             // **区間ごとの切り替え（C_GEOM_SEG）は候補から外した。**
                             // workshop 19.120 → 21.862 など、測った全部で負けた。
                             // 見積りの相手が素朴な差分だったのが原因で、実際の
                             // 勝者（幾何v4・走査・向き・回転）は区間で切り替える
                             // Pred の 8 通りよりずっと強い。記録は losses.md §16。
                             // **内挿（C_GEOM_LIFT）は候補から外してある。**
                             // 10 ファイルで測って全部負けた（+6.4〜27.0%）。
                             // 段数を 1 まで浅くしても変わらない。理由は
                             // `results/literature_2026.md` §7 と losses.md §15。
                             // 符号器は残してあるので、粗い層に強い予測子を使う形で
                             // やり直すときは候補に戻すだけでよい。
                             {C_GEOM_XYZ, {0}}, {C_GEOM_XYZ, {1}}, {C_GEOM_XYZ, {3}},
                             // 予測子だけを差し替えた 幾何v3。副情報は増えない。
                             // 直近 4 つの差分の平均は TLS と AHN3 で、中央値の
                             // 半分は autzen-2023 で残差が短かった（符号化前の見積り）。
                             {C_GEOM_XYZ, {3, 1}}, {C_GEOM_XYZ, {3, 2}},
                             // z の文脈を (k0+k1)/2 の平均でなく 2 次元の組にした版。
                             // 平均は k0=2,k1=6 と k0=6,k1=2 を混ぜてしまう。
                             {C_GEOM_XYZ, {3, 4}}, {C_GEOM_XYZ, {3, 6}},
                             {C_GEOM_XYZ, {3, 5}},
                             // 直近 5 つの差分の中央値。LASzip が座標に使う予測子で、
                             // 3 つだと 1 点の外れ値で予測が振れる。
                             {C_GEOM_XYZ, {3, 3}}, {C_GEOM_XYZ, {3, 7}},
                             // 文脈を粗くする段。LASzip は dx に 2 文脈、dy/dz に
                             // 20 文脈しか使わない。こちらは 3 軸 × 24 文脈あるので、
                             // 点数が少ないファイルでは適応模型が薄まる。
                             {C_GEOM_XYZ, {3, 16}}, {C_GEOM_XYZ, {3, 18}},
                             {C_GEOM_XYZ, {3, 34}}, {C_GEOM_XYZ, {3, 22}},
                             // 下位ビットの模型を桁数の文脈と分け、軸ごとに共有する版。
                             {C_GEOM_XYZ, {3, 64}}, {C_GEOM_XYZ, {3, 66}},
                             {C_GEOM_XYZ, {3, 70}}, {C_GEOM_XYZ, {3, 82}},
                             // 傾きの効かせ方（1/4・3/4・足さない・中央値 5 の半分）。
                             {C_GEOM_XYZ, {3, 128}}, {C_GEOM_XYZ, {3, 129}},
                             {C_GEOM_XYZ, {3, 130}}, {C_GEOM_XYZ, {3, 131}},
                             {C_GEOM_XYZ, {3, 134}}, {C_GEOM_XYZ, {3, 146}},
                             // 直近 W 点の最近傍から予測する。格納順が空間的に
                             // 連続していない入力で効く（幾何v4 の注記を参照）。
                             {C_GEOM_XYZ, {4, 4}}, {C_GEOM_XYZ, {4, 16}},
                             // 窓を広げると、格納順が空間的に飛ぶ入力で効く
                             // （vegetation は窓 16 で 27.6、4096 で 21.9 bit の見積り）。
                             // 費用は点あたり O(W) なので、実測で割に合うときだけ残る。
                             {C_GEOM_XYZ, {4, 64}}, {C_GEOM_XYZ, {4, 255}},
                             // 選んだ点に局所の傾きを足す。副情報は増えない。
                             {C_GEOM_XYZ, {4, 4, 1}}};
        if (!aux.empty()) {
            std::vector<uint8_t> pv{2};
            uint16_t l = (uint16_t)aux.size();
            pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
            pv.insert(pv.end(), aux.begin(), aux.end());
            gc.push_back({C_GEOM_XYZ, pv});
        }
        // 幾何v3（軸をまたぐ文脈）に、**予測子の状態を補助列ごとに分ける**版を足す。
        // 幾何v2 は状態を分けるが軸をまたぐ文脈を持たず、幾何v3 は逆で、
        // 両方を同時に持つ候補が無かった。LASzip は両方を持っている。
        // 1 発から複数の点が返る入力では、格納順の隣が走査線上の隣ではないので、
        // 戻り番号ごとに繋ぎ直さないと予測子が混ざる。
        {
            std::string ga;
            for (const auto& e : pre_done) if (e == "bit_fields") ga = e;
            if (ga.empty()) ga = aux;
            if (!ga.empty()) {
                uint16_t l = (uint16_t)ga.size();
                // 覆い 7 は戻り番号だけ、63 は戻り番号と戻り総数まで。
                for (uint8_t mask : {7, 63})
                    for (uint8_t pm : {8, 10, 11, 12, 14, 15}) {
                        std::vector<uint8_t> pv{3, pm, mask};
                        pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                        pv.insert(pv.end(), ga.begin(), ga.end());
                        gc.push_back({C_GEOM_XYZ, pv});
                    }
            }
        }
        // 走査モデル: gps_time と point_source_id が前に出ていれば候補にできる
        bool has_g = false, has_p = false;
        for (const auto& e : pre_done) {
            if (e == "gps_time") has_g = true;
            if (e == "point_source_id") has_p = true;
        }
        if (has_g && has_p) {
            std::string ret;
            for (const auto& e : pre_done) if (e == "bit_fields") ret = e;
            std::vector<uint8_t> pv{1};
            auto add = [&](const std::string& nm) {
                uint16_t l = (uint16_t)nm.size();
                pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                pv.insert(pv.end(), nm.begin(), nm.end());
            };
            add("gps_time"); add("point_source_id"); add(ret);
            // v2（z を中央値予測）と v3（z の文脈を面内・面外から作る）と v4 も出す。
            // 以前は既定で外していたが、**その根拠だった集計は誤りだった**
            // （results/scan_model_fitting.md 16・32 節）。測り直したところ、
            // 記号版を標本から守るようにした上で全部出すと、悪化が 1 件も無く
            // AHN4 _21 −0.13% / AHN5 _20 −0.25% / fullwave −0.28% が縮んだ。
            // 代償は符号化が 10.9 倍 → 12.7 倍。サイズが第一なので出す。
            std::vector<uint8_t> vars{1, 2, 3, 4, 5};
            if (const char* e = getenv("PCC_ALL_VARIANTS"))
                if (e[0] == '0') vars = {1, 5};
            for (uint8_t v : vars) {
                std::vector<uint8_t> q = pv; q[0] = v;
                gc.push_back({C_GEOM_SCAN, q});
            }
        }
        if (ctx && !ctx->force_geom.empty()) {
            // 名前の末尾の旗（後追いで付くもの）を剥がして元の候補を探し、旗を戻す。
            // 以前は旗つきの名前（例: 幾何v4符2）を渡すと「候補にならない」になっていた。
            std::string base = ctx->force_geom;
            uint16_t fl = 0;
            auto strip = [&](const char* sfx, uint16_t bits) {
                const size_t L = strlen(sfx);
                if (base.size() > L && base.compare(base.size() - L, L, sfx) == 0) {
                    base.resize(base.size() - L); fl |= bits; return true;
                }
                return false;
            };
            // cand_name が付ける順（内→外: 記・符・光・面・束・照・生・速）の逆に剥がす
            strip("速", C_FAST_BIT);
            strip("生", C_RAW_BIT);
            strip("照", C_MTC_BIT);
            strip("束", C_BND_BIT);
            if (!strip("面6", C_SURF_A)) if (!strip("面7", C_SURF_B)) strip("面8", C_SURF_A | C_SURF_B);
            strip("光", C_RAY_BIT);
            if (!strip("符1", C_LMS_BIT)) if (!strip("符2", C_SGN2_BIT)) strip("符4", C_LMS_BIT | C_SGN2_BIT);
            strip("記", C_FSYM_BIT);          // 記号版は best_stream が対にして作る
            std::vector<Cand> only;
            for (const auto& c : gc) {
                if (cand_name(c.codec, c.param) == ctx->force_geom) { only.push_back(c); continue; }
                if (fl && cand_name(c.codec, c.param) == base)
                    only.push_back({(uint16_t)(c.codec | fl), c.param});
            }
            if (only.empty()) {
                if (log) *log += "  （--force-geom " + ctx->force_geom +
                                 " はこの入力では候補にならない）\n";
            } else {
                gc.swap(only);
            }
        }
        // 旗「類」の相手（幾何）: 幾何より先に出る前置きの列と、gps_time の差（「Δgps_time」）。
        // 時刻が跳ぶ所（走査線の変わり目）では残差が大きい、を文脈にする。
        if (ctx && !g.empty()) {
            std::vector<std::string> cc;
            for (const auto& e : pre_done) {
                if (!f.get(e)) continue;
                cc.push_back(e);
                if (e == "gps_time") cc.push_back("\xCE\x94gps_time");
            }
            ctx->ctx_cols[g[0]] = cc;
        }
        order.push_back(defer(g, gc, true));           // 幾何だけ事前選別する
    }

    // 色 3 列が揃っていれば、可逆色変換つきの同時符号化を候補に加える。
    // 色の脱相関と空間予測は独立に効き、重ねると掛け算になる（実測 6.430 対 11.179）。
    // 色 3 列は、可逆色変換つきで**まとめて**符号化したほうが短いことがある。
    // ただし 3 列を汎用の経路から外すと、参照残差の相手を選ぶ自由が失われる。
    // **どちらが短いかは測って決める。**まとめた 1 本と、汎用の経路が出す 3 本の
    // 合計を比べ、短いほうを出す。まとめた側が長ければ何も失わない。
    //
    // 以前は「world があるとき」だけこの節に入っていたが、通常の経路は world を
    // 遅延構築にするので world は空のままで、**この候補は一度も出ていなかった**。
    // 代わりに節ごと有効にすると、汎用の経路から 3 列を外すことになり、
    // 15 件中 6 件で伸びた（AHN5 _20 +1.9%、simple1_4 +2.8%）。
    std::vector<std::string> rgb{"red", "green", "blue"};
    size_t s_joint = (size_t)-1;
    std::vector<size_t> s_rgb, s_chain;
    if (want_spatial) {
        bool all = true;
        for (const auto& c : rgb) {
            const ColSpec* sp = f.spec(c);
            if (!sp || sp->storage != Storage::Raw || !f.get(c)) all = false;
        }
        if (all) {
            std::vector<Cand> cc;
            for (uint8_t P : SPS) cc.push_back({C_ATTR_COLOR, {P}});
            s_joint = defer(rgb, cc);

            // 3 つめの案: blue → green → red の順に、直前の色だけを参照する鎖。
            // 汎用の経路は参照の相手を見積りで 2 本に絞るので、色どうしの鎖が
            // そこに入らないことがある。fullwave がそれで、鎖のほうが 0.5% 短い。
            const char* ord[3] = {"blue", "green", "red"};
            for (int i = 0; i < 3; ++i) {
                std::vector<Cand> cs = cand_attr;
                if (i > 0) {
                    std::string ref = ord[i - 1];
                    uint16_t l = (uint16_t)ref.size();
                    std::vector<uint8_t> cps{0, 100};
                    cps.insert(cps.end(), SPS.begin(), SPS.end());
                    for (uint8_t P : cps) {
                        std::vector<uint8_t> pv{P};
                        pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                        pv.insert(pv.end(), ref.begin(), ref.end());
                        cs.push_back({C_ATTR_XREF, pv});
                        if (P == 0 && SYM_ON) {   // 文脈として使う版も出す
                            for (uint8_t sh : {0, 4}) {
                                std::vector<uint8_t> sv{sh};
                                sv.insert(sv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                                sv.insert(sv.end(), ref.begin(), ref.end());
                                cs.push_back({C_RANGE_SYM, sv});
                            }
                        }
                    }
                }
                s_chain.push_back(defer({ord[i]}, cs));
            }
        }
    }

    // 参照の候補を測る段が符号化の 6 割を占めていた。列ごとに小分けに流すと、
    // 先頭の列は測る相手が 1〜2 本しか無く、その間はコアが空く。
    // **測る組を先に全部数えあげて 1 回で流す。**どの組を測るかは列の名前と
    // 並び順だけで決まるので、結果は 1 つも要らない。
    const Frame& fs_ref = (ctx && ctx->full) ? *ctx->full : f;
    auto skip_col = [&](const ColSpec& c) {
        if (joint_geom && c.role == Role::Geometry) return true;
        if (c.storage == Storage::Derived) return true;
        for (const auto& e : pre_done) if (e == c.name) return true;
        return false;
    };
    std::map<std::pair<std::string, std::string>, double> escore;
    {
        std::vector<std::string> em = pre_done;
        for (const auto& c : f.schema) {
            if (skip_col(c)) continue;
            const auto* v = f.get(c.name);
            const auto* vr = fs_ref.get(c.name);
            if (v && vr)
                for (const auto& e : em)
                    if (fs_ref.get(e)) escore[{c.name, e}] = 0.0;
            // 旗「類」の相手（参照残差と同じ規則: 前置きの列と、先に出る属性）
            // 色の 3 列は「blue → green → red」の鎖でスキーマと逆順に依存しうるので、互いを相手にしない
            // （blue が 類<green> を、green が 参照blue を取ると循環して復号できない）。
            if (ctx && v) {
                auto is_rgb = [](const std::string& x) { return x == "red" || x == "green" || x == "blue"; };
                std::vector<std::string> cc;
                for (const auto& e : em)
                    if (f.get(e) && !(is_rgb(c.name) && is_rgb(e))) cc.push_back(e);
                ctx->ctx_cols[c.name] = cc;
            }
            em.push_back(c.name);
        }
        std::vector<std::pair<const Col*, std::pair<const Col*, double*>>> tk;
        for (auto& kv : escore)
            tk.push_back({fs_ref.get(kv.first.first),
                          {fs_ref.get(kv.first.second), &kv.second}});
        if (pool && tk.size() > 1) {
            std::atomic<size_t> left{0};
            for (size_t i = 0; i < tk.size(); ++i)
                pool->add([&tk, i] {
                    *tk[i].second.second =
                        entropy_diff_sample(*tk[i].first, *tk[i].second.first, 250000);
                }, left);
            pool->help_until(left);
        } else {
            for (size_t i = 0; i < tk.size(); ++i)
                *tk[i].second.second =
                    entropy_diff_sample(*tk[i].first, *tk[i].second.first, 250000);
        }
    }

    // 空間予測を掛けたあとの乱雑さでも順位を付ける。差そのものの順位とは別物で、
    // AHN4 _20 の nir は green が差では 5 位、空間予測の後では上位に来る。
    // 座標が要るので、幾何が揃っているときだけ。
    std::map<std::pair<std::string, std::string>, double> escore_sp;
    if (want_spatial && !escore.empty()) {
        const std::vector<int32_t>* pm = nullptr;
        const std::vector<int32_t>* pdt = ctx->ensure(f.n, 3, &pm);
        if (pdt && pm) {
            for (const auto& kv : escore)
                if (f.get(kv.first.first) && f.get(kv.first.second))
                    escore_sp[kv.first] = 0.0;
            std::vector<std::pair<const Col*, std::pair<const Col*, double*>>> tk;
            for (auto& kv : escore_sp)
                tk.push_back({f.get(kv.first.first), {f.get(kv.first.second), &kv.second}});
            auto one = [&](size_t i) {
                *tk[i].second.second = entropy_spatial_diff(
                    *tk[i].first, *tk[i].second.first, *pm, *pdt, 3, f.n, 250000);
            };
            if (pool && tk.size() > 1) {
                std::atomic<size_t> left{0};
                for (size_t i = 0; i < tk.size(); ++i) pool->add([&one, i] { one(i); }, left);
                pool->help_until(left);
            } else {
                for (size_t i = 0; i < tk.size(); ++i) one(i);
            }
        }
    }

    std::vector<std::string> emitted = pre_done;              // 既に出した列（参照に使える）
    // `name#0` `name#1` …（ExtraBytes の配列型・波形パケットを 1 byte ずつ持つ列）は
    // 他の属性と同じ候補を全部試す。**絞る案は 2 つとも測って落とした。**
    //   まとめて 1 本の流れにする  … fullwave 125.8 → 137.5 bpp
    //   空間予測の候補を外す        … fullwave 125.8 → 128.9 bpp
    // どちらも符号化は速くなるが、サイズが第一なので採らない。

    for (const auto& c : f.schema) {
        if (skip_col(c)) continue;
        std::vector<Cand> cs = cand_attr;
        // **その列の自然な幅でそのまま格納する候補。**恒等候補 C_RAW64 は 64 bit
        // なので、1 byte の列の天井になっていなかった（fullwave の波形バイトが
        // 素の 8.000 より膨らんで 8.177 bpp になっていた）。
        cs.push_back({C_RAW_W, raw_w_param(c.ftype)});
        // 既出の列との残差も候補に入れる。標本で相手を 2 つに絞ってから全点で測る。
        // 参照の選定は全点の統計で行う（標本で選ぶと候補の集合が変わる）
        const auto* v = f.get(c.name);
        if (v && !emitted.empty()) {
            std::vector<std::pair<double, std::string>> sc;
            for (const auto& e : emitted) {
                auto it = escore.find({c.name, e});
                if (it != escore.end()) sc.push_back({it->second, e});
            }
            std::sort(sc.begin(), sc.end());
            std::vector<std::pair<double, std::string>> sp;
            for (const auto& e : emitted) {
                auto it = escore_sp.find({c.name, e});
                if (it != escore_sp.end()) sp.push_back({it->second, e});
            }
            std::sort(sp.begin(), sp.end());
            if (getenv("PCC_EPROF")) {
                auto dump = [&](const char* tag,
                                const std::vector<std::pair<double, std::string>>& z) {
                    std::string m = std::string("  [参照の見積り ") + tag + "] " + c.name + ":";
                    for (const auto& q : z) {
                        char b[96];
                        snprintf(b, sizeof b, " %s=%.3f", q.second.c_str(), q.first);
                        m += b;
                    }
                    fprintf(stderr, "%s\n", m.c_str());
                };
                dump("差", sc);
                dump("空間", sp);
            }
            // 参照の相手を何本まで候補にするか。既定は 4 本（2026-09-24 に 2 → 4）。
            // **2 つの見積りは別のものを測っているので、それぞれの上位を採る。**
            // 全部試したときの縮み（15 件合計 −0.030%、8/15 件）を 4 本で全部取れる。
            // 2 本だと 8 件で取りこぼし（AHN4 _20 −0.114%）、3 本でも 1 件残る（select_loss_v22.log・s123_v22.log）。
            static const size_t XKEEP = [] {
                const char* e = getenv("PCC_XREF_KEEP");
                return e ? (size_t)atol(e) : (size_t)4;
            }();
            std::vector<std::string> refs;
            auto take = [&](const std::vector<std::pair<double, std::string>>& z) {
                for (size_t i = 0; i < z.size() && i < XKEEP; ++i) {
                    bool dup = false;
                    for (const auto& r : refs) if (r == z[i].second) dup = true;
                    if (!dup) refs.push_back(z[i].second);
                }
            };
            take(sc);
            take(sp);
            for (const auto& ref : refs) {
                uint16_t l = (uint16_t)ref.size();
                std::vector<uint8_t> ps{0, 100};
                ps.insert(ps.end(), SPS.begin(), SPS.end());
                for (uint8_t P : ps) {
                    std::vector<uint8_t> pv{P};
                    pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                    pv.insert(pv.end(), ref.begin(), ref.end());
                    cs.push_back({C_ATTR_XREF, pv});
                }
                // 既出の列を「文脈」に使う字母符号器。残差ではなく文脈なので、
                // 相関が単調でなくても効く（強度は戻り総数で分布が変わる）。
                if (SYM_ON) for (uint8_t sh : {0, 4}) {
                    std::vector<uint8_t> sv{sh};
                    sv.insert(sv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                    sv.insert(sv.end(), ref.begin(), ref.end());
                    cs.push_back({C_RANGE_SYM, sv});
                }
            }
        }
        {
            size_t slot = defer({c.name}, cs);
            order.push_back(slot);
            if (c.name == "red" || c.name == "green" || c.name == "blue")
                s_rgb.push_back(slot);
        }
        emitted.push_back(c.name);
    }

    // 貯めた仕事をまとめて流す。列の仕事から出た候補の仕事も同じ待ち行列に入るので、
    // 遅い候補が 1 本残った列の横で、別の列の候補が空いたコアを埋める。
    {
        const bool eprof = getenv("PCC_EPROF") != nullptr;
        if (eprof) fprintf(stderr, "  [仕事を貯める] %.3fs  仕事 %zu\n",
                           now_sec() - t_plan0, jobs.size());
        double t_run = now_sec();
        if (!pool || jobs.size() <= 1) {
            for (auto& j : jobs) {
                double t0 = now_sec();
                try {
                    j.out = best_stream(f, j.cols, j.cands, ctx, trace_all ? &j.tr : nullptr,
                                        j.presel);
                } catch (const std::exception& e) { j.fail = e.what(); }
                j.sec = now_sec() - t0;
            }
        } else {
            std::atomic<size_t> left{0};
            for (auto& j : jobs)
                pool->add([&f, &j, ctx, trace_all] {
                    double t0 = now_sec();
                    try {
                        j.out = best_stream(f, j.cols, j.cands, ctx,
                                            trace_all ? &j.tr : nullptr, j.presel);
                    } catch (const std::exception& e) { j.fail = e.what(); }
                    j.sec = now_sec() - t0;
                }, left);
            pool->help_until(left);
        }
        // **失敗した列があれば器を作らない。**列の無い流れを書くと、--no-verify では
        // 壊れた器がそのまま残る（例: 大きな入力でメモリーが尽きたとき）。
        for (const auto& j : jobs)
            if (!j.fail.empty()) {
                std::string nm;
                for (size_t i = 0; i < j.cols.size(); ++i) nm += (i ? "+" : "") + j.cols[i];
                throw std::runtime_error("列 " + nm + " の符号化に失敗した: " + j.fail);
            }
        if (eprof) fprintf(stderr, "  [仕事を流す] %.3fs\n", now_sec() - t_run);
    }

    // 列ごとの秒は、列を並列に走らせたので重なっている。和は全体より大きくなる。
    auto emit = [&](Job& j) {
        if (log) {
            std::string nm;
            for (size_t i = 0; i < j.out.cols.size(); ++i) nm += (i ? "+" : "") + j.out.cols[i];
            char m[256];
            snprintf(m, sizeof m, "  %-22s %-8s %8.3f bpp %7.2fs\n", nm.c_str(),
                     cand_name(j.out.codec, j.out.param).c_str(),
                     f.n ? j.out.data.size() * 8.0 / f.n : 0.0, j.sec);
            *log += m;
            if (trace_all) *log += j.tr;
        }
        out.push_back(std::move(j.out));
    };
    // 色 3 列の出し方を 3 通りで比べる。0 = 汎用の経路、1 = まとめる、2 = 鎖。
    int color_pick = 0;
    if (s_joint != (size_t)-1 && s_rgb.size() == 3 && s_chain.size() == 3) {
        size_t sep = 0, jnt = jobs[s_joint].out.data.size(), chn = 0;
        for (size_t c : s_rgb) sep += jobs[c].out.data.size();
        for (size_t c : s_chain) chn += jobs[c].out.data.size();
        if (jnt && jnt < sep) { color_pick = 1; sep = jnt; }
        if (chn && chn < sep) { color_pick = 2; }
        if (log) {
            char m[200];
            snprintf(m, sizeof m,
                     "  （色 3 列: 別々 %zu / まとめて %zu / 鎖 %zu byte → %s）\n",
                     sep, jnt, chn,
                     color_pick == 0 ? "別々" : (color_pick == 1 ? "まとめる" : "鎖"));
            *log += m;
        }
    }
    bool color_out = false;
    for (size_t k : order) {
        if (color_pick && (k == s_rgb[0] || k == s_rgb[1] || k == s_rgb[2])) {
            if (!color_out) {
                if (color_pick == 1) emit(jobs[s_joint]);
                else for (size_t c : s_chain) emit(jobs[c]);
                color_out = true;
            }
            continue;
        }
        emit(jobs[k]);
    }
    return out;
}

// ---------------------------------------------------------------- 書き出し / 読み込み
enum : uint16_t {
    T_SRC_KIND = 1, T_SRC_BYTES = 2, T_SCALE = 3, T_OFFSET = 4, T_SCHEMA = 5,
    T_PLAN = 6, T_FIDELITY = 7, T_ENVELOPE = 8, T_GEOM_COLS = 9, T_GEOM_REPR = 10,
    T_EMBED = 11, T_COLDIV = 12, T_POLAR = 13
};

static void put_tag(std::vector<uint8_t>& h, uint16_t tag, const std::vector<uint8_t>& body) {
    put<uint16_t>(h, tag);
    put<uint32_t>(h, (uint32_t)body.size());
    h.insert(h.end(), body.begin(), body.end());
}

// 計画を二値にする。JSON のままだと 1 万点のファイルで 333 byte = 0.25 bpp かかる。
// 効くのは**列名を schema の添字に置き換えること**で、"classification" の 14 byte が
// 1 byte になる。直前のバイトを文脈にした符号化も試したが、この長さでは
// 256 の文脈を学習しきれず縮まなかった（333 → 330 byte）。
namespace {
const char* const OP_KINDS[] = {"drop_constant", "drop_duplicate",
                                "drop_affine", "residual_code"};
constexpr int N_OP_KINDS = 4;

void put_varint(std::vector<uint8_t>& o, uint64_t v) {
    while (v >= 0x80) { o.push_back((uint8_t)(v | 0x80)); v >>= 7; }
    o.push_back((uint8_t)v);
}
bool get_varint(const std::vector<uint8_t>& b, size_t& p, uint64_t& v) {
    v = 0; int sh = 0;
    while (p < b.size()) {
        uint8_t c = b[p++];
        v |= (uint64_t)(c & 0x7F) << sh;
        if (!(c & 0x80)) return true;
        sh += 7;
        if (sh > 63) return false;
    }
    return false;
}

// 列名 → schema の添字。無ければ 0xFFFF を置いて名前を並べる。
int name_index(const Frame& f, const std::string& nm) {
    for (size_t i = 0; i < f.schema.size(); ++i)
        if (f.schema[i].name == nm) return (int)i;
    return -1;
}

std::vector<uint8_t> plan_to_binary(const Frame& f) {
    const Plan pl = Plan::from_json(f.plan);
    std::vector<uint8_t> o;
    put<uint8_t>(o, pl.grid ? 1 : 0);
    put_varint(o, (uint64_t)(pl.grid_bits < 0 ? 0 : pl.grid_bits));
    put<double>(o, pl.max_err_m);
    put_varint(o, pl.ops.size());
    for (const auto& op : pl.ops) {
        int k = 0;
        for (int i = 0; i < N_OP_KINDS; ++i) if (op.kind == OP_KINDS[i]) k = i;
        put<uint8_t>(o, (uint8_t)k);
        for (const std::string* nm : {&op.target, &op.source}) {
            int idx = nm->empty() ? -2 : name_index(f, *nm);
            if (idx >= 0) { put<uint8_t>(o, 0); put_varint(o, (uint64_t)idx); }
            else if (idx == -2) { put<uint8_t>(o, 1); }        // 空
            else { put<uint8_t>(o, 2); put_str(o, *nm); }       // schema に無い
        }
        put_varint(o, zigzag(op.value));
        put_varint(o, zigzag(op.num));
        put_varint(o, zigzag(op.den));
        put_varint(o, zigzag(op.b));
        put<uint8_t>(o, op.exact ? 1 : 0);
    }
    return o;
}

bool plan_from_binary(const std::vector<uint8_t>& b, const Frame& f, std::string& out) {
    size_t p = 0;
    auto g8 = [&](uint8_t& v) { if (p >= b.size()) return false; v = b[p++]; return true; };
    Plan pl;
    uint8_t gr = 0;
    if (!g8(gr)) return false;
    pl.grid = gr != 0;
    uint64_t v = 0;
    if (!get_varint(b, p, v)) return false;
    pl.grid_bits = (int)v;
    if (p + 8 > b.size()) return false;
    memcpy(&pl.max_err_m, b.data() + p, 8); p += 8;
    if (!get_varint(b, p, v)) return false;
    const size_t nop = (size_t)v;
    for (size_t i = 0; i < nop; ++i) {
        Op op;
        uint8_t k = 0;
        if (!g8(k) || k >= N_OP_KINDS) return false;
        op.kind = OP_KINDS[k];
        for (std::string* nm : {&op.target, &op.source}) {
            uint8_t how = 0;
            if (!g8(how)) return false;
            if (how == 0) {
                if (!get_varint(b, p, v) || v >= f.schema.size()) return false;
                *nm = f.schema[(size_t)v].name;
            } else if (how == 1) {
                nm->clear();
            } else {
                uint16_t l = 0;
                if (p + 2 > b.size()) return false;
                memcpy(&l, b.data() + p, 2); p += 2;
                if (p + l > b.size()) return false;
                nm->assign((const char*)b.data() + p, l); p += l;
            }
        }
        uint64_t z = 0;
        if (!get_varint(b, p, z)) return false;
        op.value = unzigzag(z);
        if (!get_varint(b, p, z)) return false;
        op.num = unzigzag(z);
        if (!get_varint(b, p, z)) return false;
        op.den = unzigzag(z);
        if (!get_varint(b, p, z)) return false;
        op.b = unzigzag(z);
        uint8_t ex = 0;
        if (!g8(ex)) return false;
        op.exact = ex != 0;
        pl.ops.push_back(std::move(op));
    }
    out = pl.to_json();
    return true;
}
}  // namespace

bool reencode_matches(const Frame& f, const std::vector<Stream>& st, const CodecCtx& proto,
                      std::string& diff) {
    // 選択で温まった表（近傍表・走査の文脈・字母）を使い回すと、表の作り方が
    // 揺れていても同じ表から同じ出力が出て、揺れを見逃す。設定だけ写した新しい文脈で回す。
    CodecCtx ctx;
    ctx.world = proto.world;
    ctx.want_world = proto.want_world;
    ctx.fr = proto.fr ? proto.fr : &f;
    ctx.bitfields_first = proto.bitfields_first;
    ctx.fast_attr = proto.fast_attr;
    // 長い流れ（たいてい幾何）から先に配ると、最後に 1 本だけ残って機械が空く時間が短い
    std::vector<size_t> ord(st.size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(),
                     [&](size_t a, size_t b) { return st[a].data.size() > st[b].data.size(); });
    std::vector<std::string> werr(st.size());
    auto work1 = [&](size_t i) {
        const Stream& s = st[i];
        std::vector<const Col*> cv;
        for (const auto& c : s.cols) {
            const Col* col = f.get(c);
            if (!col) { werr[i] = "列が無い: " + c; return; }
            cv.push_back(col);
        }
        std::vector<uint8_t> blob;
        std::string e;
        if (!codec_encode(s.codec, cv, s.param, blob, e, &ctx)) { werr[i] = "符号化に失敗: " + e; return; }
        if (blob != s.data) {
            werr[i] = "バイト列が違う（" + std::to_string(s.data.size()) + " → " +
                      std::to_string(blob.size()) + " byte）";
        }
    };
    // 糸の中の例外は外へ出すと std::terminate になり、.part も消えない。理由として持ち帰る。
    auto work = [&](size_t i) {
        try {
            work1(i);
        } catch (const std::exception& ex) {
            werr[i] = std::string("符号化中の例外: ") + ex.what();
        } catch (...) {
            werr[i] = "符号化中の例外";
        }
    };
    static const size_t NT = [] {
        if (const char* e = getenv("PCC_THREADS")) { long v = atol(e); if (v > 0) return (size_t)v; }
        unsigned hw = std::thread::hardware_concurrency();
        return (size_t)(hw ? hw : 1);
    }();
    const size_t nt = NT < ord.size() ? NT : ord.size();
    std::atomic<size_t> next{0};
    auto loop = [&] { for (size_t a = next++; a < ord.size(); a = next++) work(ord[a]); };
    std::vector<std::thread> th;
    if (nt > 1) {
        th.reserve(nt);
        // 糸を立てられなかったら、立った分と呼び出し側で残りを回す（途中で投げると
        // join できる糸を抱えたまま vector が壊れて terminate する）。
        for (size_t t = 0; t + 1 < nt; ++t) {           // 呼び出し側が nt 本目
            try { th.emplace_back(loop); } catch (...) { break; }
        }
    }
    loop();                       // 呼び出し側も同じ待ち行列を回す（糸が 0 本でも全部終わる）
    for (auto& x : th) x.join();
    diff.clear();
    for (size_t i = 0; i < st.size(); ++i) {
        if (werr[i].empty()) continue;
        std::string nm;
        for (size_t c = 0; c < st[i].cols.size(); ++c) nm += (c ? "+" : "") + st[i].cols[c];
        diff += (diff.empty() ? "" : " / ") + nm + " " + cand_name(st[i].codec, st[i].param) +
                ": " + werr[i];
    }
    return diff.empty();
}

// 器の版。**読み方が変わったら上げる。**古い版は黙って読み違えるより、はっきり断る。
//   1 … 2026-09 まで
//   2 … LAS の封筒の末尾を「札つきの拡張」に改めた（ヘッダの欄・ユーザーデータを持つ）
//   3 … 二値の模型の適応の速さを出現回数で変えるようにした（全部の流れのビット列が変わる）
//   4 … 非可逆の極座標を戻す sin / cos を libm から自前の関数（dettrig.hpp）に替えた。
//       版 3 の器も読む（極座標だけ libm で戻す。それ以外は版 4 と同じ読み方）
//   5 … 旗「類」（C_CLS_BIT）を足した。版 4 の復号器は類の旗を知らないので上げる。
//       版 3・4 の器もそのまま読む（類の旗は立っていない）
static const uint16_t PCC2_VERSION = 5;

bool write_pcc2(const std::string& path, const Frame& f, const std::vector<Stream>& st,
                uint64_t& bytes_out, std::string& err) {
    std::vector<uint8_t> h, b;
    b.clear(); put_str(b, f.source_kind);              put_tag(h, T_SRC_KIND, b);
    b.clear(); put<uint64_t>(b, f.source_bytes);       put_tag(h, T_SRC_BYTES, b);
    b.clear(); for (int i = 0; i < 3; ++i) put<double>(b, f.scale[i]);   put_tag(h, T_SCALE, b);
    b.clear(); for (int i = 0; i < 3; ++i) put<double>(b, f.offset[i]);  put_tag(h, T_OFFSET, b);
    b.clear();
    put<uint32_t>(b, (uint32_t)f.schema.size());
    for (const auto& c : f.schema) {
        put_str(b, c.name); put<uint8_t>(b, (uint8_t)c.ftype);
        put<uint8_t>(b, (uint8_t)c.role); put<uint8_t>(b, c.is_extra ? 1 : 0);
        put<uint8_t>(b, (uint8_t)c.storage);
    }
    put_tag(h, T_SCHEMA, b);
    // 計画は二値にして入れる。素の JSON より短くならなければ JSON のまま入れる
    // （先頭 1 byte がどちらかを言う）。
    b.clear();
    {
        std::vector<uint8_t> pb(f.plan.begin(), f.plan.end());
        std::vector<uint8_t> bin;
        if (!f.plan.empty()) bin = plan_to_binary(f);
        if (!bin.empty() && bin.size() + 4 < pb.size()) {
            put<uint8_t>(b, 1); put_blob(b, bin);
        } else {
            put<uint8_t>(b, 0); put_blob(b, pb);
        }
    }
    put_tag(h, T_PLAN, b);
    b.clear(); put<uint8_t>(b, f.fid.exact ? 0 : 1);
               put<double>(b, f.fid.declared_eps);
               put<double>(b, f.fid.measured_max);     put_tag(h, T_FIDELITY, b);
    b.clear(); put_blob(b, f.envelope);                put_tag(h, T_ENVELOPE, b);
    b.clear(); for (int i = 0; i < 3; ++i) put_str(b, f.geom[i]);  put_tag(h, T_GEOM_COLS, b);
    b.clear(); put_str(b, f.geom_repr);                put_tag(h, T_GEOM_REPR, b);
    // 極座標格子の目盛り。非可逆の経路でしか付かない。
    if (f.geom_repr == "polar") {
        b.clear();
        for (int i = 0; i < 3; ++i) put<double>(b, f.polar_origin[i]);
        put<double>(b, f.polar_r_step);
        put<double>(b, f.polar_ang_step);
        put_str(b, f.polar_base);
        put_tag(h, T_POLAR, b);
    }
    // 列を粗い格子で割ってあれば、その (最小値, 刻み) を書く。
    if (!f.coldiv.empty()) {
        b.clear();
        put<uint32_t>(b, (uint32_t)f.coldiv.size());
        for (const auto& kv : f.coldiv) {
            put_str(b, kv.first);
            put<int64_t>(b, kv.second.first);
            put<int64_t>(b, kv.second.second);
        }
        // scale/offset は幾何を割ったときだけ要る。属性しか割っていない
        // ファイルで 48 byte 払うと、小さな入力では測れる損になる。
        bool geo = false;
        for (int i = 0; i < 3; ++i) if (f.coldiv.count(f.geom[i])) geo = true;
        put<uint8_t>(b, geo ? 1 : 0);
        if (geo) {
            for (int i = 0; i < 3; ++i) put<double>(b, f.coldiv_scale[i]);
            for (int i = 0; i < 3; ++i) put<double>(b, f.coldiv_offset[i]);
        }
        put_tag(h, T_COLDIV, b);
    }
    if (!f.embed.empty()) {
        b.clear(); put_str(b, f.embed_kind); put_blob(b, f.embed);  put_tag(h, T_EMBED, b);
    }

    std::vector<uint8_t> o;
    o.insert(o.end(), {'P', 'C', 'C', '2'});
    put<uint16_t>(o, PCC2_VERSION);
    put<uint16_t>(o, f.fid.exact ? 0 : 1);
    put<uint64_t>(o, f.n);
    put<uint32_t>(o, (uint32_t)h.size());
    o.insert(o.end(), h.begin(), h.end());
    put<uint32_t>(o, (uint32_t)st.size());
    for (const auto& s : st) {
        put_varint(o, (uint64_t)s.cols.size());
        // 列名は schema にもう入っている。添字で指す（"point_source_id" の
        // 17 byte が 1 byte になる）。schema に無い名前だけ文字列で書く。
        for (const auto& c : s.cols) {
            int idx = name_index(f, c);
            if (idx >= 0) { put<uint8_t>(o, 0); put_varint(o, (uint64_t)idx); }
            else { put<uint8_t>(o, 1); put_str(o, c); }
        }
        put<uint16_t>(o, s.codec);
        // 可変長。1 byte に切ると 255 byte を超える param が壊れる
        // （走査モデルの param は列名を 3 つ持つ）。
        put_varint(o, (uint64_t)s.param.size());
        o.insert(o.end(), s.param.begin(), s.param.end());
        put_varint(o, (uint64_t)s.data.size());     // 8 byte 固定をやめる
    }
    for (const auto& s : st) o.insert(o.end(), s.data.begin(), s.data.end());
    put<uint64_t>(o, crc64(o.data(), o.size()));

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { err = "書き込めない: " + path; return false; }
    // 書けたかどうかを確かめる（ディスクが一杯だと短い器が黙って残っていた）。
    const bool wrote = fwrite(o.data(), 1, o.size(), fp) == o.size();
    if (fclose(fp) != 0 || !wrote) { err = "書き込みが途中で失敗した: " + path; return false; }
    bytes_out = o.size();
    return true;
}

bool read_pcc2(const std::string& path, Frame& f, std::string& err) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { err = "開けない: " + path; return false; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    if (sz < 28) { fclose(fp); err = "PCC2 ではない（短すぎる）"; return false; }
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, fp) != (size_t)sz) { fclose(fp); err = "短い"; return false; }
    fclose(fp);
    if (memcmp(buf.data(), "PCC2", 4)) { err = "PCC2 ではない"; return false; }
    uint64_t want; memcpy(&want, buf.data() + sz - 8, 8);
    if (crc64(buf.data(), sz - 8) != want) { err = "crc64 が合わない"; return false; }

    Rd r{buf.data(), (size_t)sz - 8, 4};
    uint16_t ver = r.get<uint16_t>(); r.get<uint16_t>();
    if (ver != PCC2_VERSION && ver != 3 && ver != 4) {
        err = "器の版 " + std::to_string(ver) + " は読めない（この復号器は版 3〜" +
              std::to_string(PCC2_VERSION) + " だけを読む。作り直すこと）";
        return false;
    }
    f.polar_libm = (ver == 3);
    f.n = r.get<uint64_t>();
    uint32_t hl = r.get<uint32_t>();
    // 長さは**残りと比べて**確かめる（r.p + 長さ は壊れた値で桁あふれしうる）。
    if (!r.ok || hl > r.n - r.p) { err = "ヘッダの長さが器を超える"; return false; }
    size_t hend = r.p + hl;
    while (r.ok && r.p < hend) {
        uint16_t tag = r.get<uint16_t>();
        uint32_t len = r.get<uint32_t>();
        if (!r.ok || len > hend - r.p) { r.ok = false; break; }
        size_t next = r.p + len;
        switch (tag) {
        case T_SRC_KIND:  f.source_kind = r.str(); break;
        case T_SRC_BYTES: f.source_bytes = r.get<uint64_t>(); break;
        case T_SCALE:  for (int i = 0; i < 3; ++i) f.scale[i] = r.get<double>(); break;
        case T_OFFSET: for (int i = 0; i < 3; ++i) f.offset[i] = r.get<double>(); break;
        case T_SCHEMA: {
            uint32_t nc = r.get<uint32_t>();
            for (uint32_t i = 0; i < nc && r.ok; ++i) {
                ColSpec c; c.name = r.str();
                c.ftype = (FType)r.get<uint8_t>();
                c.role = (Role)r.get<uint8_t>();
                c.is_extra = r.get<uint8_t>() != 0;
                c.storage = (Storage)r.get<uint8_t>();
                f.schema.push_back(c);
            }
            break;
        }
        case T_PLAN: {
            uint8_t how = r.get<uint8_t>();
            auto pb = r.blob();
            if (how == 0) f.plan.assign(pb.begin(), pb.end());
            else if (!plan_from_binary(pb, f, f.plan)) { err = "計画が壊れている"; return false; }
            break;
        }
        case T_FIDELITY:
            f.fid.exact = r.get<uint8_t>() == 0;
            f.fid.declared_eps = r.get<double>();
            f.fid.measured_max = r.get<double>();
            break;
        case T_ENVELOPE:  f.envelope = r.blob(); break;
        case T_GEOM_COLS: for (int i = 0; i < 3; ++i) f.geom[i] = r.str(); break;
        case T_GEOM_REPR: f.geom_repr = r.str(); break;
        case T_POLAR:
            for (int i = 0; i < 3; ++i) f.polar_origin[i] = r.get<double>();
            f.polar_r_step = r.get<double>();
            f.polar_ang_step = r.get<double>();
            f.polar_base = r.str();
            break;
        case T_EMBED:     f.embed_kind = r.str(); f.embed = r.blob(); break;
        case T_COLDIV: {
            uint32_t k = r.get<uint32_t>();
            for (uint32_t i = 0; i < k && r.ok; ++i) {
                std::string nm = r.str();
                int64_t base = r.get<int64_t>(), step = r.get<int64_t>();
                f.coldiv[nm] = {base, step};
            }
            if (r.get<uint8_t>()) {
                for (int i = 0; i < 3; ++i) f.coldiv_scale[i] = r.get<double>();
                for (int i = 0; i < 3; ++i) f.coldiv_offset[i] = r.get<double>();
            } else {
                for (int i = 0; i < 3; ++i) { f.coldiv_scale[i] = f.scale[i];
                                              f.coldiv_offset[i] = f.offset[i]; }
            }
            break;
        }
        default: break;   // 知らないタグは読み飛ばす（前方互換）
        }
        // 札の中身を読み過ぎたなら壊れている（次の札の頭を読んでいた）
        if (r.p > next) r.ok = false;
        r.p = next;
    }
    if (!r.ok) { err = "ヘッダが壊れている"; return false; }

    uint32_t ns = r.get<uint32_t>();
    // 流れ 1 本の記述は少なくとも 4 byte（列数・符号器・パラメタ長・データ長）。
    // 壊れた本数で巨大な配列を確保しない。
    if (!r.ok || ns > (r.n - r.p) / 4) { err = "ストリームの本数が器に収まらない"; return false; }
    std::vector<Stream> st(ns);
    std::vector<uint64_t> dlen(ns);
    for (uint32_t i = 0; i < ns && r.ok; ++i) {
        uint64_t nc = r.varint();
        for (uint64_t k = 0; k < nc && r.ok; ++k) {
            uint8_t how = r.get<uint8_t>();
            if (how == 0) {
                uint64_t idx = r.varint();
                if (idx >= f.schema.size()) { err = "列の添字が範囲外"; return false; }
                st[i].cols.push_back(f.schema[(size_t)idx].name);
            } else {
                st[i].cols.push_back(r.str());
            }
        }
        st[i].codec = r.get<uint16_t>();
        {
            uint64_t pl = r.varint();
            if (!r.ok || pl > r.n - r.p) { r.ok = false; }
            else { st[i].param.assign(r.b + r.p, r.b + r.p + pl); r.p += (size_t)pl; }
        }
        dlen[i] = r.varint();
    }
    if (!r.ok) { err = "ストリーム記述が壊れている"; return false; }
    // 幾何は属性より先に並んでいる。幾何が揃った時点で座標を組み、
    // 以後の属性ストリームはそれを副次情報として使う（副情報は生じない）。
    CodecCtx ctx;
    ctx.fr = &f;
    std::vector<double> world;

    // 各ストリームの読み出し位置を先に出す。長さは記述子に入っているので、
    // 順に復号しなくても位置が決まる。
    std::vector<size_t> off(ns);
    for (uint32_t i = 0; i < ns; ++i) {
        if (dlen[i] > r.n - r.p) { err = "ストリームが短い"; return false; }
        off[i] = r.p; r.p += dlen[i];
    }

    // ストリームが要る列を出す。参照も走査モデルも補助列も、相手の名前は
    // パラメタに入っている。空間予測は座標（＝幾何の列）が要る。
    auto names_in = [](const std::vector<uint8_t>& p, size_t q, int cnt,
                       std::vector<std::string>& out) {
        for (int k = 0; k < cnt; ++k) {
            if (q + 2 > p.size()) return;
            uint16_t l; memcpy(&l, p.data() + q, 2); q += 2;
            if (q + l > p.size()) return;
            out.emplace_back((const char*)p.data() + q, l); q += l;
        }
    };
    std::vector<std::vector<std::string>> need(ns);
    std::vector<char> need_geom(ns, 0);
    for (uint32_t i = 0; i < ns; ++i) {
        const uint16_t id = (uint16_t)(st[i].codec & ~C_FLAG_MASK);
        // 旗「類」の param は末尾に列名が付く。元の param の依存を読む前に剥がす
        // （字母の元の param が空のとき、列名の byte を長さとして読んでしまう）。
        std::vector<uint8_t> p_orig; std::string cls_nm;
        const bool has_cls = (st[i].codec & C_CLS_BIT) && cls_split(st[i].param, p_orig, cls_nm);
        const auto& p = has_cls ? p_orig : st[i].param;
        if (id == C_ATTR_XREF) {
            names_in(p, 1, 1, need[i]);
            if (!p.empty() && p[0] != 0 && p[0] != 100) need_geom[i] = 1;
        } else if (id == C_ATTR_SPATIAL || id == C_ATTR_COLOR) {
            need_geom[i] = 1;
        } else if (id == C_GEOM_SCAN) {
            names_in(p, 1, 3, need[i]);
        } else if (id == C_GEOM_XYZ && !p.empty() && p[0] == 2) {
            names_in(p, 1, 1, need[i]);
        } else if (id == C_GEOM_XYZ && p.size() > 1 && p[0] == 3 && (p[1] & 8)) {
            names_in(p, 3, 1, need[i]);
        } else if (id == C_RANGE_SYM && !p.empty()) {
            // 参照列を文脈に使う版。param = [shift u8][名前長 u16][名前]
            names_in(p, 1, 1, need[i]);
        }
        // **光線モデルは bit_fields を名前で引く。**param に名前が無いので、
        // ここで依存を足さないと、同じ波で並列に復号されて bit_fields が
        // まだ無い（または書きかけの）まま幾何を戻してしまう。
        if (st[i].codec & C_RAY_BIT) need[i].push_back("bit_fields");
        if (has_cls) {                           // 類の列が先に戻っていること
            static const std::string D = "\xCE\x94";
            need[i].push_back(cls_nm.compare(0, D.size(), D) == 0 ? cls_nm.substr(D.size()) : cls_nm);
        }
    }

    // **入れ物を先に作ってはいけない。** f.get は「その列がもう有るか」を
    // 空でないことで判断する符号器があるので、空の入れ物を置くと
    // 未復号の列が有ることになってしまう（plane などで落ちた）。
    // 列は波が終わってから f.col に移す（下の work を参照）。

    static const size_t NT = [] {
        if (const char* e = getenv("PCC_THREADS")) { long v = atol(e); if (v > 0) return (size_t)v; }
        unsigned hw = std::thread::hardware_concurrency();
        return (size_t)(hw ? hw : 1);
    }();

    std::vector<char> done(ns, 0);
    // 定数の列は流れに入らず、容器から先に入っている。それも「有る」に数える。
    std::set<std::string> have;
    for (const auto& kv : f.col) have.insert(kv.first);
    // 流れの一覧から、この容器で実際に使われる P を集めておく。
    {
        std::vector<int>& W = const_cast<std::vector<int>&>(ctx.want_Ps);
        for (uint32_t i = 0; i < ns; ++i) {
            const uint16_t id = st[i].codec & ~C_FLAG_MASK;
            if (st[i].param.empty()) continue;
            if (id != C_ATTR_SPATIAL && id != C_ATTR_COLOR && id != C_ATTR_XREF) continue;
            int P = st[i].param[0];
            if (P <= 0 || P == 100) continue;
            if (std::find(W.begin(), W.end(), P) == W.end()) W.push_back(P);
        }
        std::sort(W.begin(), W.end());
    }
    bool world_ready = false;
    const bool dprof = getenv("PCC_DPROF") != nullptr;
    int wno = 0;
    double dsum = 0;
    uint32_t left = ns;
    while (left) {
        // いま復号できるものを集める
        std::vector<uint32_t> wave;
        for (uint32_t i = 0; i < ns; ++i) {
            if (done[i]) continue;
            if (need_geom[i] && !world_ready) continue;
            bool ok = true;
            for (const auto& c : need[i])
                if (!c.empty() && !have.count(c)) { ok = false; break; }
            if (ok) wave.push_back(i);
        }
        if (wave.empty()) { err = "ストリームの依存が解けない"; return false; }
        std::vector<std::string> werr(wave.size());
        std::vector<char> wok(wave.size(), 0);
        std::vector<double> wsec(wave.size(), 0.0);
        // **波の中では f.col に書かない。**同じ波の別の糸は、錠を取らずに
        // ctx->fr->get()（std::map の探索）で参照列を引く。挿入と探索が同時に
        // 走ると木の付け替えの途中を読みうる（未定義動作）。復号した列は波ごとの
        // 局所の入れ物に置き、波が終わってから（糸が全部合流してから）移す。
        // 同じ波の流れは互いに依らないので、置き場所を変えても結果は同じ。
        // 置き場所は幅を詰めた Col にする（int64 のまま波の終わりまで抱えると、
        // 200 万点の列 1 本で 16 MB ずつ積み上がる）。詰めるのは各糸の中で済ませる。
        std::vector<std::vector<Col>> wout(wave.size());
        size_t nt = NT < wave.size() ? NT : wave.size();
        auto work = [&](size_t a) {
            uint32_t i = wave[a];
            double t0 = now_sec();
            std::vector<std::vector<int64_t>> outc;
            // 糸の中の例外は外へ出すと std::terminate になる。理由として持ち帰る。
            try {
                if (!codec_decode(st[i].codec, st[i].param, buf.data() + off[i], dlen[i],
                                  f.n, st[i].cols.size(), outc, werr[a], &ctx)) return;
            } catch (const std::exception& e) {
                werr[a] = std::string("復号中の例外: ") + e.what(); return;
            }
            if (outc.size() != st[i].cols.size()) { werr[a] = "列の数が合わない"; return; }
            for (const auto& oc : outc)
                if (oc.size() != (size_t)f.n) { werr[a] = "列の長さが点数と合わない"; return; }
            wout[a].resize(outc.size());
            for (size_t c = 0; c < outc.size(); ++c) wout[a][c] = std::move(outc[c]);
            wsec[a] = now_sec() - t0;
            wok[a] = 1;
        };
        if (nt <= 1) {
            for (size_t a = 0; a < wave.size(); ++a) work(a);
        } else {
            std::atomic<size_t> next{0};
            std::vector<std::thread> th; th.reserve(nt);
            for (size_t t = 0; t < nt; ++t)
                th.emplace_back([&] { for (size_t a = next++; a < wave.size(); a = next++) work(a); });
            for (auto& x : th) x.join();
        }
        if (dprof) {
            double wmax = 0;
            for (size_t a = 0; a < wave.size(); ++a) wmax = std::max(wmax, wsec[a]);
            fprintf(stderr, "  [波 %d] %zu 本  最長 %.3fs\n", wno, wave.size(), wmax);
            std::vector<size_t> ord(wave.size());
            for (size_t a = 0; a < wave.size(); ++a) ord[a] = a;
            std::sort(ord.begin(), ord.end(),
                      [&](size_t x, size_t y) { return wsec[x] > wsec[y]; });
            for (size_t j = 0; j < ord.size(); ++j) {
                uint32_t i = wave[ord[j]];
                std::string nm;
                for (size_t c = 0; c < st[i].cols.size(); ++c)
                    nm += (c ? "+" : "") + st[i].cols[c];
                fprintf(stderr, "      %-24s %-22s %7.3fs\n", nm.c_str(),
                        cand_name(st[i].codec, st[i].param).c_str(), wsec[ord[j]]);
            }
            dsum += wmax;
            ++wno;
        }
        for (size_t a = 0; a < wave.size(); ++a) {
            uint32_t i = wave[a];
            if (!wok[a]) {
                err = werr[a] + "（ストリーム " + std::to_string(i) + ": ";
                for (size_t c = 0; c < st[i].cols.size(); ++c) err += (c ? "+" : "") + st[i].cols[c];
                err += " / 符号器 " + cand_name(st[i].codec, st[i].param) + "）";
                return false;
            }
            for (size_t c = 0; c < st[i].cols.size(); ++c)
                f.col[st[i].cols[c]] = std::move(wout[a][c]);
            std::vector<Col>().swap(wout[a]);
            done[i] = 1; --left;
            for (const auto& c : st[i].cols) have.insert(c);
        }
        if (!world_ready && f.get(f.geom[0]) && f.get(f.geom[1]) && f.get(f.geom[2])) {
            frame_world(f, world);
            ctx.world = &world;
            world_ready = true;
        }
    }
    if (dprof) fprintf(stderr, "  [波の合計] %.3fs（%d 波）\n", dsum, wno);
    return true;
}

} // namespace pcc
