#include "pcc/pcc2.hpp"
#include "pcc/rangecoder.hpp"
#include "pcc/attr.hpp"
#include "pcc/scanmodel.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <array>
#include <atomic>
#include <thread>

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
struct Pred {
    MedPred m;
    int64_t q[4] = {0, 0, 0, 0};
    int k = 0, mode = 0;
    inline int64_t predict() const {
        if (mode == 0) return m.predict();
        if (mode == 2) return m.prev + fdiv(m.predict() - m.prev, 2);
        // q[0] には最初の点の絶対座標が入る（prev の初期値が 0 のため）。
        // 4 つ揃っても 1 巡するまでは差分でないので使わない。
        if (k < 5) return m.predict();
        return m.prev + fdiv(q[0] + q[1] + q[2] + q[3], 4);
    }
    inline void push(int64_t x) { q[k & 3] = x - m.prev; ++k; m.push(x); }
};
} // namespace

// ---------------------------------------------------------------- 符号器
// 共通形: 1 点ごとに列を横断して符号化する（幾何 3 軸の同時符号化に必要）。
// mode: 0 そのまま / 1 1次差分 / 2 1次差分＋ビット数文脈 / 3 2次差分＋文脈
// 3 は「一定の刻みで増える列」に効く。gps_time は float64 のビットパターンを
// 整数と見なして差を取ると刻みがほぼ一定になるので、2 次差分はほぼ 0 になる。
static void enc_cols(const std::vector<const std::vector<int64_t>*>& cols, int mode,
                     std::vector<uint8_t>& out) {
    size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    Encoder e;
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<int64_t> prev(nc, 0), prev2(nc, 0);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            int64_t x = (*cols[c])[i];
            int64_t d = (mode == 0) ? x : x - prev[c];
            if (mode == 3) { int64_t t = d; d = d - prev2[c]; prev2[c] = t; }
            uint64_t z = zigzag(d);
            uc.encode(e, z, (int)c * NCTX + (mode >= 2 ? ctx[c] : 0));
            prev[c] = x;
            if (mode >= 2) ctx[c] = ctx_of(z);
        }
    out = e.finish();
}

// 差分の値が繰り返す列に効く。gps_time の差分は値の種類が少ないのに桁が大きく、
// bit 長を送る方式では同じ値が何度出ても毎回その桁ぶん払う。直近 K 個の相異なる
// 差分を表に持ち、当たれば添字だけを送る。外れたら逃げ道として従来どおり送る。
// 表は移動前置（当たった値を先頭へ）で更新するので副情報は要らない。
inline constexpr int DCACHE = 1024;

static void enc_cols_cache(const std::vector<const std::vector<int64_t>*>& cols,
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

static void enc_geom_med(const std::vector<const std::vector<int64_t>*>& cols,
                         std::vector<uint8_t>& out) {
    size_t nc = cols.size(), n = nc ? cols[0]->size() : 0;
    Encoder e;
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<MedPred> mp(nc);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            int64_t x = (*cols[c])[i];
            uint64_t z = zigzag(x - mp[c].predict());
            uc.encode(e, z, (int)c * NCTX + ctx[c]);
            mp[c].push(x);
            ctx[c] = ctx_of(z);
        }
    out = e.finish();
}
static void dec_geom_med(const uint8_t* data, size_t len, size_t n, size_t nc,
                         std::vector<std::vector<int64_t>>& out) {
    out.assign(nc, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<MedPred> mp(nc);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = uc.decode(d, (int)c * NCTX + ctx[c]);
            int64_t x = mp[c].predict() + unzigzag(z);
            out[c][i] = x;
            mp[c].push(x);
            ctx[c] = ctx_of(z);
        }
}

// 幾何 v2: 補助列（復帰番号などの状態）ごとに別の予測器状態を持つ。
// 航空 LiDAR では 1 発から複数の点が返るので、格納順の隣は走査線上の隣とは限らない。
// 同じ状態の点だけを繋ぐと逐次予測が成立する。LASzip が点レコードに対して
// しているのと同じ考え方で、復号器は補助列を先に復号していれば副情報を要しない。
static void enc_geom_aux(const std::vector<const std::vector<int64_t>*>& cols,
                         const std::vector<int64_t>& aux, std::vector<uint8_t>& out) {
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
                         const std::vector<int64_t>& aux,
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
static void enc_geom_x(const std::vector<const std::vector<int64_t>*>& cols,
                       int pmode, std::vector<uint8_t>& out) {
    size_t n = cols[0]->size();
    Encoder e;
    UIntCoder uc(3 * NCTX, 64);
    Pred mp[3];
    for (int c = 0; c < 3; ++c) mp[c].mode = pmode;
    int prev_kx = 0;
    for (size_t i = 0; i < n; ++i) {
        int k[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            int64_t x = (*cols[c])[i];
            uint64_t z = zigzag(x - mp[c].predict());
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            uc.encode(e, z, c * NCTX + ctxc);
            mp[c].push(x);
            k[c] = ctx_of(z);
        }
        prev_kx = k[0];
    }
    out = e.finish();
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
static inline int nn_back(const int64_t* X, const int64_t* Y, const int64_t* Z,
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

static void enc_geom_w(const std::vector<const std::vector<int64_t>*>& cols,
                       size_t w, int tmode, std::vector<uint8_t>& out) {
    size_t n = cols[0]->size();
    const int64_t *X = cols[0]->data(), *Y = cols[1]->data(), *Z = cols[2]->data();
    Encoder e;
    UIntCoder uc(3 * NCTX, 64);
    UIntCoder ui(NCTX, 8);                     // 添字。文脈は直前の添字
    MedPred tp[3];
    int prev_kx = 0, prev_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        int back = i ? nn_back(X, Y, Z, i, w) : 0;
        if (i) ui.encode(e, (uint64_t)back, prev_idx < NCTX ? prev_idx : NCTX - 1);
        size_t j = i ? i - 1 - (size_t)back : 0;
        const int64_t p[3] = {i ? X[j] + trend_of(tp[0], tmode) : 0,
                              i ? Y[j] + trend_of(tp[1], tmode) : 0,
                              i ? Z[j] + trend_of(tp[2], tmode) : 0};
        const int64_t v[3] = {X[i], Y[i], Z[i]};
        int k[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            uint64_t z = zigzag(v[c] - p[c]);
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            uc.encode(e, z, c * NCTX + ctxc);
            k[c] = ctx_of(z);
        }
        prev_kx = k[0];
        prev_idx = back;
        for (int c = 0; c < 3; ++c) tp[c].push(v[c]);
    }
    out = e.finish();
}

static void dec_geom_w(const uint8_t* data, size_t len, size_t n, size_t w, int tmode,
                       std::vector<std::vector<int64_t>>& out) {
    (void)w;
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc(3 * NCTX, 64);
    UIntCoder ui(NCTX, 8);
    MedPred tp[3];
    int prev_kx = 0, prev_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        int back = 0;
        if (i) back = (int)ui.decode(d, prev_idx < NCTX ? prev_idx : NCTX - 1);
        size_t j = i ? i - 1 - (size_t)back : 0;
        const int64_t p[3] = {i ? out[0][j] + trend_of(tp[0], tmode) : 0,
                              i ? out[1][j] + trend_of(tp[1], tmode) : 0,
                              i ? out[2][j] + trend_of(tp[2], tmode) : 0};
        int k[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            uint64_t z = uc.decode(d, c * NCTX + ctxc);
            out[c][i] = p[c] + unzigzag(z);
            k[c] = ctx_of(z);
        }
        prev_kx = k[0];
        prev_idx = back;
        for (int c = 0; c < 3; ++c) tp[c].push(out[c][i]);
    }
}

static void dec_geom_x(const uint8_t* data, size_t len, size_t n, int pmode,
                       std::vector<std::vector<int64_t>>& out) {
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc(3 * NCTX, 64);
    Pred mp[3];
    for (int c = 0; c < 3; ++c) mp[c].mode = pmode;
    int prev_kx = 0;
    for (size_t i = 0; i < n; ++i) {
        int k[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
            int ctxc = (c == 0) ? prev_kx : (c == 1 ? k[0] : (k[0] + k[1]) / 2);
            if (ctxc >= NCTX) ctxc = NCTX - 1;
            uint64_t z = uc.decode(d, c * NCTX + ctxc);
            int64_t x = mp[c].predict() + unzigzag(z);
            out[c][i] = x;
            mp[c].push(x);
            k[c] = ctx_of(z);
        }
        prev_kx = k[0];
    }
}

static void dec_cols(const uint8_t* data, size_t len, size_t n, size_t nc, int mode,
                     std::vector<std::vector<int64_t>>& out) {
    out.assign(nc, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<int64_t> prev(nc, 0), prev2(nc, 0);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = uc.decode(d, (int)c * NCTX + (mode >= 2 ? ctx[c] : 0));
            int64_t v = unzigzag(z);
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

const std::vector<int32_t>* CodecCtx::ensure(size_t n, int P) const {
    std::lock_guard<std::mutex> lk(mu);
    const std::vector<double>* w = world_ptr();
    if (!w || w->size() < n * 3) return nullptr;
    // 点数が変わったら作り直す。標本で順位を付けるときに同じ P で呼ばれると、
    // 標本ぶんの表を全点の符号化に使ってしまう。
    if (built_n != n) { perm.clear(); pred_by_p.clear(); built_n = n; }
    if (perm.empty()) perm = coding_order(*w, n, "morton");
    auto it = pred_by_p.find(P);
    if (it == pred_by_p.end()) {
        // 候補が使う P はいつも 1 / 3 / 5 なので、最初の 1 回でまとめて作る。
        // 近傍探索も KdTree の構築も 1 回で済む（P ごとだと 3 回になる）。
        std::vector<int> Ps;
        if (pred_by_p.empty()) { Ps = {1, 3, 5}; }
        if (std::find(Ps.begin(), Ps.end(), P) == Ps.end()) Ps.push_back(P);
        std::vector<std::vector<int32_t>> pds;
        build_causal_predictors_multi(*w, n, perm, Ps, pds);
        for (size_t a = 0; a < Ps.size(); ++a)
            pred_by_p.emplace(Ps[a], std::move(pds[a]));
        it = pred_by_p.find(P);
    }
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
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = zigzag((int64_t)res[c][i]);
            uc.encode(e, z, (int)c * NCTX + ctx[c]);
            ctx[c] = ctx_of(z);
        }
    out = e.finish();
}
// 走査モデル専用。3 列のうち z（列 2）の文脈を、同じ点の面内・面外の残差の
// ビット長から作る。幾何v3 が Z に X と Y の平均を使うのと同じ考え方。
template <class T>
static void enc_resid_x(const std::vector<std::vector<T>>& res, std::vector<uint8_t>& out) {
    size_t n = res[0].size();
    Encoder e;
    UIntCoder uc(3 * NCTX, 64);
    int p0 = 0, p1 = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = zigzag((int64_t)res[0][i]); uc.encode(e, a, 0 * NCTX + p0); int k0 = ctx_of(a);
        uint64_t b = zigzag((int64_t)res[1][i]); uc.encode(e, b, 1 * NCTX + p1); int k1 = ctx_of(b);
        uint64_t c = zigzag((int64_t)res[2][i]); uc.encode(e, c, 2 * NCTX + (k0 + k1) / 2);
        p0 = k0; p1 = k1;
    }
    out = e.finish();
}
static void dec_resid_x(const uint8_t* data, size_t len, size_t n,
                        std::vector<std::vector<int64_t>>& res) {
    res.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc(3 * NCTX, 64);
    int p0 = 0, p1 = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = uc.decode(d, 0 * NCTX + p0); res[0][i] = unzigzag(a); int k0 = ctx_of(a);
        uint64_t b = uc.decode(d, 1 * NCTX + p1); res[1][i] = unzigzag(b); int k1 = ctx_of(b);
        uint64_t c = uc.decode(d, 2 * NCTX + (k0 + k1) / 2); res[2][i] = unzigzag(c);
        p0 = k0; p1 = k1;
    }
}

static void dec_resid(const uint8_t* data, size_t len, size_t n, size_t nc,
                      std::vector<std::vector<int64_t>>& res) {
    res.assign(nc, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = uc.decode(d, (int)c * NCTX + ctx[c]);
            res[c][i] = unzigzag(z);
            ctx[c] = ctx_of(z);
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
    const std::vector<int64_t>* gcol = nullptr;
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

static bool build_scan_ctx(const CodecCtx* ctx, const std::vector<uint8_t>& param,
                           size_t n, ScanCtx& sc, std::string& err) {
    std::string ng, np, nr;
    if (!scan_names(param, ng, np, nr)) { err = "走査モデルの列名が壊れている"; return false; }
    if (!ctx || !ctx->fr) { err = "副次情報がない"; return false; }
    const auto* g = ctx->fr->get(ng);
    const auto* s = ctx->fr->get(np);
    const auto* r = nr.empty() ? nullptr : ctx->fr->get(nr);
    if (!g || !s || g->size() < n || s->size() < n) {
        err = "gps_time / point_source_id が先に復号されていない"; return false;
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
    return true;
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
static bool enc_geom_scan(const std::vector<const std::vector<int64_t>*>& cols,
                          const std::vector<uint8_t>& param, std::vector<uint8_t>& out,
                          std::string& err, const CodecCtx* ctx) {
    if (cols.size() != 3) { err = "走査モデルは 3 軸"; return false; }
    size_t n = cols[0]->size();
    const bool prof = getenv("PCC_SCAN_PROF") != nullptr;
    double tp = now_sec();
    auto mark = [&](const char* what) {
        if (prof) { fprintf(stderr, "    [走査] %-22s %6.2fs\n", what, now_sec() - tp); tp = now_sec(); }
    };
    ScanCtx sc;
    if (!build_scan_ctx(ctx, param, n, sc, err)) return false;
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
                bool fat_fail = (sp_code == 2 && sp_r0 > 1000.0 && n_fat_dumped < 40);
                if (dump_raw && ((id % 50) == 0 || fat_fail)) {
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
            int64_t ps = sp.ok ? scan_predict(sp, sc.gps(t) - g0[gidx], CZ[i])
                               : mps[gidx].predict();
            put_res(0, (size_t)t, s - ps);
            put_res(1, (size_t)t, off - po);
            mps[gidx].push(s); mpo[gidx].push(off);
            int r = (int)sc.ret[t] & 15;
            if (!hz[r]) { mz[r] = MedPred(); mz[r].prev = last_any; }
            int64_t pz = use_med ? mz[r].predict() : (hz[r] ? lz[r] : last_any);
            put_res(2, (size_t)t, CZ[i] - pz);
            mz[r].push(CZ[i]);
            lz[r] = CZ[i]; hz[r] = true; last_any = CZ[i];
        }
    }

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
    // 走査文脈はここから先で参照しない。400 万点では ord と gps と ret だけで
    // 80 MB あり、残差（3 列 × 8 バイト）と同時に抱えるとピークが立つ。
    sc.ord.clear(); sc.ord.shrink_to_fit();
    sc.ret.clear(); sc.ret.shrink_to_fit();
    L.label.clear(); L.label.shrink_to_fit();
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
    ScanCtx sc;
    if (!build_scan_ctx(ctx, param, n, sc, err)) return false;
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
    if (!rd64(lf) || p + lf > len) { err = "フラグが短い"; return false; }
    std::vector<int64_t> flags(nsw);
    decode_ints(data + p, lf, flags.data(), nsw); p += lf;
    if (!rd64(lp) || p + lp > len) { err = "パラメタが短い"; return false; }
    std::vector<int64_t> pars(nl * 8);
    decode_ints(data + p, lp, pars.data(), nl * 8); p += lp;
    if (!rd64(ll) || p + ll > len) { err = "標識が短い"; return false; }
    std::vector<int64_t> labs(nlab);
    if (nlab) {
        Decoder d(data + p, ll); UIntCoder uc(2, 2);
        int prev = 0;
        for (size_t i = 0; i < nlab; ++i) { labs[i] = (int64_t)uc.decode(d, prev); prev = (int)labs[i]; }
    }
    p += ll;
    if (!rd64(lr) || p + lr > len) { err = "残差が短い"; return false; }
    std::vector<std::vector<int64_t>> res;
    if (use_xctx) dec_resid_x(data + p, lr, n, res); else dec_resid(data + p, lr, n, 3, res);

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
          if (L.split[k]) for (int32_t t = sc.swp[k]; t < sc.swp[k + 1]; ++t)
              L.label[t] = (uint8_t)labs[q++]; }

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
    return true;
}

// param の 2 バイト目以降に入っている名前で、既に復号済みの列を引く
static const std::vector<int64_t>* aux_col(const std::vector<uint8_t>& param, const CodecCtx* ctx) {
    if (!ctx || !ctx->fr || param.size() < 3) return nullptr;
    uint16_t l; memcpy(&l, param.data() + 1, 2);
    if (param.size() < 3u + l) return nullptr;
    std::string nm((const char*)param.data() + 3, l);
    return ctx->fr->get(nm);
}

bool codec_encode(uint16_t id, const std::vector<const std::vector<int64_t>*>& cols,
                  const std::vector<uint8_t>& param, std::vector<uint8_t>& out,
                  std::string& err, const CodecCtx* ctx) {
    if (cols.empty()) { err = "列がない"; return false; }
    switch (id) {
    case C_RAW64: {
        size_t nc = cols.size(), n = cols[0]->size();
        out.resize(nc * n * 8);
        uint8_t* p = out.data();
        for (size_t c = 0; c < nc; ++c) { memcpy(p, cols[c]->data(), n * 8); p += n * 8; }
        return true;
    }
    case C_RANGE:        enc_cols(cols, 0, out); return true;
    case C_RANGE_DELTA:  enc_cols(cols, 1, out); return true;
    case C_RANGE_CTX:    enc_cols(cols, 2, out); return true;
    case C_RANGE_CTX2:   enc_cols(cols, 3, out); return true;
    case C_RANGE_MED:    enc_geom_med(cols, out); return true;
    case C_RANGE_CACHE:  enc_cols_cache(cols, out); return true;
    case C_GEOM_XYZ: {
        int var = param.empty() ? 0 : param[0];
        if (var == 4) {
            if (cols.size() != 3) { err = "幾何v4 は 3 軸"; return false; }
            size_t w = param.size() > 1 ? (size_t)param[1] : 16;
            enc_geom_w(cols, w ? w : 16, param.size() > 2 ? param[2] : 0, out);
        } else if (var == 2) {
            const std::vector<int64_t>* a = aux_col(param, ctx);
            if (!a || a->size() < cols[0]->size()) { err = "補助列がない"; return false; }
            enc_geom_aux(cols, *a, out);
        } else if (var == 3) {
            if (cols.size() != 3) { err = "幾何v3 は 3 軸"; return false; }
            enc_geom_x(cols, param.size() > 1 ? param[1] : 0, out);
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
        const std::vector<int64_t>* a = aux_col(param, ctx);
        if (!a || a->size() < cols[0]->size()) { err = "参照列がない"; return false; }
        int P = param[0];
        size_t n = cols[0]->size();
        std::vector<std::vector<int64_t>> d(1, std::vector<int64_t>(n));
        for (size_t i = 0; i < n; ++i) d[0][i] = (*cols[0])[i] - (*a)[i];
        if (P == 100) {                       // 残差にさらに格納順の 1 次差分を掛ける
            int64_t prev = 0;
            for (size_t i = 0; i < n; ++i) { int64_t v = d[0][i]; d[0][i] = v - prev; prev = v; }
        } else if (P > 0) {
            const std::vector<int32_t>* pdt = ctx->ensure(n, P);
            if (!pdt) { err = "座標がない"; return false; }
            std::vector<int64_t> r;
            spatial_residual(d[0], ctx->perm, *pdt, P, n, r);
            d[0] = std::move(r);
        }
        enc_resid(d, out);
        return true;
    }
    case C_ATTR_SPATIAL:
    case C_ATTR_COLOR: {
        if (!ctx || param.empty()) { err = "空間予測に必要な副次情報がない"; return false; }
        int P = param[0];
        size_t n = cols[0]->size();
        const std::vector<int32_t>* pdt = ctx->ensure(n, P);
            if (!pdt) { err = "座標がない"; return false; }
        std::vector<std::vector<int64_t>> src;
        if (id == C_ATTR_COLOR) {
            if (cols.size() != 3) { err = "色は 3 列でなければならない"; return false; }
            src.assign(3, std::vector<int64_t>(n));
            for (size_t i = 0; i < n; ++i)
                ycocg_fwd((*cols[0])[i], (*cols[1])[i], (*cols[2])[i],
                          src[0][i], src[1][i], src[2][i]);
        } else {
            for (auto* c : cols) src.push_back(*c);
        }
        std::vector<std::vector<int64_t>> res(src.size());
        for (size_t c = 0; c < src.size(); ++c)
            spatial_residual(src[c], ctx->perm, *pdt, P, n, res[c]);
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
    case C_GEOM_XYZ: {
        int var = param.empty() ? 0 : param[0];
        if (var == 4) {
            if (ncol != 3) { err = "幾何v4 は 3 軸"; return false; }
            size_t w = param.size() > 1 ? (size_t)param[1] : 16;
            dec_geom_w(data, len, n, w ? w : 16, param.size() > 2 ? param[2] : 0, out);
        } else if (var == 2) {
            const std::vector<int64_t>* a = aux_col(param, ctx);
            if (!a || a->size() < n) { err = "補助列がない"; return false; }
            dec_geom_aux(data, len, n, ncol, *a, out);
        } else if (var == 3) {
            if (ncol != 3) { err = "幾何v3 は 3 軸"; return false; }
            dec_geom_x(data, len, n, param.size() > 1 ? param[1] : 0, out);
        } else if (var == 1) dec_geom_med(data, len, n, ncol, out);
        else dec_cols(data, len, n, ncol, 2, out);
        return true;
    }
    case C_GEOM_SCAN:
        return dec_geom_scan(param, data, len, n, out, err, ctx);
    case C_ATTR_XREF: {
        const std::vector<int64_t>* a = aux_col(param, ctx);
        if (!a || a->size() < n) { err = "参照列がない"; return false; }
        int P = param[0];
        std::vector<std::vector<int64_t>> d;
        dec_resid(data, len, n, 1, d);
        if (P == 100) {
            int64_t acc = 0;
            for (size_t i = 0; i < n; ++i) { acc += d[0][i]; d[0][i] = acc; }
        } else if (P > 0) {
            const std::vector<int32_t>* pdt = ctx->ensure(n, P);
            if (!pdt) { err = "座標がない"; return false; }
            std::vector<int64_t> r;
            spatial_restore(d[0], ctx->perm, *pdt, P, n, r);
            d[0] = std::move(r);
        }
        out.assign(1, std::vector<int64_t>(n));
        for (size_t i = 0; i < n; ++i) out[0][i] = d[0][i] + (*a)[i];
        return true;
    }
    case C_ATTR_SPATIAL:
    case C_ATTR_COLOR: {
        if (!ctx || param.empty()) { err = "空間予測に必要な副次情報がない"; return false; }
        int P = param[0];
        const std::vector<int32_t>* pdt = ctx->ensure(n, P);
            if (!pdt) { err = "座標がない"; return false; }
        std::vector<std::vector<int64_t>> res;
        dec_resid(data, len, n, ncol, res);
        std::vector<std::vector<int64_t>> src(ncol);
        for (size_t c = 0; c < ncol; ++c)
            spatial_restore(res[c], ctx->perm, *pdt, P, n, src[c]);
        if (id == C_ATTR_COLOR) {
            if (ncol != 3) { err = "色は 3 列でなければならない"; return false; }
            out.assign(3, std::vector<int64_t>(n));
            for (size_t i = 0; i < n; ++i)
                ycocg_inv(src[0][i], src[1][i], src[2][i],
                          out[0][i], out[1][i], out[2][i]);
        } else out = std::move(src);
        return true;
    }
    }
    err = "未知の符号器";
    return false;
}

// ---------------------------------------------------------------- 候補を実測して選ぶ

// 参照相手の絞り込み。全対を全点で測ると O(列^2 * 点数) になるので標本で選ぶ。
static double entropy_diff_sample(const std::vector<int64_t>& a,
                                  const std::vector<int64_t>& b, size_t cap) {
    size_t n = std::min({a.size(), b.size(), cap});
    if (!n) return 1e30;
    std::map<int64_t, uint32_t> h;
    for (size_t i = 0; i < n; ++i) ++h[a[i] - b[i]];
    double e = 0;
    for (const auto& kv : h) { double q = (double)kv.second / n; e -= q * std::log2(q); }
    return e;
}

std::string cand_name(uint16_t c, const std::vector<uint8_t>& p) {
    char b[32];
    switch (c) {
    case C_RAW64: return "raw64";
    case C_RANGE: return "range";
    case C_RANGE_DELTA: return "delta";
    case C_RANGE_CTX: return "ctx";
    case C_RANGE_CTX2: return "ctx2";
    case C_RANGE_MED: return "med3";
    case C_RANGE_CACHE: return "差分表";
    case C_GEOM_SCAN: {
        static char b2[16];
        int v = p.empty() ? 1 : p[0];
        if (v == 5) return "走査変換";
        snprintf(b2, sizeof b2, "走査v%d", v);
        return b2;
    }
    case C_GEOM_XYZ:
        if (p.empty() || p[0] == 0) return "幾何v0";
        if (p[0] == 4) {
            static char b4[20];
            int tm = p.size() > 2 ? p[2] : 0;
            snprintf(b4, sizeof b4, "幾何v4W%d%s", p.size() > 1 ? p[1] : 16,
                     tm == 0 ? "" : (tm == 1 ? "傾" : "傾半"));
            return b4;
        }
        if (p[0] == 3 && p.size() > 1 && p[1]) {
            static char b3[20];
            snprintf(b3, sizeof b3, "幾何v3%s", p[1] == 1 ? "平均4" : "半");
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

// 候補は互いに独立に符号化できる。幾何の候補だけを並列に回す。
// 属性の候補は CodecCtx の可変メンバ（順序表・予測子表・world）を作るので
// 同時に走らせられない。幾何の候補が触るのは復号済みの列だけである。
// 出力は候補の順に並べ直してから比べるので、選ばれる符号器は並列でも変わらない。
// abort が真のときだけ、最短を超えた候補を打ち切る。
// **事前選別の標本では打ち切ってはいけない。**打ち切った候補は順位表から
// 消えるので、族の保護が「その族が 1 本も無い」と見て守れなくなる。
// 17 件のうち 1 件で、走査変換が標本で打ち切られて選ばれなくなった。
static void encode_many(const std::vector<Cand>& cs,
                        const std::vector<const std::vector<int64_t>*>& cv,
                        const CodecCtx* ctx, std::vector<std::vector<uint8_t>>& blobs,
                        std::vector<std::string>& errs, std::vector<char>& ok,
                        bool abort_long) {
    const size_t m = cs.size();
    blobs.assign(m, {}); errs.assign(m, std::string()); ok.assign(m, 0);
    static const size_t NT = [] {
        if (const char* e = getenv("PCC_THREADS")) { long v = atol(e); if (v > 0) return (size_t)v; }
        unsigned hw = std::thread::hardware_concurrency();
        return (size_t)(hw ? hw : 1);
    }();
    size_t nt = NT < m ? NT : m;
    if (nt <= 1) {
        std::atomic<size_t> best{(size_t)-1};
        enc_best = abort_long ? &best : nullptr;
        for (size_t i = 0; i < m; ++i) {
            try {
                ok[i] = codec_encode(cs[i].codec, cv, cs[i].param, blobs[i], errs[i], ctx) ? 1 : 0;
            } catch (const EncAbort&) {
                ok[i] = 0; errs[i] = "最短を超えたので打ち切り"; blobs[i].clear();
            }
            if (ok[i] && blobs[i].size() < best.load(std::memory_order_relaxed))
                best.store(blobs[i].size(), std::memory_order_relaxed);
        }
        enc_best = nullptr;
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
                }
                if (ok[i]) {
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
    std::vector<const std::vector<int64_t>*> cv;
    for (const auto& c : cols) cv.push_back(f.get(c));
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
    // 上位いくつを全点で測るか。1 でも 2 でも 17 件で出力は変わらないが、
    // 予測子を足す前は 1 だと TLS p1 が伸びた（24.252 → 25.462 bpp）ので、
    // 余裕を 1 本だけ残す。3 から 2 で符号化が 0.23 s → 0.20 s になる。
    static const size_t PRE_KEEP = [] {
        const char* e = getenv("PCC_PREKEEP");
        return e ? (size_t)atol(e) : (size_t)2;
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
    std::vector<Cand> use = candidates;
    if (preselect && PRE_SAMP && candidates.size() > PRE_KEEP + 1 &&
        ncols_n > PRE_SAMP * 2) {
        std::vector<std::vector<int64_t>> sub;
        sub.reserve(cv.size());
        for (auto* c : cv) sub.emplace_back(c->begin(), c->begin() + PRE_SAMP);
        std::vector<const std::vector<int64_t>*> scv;
        for (auto& v : sub) scv.push_back(&v);
        std::vector<std::vector<uint8_t>> sb; std::vector<std::string> se;
        std::vector<char> sok;
        if (par) {
            encode_many(candidates, scv, ctx, sb, se, sok, false);
        } else {
            sb.assign(candidates.size(), {}); se.assign(candidates.size(), std::string());
            sok.assign(candidates.size(), 0);
            for (size_t t = 0; t < candidates.size(); ++t)
                sok[t] = codec_encode(candidates[t].codec, scv, candidates[t].param,
                                      sb[t], se[t], ctx) ? 1 : 0;
        }
        std::vector<std::pair<size_t, Cand>> pre;
        for (size_t t = 0; t < candidates.size(); ++t)
            if (sok[t]) pre.push_back({sb[t].size(), candidates[t]});
        if (pre.size() > PRE_KEEP) {
            std::sort(pre.begin(), pre.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            use.clear();
            for (size_t i = 0; i < PRE_KEEP; ++i) use.push_back(pre[i].second);
            // 標本で系統的に不利になる族は、順位に入らなくても 1 つ残す。
            //   走査モデル … 掃引の数が減ると当てはめが効かない
            //                 （AHN3 の走査変換は標本では 6 位以内にも入らないのに
            //                   全点では 1 位だった）
            //   空間予測   … 点を間引くと近傍が遠くなる（30 節の nir と同じ理由）
            //   幾何v4     … 最近傍の当たり方が場所で変わる（AHN4 _21 は
            //                   先頭 12.5 万点だけ見ると 3 位以内に入らない）
            auto family = [](const Cand& c) -> int {
                if (c.codec == C_GEOM_SCAN) return 1;
                if (c.codec == C_ATTR_SPATIAL || c.codec == C_ATTR_COLOR) return 2;
                if (c.codec == C_ATTR_XREF && !c.param.empty() &&
                    c.param[0] != 0 && c.param[0] != 100) return 2;
                if (c.codec == C_GEOM_XYZ && !c.param.empty() && c.param[0] == 4) return 3;
                return 0;
            };
            // 族の保護は無条件だと重い。走査モデルは全点で 1 本 0.09 s かかり、
            // 既定の符号化時間の半分近くを占める。標本での符号長が最良から
            // 離れすぎている族は、全点で測っても勝たないと見て落とす。
            static const double FAM_THR = [] {
                const char* e = getenv("PCC_FAMTHR");
                return e ? atof(e) : 1e9;
            }();
            const double best_pre = (double)pre[0].first;
            for (int fam = 1; fam <= 3; ++fam) {
                bool have = false;
                for (const auto& c : use) if (family(c) == fam) have = true;
                if (have) continue;
                for (const auto& pr : pre)
                    if (family(pr.second) == fam) {
                        if ((double)pr.first <= best_pre * FAM_THR) use.push_back(pr.second);
                        break;
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
    if (par) encode_many(use, cv, ctx, fb, fe, fok, ABORT_ON && !on_sample);
    for (size_t ui = 0; ui < use.size(); ++ui) {
        const Cand& cd = use[ui];
        std::vector<uint8_t> blob; std::string err;
        bool got;
        if (par) { got = fok[ui] != 0; blob = std::move(fb[ui]); err = fe[ui]; }
        else got = codec_encode(cd.codec, cv, cd.param, blob, err, ctx);
        if (!got) {
            if (trace) *trace += "      " + cand_name(cd.codec, cd.param) + " 不可: " + err + "\n";
            continue;
        }
        if (trace) {
            char m[160];
            snprintf(m, sizeof m, "      %-10s %8.3f bpp\n", cand_name(cd.codec, cd.param).c_str(),
                     f.n ? blob.size() * 8.0 / f.n : 0.0);
            *trace += m;
        }
        const size_t sz = blob.size();
        rank.push_back({sz, {cd.codec, cd.param}});
        if (first || sz < best.data.size()) {
            best.codec = cd.codec; best.param = cd.param;
            best.data = std::move(blob); first = false;
        }
    }
    std::sort(rank.begin(), rank.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t i = 1; i < rank.size() && best.alt.size() < KEEP_ALT; ++i)
        best.alt.push_back(rank[i].second);
    // 標本で順位を付けると、空間予測を使う候補が系統的に不利になる。
    // 点を間引くと近傍が遠くなり、標本上では実力より悪く見えるためである
    // （AHN3 の nir は全点では参照＋空間予測が勝つのに、標本では 4 位にも入らない）。
    // 上位に入らなくても、空間予測を使う最良の候補は必ず 1 つ残す。
    auto uses_spatial = [](const Cand& c) {
        if (c.codec == C_ATTR_SPATIAL || c.codec == C_ATTR_COLOR) return true;
        return c.codec == C_ATTR_XREF && !c.param.empty() &&
               c.param[0] != 0 && c.param[0] != 100;
    };
    auto same = [](const Cand& a, const Cand& b) {
        return a.codec == b.codec && a.param == b.param;
    };
    bool have = uses_spatial({best.codec, best.param});
    for (const auto& a : best.alt) if (uses_spatial(a)) have = true;
    if (!have)
        for (const auto& r : rank)
            if (uses_spatial(r.second)) { best.alt.push_back(r.second); break; }
    return best;
}

std::vector<Stream> plan_streams(const Frame& f, bool joint_geom, std::string* log,
                                 const CodecCtx* ctx, bool trace_all) {
    std::vector<Stream> out;
    // 空間予測の近傍表は、最初にそれを呼ぶ列（たいてい intensity）が 1 人で
    // 背負っている。AHN3 _20 の 100 万点では intensity が 0.35 秒、同じ候補数の
    // red が 0.12 秒で、差の 0.23 秒がこの構築である。幾何の列を符号化している
    // 間に裏で作っておけば隠れる。表の中身は変わらないので出力は同じ。
    std::thread warm;
    if (ctx && (ctx->world || ctx->want_world))
        warm = std::thread([ctx, &f] { ctx->ensure(f.n, 1); });
    struct Joiner {
        std::thread& t;
        ~Joiner() { if (t.joinable()) t.join(); }
    } joiner{warm};
    std::vector<Cand> cand{{C_RAW64, {}}, {C_RANGE, {}}, {C_RANGE_DELTA, {}},
                           {C_RANGE_CTX, {}}, {C_RANGE_CTX2, {}}, {C_RANGE_MED, {}},
                           {C_RANGE_CACHE, {}}};
    // 幾何が揃っていれば、空間予測の候補（予測子 1 / 3 / 5 個）も加える
    std::vector<Cand> cand_attr = cand;
    if (ctx && (ctx->world || ctx->want_world))
        for (uint8_t P : {1, 3, 5}) cand_attr.push_back({C_ATTR_SPATIAL, {P}});
    // 順序の実験では幾何だけが関心で、属性の候補掃引が時間の大半を占める。
    // 絞っても往復検証は全列に掛かるので、検証の強さは落ちない。
    // 幾何より前に置く列（gps_time など）も cand から作るので、ここで一緒に絞られる。
    // 幾何の候補 gc は別に組むので影響しない。
    if (ctx && ctx->fast_attr) {
        cand = {{C_RANGE_DELTA, {}}, {C_RANGE_CTX, {}}};
        cand_attr = cand;
    }

    std::string tr;
    std::string* trp = trace_all ? &tr : nullptr;
    // 列ごとの所要を測れないと、どこを速くすればよいかが決まらない。
    double t_col = now_sec();
    auto emit = [&](Stream&& s) {
        double dt = now_sec() - t_col;
        t_col = now_sec();
        if (log) {
            std::string nm;
            for (size_t i = 0; i < s.cols.size(); ++i) nm += (i ? "+" : "") + s.cols[i];
            char m[256];
            snprintf(m, sizeof m, "  %-22s %-8s %8.3f bpp %7.2fs\n", nm.c_str(),
                     cand_name(s.codec, s.param).c_str(),
                     f.n ? s.data.size() * 8.0 / f.n : 0.0, dt);
            *log += m;
            if (trace_all) { *log += tr; tr.clear(); }
        }
        out.push_back(std::move(s));
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
        }
        emit(best_stream(f, {nm}, cs, ctx, trp));
        pre_done.push_back(nm);
    }
    std::string aux = pre.empty() ? std::string() : pre.back();

    if (joint_geom) {
        std::vector<std::string> g{f.geom[0], f.geom[1], f.geom[2]};
        std::vector<Cand> gc{{C_RAW64, {}}, {C_RANGE, {}}, {C_RANGE_DELTA, {}},
                             {C_GEOM_XYZ, {0}}, {C_GEOM_XYZ, {1}}, {C_GEOM_XYZ, {3}},
                             // 予測子だけを差し替えた 幾何v3。副情報は増えない。
                             // 直近 4 つの差分の平均は TLS と AHN3 で、中央値の
                             // 半分は autzen-2023 で残差が短かった（符号化前の見積り）。
                             {C_GEOM_XYZ, {3, 1}}, {C_GEOM_XYZ, {3, 2}},
                             // 直近 W 点の最近傍から予測する。格納順が空間的に
                             // 連続していない入力で効く（幾何v4 の注記を参照）。
                             {C_GEOM_XYZ, {4, 4}}, {C_GEOM_XYZ, {4, 16}},
                             // 選んだ点に局所の傾きを足す。副情報は増えない。
                             {C_GEOM_XYZ, {4, 4, 1}}};
        if (!aux.empty()) {
            std::vector<uint8_t> pv{2};
            uint16_t l = (uint16_t)aux.size();
            pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
            pv.insert(pv.end(), aux.begin(), aux.end());
            gc.push_back({C_GEOM_XYZ, pv});
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
            // v2（z を中央値予測）と v3（z の文脈を面内・面外から作る）と v4 は
            // 既定では出さない。ただし**その根拠だった集計は誤りだった**
            // （results/scan_model_fitting.md 16・32 節）。v3 が v1 を上回るのは
            // 1 件ではなく 2 件で、vegetation の 0.31% は誤差では片づかない。
            // PCC_ALL_VARIANTS=1 で測り直して決め直すこと。
            std::vector<uint8_t> vars{1, 5};
            if (const char* e = getenv("PCC_ALL_VARIANTS"))
                if (e[0] == '1') vars = {1, 2, 3, 4, 5};
            for (uint8_t v : vars) {
                std::vector<uint8_t> q = pv; q[0] = v;
                gc.push_back({C_GEOM_SCAN, q});
            }
        }
        if (ctx && !ctx->force_geom.empty()) {
            std::vector<Cand> only;
            for (const auto& c : gc)
                if (cand_name(c.codec, c.param) == ctx->force_geom) only.push_back(c);
            if (only.empty()) {
                if (log) *log += "  （--force-geom " + ctx->force_geom +
                                 " はこの入力では候補にならない）\n";
            } else {
                gc.swap(only);
            }
        }
        emit(best_stream(f, g, gc, ctx, trp, true));   // 幾何だけ事前選別する
    }

    // 色 3 列が揃っていれば、可逆色変換つきの同時符号化を候補に加える。
    // 色の脱相関と空間予測は独立に効き、重ねると掛け算になる（実測 6.430 対 11.179）。
    bool color_done = false;
    std::vector<std::string> rgb{"red", "green", "blue"};
    std::vector<Stream> color_streams;
    if (ctx && ctx->world) {
        bool all = true;
        for (const auto& c : rgb) {
            const ColSpec* sp = f.spec(c);
            if (!sp || sp->storage != Storage::Raw || !f.get(c)) all = false;
        }
        if (all) {
            // 候補 1: 3 列まとめて可逆色変換 + 空間予測
            std::vector<Cand> cc;
            for (uint8_t P : {1, 3, 5}) cc.push_back({C_ATTR_COLOR, {P}});
            Stream joint = best_stream(f, rgb, cc, ctx, trp);

            // 候補 2: blue → green → red の順に、直前の色を参照する鎖
            std::vector<Stream> chain;
            size_t sep = 0;
            const char* ord[3] = {"blue", "green", "red"};
            for (int i = 0; i < 3; ++i) {
                std::vector<Cand> cs = cand_attr;
                if (i > 0) {
                    std::string ref = ord[i - 1];
                    uint16_t l = (uint16_t)ref.size();
                    for (uint8_t P : {0, 100, 1, 3, 5}) {
                        std::vector<uint8_t> pv{P};
                        pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                        pv.insert(pv.end(), ref.begin(), ref.end());
                        cs.push_back({C_ATTR_XREF, pv});
                    }
                }
                chain.push_back(best_stream(f, {ord[i]}, cs, ctx, trp));
                sep += chain.back().data.size();
            }
            if (joint.data.size() <= sep) color_streams.push_back(std::move(joint));
            else for (auto& s2 : chain) color_streams.push_back(std::move(s2));
            color_done = true;
            for (auto& s2 : color_streams) emit(std::move(s2));
        }
    }

    std::vector<std::string> emitted = pre_done;              // 既に出した列（参照に使える）
    for (const auto& c : f.schema) {
        if (joint_geom && c.role == Role::Geometry) continue;
        if (c.storage == Storage::Derived) continue;          // 計画から復元するので送らない
        if (color_done && (c.name == "red" || c.name == "green" || c.name == "blue")) continue;
        bool done = false;
        for (const auto& e : pre_done) if (e == c.name) done = true;
        if (done) continue;                                   // 幾何より前に出した
        std::vector<Cand> cs = cand_attr;
        // 既出の列との残差も候補に入れる。標本で相手を 2 つに絞ってから全点で測る。
        // 参照の選定は全点の統計で行う（標本で選ぶと候補の集合が変わる）
        const Frame& fs_ref = (ctx && ctx->full) ? *ctx->full : f;
        const auto* v = f.get(c.name);
        const auto* vr = fs_ref.get(c.name);
        if (v && !emitted.empty()) {
            std::vector<std::pair<double, std::string>> sc;
            for (const auto& e : emitted) {
                const auto* w = fs_ref.get(e);
                if (w && vr) sc.push_back({entropy_diff_sample(*vr, *w, 250000), e});
            }
            std::sort(sc.begin(), sc.end());
            for (size_t i = 0; i < sc.size() && i < 2; ++i) {
                uint16_t l = (uint16_t)sc[i].second.size();
                for (uint8_t P : {0, 100, 1, 3, 5}) {
                    std::vector<uint8_t> pv{P};
                    pv.insert(pv.end(), (uint8_t*)&l, (uint8_t*)&l + 2);
                    pv.insert(pv.end(), sc[i].second.begin(), sc[i].second.end());
                    cs.push_back({C_ATTR_XREF, pv});
                }
            }
        }
        emit(best_stream(f, {c.name}, cs, ctx, trp));
        emitted.push_back(c.name);
    }
    return out;
}

// ---------------------------------------------------------------- 書き出し / 読み込み
enum : uint16_t {
    T_SRC_KIND = 1, T_SRC_BYTES = 2, T_SCALE = 3, T_OFFSET = 4, T_SCHEMA = 5,
    T_PLAN = 6, T_FIDELITY = 7, T_ENVELOPE = 8, T_GEOM_COLS = 9, T_GEOM_REPR = 10,
    T_EMBED = 11
};

static void put_tag(std::vector<uint8_t>& h, uint16_t tag, const std::vector<uint8_t>& body) {
    put<uint16_t>(h, tag);
    put<uint32_t>(h, (uint32_t)body.size());
    h.insert(h.end(), body.begin(), body.end());
}

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
    b.clear(); { std::vector<uint8_t> pb(f.plan.begin(), f.plan.end()); put_blob(b, pb); }
    put_tag(h, T_PLAN, b);
    b.clear(); put<uint8_t>(b, f.fid.exact ? 0 : 1);
               put<double>(b, f.fid.declared_eps);
               put<double>(b, f.fid.measured_max);     put_tag(h, T_FIDELITY, b);
    b.clear(); put_blob(b, f.envelope);                put_tag(h, T_ENVELOPE, b);
    b.clear(); for (int i = 0; i < 3; ++i) put_str(b, f.geom[i]);  put_tag(h, T_GEOM_COLS, b);
    b.clear(); put_str(b, f.geom_repr);                put_tag(h, T_GEOM_REPR, b);
    if (!f.embed.empty()) {
        b.clear(); put_str(b, f.embed_kind); put_blob(b, f.embed);  put_tag(h, T_EMBED, b);
    }

    std::vector<uint8_t> o;
    o.insert(o.end(), {'P', 'C', 'C', '2'});
    put<uint16_t>(o, 1);
    put<uint16_t>(o, f.fid.exact ? 0 : 1);
    put<uint64_t>(o, f.n);
    put<uint32_t>(o, (uint32_t)h.size());
    o.insert(o.end(), h.begin(), h.end());
    put<uint32_t>(o, (uint32_t)st.size());
    for (const auto& s : st) {
        put<uint16_t>(o, (uint16_t)s.cols.size());
        for (const auto& c : s.cols) put_str(o, c);
        put<uint16_t>(o, s.codec);
        put_blob(o, s.param);
        put<uint64_t>(o, (uint64_t)s.data.size());
    }
    for (const auto& s : st) o.insert(o.end(), s.data.begin(), s.data.end());
    put<uint64_t>(o, crc64(o.data(), o.size()));

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { err = "書き込めない: " + path; return false; }
    fwrite(o.data(), 1, o.size(), fp);
    fclose(fp);
    bytes_out = o.size();
    return true;
}

bool read_pcc2(const std::string& path, Frame& f, std::string& err) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { err = "開けない: " + path; return false; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, fp) != (size_t)sz) { fclose(fp); err = "短い"; return false; }
    fclose(fp);
    if (sz < 28 || memcmp(buf.data(), "PCC2", 4)) { err = "PCC2 ではない"; return false; }
    uint64_t want; memcpy(&want, buf.data() + sz - 8, 8);
    if (crc64(buf.data(), sz - 8) != want) { err = "crc64 が合わない"; return false; }

    Rd r{buf.data(), (size_t)sz - 8, 4};
    uint16_t ver = r.get<uint16_t>(); r.get<uint16_t>();
    if (ver != 1) { err = "版が違う"; return false; }
    f.n = r.get<uint64_t>();
    uint32_t hl = r.get<uint32_t>();
    size_t hend = r.p + hl;
    while (r.ok && r.p < hend) {
        uint16_t tag = r.get<uint16_t>();
        uint32_t len = r.get<uint32_t>();
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
        case T_PLAN: { auto pb = r.blob(); f.plan.assign(pb.begin(), pb.end()); break; }
        case T_FIDELITY:
            f.fid.exact = r.get<uint8_t>() == 0;
            f.fid.declared_eps = r.get<double>();
            f.fid.measured_max = r.get<double>();
            break;
        case T_ENVELOPE:  f.envelope = r.blob(); break;
        case T_GEOM_COLS: for (int i = 0; i < 3; ++i) f.geom[i] = r.str(); break;
        case T_GEOM_REPR: f.geom_repr = r.str(); break;
        case T_EMBED:     f.embed_kind = r.str(); f.embed = r.blob(); break;
        default: break;   // 知らないタグは読み飛ばす（前方互換）
        }
        r.p = next;
    }
    if (!r.ok) { err = "ヘッダが壊れている"; return false; }

    uint32_t ns = r.get<uint32_t>();
    std::vector<Stream> st(ns);
    std::vector<uint64_t> dlen(ns);
    for (uint32_t i = 0; i < ns && r.ok; ++i) {
        uint16_t nc = r.get<uint16_t>();
        for (uint16_t k = 0; k < nc; ++k) st[i].cols.push_back(r.str());
        st[i].codec = r.get<uint16_t>();
        st[i].param = r.blob();
        dlen[i] = r.get<uint64_t>();
    }
    if (!r.ok) { err = "ストリーム記述が壊れている"; return false; }
    // 幾何は属性より先に並んでいる。幾何が揃った時点で座標を組み、
    // 以後の属性ストリームはそれを副次情報として使う（副情報は生じない）。
    CodecCtx ctx;
    ctx.fr = &f;
    std::vector<double> world;
    bool world_ready = false;
    for (uint32_t i = 0; i < ns; ++i) {
        if (r.p + dlen[i] > r.n) { err = "ストリームが短い"; return false; }
        std::vector<std::vector<int64_t>> outc;
        if (!codec_decode(st[i].codec, st[i].param, buf.data() + r.p, dlen[i],
                          f.n, st[i].cols.size(), outc, err, &ctx)) {
            err += "（ストリーム " + std::to_string(i) + ": ";
            for (size_t c = 0; c < st[i].cols.size(); ++c) err += (c ? "+" : "") + st[i].cols[c];
            err += " / 符号器 " + cand_name(st[i].codec, st[i].param) + "）";
            return false;
        }
        for (size_t c = 0; c < st[i].cols.size(); ++c) f.col[st[i].cols[c]] = std::move(outc[c]);
        r.p += dlen[i];
        if (!world_ready && f.get(f.geom[0]) && f.get(f.geom[1]) && f.get(f.geom[2])) {
            frame_world(f, world);
            ctx.world = &world;
            world_ready = true;
        }
    }
    return true;
}

} // namespace pcc
