#include "pcc/pcc2.hpp"
#include "pcc/rangecoder.hpp"
#include "pcc/attr.hpp"
#include "pcc/scanmodel.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace pcc {

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
                       std::vector<uint8_t>& out) {
    size_t n = cols[0]->size();
    Encoder e;
    UIntCoder uc(3 * NCTX, 64);
    MedPred mp[3];
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
static void dec_geom_x(const uint8_t* data, size_t len, size_t n,
                       std::vector<std::vector<int64_t>>& out) {
    out.assign(3, std::vector<int64_t>(n));
    Decoder d(data, len);
    UIntCoder uc(3 * NCTX, 64);
    MedPred mp[3];
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
bool CodecCtx::ensure(size_t n, int P) const {
    if (!world || world->size() < n * 3) return false;
    if (built_P == P) return true;
    if (perm.empty()) perm = coding_order(*world, n, "morton");
    build_causal_predictors(*world, n, perm, P, P + 4, pred);
    built_P = P;
    return true;
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
static void enc_resid(const std::vector<std::vector<int64_t>>& res, std::vector<uint8_t>& out) {
    size_t nc = res.size(), n = nc ? res[0].size() : 0;
    Encoder e;
    UIntCoder uc((int)nc * NCTX, 64);
    std::vector<int> ctx(nc, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < nc; ++c) {
            uint64_t z = zigzag(res[c][i]);
            uc.encode(e, z, (int)c * NCTX + ctx[c]);
            ctx[c] = ctx_of(z);
        }
    out = e.finish();
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
// そこから走査順（psid と gps の安定ソート）と掃引の切れ目（時刻の隙間 200 ulp）が
// 再現できるので、順序と区切りには副情報が要らない。
// 1 掃引に 2 台のスキャナが混ざる場合だけ、どちらの走査線かの標識を送る。

namespace {

struct ScanCtx {
    std::vector<int32_t> ord;        // 走査順 → 格納順
    std::vector<int64_t> gps, ret;   // 走査順に並べた値
    std::vector<int32_t> swp;        // 掃引の先頭位置（末尾に n）
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
    sc.gps.resize(n); sc.ret.assign(n, 1);
    for (size_t t = 0; t < n; ++t) {
        sc.gps[t] = (*g)[sc.ord[t]];
        if (r && r->size() >= n) sc.ret[t] = (*r)[sc.ord[t]] & 0x0F;
    }
    sc.swp.clear();
    for (size_t t = 0; t < n; ++t) {
        bool brk = (t == 0);
        if (!brk) {
            int32_t a = sc.ord[t - 1], b = sc.ord[t];
            brk = ((*s)[a] != (*s)[b]) || (sc.gps[t] - sc.gps[t - 1] > 200);
        }
        if (brk) sc.swp.push_back((int32_t)t);
    }
    sc.swp.push_back((int32_t)n);
    return true;
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
    ScanCtx sc;
    if (!build_scan_ctx(ctx, param, n, sc, err)) return false;
    const auto &CX = *cols[0], &CY = *cols[1], &CZ = *cols[2];

    size_t nsw = sc.swp.size() - 1;
    LineSet L;
    L.split.assign(nsw, 0);
    L.label.assign(n, 0);
    L.line_of_sweep.assign(nsw, 0);

    std::vector<int64_t> bx, by, bz, bg, bs;
    const auto* sa_col = ctx->fr->get("scan_angle");
    // PCC_SCAN_DUMP=<path> を付けたときだけ、走査線ごとの当てはめの様子を書き出す。
    // 符号化の結果には影響しない。
    FILE* dump = nullptr; FILE* dump_raw = nullptr;
    if (const char* dp = getenv("PCC_SCAN_DUMP")) {
        dump = fopen(dp, "w");
        if (dump) fprintf(dump, "id\tn\tspan_deg\tthin_mm\theight_m\tomega_deg_s"
                                "\tmdl_bits\talt_bits\tok"
                                "\ti0\ti1\ti2\ti3\ti4\ti5\ti6\ti7\ti8\ti9\ti10\ti11\n");
        std::string rp = std::string(dp) + ".raw";
        dump_raw = fopen(rp.c_str(), "w");
        if (dump_raw) fprintf(dump_raw, "# s\tz\tshot_ulp\tscan_angle\n");
    }
    for (size_t k = 0; k < nsw; ++k) {
        int32_t a = sc.swp[k], b = sc.swp[k + 1];
        size_t m = (size_t)(b - a);
        L.line_of_sweep[k] = (int32_t)L.line.size();
        auto gather = [&](uint8_t want, bool use_label) {
            bx.clear(); by.clear(); bz.clear(); bg.clear(); bs.clear();
            for (int32_t t = a; t < b; ++t) {
                if (use_label && L.label[t] != want) continue;
                int32_t i = sc.ord[t];
                bx.push_back(CX[i]); by.push_back(CY[i]); bz.push_back(CZ[i]);
                bg.push_back(sc.gps[t]);
                bs.push_back(sa_col && sa_col->size() >= n ? (*sa_col)[i] : 0);
            }
        };
        // 2 台のスキャナが混ざるのは走査幅を広く含む掃引だけ。狭い掃引を分割すると
        // 走査線の角度幅が足りず、高度と角速度が縮退して当てはまらない。
        bool wide = true;
        if (sa_col && sa_col->size() >= n && m > 0) {
            int64_t lo = (*sa_col)[sc.ord[a]], hi = lo;
            for (int32_t t = a; t < b; ++t) {
                int64_t v = (*sa_col)[sc.ord[t]];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            wide = ((double)(hi - lo) * 0.006) >= 20.0;   // scan_angle は 0.006 度単位
        }
        if (m >= 40 && wide) {
            gather(0, false);
            double r0 = line_thinness(bx.data(), by.data(), m);
            std::vector<uint8_t> lab;
            split_two_lines(bx.data(), by.data(), m, lab);
            size_t c1 = 0; for (auto v : lab) c1 += v;
            if (c1 >= 10 && m - c1 >= 10) {
                std::vector<int64_t> x0, y0, x1, y1;
                for (size_t i = 0; i < m; ++i)
                    if (lab[i]) { x1.push_back(bx[i]); y1.push_back(by[i]); }
                    else        { x0.push_back(bx[i]); y0.push_back(by[i]); }
                double w = (x0.size() * line_thinness(x0.data(), y0.data(), x0.size()) +
                            x1.size() * line_thinness(x1.data(), y1.data(), x1.size())) / m;
                if (w < r0 * 0.5) {
                    L.split[k] = 1;
                    for (size_t i = 0; i < m; ++i) L.label[a + i] = lab[i];
                }
            }
        }
        int nl = L.split[k] ? 2 : 1;
        for (int g = 0; g < nl; ++g) {
            gather((uint8_t)g, L.split[k] != 0);
            FitDiag fd;
            SweepParam sp = bx.empty() ? SweepParam()
                : fit_scan_line(bx.data(), by.data(), bz.data(), bg.data(), bs.data(),
                                bx.size(), dump ? &fd : nullptr);
            if (dump && !bx.empty()) {
                size_t id = L.line.size();
                fprintf(dump, "%zu\t%zu\t%.3f\t%.4f\t%.1f\t%.1f\t%.4f\t%.4f\t%d",
                        id, bx.size(), fd.span_deg, fd.thin_mm, fd.height_m,
                        fd.omega_deg_s, fd.mdl_bits, fd.alt_bits, (int)sp.ok);
                for (int it = 0; it < FIT_ITERS; ++it) fprintf(dump, "\t%.4f", fd.iter_bits[it]);
                fprintf(dump, "\n");
                // 50 本に 1 本は生データも出す。Python 側で同じ走査線に
                // scipy の当てはめを掛け、C++ の結果と直接比べるために使う。
                if (dump_raw && (id % 50) == 0) {
                    fprintf(dump_raw, "# line %zu n %zu\n", id, bx.size());
                    int64_t s_i, off_i;
                    for (size_t i = 0; i < bx.size(); ++i) {
                        shear_fwd(bx[i], by[i], sp.t2, sp.sn, s_i, off_i);
                        fprintf(dump_raw, "%lld\t%lld\t%lld\t%lld\n",
                                (long long)s_i, (long long)bz[i],
                                (long long)(bg[i] - bg[0]), (long long)bs[i]);
                    }
                }
            }
            L.line.push_back(sp);
        }
    }

    if (dump) { fclose(dump); dump = nullptr; }
    if (dump_raw) { fclose(dump_raw); dump_raw = nullptr; }

    std::vector<std::vector<int64_t>> res(3, std::vector<int64_t>(n, 0));
    // z は「同じ掃引・同じ戻り番号の直前の点」から予測する。
    // その系列がまだ空なら、直前に符号化した任意の点の z を使う（掃引をまたぐ）。
    int64_t last_any = 0;
    for (size_t k = 0; k < nsw; ++k) {
        int32_t a = sc.swp[k], b = sc.swp[k + 1];
        int64_t lz[16]; bool hz[16];
        for (int i = 0; i < 16; ++i) { lz[i] = 0; hz[i] = false; }
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
                g0[gidx] = sc.gps[t]; hg[gidx] = true;
                mps[gidx] = MedPred(); mps[gidx].prev = sp.s0;
                mpo[gidx] = MedPred(); mpo[gidx].prev = sp.off0;
            }
            int32_t i = sc.ord[t];
            int64_t s, off;
            shear_fwd(CX[i], CY[i], sp.t2, sp.sn, s, off);
            int64_t po = mpo[gidx].predict();
            int64_t ps = sp.ok ? scan_predict(sp, sc.gps[t] - g0[gidx], CZ[i])
                               : mps[gidx].predict();
            res[0][t] = s - ps;
            res[1][t] = off - po;
            mps[gidx].push(s); mpo[gidx].push(off);
            int r = (int)sc.ret[t] & 15;
            int64_t pz = hz[r] ? lz[r] : last_any;
            res[2][t] = CZ[i] - pz;
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
    enc_resid(res, br);

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
                    (okl ? sok : sng)[c] += bitlen(zigzag(res[c][t]));
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
    dec_resid(data + p, lr, n, 3, res);

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
        int64_t lz[16]; bool hz[16];
        for (int i = 0; i < 16; ++i) { lz[i] = 0; hz[i] = false; }
        int64_t g0[2] = {0, 0};
        bool hg[2] = {false, false};
        MedPred mps[2], mpo[2];
        for (int32_t t = a; t < b; ++t) {
            int li = line_index(L, k, (size_t)t);
            const SweepParam& sp = L.line[li];
            int gidx = (L.split[k] && L.label[t]) ? 1 : 0;
            if (!hg[gidx]) {
                g0[gidx] = sc.gps[t]; hg[gidx] = true;
                mps[gidx] = MedPred(); mps[gidx].prev = sp.s0;
                mpo[gidx] = MedPred(); mpo[gidx].prev = sp.off0;
            }
            int r = (int)sc.ret[t] & 15;
            int64_t pz = hz[r] ? lz[r] : last_any;
            int64_t z = pz + res[2][t];
            lz[r] = z; hz[r] = true; last_any = z;
            int64_t po = mpo[gidx].predict();
            int64_t ps = sp.ok ? scan_predict(sp, sc.gps[t] - g0[gidx], z)
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
    case C_GEOM_XYZ: {
        int var = param.empty() ? 0 : param[0];
        if (var == 2) {
            const std::vector<int64_t>* a = aux_col(param, ctx);
            if (!a || a->size() < cols[0]->size()) { err = "補助列がない"; return false; }
            enc_geom_aux(cols, *a, out);
        } else if (var == 3) {
            if (cols.size() != 3) { err = "幾何v3 は 3 軸"; return false; }
            enc_geom_x(cols, out);
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
            if (!ctx->ensure(n, P)) { err = "座標がない"; return false; }
            std::vector<int64_t> r;
            spatial_residual(d[0], ctx->perm, ctx->pred, P, n, r);
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
        if (!ctx->ensure(n, P)) { err = "座標がない"; return false; }
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
            spatial_residual(src[c], ctx->perm, ctx->pred, P, n, res[c]);
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
    case C_GEOM_XYZ: {
        int var = param.empty() ? 0 : param[0];
        if (var == 2) {
            const std::vector<int64_t>* a = aux_col(param, ctx);
            if (!a || a->size() < n) { err = "補助列がない"; return false; }
            dec_geom_aux(data, len, n, ncol, *a, out);
        } else if (var == 3) {
            if (ncol != 3) { err = "幾何v3 は 3 軸"; return false; }
            dec_geom_x(data, len, n, out);
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
            if (!ctx->ensure(n, P)) { err = "座標がない"; return false; }
            std::vector<int64_t> r;
            spatial_restore(d[0], ctx->perm, ctx->pred, P, n, r);
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
        if (!ctx->ensure(n, P)) { err = "座標がない"; return false; }
        std::vector<std::vector<int64_t>> res;
        dec_resid(data, len, n, ncol, res);
        std::vector<std::vector<int64_t>> src(ncol);
        for (size_t c = 0; c < ncol; ++c)
            spatial_restore(res[c], ctx->perm, ctx->pred, P, n, src[c]);
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
    case C_GEOM_SCAN: return "走査";
    case C_GEOM_XYZ:
        if (p.empty() || p[0] == 0) return "幾何v0";
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

Stream best_stream(const Frame& f, const std::vector<std::string>& cols,
                   const std::vector<Cand>& candidates, const CodecCtx* ctx,
                   std::string* trace) {
    std::vector<const std::vector<int64_t>*> cv;
    for (const auto& c : cols) cv.push_back(f.get(c));
    Stream best; best.cols = cols; best.codec = C_RAW64;
    bool first = true;
    for (const auto& cd : candidates) {
        std::vector<uint8_t> blob; std::string err;
        if (!codec_encode(cd.codec, cv, cd.param, blob, err, ctx)) {
            if (trace) *trace += "      " + cand_name(cd.codec, cd.param) + " 不可: " + err + "\n";
            continue;
        }
        if (trace) {
            char m[160];
            snprintf(m, sizeof m, "      %-10s %8.3f bpp\n", cand_name(cd.codec, cd.param).c_str(),
                     f.n ? blob.size() * 8.0 / f.n : 0.0);
            *trace += m;
        }
        if (first || blob.size() < best.data.size()) {
            best.codec = cd.codec; best.param = cd.param;
            best.data = std::move(blob); first = false;
        }
    }
    return best;
}

std::vector<Stream> plan_streams(const Frame& f, bool joint_geom, std::string* log,
                                 const CodecCtx* ctx, bool trace_all) {
    std::vector<Stream> out;
    std::vector<Cand> cand{{C_RAW64, {}}, {C_RANGE, {}}, {C_RANGE_DELTA, {}},
                           {C_RANGE_CTX, {}}, {C_RANGE_CTX2, {}}};
    // 幾何が揃っていれば、空間予測の候補（予測子 1 / 3 / 5 個）も加える
    std::vector<Cand> cand_attr = cand;
    if (ctx && ctx->world)
        for (uint8_t P : {1, 3, 5}) cand_attr.push_back({C_ATTR_SPATIAL, {P}});

    std::string tr;
    std::string* trp = trace_all ? &tr : nullptr;
    auto emit = [&](Stream&& s) {
        if (log) {
            std::string nm;
            for (size_t i = 0; i < s.cols.size(); ++i) nm += (i ? "+" : "") + s.cols[i];
            char m[256];
            snprintf(m, sizeof m, "  %-22s %-8s %8.3f bpp\n", nm.c_str(),
                     cand_name(s.codec, s.param).c_str(), f.n ? s.data.size() * 8.0 / f.n : 0.0);
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
                             {C_GEOM_XYZ, {0}}, {C_GEOM_XYZ, {1}}, {C_GEOM_XYZ, {3}}};
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
            gc.push_back({C_GEOM_SCAN, pv});
        }
        emit(best_stream(f, g, gc, ctx, trp));
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
        const auto* v = f.get(c.name);
        if (v && !emitted.empty()) {
            std::vector<std::pair<double, std::string>> sc;
            for (const auto& e : emitted) {
                const auto* w = f.get(e);
                if (w) sc.push_back({entropy_diff_sample(*v, *w, 250000), e});
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
    T_PLAN = 6, T_FIDELITY = 7, T_ENVELOPE = 8, T_GEOM_COLS = 9, T_GEOM_REPR = 10
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
