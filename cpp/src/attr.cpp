// 属性を「格納順の1つ前」で予測する場合と「既に復号済みの空間近傍」で
// 予測する場合を、同じレンジコーダで端から端まで比べる。
//
// 因果性: 容器は幾何を先に復号するので、復号器は全点の座標を持っている。
// したがって (a) 属性を符号化する順序を幾何から導いてよいし、
// (b) その順序で先行する空間近傍を予測に使ってよい。
// 符号化器と復号器はまったく同じ順序表・近傍表を作れる。
#include "pcc/attr.hpp"
#include "pcc/las.hpp"
#include "pcc/geom.hpp"
#include "pcc/rangecoder.hpp"
#include <cstdio>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

namespace pcc {

static inline uint64_t part1by2(uint64_t x) {
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffffull;
    x = (x | x << 16) & 0x1f0000ff0000ffull;
    x = (x | x <<  8) & 0x100f00f00f00f00full;
    x = (x | x <<  4) & 0x10c30c30c30c30c3ull;
    x = (x | x <<  2) & 0x1249249249249249ull;
    return x;
}

// 符号化の順序を決める。storage は元の並び、morton は幾何から導いた並び。
std::vector<int32_t> coding_order(const std::vector<double>& xyz, size_t n,
                                  const std::string& kind) {
    std::vector<int32_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = (int32_t)i;
    if (kind != "morton") return perm;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], xyz[i*3+d]); hi[d] = std::max(hi[d], xyz[i*3+d]); }
    double side = 0;
    for (int d = 0; d < 3; ++d) side = std::max(side, hi[d] - lo[d]);
    side *= 1.0000001;
    const uint64_t M = 1ull << 21;
    std::vector<uint64_t> key(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t c[3];
        for (int d = 0; d < 3; ++d) {
            int64_t q = (int64_t)((xyz[i*3+d] - lo[d]) / side * (double)M);
            c[d] = (uint64_t)std::max<int64_t>(0, std::min<int64_t>((int64_t)M - 1, q));
        }
        key[i] = (part1by2(c[0]) << 2) | (part1by2(c[1]) << 1) | part1by2(c[2]);
    }
    std::sort(perm.begin(), perm.end(),
              [&](int32_t a, int32_t b) { return key[a] < key[b]; });
    return perm;
}

// 複数の P ぶんの表を 1 回の近傍探索で作る。
//
// P ごとに呼ぶと KdTree を毎回建て直し、1 点あたりの近傍探索も P の数だけ走る。
// P=1/3/5 なら木を 3 回、探索を 3 回である。探索は距離の昇順に返るので、
// k=9 で引いた先頭 5 つは k=5 で引いた結果と同じものになる。よって 1 回引いて
// 「先頭 P+4 個の中の先行点を最大 P 個」を P ごとに切り出せば、
// P ごとに引いたときと **同じ表** が得られる。
void build_causal_predictors_multi(const std::vector<double>& xyz, size_t n,
                                   const std::vector<int32_t>& perm,
                                   const std::vector<int>& Ps,
                                   std::vector<std::vector<int32_t>>& preds) {
    if (Ps.empty()) return;
    int kmax = 0;
    for (int P : Ps) kmax = std::max(kmax, P + 4);
    std::vector<int32_t> rank(n);
    for (size_t t = 0; t < n; ++t) rank[perm[t]] = (int32_t)t;
    preds.assign(Ps.size(), {});
    for (size_t a = 0; a < Ps.size(); ++a) preds[a].assign(n * (size_t)Ps[a], -1);
    double tb = now_sec();
    KdTree tree(xyz);
    if (getenv("PCC_DPROF")) fprintf(stderr, "        [木] %.3fs\n", now_sec() - tb);
    tb = now_sec();
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned w = 0; w < nt; ++w) {
        th.emplace_back([&, w] {
            std::vector<int64_t> idx(kmax);
            std::vector<double> d2(kmax);
            for (size_t t = w; t < n; t += nt) {
                if (t == 0) continue;
                int32_t i = perm[t];
                tree.knn(&xyz[(size_t)i * 3], kmax, idx.data(), d2.data());
                for (size_t a = 0; a < Ps.size(); ++a) {
                    const int P = Ps[a], ks = P + 4;
                    auto& pred = preds[a];
                    int got = 0;
                    for (int j = 0; j < ks && got < P; ++j) {
                        if (idx[j] < 0) continue;
                        int32_t r = rank[idx[j]];
                        if ((size_t)r < t) pred[t * (size_t)P + got++] = r;
                    }
                    if (got == 0) pred[t * (size_t)P] = (int32_t)(t - 1);
                }
            }
        });
    }
    for (auto& x : th) x.join();
    if (getenv("PCC_DPROF")) fprintf(stderr, "        [探索] %.3fs\n", now_sec() - tb);
}

// 符号化順で先行する空間近傍を最大 P 個。全フィールドで使い回す。
void build_causal_predictors(const std::vector<double>& xyz, size_t n,
                             const std::vector<int32_t>& perm, int P, int k_search,
                             std::vector<int32_t>& pred) {
    std::vector<int32_t> rank(n);
    for (size_t t = 0; t < n; ++t) rank[perm[t]] = (int32_t)t;
    pred.assign(n * P, -1);           // pred は「符号化順の位置 t」で索く
    KdTree tree(xyz);
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned w = 0; w < nt; ++w) {
        th.emplace_back([&, w] {
            std::vector<int64_t> idx(k_search);
            std::vector<double> d2(k_search);
            for (size_t t = w; t < n; t += nt) {
                if (t == 0) continue;
                int32_t i = perm[t];
                tree.knn(&xyz[(size_t)i * 3], k_search, idx.data(), d2.data());
                int got = 0;
                for (int j = 0; j < k_search && got < P; ++j) {
                    if (idx[j] < 0) continue;
                    int32_t r = rank[idx[j]];
                    if ((size_t)r < t) pred[t * P + got++] = r;   // 符号化順の位置で持つ
                }
                if (got == 0) pred[t * P] = (int32_t)(t - 1);
            }
        });
    }
    for (auto& x : th) x.join();
}

// w は符号化順に並べ替えた値の列。位置 t の予測値を返す。
template <class T>
static inline int64_t predict(const T* w, const std::vector<int32_t>& pred,
                              int P, size_t t) {
    if (t == 0) return 0;
    int64_t s = 0; int c = 0;
    for (int j = 0; j < P; ++j) {
        int32_t p = pred[t * P + j];
        if (p < 0) break;
        s += (int64_t)w[p]; ++c;
    }
    if (c == 0) return (int64_t)w[t - 1];
    return s >= 0 ? (s + c / 2) / c : -((-s + c / 2) / c);
}

// 符号化順に並べ替えた値を置く作業領域。1 列につき候補が 6〜10 本あり、
// 候補ごとに 100 万点で 8 MB を確保して捨てていた。スレッドごとに使い回す。
//
// **多くの列は int32 に収まる**（強度・分類・色・走査角）。収まる間は 32 bit で
// 持ち、溢れた列だけ 64 bit に移す。並列に走る本数だけ効くので、200 万点・
// 16 並列で 256 MB の差になる。予測は 64 bit で足すので値は変わらない。
static thread_local std::vector<int32_t> g_w32;
static thread_local std::vector<int64_t> g_w64;

// **out は v と同じ配列でよい。**値はいったん w に符号化順で写してから引くので、
// 2 つめのループは v を読まない。候補ごとに出力用の配列を別に取ると、
// 並列に走る本数だけメモリが要る（200 万点・16 並列で 256 MB）。
void spatial_residual(const std::vector<int64_t>& v, const std::vector<int32_t>& perm,
                      const std::vector<int32_t>& pred, int P, size_t n,
                      std::vector<int64_t>& out) {
    g_w32.resize(n);
    size_t t = 0;
    for (; t < n; ++t) {
        int64_t x = v[(size_t)perm[t]];
        if (x < INT32_MIN || x > INT32_MAX) break;
        g_w32[t] = (int32_t)x;
    }
    if (t == n) {
        std::vector<int64_t>().swap(g_w64);          // 広い側は抱えない
        const int32_t* w = g_w32.data();
        out.resize(n);
        for (size_t i = 0; i < n; ++i) out[i] = (int64_t)w[i] - predict(w, pred, P, i);
        return;
    }
    g_w64.resize(n);
    for (size_t i = 0; i < t; ++i) g_w64[i] = g_w32[i];
    for (size_t i = t; i < n; ++i) g_w64[i] = v[(size_t)perm[i]];
    const int64_t* w = g_w64.data();
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = w[i] - predict(w, pred, P, i);
}

void spatial_restore(const std::vector<int64_t>& res, const std::vector<int32_t>& perm,
                     const std::vector<int32_t>& pred, int P, size_t n,
                     std::vector<int64_t>& out) {
    // 復号は値が判る前に幅を決められないので 64 bit で持つ。
    g_w64.resize(n);
    int64_t* w = g_w64.data();
    for (size_t t = 0; t < n; ++t) w[t] = res[t] + predict(w, pred, P, t);
    out.resize(n);
    for (size_t t = 0; t < n; ++t) out[(size_t)perm[t]] = w[t];
}

AttrResult compare_field(const std::string& name, const std::vector<int64_t>& v,
                         const std::vector<int32_t>& perm,
                         const std::vector<int32_t>& pred, int P, size_t n) {
    AttrResult R; R.name = name;
    // 差分を取らずそのまま符号化する候補も測る。
    // 構造のない値に差分を掛けると符号長は伸びるので、この候補が要る。
    R.bpp_raw = encode_ints(v.data(), n).size() * 8.0 / n;
    std::vector<int64_t> ro(n);
    for (size_t i = 0; i < n; ++i) ro[i] = v[i] - (i ? v[i - 1] : 0);
    auto bo = encode_ints(ro.data(), n);
    R.bpp_order = bo.size() * 8.0 / n;

    std::vector<int64_t> w(n), rs(n);
    for (size_t t = 0; t < n; ++t) w[t] = v[perm[t]];
    for (size_t t = 0; t < n; ++t) rs[t] = w[t] - predict(w.data(), pred, P, t);
    auto bs = encode_ints(rs.data(), n);
    R.bpp_spatial = bs.size() * 8.0 / n;

    // 復元検証: 残差 → 符号化順の値 → 元の並び
    std::vector<int64_t> dec(n), back(n, 0);
    decode_ints(bs.data(), bs.size(), dec.data(), n);
    bool ok = true;
    for (size_t t = 0; t < n; ++t) back[t] = dec[t] + predict(back.data(), pred, P, t);
    for (size_t t = 0; t < n && ok; ++t) if (back[t] != v[perm[t]]) ok = false;
    R.roundtrip_ok = ok;
    return R;
}

} // namespace pcc
