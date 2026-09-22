#include "pcc/diff.hpp"
#include "pcc/geom.hpp"
#include "pcc/rangecoder.hpp"
#include <zstd.h>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace pcc {

static inline uint64_t ckey(int64_t x, int64_t y, int64_t z) {
    return (uint64_t)(x * 73856093) ^ (uint64_t)(y * 19349663) ^ (uint64_t)(z * 83492791);
}

std::vector<OctDiff> octree_diff(const std::vector<double>& tgt, size_t nt,
                                 const std::vector<double>& ref, size_t nr,
                                 int max_depth) {
    // 共通の立方体で両者を量子化する。座標系が同じでなければ比較にならないので、
    // 原点も一辺も両方の外接箱の和から取る。
    double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
    auto ext = [&](const std::vector<double>& p, size_t n) {
        for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], p[i*3+d]); hi[d] = std::max(hi[d], p[i*3+d]); }
    };
    ext(tgt, nt); ext(ref, nr);
    double side = 0;
    for (int d = 0; d < 3; ++d) side = std::max(side, hi[d]-lo[d]);
    side *= 1.0000001;

    // 深さ L の格子座標に落とす
    auto grid = [&](const std::vector<double>& p, size_t n, int L, std::vector<uint64_t>& out) {
        uint64_t M = 1ull << L; double s = side / (double)M;
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            uint64_t c[3];
            for (int d = 0; d < 3; ++d) {
                int64_t q = (int64_t)((p[i*3+d] - lo[d]) / s);
                c[d] = (uint64_t)std::max<int64_t>(0, std::min<int64_t>((int64_t)M-1, q));
            }
            out[i] = (c[0] << 42) | (c[1] << 21) | c[2];  // L<=21
        }
    };

    std::vector<OctDiff> res;
    for (int L = 1; L <= max_depth; ++L) {
        // 深さ L の葉を作り、その親（深さ L-1）ごとに 8 bit の占有バイトを組む
        auto bytes_of = [&](const std::vector<double>& p, size_t n,
                            std::unordered_map<uint64_t,uint8_t>& m) {
            std::vector<uint64_t> g; grid(p, n, L, g);
            m.reserve(n * 2);
            for (uint64_t k : g) {
                uint64_t x = k >> 42, y = (k >> 21) & ((1ull<<21)-1), z = k & ((1ull<<21)-1);
                uint64_t par = ((x>>1) << 42) | ((y>>1) << 21) | (z>>1);
                uint8_t bit = (uint8_t)(1u << (((x&1)<<2) | ((y&1)<<1) | (z&1)));
                m[par] |= bit;
            }
        };
        std::unordered_map<uint64_t,uint8_t> A, B;
        bytes_of(tgt, nt, A); bytes_of(ref, nr, B);

        // H(c) と H(c | ĉ_ref)。文脈は参照バイト 256 通り + 「参照になし」
        std::vector<uint64_t> cnt(256, 0);
        std::vector<std::vector<uint64_t>> joint(257, std::vector<uint64_t>(256, 0));
        std::vector<uint64_t> ctx_tot(257, 0);
        size_t hit = 0;
        for (auto& kv : A) {
            auto it = B.find(kv.first);
            int ctx = (it == B.end()) ? 256 : it->second;
            if (it != B.end()) ++hit;
            ++cnt[kv.second]; ++joint[ctx][kv.second]; ++ctx_tot[ctx];
        }
        size_t N = A.size();
        auto H0 = [&]{ double h = 0; for (int i = 0; i < 256; ++i) if (cnt[i])
                       { double p = (double)cnt[i]/N; h -= p*std::log2(p); } return h; }();
        double H1 = 0;
        for (int c = 0; c < 257; ++c) {
            if (!ctx_tot[c]) continue;
            double h = 0;
            for (int i = 0; i < 256; ++i) if (joint[c][i])
                { double p = (double)joint[c][i]/ctx_tot[c]; h -= p*std::log2(p); }
            H1 += h * ctx_tot[c];
        }
        H1 /= (double)N;

        OctDiff R;
        R.depth = L; R.leaf = side / (double)(1ull << L);
        R.nodes = N; R.H = H0; R.H_given = H1;
        R.ref_hit = N ? (double)hit / N : 0;
        // 累積の幾何ビット/点は深さごとに独立でないので、この段の寄与のみを示す
        R.bits_alone = H0 * N / (double)nt;
        R.bits_given = H1 * N / (double)nt;
        res.push_back(R);
    }
    return res;
}

static double zbits(const std::vector<std::vector<int64_t>>& st, size_t n) {
    uint64_t tot = 0;
    std::vector<uint8_t> plane(n * 8), comp(ZSTD_compressBound(n * 8));
    for (const auto& v : st) {
        int64_t prev = 0; std::vector<uint64_t> z(n);
        for (size_t i = 0; i < n; ++i) { int64_t d = v[i] - prev; prev = v[i];
                                         z[i] = ((uint64_t)d << 1) ^ (uint64_t)(d >> 63); }
        for (size_t b = 0; b < 8; ++b) for (size_t i = 0; i < n; ++i)
            plane[b*n+i] = (uint8_t)(z[i] >> (b*8));
        tot += ZSTD_compress(comp.data(), comp.size(), plane.data(), n*8, 12);
    }
    return tot * 8.0 / n;
}

ResidualDiff nearest_residual(const std::vector<double>& tgt, size_t nt,
                              const std::vector<double>& ref, size_t nr, double v) {
    ResidualDiff R; R.voxel = v;
    // 参照の点数は ref の長さから決まる（KdTree が ref をそのまま使う）。
    // 呼ぶ側が渡す nr と食い違っていたら、参照が途中で切れている。
    if (nr * 3 != ref.size() || nt * 3 > tgt.size()) return R;
    std::vector<std::vector<int64_t>> alone(3, std::vector<int64_t>(nt));
    for (size_t i = 0; i < nt; ++i) for (int d = 0; d < 3; ++d)
        alone[d][i] = (int64_t)std::llround(tgt[i*3+d] / v);
    R.bits_alone = zbits(alone, nt);

    KdTree tree(ref);
    std::vector<std::vector<int64_t>> resid(3, std::vector<int64_t>(nt));
    std::vector<double> dist(nt);
    int64_t idx[1]; double d2[1];
    for (size_t i = 0; i < nt; ++i) {
        tree.knn(&tgt[i*3], 1, idx, d2);
        dist[i] = std::sqrt(d2[0]);
        for (int d = 0; d < 3; ++d)
            resid[d][i] = (int64_t)std::llround((tgt[i*3+d] - ref[idx[0]*3+d]) / v);
    }
    R.bits_resid = zbits(resid, nt);
    std::vector<double> ds = dist;
    std::nth_element(ds.begin(), ds.begin()+nt/2, ds.end()); R.median_dist = ds[nt/2];
    std::nth_element(ds.begin(), ds.begin()+(size_t)(nt*0.95), ds.end());
    R.p95_dist = ds[(size_t)(nt*0.95)];
    return R;
}

} // namespace pcc
