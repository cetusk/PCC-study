#include "pcc/surfcode.hpp"
#include "pcc/geom.hpp"
#include "pcc/rangecoder.hpp"
#include "pcc/attr.hpp"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace pcc {

static inline uint64_t vkey(int64_t x, int64_t y, int64_t z) {
    return ((uint64_t)(x & 0x1fffff) << 42) | ((uint64_t)(y & 0x1fffff) << 21)
         | (uint64_t)(z & 0x1fffff);
}

static void bbox(const std::vector<double>& p, size_t n, double lo[3], double& side) {
    double hi[3];
    for (int d = 0; d < 3; ++d) { lo[d] = 1e300; hi[d] = -1e300; }
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], p[i*3+d]); hi[d] = std::max(hi[d], p[i*3+d]); }
    side = 0;
    for (int d = 0; d < 3; ++d) side = std::max(side, hi[d] - lo[d]);
    side *= 1.0000001;
}

uint64_t octree_bytes(const std::vector<double>& xyz, size_t n, double leaf) {
    double lo[3], side; bbox(xyz, n, lo, side);
    int L = 1;
    while (L < 21 && side / (double)(1ull << L) > leaf) ++L;
    uint64_t total = 0;
    for (int l = 1; l <= L; ++l) {
        uint64_t M = 1ull << l; double s = side / (double)M;
        std::unordered_map<uint64_t, uint8_t> cur, par;
        cur.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) {
            uint64_t c[3];
            for (int d = 0; d < 3; ++d) {
                int64_t q = (int64_t)((xyz[i*3+d] - lo[d]) / s);
                c[d] = (uint64_t)std::max<int64_t>(0, std::min<int64_t>((int64_t)M-1, q));
            }
            cur[vkey(c[0]>>1, c[1]>>1, c[2]>>1)] |=
                (uint8_t)(1u << (((c[0]&1)<<2) | ((c[1]&1)<<1) | (c[2]&1)));
        }
        if (l > 1) {
            uint64_t M2 = 1ull << (l - 1); double s2 = side / (double)M2;
            par.reserve(n * 2);
            for (size_t i = 0; i < n; ++i) {
                uint64_t c[3];
                for (int d = 0; d < 3; ++d) {
                    int64_t q = (int64_t)((xyz[i*3+d] - lo[d]) / s2);
                    c[d] = (uint64_t)std::max<int64_t>(0, std::min<int64_t>((int64_t)M2-1, q));
                }
                par[vkey(c[0]>>1, c[1]>>1, c[2]>>1)] |=
                    (uint8_t)(1u << (((c[0]&1)<<2) | ((c[1]&1)<<1) | (c[2]&1)));
            }
        }
        std::vector<int64_t> sym; std::vector<int32_t> ctx;
        for (auto& kv : cur) {
            sym.push_back(kv.second);
            if (l == 1) { ctx.push_back(0); continue; }
            uint64_t gk = vkey((int64_t)(kv.first >> 42) >> 1,
                               (int64_t)((kv.first >> 21) & 0x1fffff) >> 1,
                               (int64_t)(kv.first & 0x1fffff) >> 1);
            auto it = par.find(gk);
            ctx.push_back(it == par.end() ? 0 : it->second);
        }
        total += encode_ints(sym.data(), sym.size(), ctx.data(), 256).size();
    }
    return total;
}

static void oct_encode(const double nv[3], int bits, int32_t& a, int32_t& b) {
    double s = std::fabs(nv[0]) + std::fabs(nv[1]) + std::fabs(nv[2]);
    if (s < 1e-12) { a = b = 0; return; }
    double px = nv[0]/s, py = nv[1]/s;
    if (nv[2] < 0) {
        double tx = (1-std::fabs(py))*(px>=0?1:-1), ty = (1-std::fabs(px))*(py>=0?1:-1);
        px = tx; py = ty;
    }
    int m = (1 << bits) - 1;
    a = (int32_t)std::lround((px*0.5+0.5)*m);
    b = (int32_t)std::lround((py*0.5+0.5)*m);
}
static void oct_decode(int32_t a, int32_t b, int bits, double nv[3]) {
    int m = (1 << bits) - 1;
    double px = (double)a/m*2-1, py = (double)b/m*2-1;
    double z = 1 - std::fabs(px) - std::fabs(py);
    if (z < 0) {
        double tx = (1-std::fabs(py))*(px>=0?1:-1), ty = (1-std::fabs(px))*(py>=0?1:-1);
        px = tx; py = ty;
    }
    double l = std::sqrt(px*px+py*py+z*z);
    nv[0]=px/l; nv[1]=py/l; nv[2]=z/l;
}

// 決まった疑似乱数。節点の鍵と通し番号だけから決まるので復号側も同じ値を出せる。
static inline double hrand(uint64_t k, uint32_t i) {
    uint64_t x = k * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (double)(x >> 11) / (double)(1ull << 53);
}

struct Leaf {
    std::vector<int64_t> ids;   // この葉に属する点
    uint64_t seed;        // 決まった疑似乱数の種（節点の位置から決まる）
    double ctr[3];        // 節点の中心
    double nv[3];         // 量子化して復号した法線
    int64_t qs;           // 節点中心から平面までの符号付き距離 / eps
    int32_t qa, qb;       // 法線の量子化値
    double size, sigma;
    int cnt;
    void plane_point(double eps, double p[3]) const {
        for (int d = 0; d < 3; ++d) p[d] = ctr[d] + nv[d] * (double)qs * eps;
    }
};

static void fit(const std::vector<double>& xyz, const std::vector<int64_t>& ids,
                double m[3], double nv[3], double& rms) {
    for (int d = 0; d < 3; ++d) m[d] = 0;
    for (int64_t i : ids) for (int d = 0; d < 3; ++d) m[d] += xyz[(size_t)i*3+d];
    for (int d = 0; d < 3; ++d) m[d] /= (double)ids.size();
    double C[9] = {0,0,0,0,0,0,0,0,0};
    for (int64_t i : ids) {
        double u[3];
        for (int d = 0; d < 3; ++d) u[d] = xyz[(size_t)i*3+d] - m[d];
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C[a*3+b] += u[a]*u[b];
    }
    double ev[3], V[9];
    eigh3(C, ev, V);
    nv[0]=V[0]; nv[1]=V[3]; nv[2]=V[6];
    if (nv[2] < 0) { nv[0]=-nv[0]; nv[1]=-nv[1]; nv[2]=-nv[2]; }
    rms = std::sqrt(std::max(0.0, ev[0]) / (double)ids.size());
}

SurfResult surface_code(const std::vector<double>& xyz, size_t n, const SurfOpt& o,
                        std::vector<double>& out) {
    SurfResult R;
    double lo[3], side; bbox(xyz, n, lo, side);

    std::vector<Leaf> leaves;
    std::vector<int64_t> split_sym;     // 各占有節点で「分割したか」
    std::vector<int32_t> split_ctx;
    std::vector<int64_t> occ_sym;       // 分割した節点の子の占有バイト
    std::vector<int32_t> occ_ctx;

    // 再帰。all は節点に属する点の添字
    struct Frame { int level; uint64_t key; double org[3], sz; std::vector<int64_t> ids; };
    std::vector<Frame> stack;
    {
        Frame f; f.level = 0; f.key = 1; f.sz = side;
        for (int d = 0; d < 3; ++d) f.org[d] = lo[d];
        f.ids.resize(n);
        for (size_t i = 0; i < n; ++i) f.ids[i] = (int64_t)i;
        stack.push_back(std::move(f));
    }
    // 固定体素のときは根から voxel まで必ず分割する
    int force_level = 0;
    if (!o.adaptive) {
        double s = side;
        while (force_level < 21 && s > o.voxel) { s *= 0.5; ++force_level; }
    }

    while (!stack.empty()) {
        Frame f = std::move(stack.back()); stack.pop_back();
        bool do_split;
        if (!o.adaptive) do_split = (f.level < force_level);
        else {
            if ((int)f.ids.size() < o.min_pts || f.level >= o.max_level) do_split = false;
            else {
                double m[3], nv[3], rms; fit(xyz, f.ids, m, nv, rms);
                do_split = (rms > o.tau);
            }
            split_sym.push_back(do_split ? 1 : 0);
            split_ctx.push_back(std::min(f.level, 15));
        }
        if (do_split) {
            double h = f.sz * 0.5;
            std::vector<int64_t> ch[8];
            for (int64_t i : f.ids) {
                int b = 0;
                for (int d = 0; d < 3; ++d)
                    if (xyz[(size_t)i*3+d] >= f.org[d] + h) b |= (1 << (2 - d));
                ch[b].push_back(i);
            }
            uint8_t occ = 0;
            for (int b = 0; b < 8; ++b) if (!ch[b].empty()) occ |= (uint8_t)(1u << b);
            occ_sym.push_back(occ); occ_ctx.push_back(std::min(f.level, 15));
            for (int b = 7; b >= 0; --b) {
                if (ch[b].empty()) continue;
                Frame g; g.level = f.level + 1; g.key = (f.key << 3) | (uint64_t)b; g.sz = h;
                for (int d = 0; d < 3; ++d)
                    g.org[d] = f.org[d] + ((b >> (2 - d)) & 1 ? h : 0.0);
                g.ids = std::move(ch[b]);
                stack.push_back(std::move(g));
            }
        } else {
            double m[3], nv[3], rms; fit(xyz, f.ids, m, nv, rms);
            Leaf L;
            L.size = f.sz; L.cnt = (int)f.ids.size(); L.sigma = rms;
            for (int d = 0; d < 3; ++d) L.ctr[d] = f.org[d] + f.sz * 0.5;
            oct_encode(nv, o.nbits, L.qa, L.qb);
            oct_decode(L.qa, L.qb, o.nbits, L.nv);
            double sd = 0;
            for (int d = 0; d < 3; ++d) sd += (m[d] - L.ctr[d]) * L.nv[d];
            L.qs = (int64_t)std::llround(sd / o.eps);
            L.seed = f.key;
            L.ids = std::move(f.ids);
            leaves.push_back(std::move(L));
        }
    }
    R.n_leaf = leaves.size();

    // 葉の値をまとめて符号化する。
    // predict のときは、葉を幾何由来の順（Morton）に並べ、その順で先行する
    // 最近傍の葉から法線と面の位置を予測して残差だけを送る。
    // 復号側は節点の位置を先に復元しているので同じ予測ができる。
    std::vector<int64_t> qa, qb, qoff, qcnt, qsig;
    qa.reserve(R.n_leaf); qb.reserve(R.n_leaf); qoff.reserve(R.n_leaf);
    qcnt.reserve(R.n_leaf); qsig.reserve(R.n_leaf);
    if (!o.predict) {
        for (const auto& L : leaves) {
            qa.push_back(L.qa); qb.push_back(L.qb);
            qoff.push_back(L.qs); qcnt.push_back(L.cnt);
            if (o.jitter) qsig.push_back((int64_t)std::llround(L.sigma / o.eps));
        }
    } else {
        size_t m = leaves.size();
        std::vector<double> ctr(m * 3);
        for (size_t t = 0; t < m; ++t)
            for (int d = 0; d < 3; ++d) ctr[t*3+d] = leaves[t].ctr[d];
        auto perm = coding_order(ctr, m, "morton");
        std::vector<int32_t> pred;
        build_causal_predictors(ctr, m, perm, 1, 16, pred);
        std::vector<Leaf> ord(m);
        for (size_t t = 0; t < m; ++t) ord[t] = std::move(leaves[perm[t]]);
        for (size_t t = 0; t < m; ++t) {
            const Leaf& L = ord[t];
            int32_t pa = 0, pb = 0; int64_t ps = 0;
            if (t > 0) {
                int32_t j = pred[t];
                if (j < 0 || (size_t)j >= t) j = (int32_t)(t - 1);
                const Leaf& P = ord[j];
                pa = P.qa; pb = P.qb;
                // 先行する葉の平面を、この葉の中心で評価した符号付き距離
                double pp[3]; P.plane_point(o.eps, pp);
                double d = 0;
                for (int k = 0; k < 3; ++k) d += (L.ctr[k] - pp[k]) * P.nv[k];
                ps = (int64_t)std::llround(-d / o.eps);
            }
            qa.push_back((int64_t)L.qa - pa);
            qb.push_back((int64_t)L.qb - pb);
            qoff.push_back(L.qs - ps);
            qcnt.push_back(L.cnt);
            if (o.jitter) qsig.push_back((int64_t)std::llround(L.sigma / o.eps));
        }
        leaves.swap(ord);
    }
    R.b_normal = encode_ints(qa.data(), qa.size()).size()
               + encode_ints(qb.data(), qb.size()).size();
    R.b_offset = encode_ints(qoff.data(), qoff.size()).size();
    R.b_count  = encode_ints(qcnt.data(), qcnt.size()).size();
    if (o.jitter) R.b_sigma = encode_ints(qsig.data(), qsig.size()).size();
    if (!occ_sym.empty())
        R.b_occ = encode_ints(occ_sym.data(), occ_sym.size(), occ_ctx.data(), 16).size();
    if (!split_sym.empty())
        R.b_split = encode_ints(split_sym.data(), split_sym.size(), split_ctx.data(), 16).size();

    // 復号: 平面上に点を配置する
    out.clear(); out.reserve(n * 3);
    double gsum = 0;
    for (auto& L : leaves) {
        double pc[3]; L.plane_point(o.eps, pc);
        double e1[3], e2[3], ax[3] = {1,0,0};
        if (std::fabs(L.nv[0]) > 0.9) { ax[0]=0; ax[1]=1; }
        e1[0]=ax[1]*L.nv[2]-ax[2]*L.nv[1];
        e1[1]=ax[2]*L.nv[0]-ax[0]*L.nv[2];
        e1[2]=ax[0]*L.nv[1]-ax[1]*L.nv[0];
        double l = std::sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]);
        for (int d = 0; d < 3; ++d) e1[d] /= l;
        e2[0]=L.nv[1]*e1[2]-L.nv[2]*e1[1];
        e2[1]=L.nv[2]*e1[0]-L.nv[0]*e1[2];
        e2[2]=L.nv[0]*e1[1]-L.nv[1]*e1[0];
        int k = L.cnt, g = (int)std::ceil(std::sqrt((double)k));
        gsum += k;
        int placed = 0;
        for (int u = 0; u < g && placed < k; ++u)
        for (int v = 0; v < g && placed < k; ++v) {
            double ju = 0.5, jv = 0.5;
            if (o.strat) { ju = hrand(L.seed, (uint32_t)(placed*2));
                           jv = hrand(L.seed, (uint32_t)(placed*2+1)); }
            double fu = ((u + ju) / g - 0.5) * L.size;
            double fv = ((v + jv) / g - 0.5) * L.size;
            double jn = 0;
            if (o.jitter) {
                // 箱=ミュラーで正規乱数を 1 つ作る
                double r1 = std::max(1e-12, hrand(L.seed, (uint32_t)(1000+placed)));
                double r2 = hrand(L.seed, (uint32_t)(2000+placed));
                jn = std::sqrt(-2*std::log(r1)) * std::cos(6.283185307*r2) * L.sigma;
            }
            for (int d = 0; d < 3; ++d)
                out.push_back(pc[d] + e1[d]*fu + e2[d]*fv + L.nv[d]*jn);
            ++placed;
        }
    }
    R.n_out = out.size() / 3;
    R.mean_pts_per_leaf = R.n_leaf ? gsum / (double)R.n_leaf : 0;
    R.leaf_ids.resize(leaves.size());
    for (size_t t = 0; t < leaves.size(); ++t) R.leaf_ids[t] = std::move(leaves[t].ids);
    return R;
}

} // namespace pcc
