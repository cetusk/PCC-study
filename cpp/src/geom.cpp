#include "pcc/geom.hpp"
#include "pcc/dettrig.hpp"
#include "pcc/nanoflann.hpp"
#include <zstd.h>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <cstdio>

namespace pcc {

struct Cloud {
    const std::vector<double>* p;
    inline size_t kdtree_get_point_count() const { return p->size() / 3; }
    inline double kdtree_get_pt(const size_t i, const size_t d) const { return (*p)[i * 3 + d]; }
    template <class B> bool kdtree_get_bbox(B&) const { return false; }
};
using Tree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud, 3, int64_t>;

struct KdTree::Impl { Cloud c; Tree* t = nullptr; };

// 木の構築は 1 コアしか使わない。復号では近傍表が時間の 34% を占め、その中でも
// 構築が探索の 2 倍あった（30 万点で 木 0.030s / 探索 0.015s）。
// nanoflann の並列構築は分割の判断が点の範囲だけで決まるので、**同じ木**を返す。
static unsigned build_threads() {
    static const unsigned nt = [] {
        if (const char* e = getenv("PCC_THREADS")) { long v = atol(e); if (v > 0) return (unsigned)v; }
        unsigned hw = std::thread::hardware_concurrency();
        return hw ? hw : 1u;
    }();
    return nt;
}

KdTree::KdTree(const std::vector<double>& xyz) : p_(new Impl), n_(xyz.size() / 3) {
    p_->c.p = &xyz;
    p_->t = new Tree(3, p_->c,
                     nanoflann::KDTreeSingleIndexAdaptorParams(
                         16, nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                         build_threads()));
    p_->t->buildIndex();
}
KdTree::~KdTree() { delete p_->t; delete p_; }

void KdTree::knn(const double* q, int k, int64_t* idx, double* d2) const {
    nanoflann::KNNResultSet<double, int64_t> rs(k);
    rs.init(idx, d2);
    p_->t->findNeighbors(rs, q);
}
void KdTree::radius(const double* q, double r2, std::vector<int64_t>& out) const {
    std::vector<nanoflann::ResultItem<int64_t, double>> m;
    nanoflann::SearchParameters sp; sp.sorted = false;
    (void)p_->t->radiusSearch(q, r2, m, sp);   // 数は m.size() で分かる
    out.clear(); out.reserve(m.size());
    for (auto& e : m) out.push_back(e.first);
}

double point_spacing(const std::vector<double>& xyz, size_t sample, uint64_t seed) {
    size_t n = xyz.size() / 3;
    if (n < 2) return 0;
    KdTree t(xyz);
    std::mt19937_64 rng(seed);
    size_t m = std::min(sample, n);
    std::vector<double> d(m);
    for (size_t i = 0; i < m; ++i) {
        size_t j = (m == n) ? i : (rng() % n);
        // 見つかった近傍が 2 つ未満だと dd[1] が書かれない。0 で埋めておく
        // （自分自身しか無い＝重なった点しか無いのと同じ扱い）。
        int64_t idx[2] = {-1, -1}; double dd[2] = {0.0, 0.0};
        t.knn(&xyz[j * 3], 2, idx, dd);
        d[i] = std::sqrt(dd[1]);
    }
    std::nth_element(d.begin(), d.begin() + m / 2, d.end());
    return d[m / 2];
}

static inline uint64_t spread(uint64_t v) {
    v &= 0x1FFFFFull;
    v = (v | (v << 32)) & 0x1F00000000FFFFull;
    v = (v | (v << 16)) & 0x1F0000FF0000FFull;
    v = (v | (v << 8))  & 0x100F00F00F00F00Full;
    v = (v | (v << 4))  & 0x10C30C30C30C30C3ull;
    v = (v | (v << 2))  & 0x1249249249249249ull;
    return v;
}
std::vector<uint64_t> morton3(const std::vector<int64_t>& q, size_t n) {
    std::vector<uint64_t> m(n);
    for (size_t i = 0; i < n; ++i)
        m[i] = spread((uint64_t)q[i * 3]) | (spread((uint64_t)q[i * 3 + 1]) << 1)
             | (spread((uint64_t)q[i * 3 + 2]) << 2);
    return m;
}

void polar_forward(const std::vector<double>& xyz, size_t n, const Vec3& o,
                   double dr, double da, std::vector<int64_t>& qr,
                   std::vector<int64_t>& qa, std::vector<int64_t>& qe) {
    qr.resize(n); qa.resize(n); qe.resize(n);
    for (size_t i = 0; i < n; ++i) {
        double x = xyz[i * 3] - o[0], y = xyz[i * 3 + 1] - o[1], z = xyz[i * 3 + 2] - o[2];
        double r = std::sqrt(x * x + y * y + z * z);
        double th = std::atan2(y, x);
        double ph = std::asin(std::max(-1.0, std::min(1.0, r > 0 ? z / r : 0.0)));
        qr[i] = (int64_t)std::llround(r / dr);
        qa[i] = (int64_t)std::llround(th / da);
        qe[i] = (int64_t)std::llround(ph / da);
    }
}
void polar_inverse(const std::vector<int64_t>& qr, const std::vector<int64_t>& qa,
                   const std::vector<int64_t>& qe, const Vec3& o,
                   double dr, double da, std::vector<double>& xyz) {
    size_t n = qr.size();
    xyz.resize(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const double r = qr[i] * dr, th = qa[i] * da, ph = qe[i] * da;
        const double oo[3] = {o[0], o[1], o[2]};
        polar_point(r, th, ph, oo, false, &xyz[i * 3]);   // 復号側（frame_world）と同じ式
    }
}

// --- 候補の符号長（比較用なので候補間で方式を揃える）
static uint64_t rate_of(const std::vector<std::vector<int64_t>>& st, size_t n) {
    uint64_t tot = 0;
    std::vector<uint8_t> plane(n * 8);
    std::vector<uint8_t> comp(ZSTD_compressBound(n * 8));
    for (const auto& v : st) {
        int64_t prev = 0;
        std::vector<uint64_t> z(n);
        for (size_t i = 0; i < n; ++i) { int64_t d = v[i] - prev; prev = v[i];
                                         z[i] = ((uint64_t)d << 1) ^ (uint64_t)(d >> 63); }
        // バイト平面分離
        for (size_t b = 0; b < 8; ++b)
            for (size_t i = 0; i < n; ++i)
                plane[b * n + i] = (uint8_t)(z[i] >> (b * 8));
        size_t cs = ZSTD_compress(comp.data(), comp.size(), plane.data(), n * 8, 12);
        tot += cs;
    }
    return tot;
}

static double max_err(const std::vector<double>& a, const std::vector<double>& b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        double dx = a[i*3]-b[i*3], dy = a[i*3+1]-b[i*3+1], dz = a[i*3+2]-b[i*3+2];
        m = std::max(m, std::sqrt(dx*dx+dy*dy+dz*dz));
    }
    return m;
}

std::vector<GeomCandidate> geometry_candidates(const std::vector<double>& xyz, size_t n,
                                               double eps, bool allow_polar,
                                               double origin_max_extent) {
    std::vector<GeomCandidate> out;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    Vec3 c{{0, 0, 0}};
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) {
        double v = xyz[i * 3 + d];
        lo[d] = std::min(lo[d], v); hi[d] = std::max(hi[d], v); c[d] += v;
    }
    for (int d = 0; d < 3; ++d) c[d] /= (double)n;
    double extent = std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0]) + (hi[1]-lo[1])*(hi[1]-lo[1])
                            + (hi[2]-lo[2])*(hi[2]-lo[2]));

    // 誤差上限を使い切るまで刻みを広げる（対数二分探索）
    auto fit = [&](auto make) {
        double m = 1.0;
        auto best = make(m);
        if (best.max_err <= eps) {
            for (int i = 0; i < 8; ++i) {
                auto r2 = make(m * 2);
                if (r2.max_err > eps) break;
                m *= 2; best = r2;
            }
        }
        double l = m, h = m * 2;
        for (int i = 0; i < 14; ++i) {
            double mid = std::sqrt(l * h);
            auto r2 = make(mid);
            if (r2.max_err <= eps) { l = mid; best = r2; } else h = mid;
        }
        return best;
    };

    out.push_back(fit([&](double m) {
        GeomCandidate g; g.kind = "grid";
        g.step = m * 2 * eps / std::sqrt(3.0);
        g.streams.assign(3, std::vector<int64_t>(n));
        std::vector<double> rec(n * 3);
        for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) {
            int64_t q = (int64_t)std::llround(xyz[i*3+d] / g.step);
            g.streams[d][i] = q; rec[i*3+d] = q * g.step;
        }
        g.max_err = max_err(rec, xyz, n);
        return g;
    }));

    if (allow_polar) {
        std::vector<std::pair<std::string, Vec3>> anchors;
        double dorig = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        if (dorig <= origin_max_extent * extent) anchors.push_back({"origin", Vec3{{0,0,0}}});
        anchors.push_back({"centroid", c});
        for (auto& [nm, o] : anchors) {
            std::vector<double> r(n);
            for (size_t i = 0; i < n; ++i) {
                double x = xyz[i*3]-o[0], y = xyz[i*3+1]-o[1], z = xyz[i*3+2]-o[2];
                r[i] = std::sqrt(x*x+y*y+z*z);
            }
            std::vector<double> rs = r;
            std::nth_element(rs.begin(), rs.begin() + (size_t)(n * 0.99), rs.end());
            double r99 = rs[(size_t)(n * 0.99)];
            if (r99 <= 0) continue;
            out.push_back(fit([&, nm = nm, o = o, r99 = r99](double m) {
                GeomCandidate g; g.kind = "polar"; g.anchor = nm; g.origin = o;
                g.r_step = m * eps / std::sqrt(3.0);
                g.ang_step = m * eps / (std::sqrt(3.0) * r99);
                g.streams.assign(3, {});
                polar_forward(xyz, n, o, g.r_step, g.ang_step,
                              g.streams[0], g.streams[1], g.streams[2]);
                std::vector<double> rec;
                polar_inverse(g.streams[0], g.streams[1], g.streams[2], o,
                              g.r_step, g.ang_step, rec);
                g.max_err = max_err(rec, xyz, n);
                return g;
            }));
        }
    }
    return out;
}

GeomCandidate choose_geometry(const std::vector<double>& xyz, size_t n, double eps,
                              bool allow_polar, size_t sample, bool verbose, const std::string& kind) {
    auto cands = geometry_candidates(xyz, n, eps, allow_polar);
    size_t m = std::min(sample, n);
    GeomCandidate* best = nullptr;
    for (auto& c : cands) {
        if (c.max_err > eps * 1.02) continue;
        std::vector<std::vector<int64_t>> head(3);
        for (int d = 0; d < 3; ++d) head[d].assign(c.streams[d].begin(), c.streams[d].begin() + m);
        c.bytes = rate_of(head, m);
        if (!best || c.bytes < best->bytes) best = &c;
    }
    // 実験: PCC_GEOM_KIND=grid / polar/origin / polar/centroid で格子の種類を固定する（符号化側だけ。
    // 代理の符号長による選択がどれだけ損をしているかを、器で実際に符号化して測るため）。
    std::string want_s = kind;
    if (want_s.empty())
        if (const char* want = getenv("PCC_GEOM_KIND")) want_s = want;
    if (!want_s.empty()) {
        GeomCandidate* hit = nullptr;
        for (auto& c : cands) {
            const std::string nm = c.kind + (c.anchor.empty() ? "" : "/" + c.anchor);
            if (nm == want_s && c.max_err <= eps * 1.02) { hit = &c; break; }
        }
        if (!kind.empty() && !hit) return GeomCandidate{};   // 指定の種類は上限を守れない
        if (hit) best = hit;
    }
    if (verbose)
        for (auto& c : cands)
            printf("    候補 %-16s %7.3f bpp  誤差 %7.3fmm%s\n",
                   (c.kind + (c.anchor.empty() ? "" : "/" + c.anchor)).c_str(),
                   c.bytes * 8.0 / m, c.max_err * 1000, (&c == best) ? "  ← 採用" : "");
    return best ? *best : cands.front();
}

// Jacobi 法による 3x3 対称行列の固有分解（昇順）
void eigh3(const double Cin[9], double ev[3], double V[9]) {
    double A[9]; memcpy(A, Cin, sizeof(A));
    for (int i = 0; i < 9; ++i) V[i] = (i % 4 == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 12; ++sweep) {
        double off = A[1]*A[1] + A[2]*A[2] + A[5]*A[5];
        if (off < 1e-30) break;
        for (int p = 0; p < 2; ++p) for (int q = p + 1; q < 3; ++q) {
            double apq = A[p*3+q];
            if (std::fabs(apq) < 1e-300) continue;
            double theta = (A[q*3+q] - A[p*3+p]) / (2 * apq);
            double t = (theta >= 0 ? 1.0 : -1.0) /
                       (std::fabs(theta) + std::sqrt(theta * theta + 1));
            double c = 1 / std::sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < 3; ++k) {
                double akp = A[k*3+p], akq = A[k*3+q];
                A[k*3+p] = c * akp - s * akq; A[k*3+q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k) {
                double apk = A[p*3+k], aqk = A[q*3+k];
                A[p*3+k] = c * apk - s * aqk; A[q*3+k] = s * apk + c * aqk;
                double vkp = V[k*3+p], vkq = V[k*3+q];
                V[k*3+p] = c * vkp - s * vkq; V[k*3+q] = s * vkp + c * vkq;
            }
        }
    }
    int idx[3] = {0, 1, 2};
    double d[3] = {A[0], A[4], A[8]};
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j)
        if (d[idx[j]] < d[idx[i]]) std::swap(idx[i], idx[j]);
    double Vt[9], et[3];
    for (int i = 0; i < 3; ++i) { et[i] = d[idx[i]];
        for (int k = 0; k < 3; ++k) Vt[k*3+i] = V[k*3+idx[i]]; }
    memcpy(ev, et, sizeof(et)); memcpy(V, Vt, sizeof(Vt));
}

} // namespace pcc
