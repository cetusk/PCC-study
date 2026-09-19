#include "pcc/distort.hpp"
#include "pcc/geom.hpp"
#include <thread>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace pcc {

// 点群 Q に対する P の各点の最近傍距離
static void nn_dists(const std::vector<double>& P, size_t np, const KdTree& tq,
                     std::vector<double>& d) {
    d.resize(np);
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned w = 0; w < nt; ++w) {
        th.emplace_back([&, w] {
            int64_t idx[1]; double d2[1];
            for (size_t i = w; i < np; i += nt) {
                tq.knn(&P[i * 3], 1, idx, d2);
                d[i] = std::sqrt(d2[0]);
            }
        });
    }
    for (auto& x : th) x.join();
}

// B の点を A の局所平面に射影した距離。面の再現を測るときはこちらが本質的で、
// 接平面方向に点がずれても罰しない。
static void plane_dists(const std::vector<double>& A, size_t na, const KdTree& ta,
                        const std::vector<double>& B, size_t nb, int k,
                        std::vector<double>& d) {
    d.resize(nb);
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned w = 0; w < nt; ++w) {
        th.emplace_back([&, w] {
            std::vector<int64_t> idx(k); std::vector<double> d2(k);
            for (size_t i = w; i < nb; i += nt) {
                ta.knn(&B[i * 3], k, idx.data(), d2.data());
                double m[3] = {0, 0, 0};
                int c = 0;
                for (int j = 0; j < k; ++j) { if (idx[j] < 0) continue;
                    for (int t = 0; t < 3; ++t) m[t] += A[(size_t)idx[j]*3+t]; ++c; }
                if (c < 3) { d[i] = std::sqrt(d2[0]); continue; }
                for (int t = 0; t < 3; ++t) m[t] /= c;
                double C[9] = {0,0,0,0,0,0,0,0,0};
                for (int j = 0; j < k; ++j) { if (idx[j] < 0) continue;
                    double u[3];
                    for (int t = 0; t < 3; ++t) u[t] = A[(size_t)idx[j]*3+t] - m[t];
                    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C[a*3+b] += u[a]*u[b];
                }
                double ev[3], V[9];
                eigh3(C, ev, V);                       // 昇順
                double nx = V[0], ny = V[3], nz = V[6];   // 最小固有値に対応する列＝法線
                double dx = B[i*3] - m[0], dy = B[i*3+1] - m[1], dz = B[i*3+2] - m[2];
                d[i] = std::fabs(dx*nx + dy*ny + dz*nz);
            }
        });
    }
    for (auto& x : th) x.join();
}

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    size_t k = (size_t)(v.size() * q);
    if (k >= v.size()) k = v.size() - 1;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

Distortion distortion(const std::vector<double>& A, size_t na,
                      const std::vector<double>& B, size_t nb, int plane_k) {
    Distortion R;
    KdTree ta(A), tb(B);
    std::vector<double> ab, ba, pd;
    nn_dists(A, na, tb, ab);        // A の各点 → B
    nn_dists(B, nb, ta, ba);        // B の各点 → A
    R.a2b_mean = std::accumulate(ab.begin(), ab.end(), 0.0) / na;
    R.b2a_mean = std::accumulate(ba.begin(), ba.end(), 0.0) / nb;
    R.a2b_max = *std::max_element(ab.begin(), ab.end());
    R.b2a_max = *std::max_element(ba.begin(), ba.end());
    R.chamfer = 0.5 * (R.a2b_mean + R.b2a_mean);
    R.hausdorff = std::max(R.a2b_max, R.b2a_max);
    std::vector<double> both = ab; both.insert(both.end(), ba.begin(), ba.end());
    R.p95 = pct(both, 0.95);
    plane_dists(A, na, ta, B, nb, plane_k, pd);
    R.plane_mean = std::accumulate(pd.begin(), pd.end(), 0.0) / nb;
    R.plane_p95 = pct(pd, 0.95);
    R.plane_max = *std::max_element(pd.begin(), pd.end());
    return R;
}

} // namespace pcc
