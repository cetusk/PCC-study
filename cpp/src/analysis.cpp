#include "pcc/analysis.hpp"
#include "pcc/normalize.hpp"
#include <zstd.h>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdio>

namespace pcc {

static double ent(const std::vector<int64_t>& v) { return entropy0(v); }

static double cond_ent(const std::vector<int64_t>& a, const std::vector<int64_t>& c) {
    std::unordered_map<int64_t, uint64_t> jc, cc;
    for (size_t i = 0; i < a.size(); ++i) { ++jc[(c[i] << 8) | (a[i] & 0xFF)]; ++cc[c[i]]; }
    double n = (double)a.size(), hj = 0, hc = 0;
    for (auto& kv : jc) { double p = kv.second / n; hj -= p * std::log2(p); }
    for (auto& kv : cc) { double p = kv.second / n; hc -= p * std::log2(p); }
    return hj - hc;
}

OctantResult octant_experiment(const std::vector<double>& xyz, size_t n, double v,
                               int radius, size_t sample, uint64_t seed) {
    OctantResult R;
    std::vector<int64_t> par(n * 3), oct(n);
    std::unordered_set<uint64_t> occ;
    occ.reserve(n * 2);
    auto key = [](int64_t x, int64_t y, int64_t z) {
        return (uint64_t)(x * 73856093) ^ (uint64_t)(y * 19349663) ^ (uint64_t)(z * 83492791);
    };
    for (size_t i = 0; i < n; ++i) {
        int64_t px = (int64_t)std::floor(xyz[i*3] / (2 * v));
        int64_t py = (int64_t)std::floor(xyz[i*3+1] / (2 * v));
        int64_t pz = (int64_t)std::floor(xyz[i*3+2] / (2 * v));
        par[i*3] = px; par[i*3+1] = py; par[i*3+2] = pz;
        int64_t cx = (int64_t)std::floor(xyz[i*3] / v) & 1;
        int64_t cy = (int64_t)std::floor(xyz[i*3+1] / v) & 1;
        int64_t cz = (int64_t)std::floor(xyz[i*3+2] / v) & 1;
        oct[i] = cx | (cy << 1) | (cz << 2);
        occ.insert(key(px, py, pz));
    }
    std::mt19937_64 rng(seed);
    size_t m = std::min(sample, n);
    std::vector<int64_t> A, P;
    A.reserve(m); P.reserve(m);
    size_t used_sum = 0, covered = 0, hit_acc = 0;
    std::vector<std::array<double,3>> C;
    for (size_t t = 0; t < m; ++t) {
        size_t i = (m == n) ? t : (rng() % n);
        int64_t px = par[i*3], py = par[i*3+1], pz = par[i*3+2];
        C.clear();
        for (int dx = -radius; dx <= radius; ++dx)
        for (int dy = -radius; dy <= radius; ++dy)
        for (int dz = -radius; dz <= radius; ++dz) {
            if (!dx && !dy && !dz) continue;
            if (occ.count(key(px+dx, py+dy, pz+dz)))
                C.push_back({(px+dx+0.5)*2*v, (py+dy+0.5)*2*v, (pz+dz+0.5)*2*v});
        }
        if (C.size() < 6) continue;
        ++covered;
        double c[3] = {0,0,0};
        for (auto& q : C) for (int d = 0; d < 3; ++d) c[d] += q[d];
        for (int d = 0; d < 3; ++d) c[d] /= (double)C.size();
        double M[9] = {0,0,0,0,0,0,0,0,0};
        for (auto& q : C) { double u[3] = {q[0]-c[0], q[1]-c[1], q[2]-c[2]};
            for (int a2 = 0; a2 < 3; ++a2) for (int b2 = 0; b2 < 3; ++b2) M[a2*3+b2] += u[a2]*u[b2]; }
        for (int k2 = 0; k2 < 9; ++k2) M[k2] /= (double)C.size();
        double ev[3], V[9]; eigh3(M, ev, V);
        double nv[3] = {V[0], V[3], V[6]};      // 最小固有値の固有ベクトル
        double base[3] = {px*2*v, py*2*v, pz*2*v};
        int bestj = 0; double bestd = 1e300;
        for (int j = 0; j < 8; ++j) {
            double cc[3] = {base[0] + ((j&1)+0.5)*v, base[1] + (((j>>1)&1)+0.5)*v,
                            base[2] + (((j>>2)&1)+0.5)*v};
            double d = std::fabs((cc[0]-c[0])*nv[0] + (cc[1]-c[1])*nv[1] + (cc[2]-c[2])*nv[2]);
            if (d < bestd) { bestd = d; bestj = j; }
        }
        A.push_back(oct[i]); P.push_back(bestj);
        used_sum += C.size();
        if (oct[i] == bestj) ++hit_acc;
    }
    R.n = A.size();
    R.coverage = m ? (double)covered / m : 0;
    if (!A.empty()) {
        R.H = ent(A); R.H_cond = cond_ent(A, P);
        R.acc = (double)hit_acc / A.size();
        R.used = (double)used_sum / A.size();
    }
    return R;
}

double cv_first_neighbor(const std::vector<double>& xyz, size_t n, size_t sample, uint64_t seed) {
    KdTree t(xyz);
    std::mt19937_64 rng(seed);
    size_t m = std::min(sample, n);
    double s = 0, s2 = 0;
    for (size_t i = 0; i < m; ++i) {
        size_t j = (m == n) ? i : (rng() % n);
        int64_t idx[2]; double d2[2];
        t.knn(&xyz[j*3], 2, idx, d2);
        double d = std::sqrt(d2[1]); s += d; s2 += d * d;
    }
    double mu = s / m, var = s2 / m - mu * mu;
    return mu > 0 ? std::sqrt(std::max(0.0, var)) / mu : 0;
}

double hexatic(const std::vector<double>& xyz, size_t n, int mm, int k,
               size_t sample, uint64_t seed) {
    KdTree t(xyz);
    std::mt19937_64 rng(seed);
    size_t m = std::min(sample, n);
    std::vector<int64_t> idx(k + 1); std::vector<double> d2(k + 1);
    double acc = 0; size_t cnt = 0;
    for (size_t s = 0; s < m; ++s) {
        size_t i = (m == n) ? s : (rng() % n);
        t.knn(&xyz[i*3], k + 1, idx.data(), d2.data());
        double c[3] = {0,0,0};
        for (int j = 1; j <= k; ++j) for (int d = 0; d < 3; ++d) c[d] += xyz[idx[j]*3+d];
        for (int d = 0; d < 3; ++d) c[d] /= (double)k;
        double M[9] = {0,0,0,0,0,0,0,0,0};
        for (int j = 1; j <= k; ++j) {
            double u[3] = {xyz[idx[j]*3]-c[0], xyz[idx[j]*3+1]-c[1], xyz[idx[j]*3+2]-c[2]};
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) M[a*3+b] += u[a]*u[b];
        }
        for (int q = 0; q < 9; ++q) M[q] /= (double)k;
        double ev[3], V[9]; eigh3(M, ev, V);
        double e1[3] = {V[2], V[5], V[8]}, e2[3] = {V[1], V[4], V[7]};  // 大きい2つ
        double re = 0, im = 0;
        for (int j = 1; j <= k; ++j) {
            double u[3] = {xyz[idx[j]*3]-xyz[i*3], xyz[idx[j]*3+1]-xyz[i*3+1],
                           xyz[idx[j]*3+2]-xyz[i*3+2]};
            double x = u[0]*e1[0]+u[1]*e1[1]+u[2]*e1[2];
            double y = u[0]*e2[0]+u[1]*e2[1]+u[2]*e2[2];
            double th = std::atan2(y, x) * mm;
            re += std::cos(th); im += std::sin(th);
        }
        acc += std::sqrt(re*re + im*im) / k; ++cnt;
    }
    return cnt ? acc / cnt : 0;
}

static uint64_t zstd_bits(const std::vector<std::vector<int64_t>>& st, size_t n) {
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
    return tot;
}

double morton_delta(const std::vector<double>& xyz, size_t n, double v) {
    double lo[3] = {1e300,1e300,1e300};
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) lo[d] = std::min(lo[d], xyz[i*3+d]);
    std::vector<int64_t> q(n * 3);
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d)
        q[i*3+d] = (int64_t)std::llround((xyz[i*3+d] - lo[d]) / v);
    std::vector<std::vector<int64_t>> cols(3, std::vector<int64_t>(n));
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) cols[d][i] = q[i*3+d];
    double a = zstd_bits(cols, n) * 8.0 / n;
    auto mo = morton3(q, n);
    std::vector<size_t> ord(n);
    for (size_t i = 0; i < n; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t x, size_t y) { return mo[x] < mo[y]; });
    std::vector<std::vector<int64_t>> mc(3, std::vector<int64_t>(n));
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) mc[d][i] = q[ord[i]*3+d];
    double b = zstd_bits(mc, n) * 8.0 / n;
    return b - a;
}

static void rand_rot(std::mt19937_64& rng, double R[9]) {
    std::normal_distribution<double> g(0, 1);
    double a[9]; for (int i = 0; i < 9; ++i) a[i] = g(rng);
    // Gram-Schmidt
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) {
            double d = 0; for (int k = 0; k < 3; ++k) d += a[i*3+k]*a[j*3+k];
            for (int k = 0; k < 3; ++k) a[i*3+k] -= d * a[j*3+k];
        }
        double nn = 0; for (int k = 0; k < 3; ++k) nn += a[i*3+k]*a[i*3+k];
        nn = std::sqrt(nn);
        for (int k = 0; k < 3; ++k) a[i*3+k] /= nn;
    }
    memcpy(R, a, sizeof(a));
}

AcqScore acquisition_score(const std::vector<double>& W, size_t n_all,
                           size_t sample, uint64_t seed) {
    auto t0 = std::chrono::steady_clock::now();
    AcqScore S;
    size_t n = std::min(sample, n_all);       // 並び順を見るので先頭を使う
    std::vector<double> raw(W.begin(), W.begin() + n * 3);
    double c[3] = {0,0,0};
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) c[d] += raw[i*3+d];
    for (int d = 0; d < 3; ++d) c[d] /= (double)n;
    std::vector<double> w(n * 3);
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) w[i*3+d] = raw[i*3+d] - c[d];

    double sp = point_spacing(w, std::min<size_t>(50000, n), seed);
    S.sig.spacing = sp;
    S.sig.cv_1nn = cv_first_neighbor(w, n, std::min<size_t>(50000, n), seed);
    S.sig.psi6 = hexatic(w, n, 6, 6, std::min<size_t>(20000, n), seed);
    S.sig.psi6_ref = 1.0 / std::sqrt(6.0);

    // 向きの特権性: 元の向きが、適当な向きよりどれだけ得か
    double v8 = sp / 8;
    std::vector<std::vector<int64_t>> cols(3, std::vector<int64_t>(n));
    auto bits_of = [&](const std::vector<double>& p) {
        double lo[3] = {1e300,1e300,1e300};
        for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) lo[d] = std::min(lo[d], p[i*3+d]);
        for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d)
            cols[d][i] = (int64_t)std::llround((p[i*3+d] - lo[d]) / v8);
        return zstd_bits(cols, n) * 8.0 / n;
    };
    double base = bits_of(w);
    std::mt19937_64 rng(seed);
    double rsum = 0, rmin = 1e300;
    for (int t = 0; t < 6; ++t) {
        double R[9]; rand_rot(rng, R);
        std::vector<double> p(n * 3);
        for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d)
            p[i*3+d] = w[i*3]*R[d*3] + w[i*3+1]*R[d*3+1] + w[i*3+2]*R[d*3+2];
        double b = bits_of(p); rsum += b; rmin = std::min(rmin, b);
    }
    double rmean = rsum / 6;
    S.sig.frame_gain = (rmean - base) / std::max(rmean, 1e-30);
    S.sig.rot_gain = (base - rmin) / std::max(base, 1e-30);
    S.sig.morton_delta = morton_delta(w, n, v8);

    // 走査軸の痕跡: 刻みが点間隔と同程度なら走査軸、ずっと細かければ格納格子
    for (int d = 0; d < 3; ++d) {
        std::vector<double> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = raw[i*3+d];
        std::vector<double> u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        if (u.size() < 3 || u.size() > n / 2) continue;
        std::vector<double> diffs;
        for (size_t i = 1; i < u.size(); ++i) if (u[i] > u[i-1]) diffs.push_back(u[i]-u[i-1]);
        if (diffs.empty()) continue;
        std::nth_element(diffs.begin(), diffs.begin()+diffs.size()/2, diffs.end());
        double step = diffs[diffs.size()/2];
        double amax = 0; for (double x : v) amax = std::max(amax, std::fabs(x));
        if (step <= 0 || amax / step > 1e9) continue;
        size_t on = 0;
        for (double x : v) { double k = x / step; if (std::fabs(k - std::round(k)) < 0.01) ++on; }
        double frac = (double)on / n, ratio = step / std::max(sp, 1e-30);
        if (frac > 0.99 && ratio > 0.1 && frac > S.sig.scan_on_grid) {
            S.sig.scan_axis = d; S.sig.scan_on_grid = frac;
            S.sig.scan_step_ratio = ratio; S.sig.scan_levels = (int64_t)u.size();
        }
    }

    double extent = 0;
    { double lo[3]={1e300,1e300,1e300}, hi[3]={-1e300,-1e300,-1e300};
      for (size_t i=0;i<n;++i) for(int d=0;d<3;++d){lo[d]=std::min(lo[d],raw[i*3+d]);hi[d]=std::max(hi[d],raw[i*3+d]);}
      extent = std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0])+(hi[1]-lo[1])*(hi[1]-lo[1])+(hi[2]-lo[2])*(hi[2]-lo[2])); }
    S.sig.origin_over_extent = std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]) / std::max(extent, 1e-30);
    // 方位角の単調性は、原点がセンサ位置と解釈できるときだけ意味を持つ
    if (S.sig.origin_over_extent <= 10.0) {
        size_t up = 0;
        for (size_t i = 1; i < n; ++i) {
            double a0 = std::atan2(raw[(i-1)*3+1], raw[(i-1)*3]);
            double a1 = std::atan2(raw[i*3+1], raw[i*3]);
            double d = a1 - a0;
            while (d > M_PI) d -= 2*M_PI;
            while (d < -M_PI) d += 2*M_PI;
            if (d > 0) ++up;
        }
        S.sig.azimuth_monotonic = (double)up / (n - 1);
    } else S.sig.azimuth_monotonic = std::nan("");

    auto clamp01 = [](double x) { return std::max(0.0, std::min(1.0, x)); };
    S.comp["距離の規則性"] = clamp01((0.52 - S.sig.cv_1nn) / 0.45);
    S.comp["角度の規則性"] = clamp01((S.sig.psi6 - S.sig.psi6_ref) / (1.0 - S.sig.psi6_ref));
    S.comp["向きの特権性"] = clamp01(S.sig.frame_gain / 0.30);
    S.comp["順序の意味"]   = clamp01(S.sig.morton_delta / 4.0);
    S.comp["走査軸の痕跡"] = S.sig.scan_axis >= 0 ? S.sig.scan_on_grid : 0.0;
    S.comp["回転走査の痕跡"] = std::isnan(S.sig.azimuth_monotonic) ? 0.0
                          : clamp01((S.sig.azimuth_monotonic - 0.5) / 0.45);
    double s = 0; for (auto& kv : S.comp) s += kv.second;
    S.score = s / S.comp.size();

    S.keep_storage_order = S.sig.morton_delta > 0.3;
    S.consider_morton = !S.keep_storage_order;
    S.consider_polar = (S.sig.origin_over_extent <= 10.0) &&
                       !std::isnan(S.sig.azimuth_monotonic) && S.sig.azimuth_monotonic > 0.95;
    S.consider_rotation = S.sig.rot_gain > 0.01;

    char buf[256];
    if (S.keep_storage_order)
        snprintf(buf, sizeof(buf), "格納順が Morton より %.2f bpp 良い。並べ替えないこと", S.sig.morton_delta);
    else
        snprintf(buf, sizeof(buf), "格納順に意味が見えない（Morton 差 %+.2f bpp）。Morton を候補に", S.sig.morton_delta);
    S.notes.push_back(buf);
    if (S.sig.scan_axis >= 0) {
        snprintf(buf, sizeof(buf), "%c 軸が %lld 段階の格子に %.0f%% 乗り、刻みが点間隔の %.2f 倍。走査軸の可能性",
                 "xyz"[S.sig.scan_axis], (long long)S.sig.scan_levels,
                 S.sig.scan_on_grid * 100, S.sig.scan_step_ratio);
        S.notes.push_back(buf);
    }
    if (S.sig.origin_over_extent > 10.0) {
        snprintf(buf, sizeof(buf), "座標原点が広がりの %.0f 倍遠い。極座標は縮退するので除外（方位角の判定も無効）",
                 S.sig.origin_over_extent);
        S.notes.push_back(buf);
    } else if (S.sig.azimuth_monotonic > 0.95) {
        snprintf(buf, sizeof(buf), "方位角が %.0f%% 単調。回転式センサの 1 スイープの可能性",
                 S.sig.azimuth_monotonic * 100);
        S.notes.push_back(buf);
    }
    if (S.sig.rot_gain > 0.01)
        snprintf(buf, sizeof(buf), "回転すると %.1f%% 得になる。回転探索の余地", S.sig.rot_gain * 100);
    else
        snprintf(buf, sizeof(buf), "元の向きが最良（適当な向きより %.1f%% 良い）。回転させないこと",
                 S.sig.frame_gain * 100);
    S.notes.push_back(buf);
    S.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return S;
}

} // namespace pcc
