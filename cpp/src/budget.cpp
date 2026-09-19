#include "pcc/budget.hpp"
#include <algorithm>
#include <random>
#include <queue>
#include <cstring>
#include <cmath>

namespace pcc {

void estimate_normals(const std::vector<double>& xyz, size_t n, int k,
                      const double* up, std::vector<double>& N) {
    KdTree t(xyz);
    N.assign(n * 3, 0.0);
    std::vector<int64_t> idx(k + 1); std::vector<double> d2(k + 1);
    double c0[3] = {0, 0, 0};
    if (!up) { for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) c0[d] += xyz[i*3+d];
               for (int d = 0; d < 3; ++d) c0[d] /= (double)n; }
    for (size_t i = 0; i < n; ++i) {
        t.knn(&xyz[i*3], k + 1, idx.data(), d2.data());
        double c[3] = {0, 0, 0};
        for (int j = 1; j <= k; ++j) for (int d = 0; d < 3; ++d) c[d] += xyz[idx[j]*3+d];
        for (int d = 0; d < 3; ++d) c[d] /= (double)k;
        double M[9] = {0,0,0,0,0,0,0,0,0};
        for (int j = 1; j <= k; ++j) {
            double u[3] = {xyz[idx[j]*3]-c[0], xyz[idx[j]*3+1]-c[1], xyz[idx[j]*3+2]-c[2]};
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) M[a*3+b] += u[a]*u[b];
        }
        for (int q = 0; q < 9; ++q) M[q] /= (double)k;
        double ev[3], V[9]; eigh3(M, ev, V);
        double nv[3] = {V[0], V[3], V[6]};
        double ref[3];
        if (up) { ref[0] = up[0]; ref[1] = up[1]; ref[2] = up[2]; }
        else { for (int d = 0; d < 3; ++d) ref[d] = xyz[i*3+d] - c0[d]; }
        if (nv[0]*ref[0] + nv[1]*ref[1] + nv[2]*ref[2] < 0) for (int d = 0; d < 3; ++d) nv[d] = -nv[d];
        for (int d = 0; d < 3; ++d) N[i*3+d] = nv[d];
    }
}

bool build_tsdf(const std::vector<double>& xyz, const std::vector<double>& NN,
                size_t n, double v, Tsdf& T, std::string& err,
                double trunc_mult, int knn) {
    T.voxel = v; T.trunc = trunc_mult * v;
    double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
    for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], xyz[i*3+d]); hi[d] = std::max(hi[d], xyz[i*3+d]); }
    for (int d = 0; d < 3; ++d) { lo[d] -= 3*v; hi[d] += 3*v; T.lo[d] = lo[d];
        T.dims[d] = std::max(2, (int)std::ceil((hi[d]-lo[d])/v) + 1); }
    size_t total = (size_t)T.dims[0]*T.dims[1]*T.dims[2];
    if (total > 120000000ull) { err = "格子が大きすぎる"; return false; }
    const float NA = std::numeric_limits<float>::quiet_NaN();
    T.v.assign(total, NA);

    int rad = (int)std::ceil(T.trunc / v);
    KdTree tree(xyz);
    std::vector<int64_t> idx(knn); std::vector<double> d2(knn);
    // narrow band のボクセルを点から生成
    std::vector<uint8_t> touched(total, 0);
    for (size_t i = 0; i < n; ++i) {
        int gi[3];
        for (int d = 0; d < 3; ++d) gi[d] = (int)std::lround((xyz[i*3+d]-lo[d])/v);
        for (int a = -rad; a <= rad; ++a) for (int b = -rad; b <= rad; ++b)
        for (int c = -rad; c <= rad; ++c) {
            int x = gi[0]+a, y = gi[1]+b, z = gi[2]+c;
            if (x < 0 || y < 0 || z < 0 || x >= T.dims[0] || y >= T.dims[1] || z >= T.dims[2]) continue;
            touched[T.idx(x,y,z)] = 1;
        }
    }
    for (int x = 0; x < T.dims[0]; ++x) for (int y = 0; y < T.dims[1]; ++y)
    for (int z = 0; z < T.dims[2]; ++z) {
        size_t id = T.idx(x,y,z);
        if (!touched[id]) continue;
        double q[3] = {lo[0]+x*v, lo[1]+y*v, lo[2]+z*v};
        tree.knn(q, knn, idx.data(), d2.data());
        double wsum = 0, acc = 0;
        for (int j = 0; j < knn; ++j) {
            double dist = std::sqrt(d2[j]);
            if (dist > T.trunc) continue;
            double w = std::exp(-(dist/v)*(dist/v));
            double proj = (q[0]-xyz[idx[j]*3])*NN[idx[j]*3]
                        + (q[1]-xyz[idx[j]*3+1])*NN[idx[j]*3+1]
                        + (q[2]-xyz[idx[j]*3+2])*NN[idx[j]*3+2];
            wsum += w; acc += w * proj;
        }
        if (wsum > 1e-12) T.v[id] = (float)std::max(-T.trunc, std::min(T.trunc, acc/wsum));
    }
    // 未定義ボクセルの符号: 外周に繋がるものは +trunc、閉じたものは −trunc
    std::vector<uint8_t> vis(total, 0);
    std::queue<size_t> Q;
    auto push_if = [&](int x, int y, int z) {
        if (x<0||y<0||z<0||x>=T.dims[0]||y>=T.dims[1]||z>=T.dims[2]) return;
        size_t id = T.idx(x,y,z);
        if (vis[id] || !std::isnan(T.v[id])) return;
        vis[id] = 1; Q.push(id);
    };
    for (int x = 0; x < T.dims[0]; ++x) for (int y = 0; y < T.dims[1]; ++y) {
        push_if(x,y,0); push_if(x,y,T.dims[2]-1); }
    for (int x = 0; x < T.dims[0]; ++x) for (int z = 0; z < T.dims[2]; ++z) {
        push_if(x,0,z); push_if(x,T.dims[1]-1,z); }
    for (int y = 0; y < T.dims[1]; ++y) for (int z = 0; z < T.dims[2]; ++z) {
        push_if(0,y,z); push_if(T.dims[0]-1,y,z); }
    while (!Q.empty()) {
        size_t id = Q.front(); Q.pop();
        int z = (int)(id % T.dims[2]); size_t r = id / T.dims[2];
        int y = (int)(r % T.dims[1]); int x = (int)(r / T.dims[1]);
        push_if(x+1,y,z); push_if(x-1,y,z); push_if(x,y+1,z);
        push_if(x,y-1,z); push_if(x,y,z+1); push_if(x,y,z-1);
    }
    for (size_t i = 0; i < total; ++i)
        if (std::isnan(T.v[i])) T.v[i] = vis[i] ? (float)T.trunc : (float)-T.trunc;
    return true;
}

void extract_surface(const Tsdf& T, std::vector<double>& P, std::vector<double>& N) {
    P.clear(); N.clear();
    auto grad = [&](int x, int y, int z, double g[3]) {
        auto at = [&](int a, int b, int c) {
            a = std::max(0, std::min(T.dims[0]-1, a));
            b = std::max(0, std::min(T.dims[1]-1, b));
            c = std::max(0, std::min(T.dims[2]-1, c));
            return (double)T.v[T.idx(a,b,c)];
        };
        g[0] = (at(x+1,y,z) - at(x-1,y,z)) * 0.5;
        g[1] = (at(x,y+1,z) - at(x,y-1,z)) * 0.5;
        g[2] = (at(x,y,z+1) - at(x,y,z-1)) * 0.5;
    };
    const int off[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int x = 0; x < T.dims[0]-1; ++x) for (int y = 0; y < T.dims[1]-1; ++y)
    for (int z = 0; z < T.dims[2]-1; ++z) {
        double a = T.v[T.idx(x,y,z)];
        for (int e = 0; e < 3; ++e) {
            int nx = x+off[e][0], ny = y+off[e][1], nz = z+off[e][2];
            double b = T.v[T.idx(nx,ny,nz)];
            if ((a <= 0 && b > 0) || (a > 0 && b <= 0)) {
                double t = a / (a - b);
                double p[3] = {T.lo[0] + (x + t*off[e][0]) * T.voxel,
                               T.lo[1] + (y + t*off[e][1]) * T.voxel,
                               T.lo[2] + (z + t*off[e][2]) * T.voxel};
                double g0[3], g1[3];
                grad(x,y,z,g0); grad(nx,ny,nz,g1);
                double g[3];
                for (int d = 0; d < 3; ++d) g[d] = g0[d]*(1-t) + g1[d]*t;
                double nn = std::sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2]);
                if (nn < 1e-30) continue;
                for (int d = 0; d < 3; ++d) { P.push_back(p[d]); N.push_back(g[d]/nn); }
            }
        }
    }
}

std::vector<double> shrinking_ball_lfs(const std::vector<double>& P,
                                       const std::vector<double>& N,
                                       double r_init, int n_iter,
                                       double sep_angle_deg, double rel_tol) {
    size_t n = P.size() / 3;
    KdTree t(P);
    std::vector<double> r(n, r_init);
    std::vector<uint8_t> shrunk(n, 0);
    double cos_min = std::cos(sep_angle_deg * M_PI / 180.0);
    int64_t idx[2]; double d2[2];
    for (int it = 0; it < n_iter; ++it) {
        bool any = false;
        for (size_t i = 0; i < n; ++i) {
            double c[3];
            for (int d = 0; d < 3; ++d) c[d] = P[i*3+d] - N[i*3+d]*r[i];
            t.knn(c, 2, idx, d2);
            size_t j = (idx[0] == (int64_t)i) ? (size_t)idx[1] : (size_t)idx[0];
            double dq = std::sqrt((idx[0] == (int64_t)i) ? d2[1] : d2[0]);
            if (!(dq < r[i] - 1e-12)) continue;
            double vp[3], vq[3];
            for (int d = 0; d < 3; ++d) { vp[d] = P[i*3+d]-c[d]; vq[d] = P[j*3+d]-c[d]; }
            double np_ = std::sqrt(vp[0]*vp[0]+vp[1]*vp[1]+vp[2]*vp[2]);
            double nq_ = std::sqrt(vq[0]*vq[0]+vq[1]*vq[1]+vq[2]*vq[2]);
            double cs = (vp[0]*vq[0]+vp[1]*vq[1]+vp[2]*vq[2]) / std::max(np_*nq_, 1e-30);
            // 初回だけ分離角を課さない（初期半径が大きいと必ず弾かれて縮まない）
            if (shrunk[i] && !(cs < cos_min)) continue;
            double df[3];
            for (int d = 0; d < 3; ++d) df[d] = P[i*3+d]-P[j*3+d];
            double dist2 = df[0]*df[0]+df[1]*df[1]+df[2]*df[2];
            double dist = std::sqrt(std::max(dist2, 1e-30));
            double cosang = (N[i*3]*df[0]+N[i*3+1]*df[1]+N[i*3+2]*df[2]) / dist;
            if (cosang <= 1e-12) continue;
            double rn = dist2 / (2.0 * cosang * dist);
            if (!(rn > 0) || !(rn < r[i]*(1-rel_tol))) continue;
            r[i] = rn; shrunk[i] = 1; any = true;
        }
        if (!any) break;
    }
    return r;
}

double noise_floor_estimate(const std::vector<double>& xyz, size_t n, int k,
                            size_t sample, uint64_t seed) {
    KdTree t(xyz);
    std::mt19937_64 rng(seed);
    size_t m = std::min(sample, n);
    std::vector<int64_t> idx(k + 1); std::vector<double> d2(k + 1);
    std::vector<double> resid; resid.reserve(m);
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
        double sum = ev[0]+ev[1]+ev[2];
        if (sum <= 0 || ev[0]/sum >= 0.02) continue;      // 平面的な点に限る
        double u[3] = {xyz[i*3]-c[0], xyz[i*3+1]-c[1], xyz[i*3+2]-c[2]};
        resid.push_back(u[0]*V[0] + u[1]*V[3] + u[2]*V[6]);
    }
    if (resid.size() < 100) return 0;
    double mu = 0; for (double x : resid) mu += x; mu /= resid.size();
    double var = 0; for (double x : resid) var += (x-mu)*(x-mu);
    return std::sqrt(var / resid.size());
}

} // namespace pcc
