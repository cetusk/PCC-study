#include "pcc/scanmodel.hpp"

namespace pcc {

// atan(2^-i) を角度単位で表した定数（[-45,45] 度 ↔ [-2^31, 2^31]）。
// 値は生成して埋め込んである。実行時に libm を呼ばないので処理系に依存しない。
static const int64_t CORDIC_ATAN[32] = {
    2147483648LL, 1267733622LL, 669835629LL, 340019024LL,
    170669324LL, 85417861LL, 42719353LL, 21360980LL,
    10680653LL, 5340347LL, 2670176LL, 1335088LL,
    667544LL, 333772LL, 166886LL, 83443LL,
    41722LL, 20861LL, 10430LL, 5215LL,
    2608LL, 1304LL, 652LL, 326LL,
    163LL, 81LL, 41LL, 20LL,
    10LL, 5LL, 3LL, 1LL,
};
static const int64_t CORDIC_K = 652032874LL;    // 1/利得 を 2^30 で

// 回転モードの CORDIC。整数のみ。
static void cordic_sincos(int64_t z, int64_t& sin_out, int64_t& cos_out) {
    int64_t x = CORDIC_K, y = 0;
    for (int i = 0; i < 32; ++i) {
        int64_t d = (z >= 0) ? 1 : -1;
        int64_t x2 = x - ((d * y) >> i);
        y = y + ((d * x) >> i);
        x = x2;
        z -= d * CORDIC_ATAN[i];
    }
    sin_out = y; cos_out = x;
}

// 表は [-2^31, 2^31] を 2^16 分割。線形補間も整数で行う。
static constexpr int TAN_BITS = 16;
static int64_t g_tab[(1 << TAN_BITS) + 1];
static bool g_tab_ready = false;

static void build_tab() {
    for (int i = 0; i <= (1 << TAN_BITS); ++i) {
        int64_t z = ((int64_t)i << (32 - TAN_BITS)) - SCAN_ANG_MAX;
        if (z > SCAN_ANG_MAX) z = SCAN_ANG_MAX;
        int64_t s, c;
        cordic_sincos(z, s, c);
        g_tab[i] = c ? ((s << SCAN_TAN_SH) / c) : 0;
    }
    g_tab_ready = true;
}

int64_t tan_fx(int64_t theta_fx) {
    if (!g_tab_ready) build_tab();
    int64_t u = theta_fx + SCAN_ANG_MAX;
    if (u < 0) u = 0;
    if (u > ((int64_t)1 << 32) - 1) u = ((int64_t)1 << 32) - 1;
    int64_t idx = u >> (32 - TAN_BITS);
    int64_t frac = u - (idx << (32 - TAN_BITS));
    if (idx >= (1 << TAN_BITS)) { idx = (1 << TAN_BITS) - 1; frac = (1 << (32 - TAN_BITS)) - 1; }
    int64_t a = g_tab[idx], b = g_tab[idx + 1];
    return a + (((b - a) * frac) >> (32 - TAN_BITS));
}

} // namespace pcc

// ============================================================ 符号化器側の当てはめ
// ここから下は符号化器だけが使う。復号器は送られた整数パラメタしか見ないので、
// 浮動小数を使ってよい。決定性が要るのは scan_predict の側である。

#include <cmath>
#include <algorithm>

namespace pcc {

// (x,y) の主軸を返す（2x2 共分散の最大固有ベクトル）
static void principal_axis(const int64_t* X, const int64_t* Y, size_t n,
                           double& dx, double& dy, double& cx, double& cy) {
    double sx = 0, sy = 0;
    for (size_t i = 0; i < n; ++i) { sx += (double)X[i]; sy += (double)Y[i]; }
    cx = sx / n; cy = sy / n;
    double a = 0, b = 0, c = 0;
    for (size_t i = 0; i < n; ++i) {
        double u = (double)X[i] - cx, v = (double)Y[i] - cy;
        a += u * u; b += u * v; c += v * v;
    }
    double tr = a + c, det = a * c - b * b;
    double disc = std::sqrt(std::max(0.0, tr * tr / 4 - det));
    double l1 = tr / 2 + disc;
    if (std::fabs(b) > 1e-12) { dx = l1 - c; dy = b; }
    else { dx = (a >= c) ? 1.0 : 0.0; dy = (a >= c) ? 0.0 : 1.0; }
    double nn = std::hypot(dx, dy);
    if (nn > 0) { dx /= nn; dy /= nn; }
}

double line_thinness(const int64_t* X, const int64_t* Y, size_t n) {
    if (n < 5) return 1e18;
    double dx, dy, cx, cy;
    principal_axis(X, Y, n, dx, dy, cx, cy);
    double s2 = 0;
    for (size_t i = 0; i < n; ++i) {
        double u = (double)X[i] - cx, v = (double)Y[i] - cy;
        double d = -u * dy + v * dx;
        s2 += d * d;
    }
    return std::sqrt(s2) / std::sqrt((double)n);
}

void split_two_lines(const int64_t* X, const int64_t* Y, size_t n,
                     std::vector<uint8_t>& label) {
    label.assign(n, 0);
    if (n < 40) return;
    for (size_t i = 0; i < n; ++i) label[i] = (uint8_t)(i & 1);
    std::vector<int64_t> bx, by;
    for (int it = 0; it < 12; ++it) {
        double p[2][4];
        bool bad = false;
        for (int g = 0; g < 2; ++g) {
            bx.clear(); by.clear();
            for (size_t i = 0; i < n; ++i) if (label[i] == g) { bx.push_back(X[i]); by.push_back(Y[i]); }
            if (bx.size() < 5) { bad = true; break; }
            double dx, dy, cx, cy;
            principal_axis(bx.data(), by.data(), bx.size(), dx, dy, cx, cy);
            p[g][0] = cx; p[g][1] = cy; p[g][2] = -dy; p[g][3] = dx;
        }
        if (bad) { label.assign(n, 0); return; }
        bool changed = false;
        for (size_t i = 0; i < n; ++i) {
            double u = (double)X[i], v = (double)Y[i];
            double d0 = std::fabs((u - p[0][0]) * p[0][2] + (v - p[0][1]) * p[0][3]);
            double d1 = std::fabs((u - p[1][0]) * p[1][2] + (v - p[1][1]) * p[1][3]);
            uint8_t nl = (d1 < d0) ? 1 : 0;
            if (nl != label[i]) { label[i] = nl; changed = true; }
        }
        if (!changed) break;
    }
    size_t c1 = 0;
    for (size_t i = 0; i < n; ++i) c1 += label[i];
    if (c1 < 10 || n - c1 < 10) label.assign(n, 0);
}

SweepParam fit_scan_line(const int64_t* X, const int64_t* Y, const int64_t* Z,
                         const int64_t* gps, const int64_t* sa, size_t n) {
    SweepParam p;
    p.count = (int64_t)n;
    if (n < 8) return p;
    double dx, dy, cx, cy;
    principal_axis(X, Y, n, dx, dy, cx, cy);
    double th = std::atan2(dy, dx);
    th -= M_PI * std::round(th / M_PI);                 // (-90, 90] に畳む
    p.t2 = (int64_t)llround(std::tan(-th / 2) * ((int64_t)1 << SCAN_SH));
    p.sn = (int64_t)llround(std::sin(-th) * ((int64_t)1 << SCAN_SH));

    std::vector<int64_t> s(n), off(n);
    for (size_t i = 0; i < n; ++i) shear_fwd(X[i], Y[i], p.t2, p.sn, s[i], off[i]);
    std::vector<int64_t> tmp(off);
    std::nth_element(tmp.begin(), tmp.begin() + n / 2, tmp.end());
    p.off0 = tmp[n / 2];

    // 素朴な差分（退避）の散らばり
    double m1 = 0, m2 = 0;
    for (size_t i = 1; i < n; ++i) { double d = (double)(s[i] - s[i - 1]); m1 += d; m2 += d * d; }
    double alt_sd = std::sqrt(std::max(0.0, m2 / (n - 1) - (m1 / (n - 1)) * (m1 / (n - 1))));

    // 初期値: 記録角から th0, om を線形に、Sz は広がりの比から
    std::vector<double> shot(n), zz(n), sf(n), ang(n);
    for (size_t i = 0; i < n; ++i) {
        shot[i] = (double)(gps[i] - gps[0]);
        zz[i] = (double)Z[i];
        sf[i] = (double)s[i];
        ang[i] = (double)sa[i] * 0.006 * M_PI / 180.0;
    }
    double sw = 0, sw2 = 0, sa_ = 0, swa = 0;
    for (size_t i = 0; i < n; ++i) { sw += shot[i]; sw2 += shot[i] * shot[i]; sa_ += ang[i]; swa += shot[i] * ang[i]; }
    double den = n * sw2 - sw * sw;
    double om = (std::fabs(den) > 1e-9) ? (n * swa - sw * sa_) / den : 0.0;
    double th0 = (sa_ - om * sw) / n;
    double tmin = 1e18, tmax = -1e18, smin = 1e18, smax = -1e18, zmean = 0;
    for (size_t i = 0; i < n; ++i) {
        double t = std::tan(ang[i]);
        tmin = std::min(tmin, t); tmax = std::max(tmax, t);
        smin = std::min(smin, sf[i]); smax = std::max(smax, sf[i]);
        zmean += zz[i];
    }
    zmean /= n;
    double Sz = zmean + (smax - smin) / std::max(tmax - tmin, 1e-6);
    Sz = std::min(std::max(Sz, zmean + 1e5), zmean + 4e6);
    double s0 = 0;
    for (size_t i = 0; i < n; ++i) s0 += sf[i];
    s0 /= n;

    // 交互最適化: (s0, Sz) は線形、(th0, om) はガウス・ニュートン
    for (int it = 0; it < 12; ++it) {
        double A00 = 0, A01 = 0, A11 = 0, b0 = 0, b1 = 0;
        for (size_t i = 0; i < n; ++i) {
            double T = std::tan(th0 + om * shot[i]);
            double t0 = 1.0, t1 = T, y = sf[i] + zz[i] * T;
            A00 += t0 * t0; A01 += t0 * t1; A11 += t1 * t1;
            b0 += t0 * y;  b1 += t1 * y;
        }
        double d = A00 * A11 - A01 * A01;
        if (std::fabs(d) > 1e-9) {
            double ns0 = (b0 * A11 - b1 * A01) / d;
            double nSz = (A00 * b1 - A01 * b0) / d;
            if (std::isfinite(ns0) && std::isfinite(nSz) &&
                nSz > zmean + 5e4 && nSz < zmean + 4e6) { s0 = ns0; Sz = nSz; }
        }
        double B00 = 0, B01 = 0, B11 = 0, g0 = 0, g1 = 0;
        for (size_t i = 0; i < n; ++i) {
            double T = std::tan(th0 + om * shot[i]);
            double r = sf[i] - (s0 + (Sz - zz[i]) * T);
            double j = -(Sz - zz[i]) * (1.0 + T * T);
            double j0 = j, j1 = j * shot[i];
            B00 += j0 * j0; B01 += j0 * j1; B11 += j1 * j1;
            g0 -= j0 * r;  g1 -= j1 * r;
        }
        double e = B00 * B11 - B01 * B01;
        if (std::fabs(e) > 1e-9) {
            double dth = (g0 * B11 - g1 * B01) / e;
            double dom = (B00 * g1 - B01 * g0) / e;
            if (std::isfinite(dth) && std::isfinite(dom)) {
                double nt = th0 - dth, no = om - dom;
                if (std::fabs(nt) < M_PI / 4 - 1e-6) { th0 = nt; om = no; }
            }
        }
    }

    // 整数へ。角度は [-45,45] 度 を [-2^31, 2^31] に写す
    const double U = (M_PI / 4) / (double)SCAN_ANG_MAX;
    p.s0 = (int64_t)llround(s0);
    p.Sz = (int64_t)llround(Sz);
    p.th0 = (int64_t)llround(th0 / U);
    p.om = (int64_t)llround(om / U);
    if (p.th0 > SCAN_ANG_MAX - 1) p.th0 = SCAN_ANG_MAX - 1;
    if (p.th0 < -SCAN_ANG_MAX) p.th0 = -SCAN_ANG_MAX;

    // 整数の経路で残差を作り、素朴な差分に勝つかを見る
    double q1 = 0, q2 = 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t r = s[i] - scan_predict(p, gps[i] - gps[0], Z[i]);
        q1 += (double)r; q2 += (double)r * (double)r;
    }
    double sd = std::sqrt(std::max(0.0, q2 / n - (q1 / n) * (q1 / n)));
    p.ok = (sd < alt_sd) ? 1 : 0;
    return p;
}

} // namespace pcc
