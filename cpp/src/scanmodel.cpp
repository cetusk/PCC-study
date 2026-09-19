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
#include "pcc/rangecoder.hpp"

namespace pcc {

// 4x4 の連立一次方程式を部分ピボットのガウス消去で解く。M は [A | b] の 4x5。
static bool solve4(double M[4][5], double out[4]) {
    for (int c = 0; c < 4; ++c) {
        int piv = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
        if (std::fabs(M[piv][c]) < 1e-300) return false;
        if (piv != c) for (int k = c; k < 5; ++k) std::swap(M[c][k], M[piv][k]);
        for (int r = c + 1; r < 4; ++r) {
            double f = M[r][c] / M[c][c];
            for (int k = c; k < 5; ++k) M[r][k] -= f * M[c][k];
        }
    }
    for (int r = 3; r >= 0; --r) {
        double v = M[r][4];
        for (int k = r + 1; k < 4; ++k) v -= M[r][k] * out[k];
        out[r] = v / M[r][r];
        if (!std::isfinite(out[r])) return false;
    }
    return true;
}

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
                         const int64_t* gps, const int64_t* sa, size_t n,
                         FitDiag* diag) {
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

    // 費用は zigzag のビット長で測る（符号長を決めるのは散らばりではなく裾の重さ）
    auto blen = [](int64_t v) { uint64_t z = zigzag(v); int k = 0; while (z) { ++k; z >>= 1; } return k; };

    auto blen_d = [](double r) {
        int64_t v = (int64_t)llround(r);
        uint64_t z = zigzag(v); int k = 0; while (z) { ++k; z >>= 1; } return (double)k;
    };

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

    // 角度は符号化器（scan_predict）と同じく [-45, 45] 度で打ち切る。
    // 当てはめの目的関数が符号化器の計算と違うと、実態に合わない最適化になる。
    const double TH_LIM = M_PI / 4 - 1e-9;
    auto clamp_th = [&](double t) { return t > TH_LIM ? TH_LIM : (t < -TH_LIM ? -TH_LIM : t); };
    auto cost_of = [&](double S0, double SZ, double TH, double OM) {
        double c = 0;
        for (size_t i = 0; i < n; ++i) {
            double r = sf[i] - (S0 + (SZ - zz[i]) * std::tan(clamp_th(TH + OM * shot[i])));
            c += r * r;
        }
        return c;
    };

    // 初期値の詰め: 角度を固定して (s0, Sz) を線形最小二乗で合わせる。
    //   s + z T = s0 + Sz T  なので、計画行列 [1, T]、目標 s + z T の 2 変数
    {
        double A00 = 0, A01 = 0, A11 = 0, b0 = 0, b1 = 0;
        for (size_t i = 0; i < n; ++i) {
            double T = std::tan(clamp_th(th0 + om * shot[i]));
            double y = sf[i] + zz[i] * T;
            A00 += 1; A01 += T; A11 += T * T; b0 += y; b1 += y * T;
        }
        double det = A00 * A11 - A01 * A01;
        if (std::fabs(det) > 1e-9) {
            double ns0 = (b0 * A11 - b1 * A01) / det;
            double nSz = (A00 * b1 - A01 * b0) / det;
            if (std::isfinite(ns0) && std::isfinite(nSz) &&
                nSz > zmean + 5e4 && nSz < zmean + 4e6) { s0 = ns0; Sz = nSz; }
        }
    }

    // レーベンバーグ・マルカート法で 4 パラメタを同時に合わせる。
    //   (JᵀJ + λ diag(JᵀJ)) δ = -Jᵀr,  p ← p + δ
    // 列のスケールが 6 桁違う（shot は最大 7 万、Sz は 10^6 mm）ので、
    // D = diag(sqrt(A_kk)) で正規化してから解く。D⁻¹AD⁻¹ は対角が 1 になる。
    double lambda = 1e-3;
    double cost = cost_of(s0, Sz, th0, om);
    for (int it = 0; it < FIT_ITERS; ++it) {
        double A[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
        double g[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < n; ++i) {
            double raw = th0 + om * shot[i];
            bool cl = (raw > TH_LIM || raw < -TH_LIM);       // 打ち切られた点は角度の勾配が 0
            double T = std::tan(clamp_th(raw));
            double dz = Sz - zz[i];
            double r = sf[i] - (s0 + dz * T);
            double w = cl ? 0.0 : dz * (1.0 + T * T);
            double J[4] = {-1.0, -T, -w, -w * shot[i]};
            for (int a2 = 0; a2 < 4; ++a2) {
                g[a2] -= J[a2] * r;
                for (int b2 = a2; b2 < 4; ++b2) A[a2][b2] += J[a2] * J[b2];
            }
        }
        for (int a2 = 0; a2 < 4; ++a2)
            for (int b2 = 0; b2 < a2; ++b2) A[a2][b2] = A[b2][a2];

        double sc[4];
        for (int a2 = 0; a2 < 4; ++a2) sc[a2] = (A[a2][a2] > 0) ? std::sqrt(A[a2][a2]) : 1.0;

        bool stepped = false;
        for (int tr = 0; tr < 8 && !stepped; ++tr) {
            double M[4][5];
            for (int a2 = 0; a2 < 4; ++a2) {
                for (int b2 = 0; b2 < 4; ++b2) M[a2][b2] = A[a2][b2] / (sc[a2] * sc[b2]);
                M[a2][a2] += lambda;
                M[a2][4] = g[a2] / sc[a2];
            }
            double dd[4];
            if (solve4(M, dd)) {
                double ns0 = s0 + dd[0] / sc[0], nSz = Sz + dd[1] / sc[1];
                double nth = th0 + dd[2] / sc[2], nom = om + dd[3] / sc[3];
                if (nSz < zmean + 5e4) nSz = zmean + 5e4;
                if (nSz > zmean + 4e6) nSz = zmean + 4e6;
                nth = clamp_th(nth);
                double nc = cost_of(ns0, nSz, nth, nom);
                if (std::isfinite(nc) && nc < cost) {
                    s0 = ns0; Sz = nSz; th0 = nth; om = nom;
                    cost = nc;
                    lambda = std::max(lambda * 0.1, 1e-12);
                    stepped = true;
                }
            }
            if (!stepped) lambda = std::min(lambda * 10.0, 1e12);
        }
        if (diag) {
            double bb = 0;
            for (size_t i = 0; i < n; ++i)
                bb += blen_d(sf[i] - (s0 + (Sz - zz[i]) * std::tan(clamp_th(th0 + om * shot[i]))));
            for (int j = it; j < FIT_ITERS; ++j) diag->iter_bits[j] = bb / n;
        }
        if (!stepped) break;
    }

    // 整数へ。角度は [-45,45] 度 を [-2^31, 2^31] に写す
    const double U = (M_PI / 4) / (double)SCAN_ANG_MAX;
    p.s0 = (int64_t)llround(s0);
    p.Sz = (int64_t)llround(Sz);
    p.th0 = (int64_t)llround(th0 / U);
    p.om = (int64_t)llround(om / U);
    if (p.th0 > SCAN_ANG_MAX - 1) p.th0 = SCAN_ANG_MAX - 1;
    if (p.th0 < -SCAN_ANG_MAX) p.th0 = -SCAN_ANG_MAX;

    // 整数の経路で残差を作り、退避路に勝つかを見る。
    // 退避路は符号化器が実際に使うもの（s0 から始める中央値予測）と同一にする。
    // ここが実態とずれると、また実態に合わない代理指標で選ぶことになる。
    // 面外の符号化は採用・退避で変えないので、比べるのは面内だけでよい。
    double mdl_bits = 0, alt_bits = 0;
    MedPred mp;
    mp.prev = p.s0;
    for (size_t i = 0; i < n; ++i) {
        mdl_bits += blen(s[i] - scan_predict(p, gps[i] - gps[0], Z[i]));
        alt_bits += blen(s[i] - mp.predict());
        mp.push(s[i]);
    }
    p.ok = (mdl_bits < alt_bits) ? 1 : 0;

    // モデルを採らない線では、せん断回転は退避路の費用だけで決めてよい。
    // 帯状に広がっていて主軸に意味がない線では、回転しないほうが安い。
    // 両方を符号化してみて短い方を採る。
    if (!p.ok) {
        auto fb_bits = [&](const int64_t* u, const int64_t* v) {
            MedPred a2, b2;
            a2.prev = u[0]; b2.prev = v[0];
            double c = 0;
            for (size_t i = 0; i < n; ++i) {
                c += blen(u[i] - a2.predict()) + blen(v[i] - b2.predict());
                a2.push(u[i]); b2.push(v[i]);
            }
            return c;
        };
        if (fb_bits(X, Y) < fb_bits(s.data(), off.data())) {
            p.t2 = 0; p.sn = 0;                       // 恒等（s = X, off = Y）
            p.s0 = X[0];
            p.off0 = Y[0];
        } else {
            p.s0 = s[0];
            p.off0 = off[0];
        }
    }

    if (diag) {
        diag->span_deg = (double)(*std::max_element(sa, sa + n)
                                - *std::min_element(sa, sa + n)) * 0.006;
        diag->thin_mm = line_thinness(X, Y, n);
        diag->height_m = ((double)p.Sz - zmean) * 0.001;
        // 角度単位は [-45,45] 度 ↔ [-2^31,2^31]、時間単位は gps の ulp（59.6046 ns）
        diag->omega_deg_s = (double)p.om * (45.0 / 2147483648.0) / 59.6046e-9;
        diag->mdl_bits = mdl_bits / n;
        diag->alt_bits = alt_bits / n;
    }
    return p;
}

} // namespace pcc
