#include "pcc/grid.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace pcc {
namespace {

// PCC_GRID_DEBUG=1 を付けたときだけ、刻みの探索と証明の様子を出す。
bool grid_dbg() {
    static const bool on = getenv("PCC_GRID_DEBUG") != nullptr;
    return on;
}

// 標本。全点を並べ替えると大きな入力で重いので、等間隔に間引く。
// 間引き方は点数だけで決まるので、同じ入力なら必ず同じ標本になる。
std::vector<double> take(const double* v, size_t n, size_t cap) {
    std::vector<double> s;
    size_t stride = n > cap ? (n + cap - 1) / cap : 1;
    s.reserve(n / stride + 1);
    for (size_t i = 0; i < n; i += stride) s.push_back(v[i]);
    return s;
}

// その値域での浮動小数の刻み。これより細かい「格子」は表現そのものであって、
// データの性質ではない（同語反復になる）。
double ulp_of(const std::vector<double>& s, bool f32) {
    double m = 0;
    for (double x : s) m = std::max(m, std::fabs(x));
    if (m <= 0) return 0;
    if (f32) {
        float mf = (float)m;
        return (double)(std::nextafterf(mf, INFINITY) - mf);
    }
    return std::nextafter(m, INFINITY) - m;
}

// 格子らしさ。位相（原点のずれ）には依らない。
// **段の数を要求する。**刻みが値域より大きいと全部が同じ段に落ち、
// 格子らしさが 1 になってしまう（自明に成り立つだけで何も言っていない）。
double gridness(const std::vector<double>& s, double step) {
    if (!(step > 0) || !std::isfinite(step)) return 0;
    auto mm = std::minmax_element(s.begin(), s.end());
    if (*mm.second - *mm.first < 32.0 * step) return 0;
    double re = 0, im = 0;
    for (double x : s) {
        double t = 2.0 * M_PI * (x / step);
        re += std::cos(t); im += std::sin(t);
    }
    return std::sqrt(re * re + im * im) / (double)s.size();
}

// 差のおよその最大公約数を刻みとする。
// 格子に乗っているなら、並べ替えた値の差はすべて刻みの整数倍になる。
double gcd_step(std::vector<double> u, double floor_step) {
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    // 値の種類が少ない列でも格子はある（KITTI の強度は 97 種）。
    // 採否は証明が決めるので、ここは緩くてよい。刻みを外せば証明が落ちる。
    if (u.size() < 16) return 0;
    std::vector<double> d;
    d.reserve(u.size());
    for (size_t i = 1; i < u.size(); ++i) {
        double t = u[i] - u[i - 1];
        if (t > floor_step) d.push_back(t);
    }
    if (d.size() < 8) return 0;
    double s = *std::min_element(d.begin(), d.end());
    for (int it = 0; it < 6; ++it) {
        double num = 0, den = 0;
        for (double t : d) {
            double q = t / s, k = std::round(q);
            if (k >= 1 && k <= 4096 && std::fabs(q - k) < 0.25) { num += t; den += k; }
        }
        if (den < 8) break;
        double sn = num / den;
        if (!std::isfinite(sn) || sn <= 0) break;
        if (std::fabs(sn - s) <= 1e-15 * s) { s = sn; break; }
        s = sn;
    }
    return s;
}

// 刻みの相対誤差は値域の端で積もる。値そのものから最小二乗で磨く。
void polish(const std::vector<double>& s, double& step) {
    for (int it = 0; it < 3; ++it) {
        double num = 0, den = 0;
        for (double x : s) {
            double k = std::round(x / step);
            if (k == 0) continue;
            num += x * k; den += k * k;
        }
        if (den <= 0) return;
        double sn = num / den;
        if (!std::isfinite(sn) || sn <= 0) return;
        if (std::fabs(sn - step) <= 1e-16 * step) { step = sn; return; }
        step = sn;
    }
}

// 候補の刻みを 1 つ選ぶ。採否はここでは決めない。
double candidate_step(const double* v, size_t n, bool f32) {
    std::vector<double> s = take(v, n, 200000);
    if (s.size() < 100) return 0;
    double u4 = 4.0 * ulp_of(s, f32);
    double step = gcd_step(s, u4);
    if (!(step > u4)) return 0;
    polish(s, step);
    double g = gridness(s, step);
    // 見つけた刻みが本当の刻みの約数（倍音）のことがある。整数倍に上げる。
    if (g > 0.98) {
        for (int m = 2; m <= 32; ++m) {
            double gm = gridness(s, step * m);
            if (gm > 0.98) { step *= m; g = gm; }
        }
    }
    // 十進で書かれた文字列から来た値は 10^-k（や 2.5·10^-k）の格子に乗る。
    // 差分から出した種がその約数でないと、整数倍に上げても届かない。
    if (g <= 0.95) {
        for (int k = -3; k < 12 && g <= 0.95; ++k) {
            for (double c : {1.0, 2.5, 5.0}) {
                double t = c * std::pow(10.0, -k);
                if (t <= u4) continue;
                double gt = gridness(s, t);
                if (gt > 0.90) { step = t; g = gt; break; }
            }
        }
    }
    return step;
}

// 磨いた刻みは真の値から 10^-13 ほどずれる。ふつうは無害だが、値が隣り合う
// 2 つの浮動小数のちょうど中間に落ちる点では丸めが反転し、証明が落ちる。
// 十進に丸めた候補も並べ、証明を通った最初のものを採る。
std::vector<double> step_candidates(double step) {
    std::vector<double> c;
    for (int k = 1; k <= 15; ++k) {
        double p10 = std::pow(10.0, k);
        double t = std::round(step * p10) / p10;
        if (!(t > 0) || std::fabs(t / step - 1.0) >= 1e-6) continue;
        bool dup = false;
        for (double x : c) if (x == t) dup = true;
        if (!dup) c.push_back(t);
    }
    c.push_back(step);          // 十進に乗らない刻みもあるので、磨いた値も試す
    return c;
}

// 刻み step で全点のビット列が再生できるか。
// **整数を経る道順は、実際に格納するときと同じでなければならない。**
// double の round はゼロの符号を残すが、int64 に落とすと消える。
// 残る側で試すと、消える例外を数え落とす。
template <class F, class U>
bool prove(const F* v, size_t n, double step, int dec_exp,
           std::vector<uint64_t>& neg_zero) {
    neg_zero.clear();
    const double p10 = dec_exp >= 0 ? std::pow(10.0, dec_exp) : 0.0;
    for (size_t i = 0; i < n; ++i) {
        double q = dec_exp >= 0 ? std::round((double)v[i] * p10)
                                : std::round((double)v[i] / step);
        if (!(std::fabs(q) <= 9.0e15)) return false;
        int64_t k = (int64_t)q;
        F back = dec_exp >= 0 ? (F)((double)k / p10) : (F)((double)k * step);
        U a, b;
        std::memcpy(&a, &back, sizeof(U));
        std::memcpy(&b, &v[i], sizeof(U));
        if (a == b) continue;
        // 負のゼロだけは整数を経ると符号が消える。値は等しいので別に数える。
        if (v[i] == (F)0 && back == (F)0) { neg_zero.push_back((uint64_t)i); continue; }
        return false;
    }
    return true;
}

template <class F, class U>
GridFit fit_impl(const F* v, const double* d, size_t n, bool f32, const char* tag) {
    GridFit r;
    if (n < 100) return r;
    double step = candidate_step(d, n, f32);
    if (!(step > 0)) {
        if (grid_dbg()) fprintf(stderr, "[格子] %s 刻みが見つからない\n", tag);
        return r;
    }
    // 掛け算で戻る刻みを先に試す（KITTI のような 2 進で素直な場合）。
    for (double c : step_candidates(step)) {
        std::vector<uint64_t> nz;
        if (!prove<F, U>(v, n, c, -1, nz)) continue;
        r.ok = true; r.step = c; r.dec_exp = -1; r.neg_zero = std::move(nz);
        std::vector<double> sm = take(d, n, 200000);
        r.gridness = gridness(sm, c);
        if (grid_dbg())
            fprintf(stderr, "[格子] %s 通った: step=%.17g 格子らしさ=%.4f -0.0=%zu\n",
                    tag, c, r.gridness, r.neg_zero.size());
        return r;
    }
    // 十進の文字列から来た値は割り算で戻る。**粗いほうから**試して、
    // 全点が戻る最初の桁を採る（細かい桁を採ると整数が無駄に大きくなる）。
    //
    // ここにも表現の刻みの保護が要る。**桁を増やせばどんな値でも「格子に乗る」**
    // ので、保護が無いと float32 の並びが 10^-11 の格子と判定され、
    // 整数が無駄に大きくなって元より伸びる。
    std::vector<double> sm0 = take(d, n, 200000);
    const double u4 = 4.0 * ulp_of(sm0, f32);
    for (int de = 1; de <= 15; ++de) {
        const double st = std::pow(10.0, -de);
        if (st <= u4) break;
        std::vector<uint64_t> nz;
        if (!prove<F, U>(v, n, 0.0, de, nz)) continue;
        r.ok = true; r.step = std::pow(10.0, -de); r.dec_exp = de;
        r.neg_zero = std::move(nz);
        std::vector<double> sm = take(d, n, 200000);
        r.gridness = gridness(sm, r.step);
        if (grid_dbg())
            fprintf(stderr, "[格子] %s 通った: 10^-%d で割る 格子らしさ=%.4f -0.0=%zu\n",
                    tag, de, r.gridness, r.neg_zero.size());
        return r;
    }
    if (grid_dbg())
        fprintf(stderr, "[格子] %s 落ちた: 候補 %.17g から作った刻みがどれも通らない\n",
                tag, step);
    return GridFit{};
}

}  // namespace

GridFit fit_grid_f32(const float* v, size_t n) {
    if (n < 100) return GridFit{};
    std::vector<double> d(n);
    for (size_t i = 0; i < n; ++i) d[i] = (double)v[i];
    return fit_impl<float, uint32_t>(v, d.data(), n, true, "f32");
}

GridFit fit_grid_f64(const double* v, size_t n) {
    return fit_impl<double, uint64_t>(v, v, n, false, "f64");
}

}  // namespace pcc
