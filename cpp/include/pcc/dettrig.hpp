// libm に依らない sin / cos と、極座標格子から直交座標への逆変換。
//
// **非可逆の極座標の復号は、別の機械でも同じビット列の座標を出さなければならない。**
// 座標は空間予測の順序表と近傍表に入るので、1 ulp ずれるだけで可逆のはずの属性
// （KITTI の強度など）が戻らなくなる。libm の sin / cos は実装（glibc・musl・macOS・MSVC）や
// 実行時に選ばれる版（FMA の有無）で最後の桁が変わりうる。
// ここでは四則演算と floor だけで組む（fdlibm の多項式）。-ffp-contract=off の下では
// IEEE 754 の倍精度を持つどの機械でも同じ値になる。誤差は 1〜2 ulp（角度は |x| < 2^19 を想定）。
#pragma once
#include <cmath>

namespace pcc {

namespace dettrig_detail {
// |x| <= pi/4 での sin（fdlibm の __kernel_sin、y = 0）
inline double ksin(double x) {
    const double S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03,
                 S3 = -1.98412698298579493134e-04, S4 = 2.75573137070700676789e-06,
                 S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
    const double z = x * x, v = z * x;
    const double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    return x + v * (S1 + z * r);
}
// |x| <= pi/4 での cos（fdlibm の __kernel_cos、y = 0）
inline double kcos(double x) {
    const double C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03,
                 C3 = 2.48015872894767294178e-05, C4 = -2.75573143513906633035e-07,
                 C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
    const double z = x * x;
    const double r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    const double hz = 0.5 * z, w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + z * r);
}
}  // namespace dettrig_detail

// sin と cos を同時に返す。四則演算と floor（どちらも IEEE 754 で結果が一意）だけを使う。
inline void det_sincos(double x, double& s, double& c) {
    using namespace dettrig_detail;
    const double invpio2 = 6.36619772367581382433e-01;
    const double pio2_1 = 1.57079632673412561417e+00;   // pi/2 の上位 33 bit
    const double pio2_1t = 6.07710050650619224932e-11;  // pi/2 - pio2_1
    const double fn = std::floor(x * invpio2 + 0.5);
    const double y = (x - fn * pio2_1) - fn * pio2_1t;
    const double sy = ksin(y), cy = kcos(y);
    const long q = (long)fn & 3;
    switch (q) {
    case 0: s = sy;  c = cy;  break;
    case 1: s = cy;  c = -sy; break;
    case 2: s = -sy; c = -cy; break;
    default: s = -cy; c = sy; break;
    }
}

// 極座標格子の 1 点を直交座標に戻す。符号化側（geom.cpp の polar_inverse）と
// 復号側（frame.cpp の frame_world）が**必ずこれを通る**ので、式は 1 か所にしかない。
// legacy_libm は版 3 の器を読むときだけ立てる（版 3 は libm の sin / cos で書かれた）。
inline void polar_point(double r, double th, double ph, const double o[3], bool legacy_libm,
                        double* out) {
    double st, ct, sp, cp;
    if (legacy_libm) {
        st = std::sin(th); ct = std::cos(th); sp = std::sin(ph); cp = std::cos(ph);
    } else {
        det_sincos(th, st, ct);
        det_sincos(ph, sp, cp);
    }
    out[0] = r * cp * ct + o[0];
    out[1] = r * cp * st + o[1];
    out[2] = r * sp + o[2];
}

}  // namespace pcc
