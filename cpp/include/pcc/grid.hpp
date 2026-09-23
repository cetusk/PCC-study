// 浮動小数で書かれた座標が、実は一様な格子の整数倍かどうかを見つける。
//
// 点群を float で持つ器（KITTI .bin、PLY など）は、中身が整数のことがある。
// そのまま符号化するとビットパターンを詰めることになり、値の構造が使えない。
// 刻みを見つけて整数に直せば、LAS と同じ経路に載る。
//
// **候補は統計で探し、採否はビット単位の証明で決める。**
// 刻み s が正しければ、すべての点で llround(v/s) から v のビット列が再生できる。
// 再生できない点があれば採らない（負のゼロだけは符号が消えるので別に数える）。
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pcc {

struct GridFit {
    bool ok = false;           // ビット単位の証明を通ったか
    double step = 0;           // 見つけた刻み（dec_exp >= 0 なら 10^-dec_exp）
    // 十進の文字列から来た値は、10^-d を**掛ける**と戻らない。10^d で**割る**と戻る
    // （2 進で表せない 10^-d を先に丸めてしまうため）。どちらの道順かを持つ。
    int dec_exp = -1;          // >= 0 なら「10^dec_exp で割る」。-1 なら「step を掛ける」
    double gridness = 0;       // 格子らしさ（候補探しの指標。採否には使わない）
    std::vector<uint64_t> neg_zero;   // -0.0 だった位置。整数を経ると符号が消える
};

// 10^d を正確に返す（0 <= d <= 22 は double で正確に表せる）。
// **復号側で libm の pow を呼ばないため。**pow の結果は libm の実装で 1 ulp 揺れうるので、
// 別の機械で復号すると十進の格子から戻した座標がずれうる。22 を超える d は
// 符号化側が作らない（1〜15 しか試さない）ので、壊れた器として 0 を返す。
inline double exact_pow10(int d) {
    static const double T[23] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,
                                 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                                 1e20, 1e21, 1e22};
    return (d >= 0 && d <= 22) ? T[d] : 0.0;
}

// float32 の並びに対して刻みを探し、証明する。
GridFit fit_grid_f32(const float* v, size_t n);
// float64 の並び（PLY の ascii など）に対して同じことをする。
GridFit fit_grid_f64(const double* v, size_t n);

} // namespace pcc
