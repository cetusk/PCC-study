// パーシステントホモロジー（GUDHI の Alpha 複体）。
//
// 位相そのものは誤差予算を決められなかった（安定性定理の上界をなぞるだけだった）。
// 正しい使い方は「寿命 L の特徴は摂動 δ < L/2 で保存される」で、
// 位相は数値ではなく**特徴と寿命の一覧**を返すのが役目。
#include "pcc/persist.hpp"
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <CGAL/Epick_d.h>
#include <algorithm>
#include <cmath>

namespace pcc {

using Kernel = CGAL::Epick_d<CGAL::Dimension_tag<3>>;
using Point = Kernel::Point_d;
using SimplexTree = Gudhi::Simplex_tree<>;
using Persistence = Gudhi::persistent_cohomology::Persistent_cohomology<
    SimplexTree, Gudhi::persistent_cohomology::Field_Zp>;

std::vector<Feature> persistence_features(const std::vector<double>& xyz, size_t n,
                                          double max_alpha_sq) {
    std::vector<Point> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double c[3] = {xyz[i*3], xyz[i*3+1], xyz[i*3+2]};
        pts.emplace_back(c, c + 3);
    }
    Gudhi::alpha_complex::Alpha_complex<Kernel> ac(pts);
    SimplexTree st;
    if (!ac.create_complex(st, max_alpha_sq)) return {};
    Persistence pcoh(st);
    pcoh.init_coefficients(2);
    pcoh.compute_persistent_cohomology(0);
    std::vector<Feature> out;
    for (const auto& iv : pcoh.get_persistent_pairs()) {
        int dim = st.dimension(std::get<0>(iv));
        double b = st.filtration(std::get<0>(iv));
        double d = (std::get<1>(iv) == st.null_simplex())
                 ? std::numeric_limits<double>::infinity()
                 : st.filtration(std::get<1>(iv));
        // alpha 複体の値は半径の二乗
        double bb = std::sqrt(std::max(0.0, b));
        double dd = std::isinf(d) ? d : std::sqrt(std::max(0.0, d));
        if (dim < 1 || dim > 2) continue;
        if (std::isinf(dd)) continue;
        out.push_back(Feature{dim, bb, dd, dd - bb});
    }
    std::sort(out.begin(), out.end(),
              [](const Feature& a, const Feature& b) { return a.life > b.life; });
    return out;
}

} // namespace pcc
