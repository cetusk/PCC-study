// 反復測量の差分符号化 — 参照点群が与えられたとき、対象点群は何ビット安くなるか。
//
// 予測: 参照が効くのは「表面の情報」だけで「サンプリング位置」には効かない。
// 表面は幾何のビットの 11〜22% しかないので、削減はその範囲に収まるはず。
// この予測が外れるかどうかが、この実験の意味である。
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>

namespace pcc {

struct DiffResult {
    double voxel = 0;
    size_t n_tgt = 0, n_ref = 0;
    size_t occ_tgt = 0, occ_ref = 0, occ_both = 0;
    double H_alone = 0;        // 参照なしで占有を送るビット/点
    double H_given = 0;        // 参照ありのビット/点
    double overlap = 0;        // 対象の占有セルのうち参照にもあった割合
};

// オクトツリーを降りながら、対象の占有バイト c を符号化する。
// 文脈は「同じノードにおける参照側の占有バイト ĉ_ref」。
// 符号化器も復号器も対象のツリーを同じ順で降りるので、ĉ_ref は双方が持てる。
// 候補セルを対象自身から作らないので、情報の漏れがない。
struct OctDiff {
    double leaf = 0;           // 葉の一辺
    int depth = 0;
    size_t nodes = 0;          // 占有ノード数（＝符号化される記号数）
    double H = 0;              // 文脈なしの H(c)  [bit/記号]
    double H_given = 0;        // 文脈ありの H(c | ĉ_ref)
    double bits_alone = 0;     // 幾何 [bit/点]
    double bits_given = 0;
    double ref_hit = 0;        // 参照側も占有だったノードの割合
};
std::vector<OctDiff> octree_diff(const std::vector<double>& tgt, size_t nt,
                                 const std::vector<double>& ref, size_t nr,
                                 int max_depth);

// 対象の各点について参照の最近傍点との残差を取り、符号化コストを測る
struct ResidualDiff {
    double voxel = 0;
    double bits_alone = 0;     // 対象だけを量子化して符号化
    double bits_resid = 0;     // 参照の最近傍との残差を符号化
    double median_dist = 0, p95_dist = 0;
};
ResidualDiff nearest_residual(const std::vector<double>& tgt, size_t nt,
                              const std::vector<double>& ref, size_t nr, double v);

} // namespace pcc
