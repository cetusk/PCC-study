// 面を符号化して点を引き直す方式と、点をそのまま符号化する方式を、
// 同じエントロピー符号器で比べる。
#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace pcc {

uint64_t octree_bytes(const std::vector<double>& xyz, size_t n, double leaf);

struct SurfOpt {
    double voxel = 0;        // 固定体素のとき使う一辺（適応のときは根の一辺）
    double eps = 0.01;       // 面までの距離の刻み
    int nbits = 8;           // 法線の量子化ビット
    bool adaptive = false;   // 平面から外れる節点を分割する
    double tau = 0.02;       // 分割の閾値（平面残差 RMS）
    int min_pts = 8;         // これ未満なら分割しない
    int max_level = 12;
    bool jitter = false;     // 法線方向の粗さを復元する
    bool strat = false;      // 接平面方向の配置を層化ジッタにする
    bool predict = false;    // 葉の法線と面位置を、既に符号化した近傍から予測する
};

struct SurfResult {
    size_t n_leaf = 0, n_out = 0;
    uint64_t b_occ = 0, b_split = 0, b_normal = 0, b_offset = 0, b_count = 0, b_sigma = 0;
    double mean_pts_per_leaf = 0;
    // 葉ごとの点の添字（最終的な葉の順序で並ぶ）。属性を面上の場にするときに使う
    std::vector<std::vector<int64_t>> leaf_ids;
    uint64_t total() const { return b_occ + b_split + b_normal + b_offset + b_count + b_sigma; }
};

SurfResult surface_code(const std::vector<double>& xyz, size_t n, const SurfOpt& o,
                        std::vector<double>& out);

} // namespace pcc
