// 対応関係を仮定しない歪み尺度。
// 復号器が元と別の点を出してよい設計では、点ごとの誤差は定義できない。
#pragma once
#include <vector>
#include <cstddef>

namespace pcc {

struct Distortion {
    double chamfer = 0;      // 双方向の最近傍距離の平均
    double hausdorff = 0;    // 双方向の最大
    double p95 = 0;          // 双方向の 95 パーセンタイル
    double a2b_mean = 0, b2a_mean = 0;
    double a2b_max = 0, b2a_max = 0;
    // 面に対する誤差（点ごとに近傍で平面を当て、その平面までの距離）
    double plane_mean = 0, plane_p95 = 0, plane_max = 0;
};

Distortion distortion(const std::vector<double>& A, size_t na,
                      const std::vector<double>& B, size_t nb,
                      int plane_k = 12);

} // namespace pcc
