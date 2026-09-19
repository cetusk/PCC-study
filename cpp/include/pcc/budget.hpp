// 誤差予算: センサ床（局所平面残差）と物体側（TSDF 経由の局所特徴サイズ）。
//
// 中軸はノイズに不安定なので、生の点にそのまま当てない。正則ボクセル格子で
// 符号付き距離を作り、零交差から曲面を取り出してから shrinking ball を回す。
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "pcc/geom.hpp"

namespace pcc {

// 近傍 k 点から法線を推定する（up が非ゼロならその向きに揃える）
void estimate_normals(const std::vector<double>& xyz, size_t n, int k,
                      const double* up, std::vector<double>& normals);

struct Tsdf {
    std::vector<float> v;              // dims[0]*dims[1]*dims[2]
    int dims[3] = {0, 0, 0};
    double lo[3] = {0, 0, 0};
    double voxel = 0, trunc = 0;
    inline size_t idx(int i, int j, int k) const {
        return ((size_t)i * dims[1] + j) * dims[2] + k;
    }
};

// narrow band の符号付き距離場。未定義ボクセルは外周から塗り分けて符号を決める
// （一律 +trunc で埋めると物体内部に偽の零交差ができ、球の lfs 誤差が 2.1%→23% に悪化した）
bool build_tsdf(const std::vector<double>& xyz, const std::vector<double>& normals,
                size_t n, double voxel, Tsdf& out, std::string& err,
                double trunc_mult = 3.0, int knn = 12);

// 格子の辺上の零交差を曲面点として取り出し、勾配から法線を作る
void extract_surface(const Tsdf& t, std::vector<double>& pts, std::vector<double>& normals);

// 最大内接球の半径（= 中軸までの距離）。分離角で潰れを防ぐ。
std::vector<double> shrinking_ball_lfs(const std::vector<double>& pts,
                                       const std::vector<double>& normals,
                                       double r_init, int n_iter = 60,
                                       double sep_angle_deg = 32.0,
                                       double rel_tol = 1e-3);

// センサ床の目安: 近傍から当てた局所平面からの残差の標準偏差（平面的な点に限る）
double noise_floor_estimate(const std::vector<double>& xyz, size_t n, int k = 8,
                            size_t sample = 100000, uint64_t seed = 0);

} // namespace pcc
