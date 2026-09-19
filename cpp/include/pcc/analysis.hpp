// 計測器: エントロピー・オクタント予測・表面/サンプリング分解・
//         取得構造スコア（距離の規則性/角度の規則性/向きの特権性/順序の意味/走査の痕跡）
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include "pcc/geom.hpp"

namespace pcc {

// 粗レベルの占有から局所平面を当て、子オクタントを予測できるか
struct OctantResult {
    size_t n = 0; double coverage = 0, H = 0, H_cond = 0, acc = 0, used = 0;
    double mi() const { return H - H_cond; }
};
OctantResult octant_experiment(const std::vector<double>& xyz, size_t n, double v,
                               int radius = 2, size_t sample = 150000, uint64_t seed = 0);

// 表面（点間隔の格子での占有）とサンプリング（そこから細かくする分）の分解
struct SurfaceSplit { double spacing, b_surf, b_fine, b_sample, frac_surf; };

// 取得構造スコア
struct AcqSignals {
    double cv_1nn = 0, psi6 = 0, psi6_ref = 0, frame_gain = 0, rot_gain = 0;
    double morton_delta = 0, azimuth_monotonic = 0, origin_over_extent = 0, spacing = 0;
    int scan_axis = -1; double scan_on_grid = 0, scan_step_ratio = 0; int64_t scan_levels = 0;
};
struct AcqScore {
    double score = 0;
    std::map<std::string, double> comp;
    AcqSignals sig;
    bool keep_storage_order = true, consider_morton = false;
    bool consider_polar = false, consider_rotation = false;
    std::vector<std::string> notes;
    double seconds = 0;
};
AcqScore acquisition_score(const std::vector<double>& xyz_world, size_t n,
                           size_t sample = 60000, uint64_t seed = 0);

double cv_first_neighbor(const std::vector<double>& xyz, size_t n, size_t sample, uint64_t seed);
double hexatic(const std::vector<double>& xyz, size_t n, int m, int k,
               size_t sample, uint64_t seed);
double morton_delta(const std::vector<double>& xyz, size_t n, double v);

} // namespace pcc
