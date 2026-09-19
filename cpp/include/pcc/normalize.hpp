// 表現の正規化。フィールド名をハードコードせず、関係をデータから見つける。
// 採用は全点検証を通ったものだけ。Python 版 (python/normalize.py) と同じ規則。
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "pcc/las.hpp"

namespace pcc {

struct Op {
    std::string kind;          // drop_constant | drop_duplicate | drop_affine | residual_code
    std::string target;
    std::string source;
    int64_t value = 0;         // drop_constant
    int64_t num = 1, den = 1, b = 0;   // drop_affine
    bool exact = true;
    std::string note;
    std::string describe() const;
};

struct Plan {
    std::vector<Op> ops;
    // 幾何側（bounded）
    bool grid = false;
    int grid_bits = 0;
    double max_err_m = 0.0;
    std::string to_json() const;
    static Plan from_json(const std::string& s);
    std::string report() const;
};

Plan analyze(const PointCloud& pc, bool enable_residual = true);

// 容器に残すフィールド名と、外部符号化に回す残差
void apply_plan(const PointCloud& pc, const Plan& plan,
                std::vector<std::string>& keep,
                std::map<std::string, std::vector<int64_t>>& external);

// 依存順に反復解決して元のフィールドを復元する
bool invert_plan(std::map<std::string, std::vector<int64_t>>& fields,
                 const std::map<std::string, std::vector<int64_t>>& external,
                 const Plan& plan, size_t n, std::string& err);

double entropy0(const std::vector<int64_t>& v);
double entropy0_diff(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

} // namespace pcc
