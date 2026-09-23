// 幾何の道具立て: KD木・点間隔・Morton・極座標・格子量子化・表現の選択。
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace pcc {

using Vec3 = std::array<double, 3>;

// ---- KD木（nanoflann の薄い包み）
class KdTree {
public:
    explicit KdTree(const std::vector<double>& xyz);   // N*3 の連続配列
    ~KdTree();
    // k 近傍。out_idx / out_d2 は k 要素
    void knn(const double* q, int k, int64_t* out_idx, double* out_d2) const;
    // 半径内の点（二乗半径で指定）
    void radius(const double* q, double r2, std::vector<int64_t>& out) const;
    size_t size() const { return n_; }
private:
    struct Impl; Impl* p_;
    size_t n_;
};

double point_spacing(const std::vector<double>& xyz, size_t sample = 50000,
                     uint64_t seed = 0);

std::vector<uint64_t> morton3(const std::vector<int64_t>& q, size_t n);

// 極座標（原点 o まわり）
void polar_forward(const std::vector<double>& xyz, size_t n, const Vec3& o,
                   double dr, double da,
                   std::vector<int64_t>& qr, std::vector<int64_t>& qa,
                   std::vector<int64_t>& qe);
void polar_inverse(const std::vector<int64_t>& qr, const std::vector<int64_t>& qa,
                   const std::vector<int64_t>& qe, const Vec3& o,
                   double dr, double da, std::vector<double>& xyz);

// 表現の候補と選択
struct GeomCandidate {
    std::string kind;              // grid | polar
    std::string anchor;            // polar のときの原点の取り方
    double step = 0, r_step = 0, ang_step = 0;
    Vec3 origin{{0, 0, 0}};
    double max_err = 0;
    uint64_t bytes = 0;
    std::vector<std::vector<int64_t>> streams;
};

// 誤差上限 eps を「使い切る」ように刻みを二分探索で広げてから比較する。
// allow_polar=false なら極座標候補を作らない。
std::vector<GeomCandidate> geometry_candidates(const std::vector<double>& xyz, size_t n,
                                               double eps, bool allow_polar = true,
                                               double origin_max_extent = 10.0);
// kind を渡すと（"grid" / "polar/origin" / "polar/centroid"）その種類に固定する。誤差上限を守る候補に
// その種類が無ければ、種類 "" の候補（kind も空）を返す。空なら代理の符号長で選ぶ。
GeomCandidate choose_geometry(const std::vector<double>& xyz, size_t n, double eps,
                              bool allow_polar = true, size_t sample = 200000,
                              bool verbose = false, const std::string& kind = "");

// 3x3 対称行列の固有分解（昇順）。法線推定などで使う
void eigh3(const double C[9], double evals[3], double evecs[9]);

} // namespace pcc
