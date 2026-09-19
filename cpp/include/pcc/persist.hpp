#pragma once
#include <vector>
#include <cstddef>
#include <limits>
namespace pcc {
struct Feature { int dim; double birth, death, life; };
// 有限寿命の特徴を寿命の長い順に返す（H1, H2）
std::vector<Feature> persistence_features(const std::vector<double>& xyz, size_t n,
                                          double max_alpha_sq
                                              = std::numeric_limits<double>::infinity());
}
