// 属性の空間予測 — 幾何から導いた順序で、既に復号済みの近傍から予測する。
#pragma once
#include "pcc/col.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>

namespace pcc {

std::vector<int32_t> coding_order(const std::vector<double>& xyz, size_t n,
                                  const std::string& kind);
void build_causal_predictors(const std::vector<double>& xyz, size_t n,
                             const std::vector<int32_t>& perm, int P, int k_search,
                             std::vector<int32_t>& pred);

// 複数の P ぶんを 1 回の近傍探索で作る（木も 1 回だけ建てる）。
// P ごとに呼んだときと同じ表を返す。
void build_causal_predictors_multi(const std::vector<double>& xyz, size_t n,
                                   const std::vector<int32_t>& perm,
                                   const std::vector<int>& Ps,
                                   std::vector<std::vector<int32_t>>& preds);

struct AttrResult {
    std::string name;
    double bpp_raw = 0;        // 差分を取らずそのまま符号化
    double bpp_order = 0;      // 格納順の差分
    double bpp_spatial = 0;    // 幾何由来の順序での空間予測
    bool roundtrip_ok = false;
};

// 符号化順に並べ替えて予測残差を作る / 残差から元の並びへ戻す
void spatial_residual(const Col& v, const std::vector<int32_t>& perm,
                      const std::vector<int32_t>& pred, int P, size_t n,
                      std::vector<int64_t>& out);
// 残差を int32 で受ける版。値も残差も収まったときだけ真を返す。
// 収まらなければ out には触れず、呼び手は 64 bit の版に落とす。
bool spatial_residual32(const Col& v, const std::vector<int32_t>& perm,
                        const std::vector<int32_t>& pred, int P, size_t n,
                        std::vector<int32_t>& out);
// **符号化順に値が並んだ配列を、その場で残差に変える。**
// predict は自分より前の位置しか見ないので、後ろから計算すれば前はまだ値のまま
// であり、並べ替え用の作業配列が要らない（200 万点・16 並列で 128 MB）。
// int32 版は途中で溢れたら偽を返す。そのとき配列の中身は壊れている。
bool residual_backward32(std::vector<int32_t>& a, const std::vector<int32_t>& pred,
                         int P, size_t n);
void residual_backward64(std::vector<int64_t>& a, const std::vector<int32_t>& pred,
                         int P, size_t n);
void spatial_restore(const std::vector<int64_t>& res, const std::vector<int32_t>& perm,
                     const std::vector<int32_t>& pred, int P, size_t n,
                     std::vector<int64_t>& out);

AttrResult compare_field(const std::string& name, const std::vector<int64_t>& v,
                         const std::vector<int32_t>& perm,
                         const std::vector<int32_t>& pred, int P, size_t n);

} // namespace pcc
