// LAS/LAZ の読み書き（LASzip 経由）。
//
// 正規化レイヤーが扱うのは「ファイルに実際に格納されている生の値」なので、
// scale/offset を適用しない整数のまま列指向で取り出す。
// ExtraBytes は VLR を自前で解釈して、フィールド名・型・scale を保持する。
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace pcc {

enum class FType : int {
    U8 = 1, I8 = 2, U16 = 3, I16 = 4, U32 = 5, I32 = 6,
    U64 = 7, I64 = 8, F32 = 9, F64 = 10
};

int ftype_size(FType t);

struct ExtraDim {
    std::string name;
    FType type = FType::U16;
    bool has_scale = false, has_offset = false;
    double scale[3] = {1, 1, 1}, offset[3] = {0, 0, 0};
    std::string description;
    int byte_offset = 0;          // 点レコード内の extra_bytes 先頭からの位置
};

// 列指向の点群。値は全て int64 に正規化して持つ（元の型は ftype に残す）。
// 浮動小数フィールド（gps_time など）はビットパターンを int64 として持つ。
struct Field {
    FType ftype = FType::I64;
    bool is_extra = false;
    std::vector<int64_t> v;
    // 値が 1 種類しかない列は実体を持たない。幾何だけの PF6 では大半がそうで、
    // 400 万点なら 1 列あたり 32 MB を節約できる。
    bool is_const = false;
    int64_t cval = 0;
    inline int64_t at(size_t i) const { return is_const ? cval : v[i]; }
    inline size_t count(size_t n) const { return is_const ? n : v.size(); }
    // 実体が要る場面で展開する（定数のままでは扱えない経路のため）
    void materialize(size_t n) {
        if (is_const) { v.assign(n, cval); is_const = false; }
    }
};

struct PointCloud {
    size_t n = 0;
    std::vector<int32_t> X, Y, Z;          // scale/offset 適用前
    double scale[3] = {1, 1, 1};
    double offset[3] = {0, 0, 0};
    std::vector<std::string> order;        // フィールドの出現順（循環回避に使う）
    std::map<std::string, Field> fields;
    // 書き戻しに必要なヘッダ情報
    uint8_t point_format = 0;
    uint16_t point_record_len = 0;
    uint8_t version_minor = 4;
    std::vector<ExtraDim> extra;
    std::string src_path;
    uint64_t src_bytes = 0;
};

bool read_las(const std::string& path, PointCloud& pc, std::string& err,
              size_t max_points = 0);

// keep に無い ExtraBytes 次元は VLR ごと落とす。
// keep に無い標準次元は 0 で埋める（LASzip は定数列をほぼ無コストで潰す）。
bool write_las(const std::string& path, const PointCloud& pc,
               const std::vector<std::string>& keep, std::string& err,
               const std::vector<int32_t>* Xq = nullptr,
               const std::vector<int32_t>* Yq = nullptr,
               const std::vector<int32_t>* Zq = nullptr,
               const double* scale_override = nullptr);

// 解析系サブコマンド用: LAS/LAZ・PLY・KITTI .bin から座標だけ読む
bool read_points_any(const std::string& path, std::vector<double>& xyz,
                     size_t& n, std::string& err, size_t max_points = 0);

} // namespace pcc
