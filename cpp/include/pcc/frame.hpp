// Frame — 出所に依存しない列指向の点群。
//
// LAS/LAZ・KITTI .bin・PLY を同じ器に載せるための中間表現。
// 値はすべて int64 の列として持つ（浮動小数の列はビットパターンを入れる）。
// 幾何も属性と同じく「列」であり、特別扱いは geom[3] の名前だけが持つ。
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include "pcc/las.hpp"

namespace pcc {

enum class Role : uint8_t { Geometry = 0, Attribute = 1, Time = 2 };

// 列がどう格納されているか。正規化の計画と対で使う。
//   Raw      そのまま 1 ストリームとして入っている
//   Derived  入っていない（計画から復元する）
//   Residual 他フィールドとの残差として入っている
enum class Storage : uint8_t { Raw = 0, Derived = 1, Residual = 2 };

struct ColSpec {
    std::string name;
    FType ftype = FType::I64;
    Role role = Role::Attribute;
    bool is_extra = false;
    Storage storage = Storage::Raw;
};

// 精度の申告。bounded の上限は推定ではなく全点での実測値を書く。
struct Fidelity {
    bool exact = true;
    double declared_eps = 0;   // 宣言した誤差上限 [m]
    double measured_max = 0;   // 実測した最大誤差 [m]
};

// 幾何の表し方。
//   "int"     値 = scale*q + offset（LAS と同じ整数格子）
//   "f32bits" float32 のビットパターンをそのまま int64 に入れてある
//   "f64bits" float64 のビットパターン
//   "polar"   極座標格子（scale/offset の意味は封筒が持つ）
struct Frame {
    uint64_t n = 0;
    std::vector<ColSpec> schema;                          // 出現順（＝取得順の列の並び）
    std::map<std::string, std::vector<int64_t>> col;
    double scale[3] = {1, 1, 1}, offset[3] = {0, 0, 0};
    std::string geom[3] = {"X", "Y", "Z"};
    std::string geom_repr = "int";
    std::string source_kind = "raw";
    uint64_t source_bytes = 0;
    std::string plan;                                     // 正規化の副情報（JSON）
    Fidelity fid;
    std::vector<uint8_t> envelope;                        // 元の器を再生成する情報

    const std::vector<int64_t>* get(const std::string& k) const {
        auto it = col.find(k); return it == col.end() ? nullptr : &it->second;
    }
    const ColSpec* spec(const std::string& k) const {
        for (const auto& s : schema) if (s.name == k) return &s;
        return nullptr;
    }
};

// LAS の PointCloud との相互変換（封筒に点形式・ExtraBytes 定義を格納する）
bool frame_from_las(const PointCloud& pc, Frame& f, std::string& err);
bool frame_to_las(const Frame& f, PointCloud& pc, std::string& err);

// 拡張子で振り分けて読む（.las/.laz/.bin/.ply）
bool load_frame(const std::string& path, Frame& f, std::string& err, size_t max_points = 0);

// 全列の完全一致を確認する。違えば diff に最初の差異を書く。
bool frames_equal(const Frame& a, const Frame& b, std::string& diff);

// 世界座標（解析用）。geom_repr に応じて復元する。
void frame_world(const Frame& f, std::vector<double>& xyz);

// 先頭 n 点だけを持つ Frame を作る（候補選択を標本で行うときに使う）。
// 注意: 空間予測の利得そのものは部分標本では測れない（既報 2.42% 対 6.07%）。
// これは「どの符号器を選ぶか」にだけ使い、報告する数字は必ず全点で取り直す。
Frame truncate_frame(const Frame& f, size_t n);

// 正規化の計画を Frame に適用する / 復元する。
// 適用後もスキーマは元の全列を保持し、各列の storage だけが変わる。
// これにより復号側は元の列の並びをそのまま取り戻せる。
bool normalize_frame(Frame& f, const PointCloud& pc, std::string& err,
                     bool residual_ops = false);
bool denormalize_frame(Frame& f, std::string& err);

} // namespace pcc
