// Frame — 出所に依存しない列指向の点群。
//
// LAS/LAZ・KITTI .bin・PLY を同じ器に載せるための中間表現。
// 値はすべて int64 の列として持つ（浮動小数の列はビットパターンを入れる）。
// 幾何も属性と同じく「列」であり、特別扱いは geom[3] の名前だけが持つ。
#pragma once
#include "pcc/col.hpp"
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
    std::map<std::string, Col> col;
    double scale[3] = {1, 1, 1}, offset[3] = {0, 0, 0};
    std::string geom[3] = {"X", "Y", "Z"};
    std::string geom_repr = "int";
    std::string source_kind = "raw";
    uint64_t source_bytes = 0;
    std::string plan;                                     // 正規化の副情報（JSON）
    Fidelity fid;
    std::vector<uint8_t> envelope;                        // 元の器を再生成する情報
    // 列が粗い格子に乗っているときの (最小値, 刻み)。値 = base + step * 格納値。
    // 宣言された分解能が実際の分解能とは限らない（8 bit の色を 16 bit 欄に
    // 入れた LAS など）。格子で割ってから符号化すると、その分だけ短くなる。
    std::map<std::string, std::pair<int64_t, int64_t>> coldiv;
    // 幾何の scale/offset は格子に合わせて動かす。掛け戻すときに計算で戻すと
    // 丸めで漂うので、元の値をそのまま覚えておく。
    double coldiv_scale[3] = {1, 1, 1}, coldiv_offset[3] = {0, 0, 0};
    // 極座標格子（geom_repr == "polar" のときだけ意味を持つ）。
    // 列には (距離, 方位角, 仰角) を刻みで割った整数が入る。**非可逆**。
    // polar_base は極座標に直す前の器で、戻すときにどちらの浮動小数で書くかを言う。
    double polar_origin[3] = {0, 0, 0};
    double polar_r_step = 0, polar_ang_step = 0;
    std::string polar_base = "f32bits";
    // 自前の符号器より元の器のほうが短いときに、その中身をそのまま包んで運ぶ。
    // 空でなければ列ストリームは無く、復号はこの中身を書き出して読み直す。
    std::vector<uint8_t> embed;
    std::string embed_kind;

    const Col* get(const std::string& k) const {
        auto it = col.find(k); return it == col.end() ? nullptr : &it->second;
    }
    const ColSpec* spec(const std::string& k) const {
        for (const auto& s : schema) if (s.name == k) return &s;
        return nullptr;
    }
};

// LAS の PointCloud との相互変換（封筒に点形式・ExtraBytes 定義を格納する）
bool frame_from_las(const PointCloud& pc, Frame& f, std::string& err);
// 計画を先に立てて、残す列を pc から移して Frame を作る。
// pc と f が同じ列を同時に持たないので、ピークメモリが約半分になる。
bool frame_from_las_normalized(PointCloud& pc, Frame& f, bool residual_ops,
                               std::string& err);
bool frame_to_las(const Frame& f, PointCloud& pc, std::string& err);

// 拡張子で振り分けて読む（.las/.laz/.bin/.ply）
bool load_frame(const std::string& path, Frame& f, std::string& err, size_t max_points = 0);

// 整数に直して取り込んだ KITTI .bin を、元のビット列に戻して書く。
// 封筒（'FGRD'）が持つ刻み・道順・-0.0 の位置から再生する。
// 負のゼロまで含めてビット完全に戻る。
bool frame_to_kitti_bin(const Frame& f, const std::string& path, std::string& err);
// 浮動小数の器の格子情報（封筒の FGRD）から、名前を挙げた列を外す。
// 非可逆の量子化で幾何の刻みが変わったとき、古い刻みが残らないようにする。
void drop_grid_cols(Frame& f, const std::vector<std::string>& names);

// 整数列が粗い格子に乗っていれば、(最小値, 刻み) を見つけて列を割る。
// 幾何は scale/offset も合わせて動かすので、世界座標は 1 ミリも変わらない。
// 見つけた組は f.coldiv に入る。逆変換は restore_column_grid。
void apply_column_grid(Frame& f);
void restore_column_grid(Frame& f);

// 見つけた格子を人が読める形で返す。無ければ空文字列。
std::string coldiv_summary(const Frame& f);

// 浮動小数の器を整数に直したときの内訳を人が読める形で返す。
// 直していなければ空文字列。封筒に書いた 'FGRD' を読む。
std::string grid_summary(const Frame& f);

// 全列の完全一致を確認する。違えば diff に最初の差異を書く。
bool frames_equal(const Frame& a, const Frame& b, std::string& diff);

// 世界座標（解析用）。geom_repr に応じて復元する。
void frame_world(const Frame& f, std::vector<double>& xyz);

// 先頭 n 点だけを持つ Frame を作る（候補選択を標本で行うときに使う）。
// 注意: 空間予測の利得そのものは部分標本では測れない（既報 2.42% 対 6.07%）。
// これは「どの符号器を選ぶか」にだけ使い、報告する数字は必ず全点で取り直す。
Frame truncate_frame(const Frame& f, size_t n);
// 等間隔に置いた chunks 個の連続塊から合計 n 点を取る。
// 先頭だけを見ると分布を代表しないファイルがある。
Frame sample_frame(const Frame& f, size_t n, int chunks);

// 正規化の計画を Frame に適用する / 復元する。
// 適用後もスキーマは元の全列を保持し、各列の storage だけが変わる。
// これにより復号側は元の列の並びをそのまま取り戻せる。
bool normalize_frame(Frame& f, const PointCloud& pc, std::string& err,
                     bool residual_ops = false);
bool denormalize_frame(Frame& f, std::string& err);

} // namespace pcc
