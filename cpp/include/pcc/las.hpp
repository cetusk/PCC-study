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
    // LAS の ExtraBytes には配列型（data_type 11-30）と「未定義バイト列」
    // （data_type 0）がある。値として解釈できないので、**生のバイト列として運ぶ**。
    // nbytes > 0 のとき、この次元は nbytes 個の U8 の列（name#0..）になる。
    // 元の VLR をそのまま作り直せるように data_type と options も持つ。
    int nbytes = 0;
    uint8_t raw_dt = 0, raw_opts = 0;
    bool has_scale = false, has_offset = false;
    double scale[3] = {1, 1, 1}, offset[3] = {0, 0, 0};
    std::string description;
    int byte_offset = 0;          // 点レコード内の extra_bytes 先頭からの位置
    // ExtraBytes の VLR に**記述されていない**余りのバイト（点レコードが基本の大きさ
    // ＋記述された次元より長いときの残り）。VLR には書かず、点レコードにだけ書き戻す。
    // 以前は 0 で書いていた（中身が落ちていた）。
    bool undocumented = false;
    // **列の名前**（pc.fields の鍵）。ふつうは name と同じだが、name が標準の列名
    // （X・intensity など）や別の ExtraBytes の名前とぶつかる、または空のときは
    // 一意な名前にする。以前は name をそのまま鍵にしていたので、ぶつかると同じ列を
    // 上書きして片方が黙って消えた（名前が X の ExtraBytes で幾何 X が置き換わった）。
    // 器には書かない。読み込みと封筒の復元で assign_extra_cols が同じ規則で作り直す。
    std::string col;
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
    // 元のファイルの VLR をそのまま持つ（測地系・WKT など）。
    // LASzip 自身が作る "laszip encoded" と、こちらが作り直す ExtraBytes は除く。
    struct RawVlr { std::string user_id, description; uint16_t record_id = 0;
                    std::vector<uint8_t> data; };
    std::vector<RawVlr> vlrs;
    // EVLR（LAS 1.4 の拡張 VLR）。点データの後ろに置かれ、位置はヘッダが持つ。
    // **laszip の C API は EVLR を配列で出さない**ので、ファイルを直に読み書きする。
    std::vector<RawVlr> evlrs;
    // 元のファイルのヘッダの欄。落とすと器が変わる
    // （global_encoding は LAS 1.4 の点形式 6 以上で WKT の有無を言う必須ビット）。
    uint16_t file_source_id = 0, global_encoding = 0;
    uint16_t creation_day = 0, creation_year = 0;
    std::string system_identifier, generating_software;
    // ExtraBytes の VLR は 192 byte のレコードの並び。作り直すと scale / no_data /
    // min / max / reserved が既定値になるので、元のものをそのまま持つ。
    std::vector<uint8_t> extra_vlr;
    std::string extra_vlr_desc;
    // ExtraBytes の VLR が元の並びで何番目にあったか（それより前にある
    // 保存対象の VLR の数）。書き戻すときにこの位置へ入れる。以前は常に先頭に
    // 書いていたので、AHN4 のように LASF_Spec 7 の後ろにあるファイルで
    // VLR の並びが入れ替わっていた（中身のバイトは一致していた）。
    uint32_t extra_vlr_pos = 0;
    // **中身から決まるヘッダの欄**（範囲・戻り番号ごとの点数・旧形式の点数・GUID・
    // ユーザーデータ・LAS 1.3 の波形データの位置）。以前は書き戻すときに全部 0 に
    // していた。書き戻すときは中身から数え直し（recount_header）、元のヘッダが
    // 数え直した値と食い違っていた欄（hdr_mask）だけ元の値で上書きする。
    // 封筒に載るのも食い違った欄だけなので、大半のファイルでは 1 byte も増えない。
    bool hdr_known = false;
    uint64_t hdr_n = 0;                    // 読んだときのファイル全体の点数
    // どの欄を元の値で上書きするか（数え直した値と食い違う欄だけ立てる）。
    //   bit 0-5 範囲（max_x, min_x, max_y, min_y, max_z, min_z）/ 6 旧形式の点数 /
    //   7 戻り番号ごとの点数（旧）/ 8 同（拡張）/ 9 GUID / 10 波形データの位置 /
    //   11 拡張の点数
    uint16_t hdr_mask = 0;
    uint64_t hdr_ext_n = 0;
    double hdr_minmax[6] = {0, 0, 0, 0, 0, 0};   // max_x, min_x, max_y, min_y, max_z, min_z
    uint32_t hdr_legacy_n = 0;
    uint32_t hdr_by_ret[5] = {0, 0, 0, 0, 0};
    uint64_t hdr_ext_by_ret[15] = {0};
    uint8_t hdr_guid[16] = {0};
    uint64_t hdr_waveform_start = 0;
    std::vector<uint8_t> user_in_header, user_after_header;
    std::string src_path;
    uint64_t src_bytes = 0;
};

// ExtraBytes の列の名前（ExtraDim::col）を決める。標準の列名・先に決めた列名と
// ぶつかるものと空の名前は "eb<番号>:<元の名前>" にする。
void assign_extra_cols(PointCloud& pc);

// 点の中身から、範囲・旧形式の点数・戻り番号ごとの点数を数え直す。
// 読み込み（元のヘッダと比べて hdr_mask を決める）と書き出し（数え直した値を書く）で
// **同じ関数**を使うので、食い違わない限り元のヘッダとビット一致する。
void recount_header(const PointCloud& pc, double mm[6], uint32_t& legacy_n,
                    uint32_t by_ret[5], uint64_t ext_by_ret[15], uint64_t& ext_n);
// 元のヘッダの値と数え直した値を比べて hdr_mask を決める（全点を読んだときだけ）。
void decide_header_mask(PointCloud& pc);

bool read_las(const std::string& path, PointCloud& pc, std::string& err,
              size_t max_points = 0);

// 読み込んだ 2 つの点群が、器として同じか（点の全次元・ExtraBytes・VLR/EVLR・
// ヘッダの欄・ユーザーデータ）。counts が偽なら範囲と点数の欄は比べない
// （途中で切って読んだものは元のヘッダと合わないため）。
bool pointclouds_equal(const PointCloud& a, const PointCloud& b, bool counts,
                       std::string& diff);

// keep に無い ExtraBytes 次元は VLR ごと落とす。
// keep に無い標準次元は 0 で埋める（LASzip は定数列をほぼ無コストで潰す）。
bool write_las(const std::string& path, const PointCloud& pc,
               const std::vector<std::string>& keep, std::string& err,
               const std::vector<int32_t>* Xq = nullptr,
               const std::vector<int32_t>* Yq = nullptr,
               const std::vector<int32_t>* Zq = nullptr,
               const double* scale_override = nullptr);

// 解析系サブコマンド用: LAS/LAZ・PLY・KITTI .bin から座標だけ読む。
// src_bytes を渡すと、軸ごとに**元のファイルでの浮動小数の幅**（4 か 8）を返す。
// 4 なら値は float32 の全体であり、その幅で格子を調べないと証明が通らない
// （double に広げてしまうと、元のビット列を再現する条件が変わる）。
// ascii の PLY は文字列を strtod で読むので、宣言が float でも 8 を返す。
bool read_points_any(const std::string& path, std::vector<double>& xyz,
                     size_t& n, std::string& err, size_t max_points = 0,
                     int* src_bytes = nullptr);

} // namespace pcc
