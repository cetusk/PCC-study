// PCC2 — 自前符号器のコンテナ。設計は notes/02_codec_container.md を参照。
//
// PCC1 との違い:
//   * 完全に自己記述的（復号に点数やスキーマを外から渡さない）
//   * 幾何も属性と同じ「列」として扱い、符号器を差し替えられる
//   * 精度の申告と副情報を器が必ず持ち、ベンチで勘定に入る
//   * 恒等符号器 RAW64 を必ず候補に含めるので、器の長さは有限で閉じる
#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <string>
#include <vector>
#include "pcc/frame.hpp"

namespace pcc {

// 符号器 id
inline constexpr uint16_t C_RAW64       = 0;   // 恒等候補
inline constexpr uint16_t C_RANGE       = 1;   // zigzag + レンジ符号
inline constexpr uint16_t C_RANGE_DELTA = 2;   // 1 次差分 + レンジ符号
inline constexpr uint16_t C_RANGE_CTX   = 3;   // 1 次差分 + 直前値のビット数を文脈に
inline constexpr uint16_t C_RANGE_CTX2  = 4;   // 2 次差分＋文脈（一定刻みの列に効く）
inline constexpr uint16_t C_RANGE_MED   = 5;   // 直近 3 差分の中央値を予測子に＋文脈
inline constexpr uint16_t C_RANGE_CACHE = 6;   // 直近 K 個の相異なる差分の表を指す
inline constexpr uint16_t C_GEOM_XYZ    = 10;
inline constexpr uint16_t C_GEOM_SCAN   = 11;  // ALS の走査モデル経由（scanmodel.hpp）  // 幾何 3 軸を同時に
inline constexpr uint16_t C_ATTR_SPATIAL= 20;  // 幾何由来の順序で先行する近傍 P 個から予測
inline constexpr uint16_t C_ATTR_COLOR  = 21;  // 3 列に可逆色変換を掛けてから同上
inline constexpr uint16_t C_ATTR_XREF   = 22;  // 既に復号済みの別の列との残差を符号化
// 符号器 id の最上位ビットを「下位ビットを 1 記号で送る」の旗に使う。
// 流れに書かれるので復号側は迷わない。立っていなければ従来どおり。
inline constexpr uint16_t C_FSYM_BIT     = 0x8000;

// 符号器が使ってよい副次情報。幾何は属性より先に復号されるので、
// 属性の符号化時には座標が揃っている（命題: 副情報の不要性）。
struct Frame;
struct CodecCtx {
    // 座標の実数表現 n*3。空間予測にしか使わないのに常に作ると、100 万点で
    // 24 MB が無駄になる（幾何だけを符号化するときは一度も参照されない）。
    // want_world を立てておけば、最初に必要になった時点で fr から作る。
    const std::vector<double>* world = nullptr;
    bool want_world = false;                      // 遅延構築を許すか
    mutable std::vector<double> world_own;        // 遅延構築したときの実体
    const Frame* fr = nullptr;                    // 既に復号済みの列を引くため
    // 標本で順位を付けるとき、候補の**集合**まで標本から決めると、全点での
    // 勝者が候補に入らないことがある（AHN3 の nir が実際にそうだった）。
    // 参照列の選定だけは全点の統計で行うため、全点の Frame をここに置く。
    const Frame* full = nullptr;
    // 近傍の予測子表は P ごとに違う。候補ごとに作り直すと、1 つの列で
    // sp(P=1) / sp(P=3) / sp(P=5) …と何度も建て直すことになる
    // （AHN3 _20 の 14 列では 30 回を超えていた）。
    // P ごとに取っておく。表は n*P の int32 なので 100 万点・P=5 で 20 MB。
    // 点数でも分けて持つ。標本（版を選ぶための 2.5 万点）と全点が交互に来るので、
    // 片方しか持たないと毎回作り直しになる（実測で全体が 2 倍になった）。
    mutable std::map<size_t, std::vector<int32_t>> perm_by_n;
    mutable std::map<std::pair<size_t, int>, std::vector<int32_t>> pred_by_np;
    mutable std::mutex mu;        // 候補を並列に符号化するときのため
    // 順序表と、P に対応する予測子表を返す。作っていなければ作る。
    // 予測子表を返し、perm_out に順序表を入れる（どちらも点数ごとに持つ）。
    const std::vector<int32_t>* ensure(size_t n, int P,
                                       const std::vector<int32_t>** perm_out) const;
    const std::vector<double>* world_ptr() const; // 無ければ作る
    // 報告する構成を往復検証に通すための指定。空なら通常どおり全候補を実測して選ぶ。
    std::string force_geom;                       // X+Y+Z をこの候補名に固定する
    bool fast_attr = false;                       // 属性列の候補を絞って時間を詰める
};

// 候補（符号器とそのパラメタ）
struct Cand { uint16_t codec; std::vector<uint8_t> param; };

struct Stream {
    std::vector<std::string> cols;
    uint16_t codec = C_RAW64;
    std::vector<uint8_t> param;
    std::vector<uint8_t> data;
    // 標本で順位を付けたときの次点。標本の 1 位が全点でも 1 位とは限らないので、
    // 上位だけを全点で測り直して短い方を採るために使う。data には入らない。
    std::vector<Cand> alt;
};

// 単一符号器の符号化・復号（cols は同じ長さの列）
bool codec_encode(uint16_t id, const std::vector<const std::vector<int64_t>*>& cols,
                  const std::vector<uint8_t>& param, std::vector<uint8_t>& out,
                  std::string& err, const CodecCtx* ctx = nullptr);
bool codec_decode(uint16_t id, const std::vector<uint8_t>& param,
                  const uint8_t* data, size_t len, size_t n, size_t ncol,
                  std::vector<std::vector<int64_t>>& out, std::string& err,
                  const CodecCtx* ctx = nullptr);

// 候補を全部実際に符号化して、最も短いものを返す（推定は使わない）
// preselect: 候補が多いとき、標本で順位を付けてから上位だけを全点で測る。
// 幾何では bpp が完全に不変のまま 2 倍速くなったが、属性では標本の誤順位で
// USGS NY が +3.7% 悪化した。サイズが第一なので、幾何だけで使う。
Stream best_stream(const Frame& f, const std::vector<std::string>& cols,
                   const std::vector<Cand>& candidates, const CodecCtx* ctx = nullptr,
                   std::string* trace = nullptr, bool preselect = false);
std::string cand_name(uint16_t codec, const std::vector<uint8_t>& param);

// 既定の計画: 幾何 3 列を 1 ストリームに、属性は列ごとに、それぞれ実測で選ぶ
std::vector<Stream> plan_streams(const Frame& f, bool joint_geom, std::string* log = nullptr,
                                 const CodecCtx* ctx = nullptr, bool trace_all = false);

bool write_pcc2(const std::string& path, const Frame& f, const std::vector<Stream>& st,
                uint64_t& bytes_out, std::string& err);
bool read_pcc2(const std::string& path, Frame& f, std::string& err);

uint64_t crc64(const uint8_t* p, size_t n);

} // namespace pcc
