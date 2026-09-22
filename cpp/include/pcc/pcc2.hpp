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
#include <memory>
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
// 値そのものを、**直前の値**を文脈にして送る。値の種類が少なく系列に型がある列
// （戻り番号のように 1,2,3,1,2 と繰り返す詰め込みビット）に効く。
// 既存の ctx / ctx2 は「直前の差分の桁数」を文脈にするので、この型を掴めない。
inline constexpr uint16_t C_RANGE_PREV = 7;
// 値の種類が少ない列を、**字母をそのまま**直前の記号を文脈にして送る。
// UIntCoder は値を「桁数＋ビット」に分けて送るので、55 種のような小さな字母では
// 条件付きエントロピー（1.53 bit）に対して 2.22 bit かかっていた。
inline constexpr uint16_t C_RANGE_SYM = 8;
// 3 軸を順に送り、**後の軸の文脈に前の軸の差の桁数を使う**。
// 既存の符号器は軸ごとに自分の過去しか見ていないが、実測では
// H(dz) 7.913 → H(dz | dx,dy の桁) 5.036 bit（vegetation）と大きく効く。
inline constexpr uint16_t C_GEOM_CROSS  = 9;
inline constexpr uint16_t C_GEOM_XYZ    = 10;
// y を「その点の dx」から予測する。1 点の中で x は y より先に送られるので、
// 復号側も同じものを持っている（副情報は生じない）。走査線やリングに沿って
// 並んだ入力では進む向きが滑らかに変わるので、dy ≒ dx × (前の dy / 前の dx)。
// 既存の幾何v3 は dx の**桁数**しか文脈に使わず、値としては使っていない。
inline constexpr uint16_t C_GEOM_DIR    = 12;
// 列を**その列の自然な幅**でそのまま格納する。恒等候補の C_RAW64 は 1 値 64 bit
// なので、1 byte の列の天井になっていなかった（fullwave の波形バイトを 8.177 bpp
// と、素の 8.000 より膨らませて送っていた）。param = [幅 byte][符号つきか]。
// **これで「どの列も自然な幅を超えない」が保証される。**
inline constexpr uint16_t C_RAW_W       = 13;
// **外挿でなく内挿する。**他の幾何符号器はどれも格納順に厳密に因果的で、
// 各点を「それまでに復号した点」から外挿する。こちらは粗い層を先に送り、
// 細かい層を**前後の点で挟んで内挿**する（多段の持ち上げ変換）。
// 走査線やリングに沿った入力では内挿の残差が桁違いに小さい
// （AHN3 で −13.0%、AHN4 −11.7%、TLS p1 −10.0%、KITTI −4.9% の見積り）。
// 位置は 2 の冪で決まるので添字表は要らない。param = [段数]。
inline constexpr uint16_t C_GEOM_LIFT   = 14;
// 座標系を回してから符号化する。差分ベクトルの第 1 主成分が 90〜99.9% を占め、
// 主軸は座標軸から 3〜14 度ずれている。軸に揃えると、ずれているぶんが他の軸へ
// 漏れなくなる（AHN3 −5.0%、TLS p1 −2.5% の実測）。
// **回転はせん断 3 回に分解すると整数のまま完全に可逆になる。**
// 復号側で三角関数を計算すると桁が食い違うので、**せん断の係数そのもの**を
// 流れの先頭に格納する（3 平面 × 2 係数 × int32 = 24 byte）。
inline constexpr uint16_t C_GEOM_ROT    = 15;
inline constexpr int      ROT_SH        = 20;   // 係数の固定小数の桁
// 区間ごとに予測子を切り替える。ファイルの途中で性質は変わる（飛行線が変わる、
// 地形が変わる、走査が切れる）。**模型は通しで持ち、予測子だけ切り替える。**
// 選択表は区間ごとに 3 bit しか要らない（蒸留の形）。
// 点ごとに選ぶのは割に合わない（どれが勝ったかを毎点送ることになる）。
inline constexpr uint16_t C_GEOM_SEG    = 16;
inline constexpr uint16_t C_GEOM_SCAN   = 11;  // ALS の走査モデル経由（scanmodel.hpp）  // 幾何 3 軸を同時に
inline constexpr uint16_t C_ATTR_SPATIAL= 20;  // 幾何由来の順序で先行する近傍 P 個から予測
inline constexpr uint16_t C_ATTR_COLOR  = 21;  // 3 列に可逆色変換を掛けてから同上
inline constexpr uint16_t C_ATTR_XREF   = 22;  // 既に復号済みの別の列との残差を符号化
// 符号器 id の最上位ビットを「下位ビットを 1 記号で送る」の旗に使う。
// 流れに書かれるので復号側は迷わない。立っていなければ従来どおり。
inline constexpr uint16_t C_FSYM_BIT     = 0x8000;
// 仮数部の下位を**模型に通さず素通しで**書く旗。一様に近いビットに適応模型を
// 掛けると 1 ビットが 1.0119 ビットに付く（rate=5 での実測）。素通しにすると
// その 1.19% が返る。逆に下位に偏りがある列では模型が勝つので、旗にして
// 実測で選ばせる。上から 1 ビットだけは模型に通す（分布の形はそこに出る）。
inline constexpr uint16_t C_RAW_BIT      = 0x4000;
inline constexpr int      C_RAW_KEEP     = 1;
// 桁長の並びに**照合模型**を掛ける旗。直前 3 つの桁長から合図を作り、
// 同じ合図のときに前回来た桁長を予測として使う。当たり外れを 1 ビットで送り、
// 当たれば桁長そのものは送らない。合図と表は符号側・復号側が同じ履歴から
// 同じものを作るので、**送るものは増えない**。
inline constexpr uint16_t C_MTC_BIT      = 0x2000;
// **文脈を束ねる**旗。文脈ごとの表は、その文脈をまだ数えるほど見ていないうちは
// 当てにならない。見た数が BND_T に届くまでは**全文脈に共通の表**で符号化し、
// 届いてから自分の表に移る。どちらの表も毎回更新するので、移った時点で
// 自分の表は既に温まっている。見た数は両側が同じ履歴から数えるので、
// **送るものは増えない**（値ごとの旗も要らない）。
inline constexpr uint16_t C_BND_BIT      = 0x1000;
// 符号器 id に重ねる旗の全体。id を取り出すときは必ずこれで落とす。
// **文脈に直前の残差の符号を足す**旗（既定。`PCC_LMS_KIND` で中身を替えられる）。
// 我々の文脈は `ctx_of(zigzag(残差))`＝残差の桁数で、**符号を捨てている**。
// JPEG-LS が条件にしているのは符号つきの勾配である。符号は両側が同じ復号済みの
// 残差から作るので、**送るバイトは 1 つも増えない**（長さの指定は既存の符号器 id の
// 空きビットに乗る）。効くのは**実装した 3 経路**だけ:
//   enc_geom_x（幾何v3・回転）/ enc_geom_w（幾何v4）/ enc_resid_x（走査 var 3・4）。
// `PCC_LMS_KIND=1` で適応フィルタ、`=0` で文脈ごとの偏り補正に切り替わる
// （どちらも測って外した。losses.md §22）。
inline constexpr uint16_t C_LMS_BIT      = 0x0800;
// 符号の履歴を何個取るかを、もう 1 ビットで言う。C_LMS_BIT と組で
//   0 = 切、1 = 直前 1 個、2 = 直前 2 個、3 = 直前 4 個
// 最適な長さはファイルで違う（USGS NY は 2 個、red-rocks は 4 個）ので実測で選ぶ。
inline constexpr uint16_t C_SGN2_BIT     = 0x0400;
// **パルス構造**の旗（名前は「光」）。LiDAR の 1 パルスが返す複数の戻りは、
// 物理的に 1 本の直線（光線）の上に乗り、隣り合うパルスの戻り間ベクトルは
// 実測で cos 1.00000 と厳密に平行である。ただし**その平行性を予測に使っても
// 効かなかった**（実際の勝者が続きの点を既に同じくらい送っている）。
// 効いたのは構造のほうで、「同じパルスの続き」の点を
//   ・予測子の**履歴に入れない**（戻り間の跳びで傾きが汚れるのを防ぐ）
//   ・**別の文脈**で送る
// の 2 つ。続きかどうかは `bit_fields`（戻り番号と戻り総数）から分かり、
// bit_fields は**幾何より先に復号される**ので、送るものは増えない。
inline constexpr uint16_t C_RAY_BIT      = 0x0200;
// **曲面による z の予測**（2 ビットの組で 0 = 切・1/2/3 = 格子 2^6 / 2^7 / 2^8）。
// 幾何は x → y → z の順に送るので、z を送る時点でその点の (x, y) は復号側も
// 知っている。そこで z を「(x, y) の近くにある復号済みの点」から予測できる:
//   いまの予測子 / (x, y) の最近傍の z / 近傍 9 マスの代表点に当てた平面
// の 3 つを持ち、**直近の減衰誤差が最も小さいもの**を使う（復号済みの値だけから
// 決まるので副情報は要らない）。単独ではどれも いまの予測子 に負けるが、
// 点ごとに当たる所が違う。平面は**整数演算で厳密に**解く（浮動小数だと別の
// 機械で復号がずれうる）。効くのは enc_geom_w（幾何v4）だけ。
inline constexpr uint16_t C_SURF_A       = 0x0100;
inline constexpr uint16_t C_SURF_B       = 0x0080;
inline constexpr int      LMS_ORD        = 16;   // 履歴の長さ
inline constexpr int      LMS_SH         = 12;   // 重みの固定小数の桁

inline constexpr uint16_t C_FLAG_MASK =
    (uint16_t)(C_FSYM_BIT | C_RAW_BIT | C_MTC_BIT | C_BND_BIT |
               C_LMS_BIT | C_SGN2_BIT | C_RAY_BIT | C_SURF_A | C_SURF_B);

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
    // 走査モデルの文脈（走査順・戻り番号・掃引の切れ目）は、**変種によらず同じ**で
    // ある。候補ごとに作り直すと、200 万点の安定ソートを走査候補の本数だけ繰り返し、
    // 同じ配列を同時に何本も抱えることになる。列名と点数で引けるようにして共有する。
    // 中身は pcc2.cpp にしかないので、ここでは持ち主だけを置く。
    mutable std::shared_ptr<void> scan_cache;
    // 字母符号器の字母は**候補によらず列だけで決まる**。候補ごとに作り直すと、
    // 200 万点の集合への挿入を候補の数だけ繰り返すことになる（実測で符号化が
    // 1.7 倍になった）。列ごとに一度だけ作って使い回す。
    mutable std::shared_ptr<void> sym_cache;
    // 符号化は P=1/3/5 を全部試すので全部要る。**復号は選ばれた P しか通らない。**
    // 復号器は流れの一覧を先に読めるので、そこに出てくる P だけをここに入れる。
    // 近傍探索の k は max(P)+4 なので、P=3 だけの回では 9 が 7 に下がる。
    std::vector<int> want_Ps;
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

// 勝者に旗（生・符号・光線・束ね・曲面）を重ねた版を測り、短ければ置き換える。
// best_stream は内部で呼ぶ。標本で選んだ符号器を全点で測り直した後にも呼ぶこと。
void apply_post_flags(Stream& s, const std::vector<const Col*>& cv, const CodecCtx* ctx,
                      std::string* trace);

// 単一符号器の符号化・復号（cols は同じ長さの列）
bool codec_encode(uint16_t id, const std::vector<const Col*>& cols,
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
