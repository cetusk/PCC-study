// ALS の走査モデル — 掃引内の走査角は時刻の 1 次関数である。
//
// 測定の裏付けは results/als_scan_structure.md、設計は notes/03_scan_model_codec.md。
//   * 掃引の点は厚さ 3.6 mm の鉛直面に乗る
//   * 幅 278 m の掃引を 4 パラメタで 1.4 cm まで再現できる
//   * 発射番号は gps_time から決まるので、角度 2 自由度は副情報を生まない
//
// 可逆・決定論のため三角関数は復号側で呼ばない。
//   * 回転は 3 回のせん断（各段が Z^2 上の全単射）
//   * 正接は整数 CORDIC で作った固定小数の表（誤差 1200 m 先で 0.051 mm）
#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace pcc {

inline constexpr int SCAN_SH = 30;       // せん断定数の固定小数
inline constexpr int SCAN_TAN_SH = 30;   // 正接の固定小数
// 角度は [-45, +45] 度を [-2^31, +2^31] に写した整数で持つ
inline constexpr int64_t SCAN_ANG_MAX = (int64_t)1 << 31;

// 決定論的な正接。整数のみ。
int64_t tan_fx(int64_t theta_fx);

// 3 回のせん断による整数回転。(t2, sn) = (tan(-θ/2), sin(-θ)) を 2^SCAN_SH で。
inline void shear_fwd(int64_t X, int64_t Y, int64_t t2, int64_t sn,
                      int64_t& s, int64_t& off) {
    int64_t x = X - ((t2 * Y) >> SCAN_SH);
    int64_t y = Y + ((sn * x) >> SCAN_SH);
    s = x - ((t2 * y) >> SCAN_SH);
    off = y;
}
inline void shear_inv(int64_t s, int64_t off, int64_t t2, int64_t sn,
                      int64_t& X, int64_t& Y) {
    int64_t x1 = s + ((t2 * off) >> SCAN_SH);
    Y = off - ((sn * x1) >> SCAN_SH);
    X = x1 + ((t2 * Y) >> SCAN_SH);
}

// 1 掃引ぶんのモデル。すべて整数で、そのまま副情報に書ける。
struct SweepParam {
    int64_t t2 = 0, sn = 0;      // 掃引面への回転（せん断定数）
    int64_t s0 = 0;              // 面内のセンサ位置
    int64_t Sz = 0;              // センサ高度
    int64_t th0 = 0;             // 掃引の先頭での走査角
    int64_t om = 0;              // 発射あたりの角度変化
    int64_t off0 = 0;            // 面外の基準
    int64_t first = 0;           // 符号化順での最初の点
    int64_t count = 0;
    uint8_t ok = 0;              // モデルを採用するか（しない掃引は幾何v3 に落とす）
};

// 掃引内の位置 shot（掃引先頭からの ulp 数）での面内座標の予測
inline int64_t scan_predict(const SweepParam& p, int64_t shot, int64_t z) {
    int64_t th = p.th0 + p.om * shot;
    if (th > SCAN_ANG_MAX - 1) th = SCAN_ANG_MAX - 1;
    if (th < -SCAN_ANG_MAX) th = -SCAN_ANG_MAX;
    return p.s0 + (((p.Sz - z) * tan_fx(th)) >> SCAN_TAN_SH);
}

// ---- 以下は符号化器だけが使う（復号器は送られた整数パラメタしか見ない）。
//      当てはめには浮動小数を使ってよい。決定性が要るのは予測の計算だけである。

// 走査線 1 本の点から SweepParam を作る。
//   xy から鉛直面への回転（t2, sn）を決め、面内座標 s に落としてから
//   s = s0 + (Sz - z) tan(th0 + om * shot) を当てはめる。
//   sa は記録されている走査角（0.006 度単位）。初期値にのみ使う。
//   当てはめが素朴な差分に勝たなければ ok=0 を返す。
SweepParam fit_scan_line(const int64_t* X, const int64_t* Y, const int64_t* Z,
                         const int64_t* gps, const int64_t* sa, size_t n);

// 1 掃引に 2 台のスキャナの走査線が混ざっているとき、EM で 2 群に分ける。
// 戻り値は 0/1 のラベル。分けても薄くならなければ全部 0 を返す。
void split_two_lines(const int64_t* X, const int64_t* Y, size_t n,
                     std::vector<uint8_t>& label);

// 鉛直面への当てはまりの薄さ（点の広がりの短軸 / sqrt(n)）
double line_thinness(const int64_t* X, const int64_t* Y, size_t n);

} // namespace pcc
