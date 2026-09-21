#pragma once
// 列を「その列に要る幅」で持つ。
//
// 以前は全部 int64 だった。intensity が u16、classification が u8 でも 1 点
// 8 byte を使う。200 万点・17 列で 272 MB になり、復号のピークの 46% を
// 占めていた。値の最小値を base に出し、差を必要な byte 数だけ詰める。
//
// **読み出しに分岐を入れない。**末尾に 8 byte の余白を置いて必ず 8 byte 読み、
// 幅ぶんだけ mask で残す。幅が 0（定数列）のときは添字を掛けても 0 になり、
// 余白の 0 を読んで base が返る。分岐も特別扱いも要らない。
#include <cstdint>
#include <cstring>
#include <vector>

namespace pcc {

struct Col {
    int64_t base = 0;              // 値 = base + （詰めた符号なし整数）
    uint64_t mask = 0;             // 幅ぶんのビットだけ残す
    uint8_t w = 0;                 // 1 点あたりの byte 数。0 は定数列
    std::vector<uint8_t> b{8, 0};  // 実体。末尾に 8 byte の余白

    inline int64_t operator[](size_t i) const {
        uint64_t v;
        std::memcpy(&v, b.data() + i * (size_t)w, 8);
        return base + (int64_t)(v & mask);
    }
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

    // 値の並びから幅を決めて詰める。
    void from(const std::vector<int64_t>& v) {
        n_ = v.size();
        if (n_ == 0) { set_const(0, 0); return; }
        int64_t mn = v[0], mx = v[0];
        for (size_t i = 1; i < n_; ++i) {
            if (v[i] < mn) mn = v[i];
            if (v[i] > mx) mx = v[i];
        }
        // **符号なしで引く。**int64 のまま引くと桁あふれする範囲がある
        // （gps_time は実数のビット列をそのまま持つので端から端まで使う）。
        uint64_t range = (uint64_t)mx - (uint64_t)mn;
        if (range == 0) { set_const(mn, n_); return; }
        if (range <= 0xFFull)            pack(v, mn, 1, 0xFFull);
        else if (range <= 0xFFFFull)     pack(v, mn, 2, 0xFFFFull);
        else if (range <= 0xFFFFFFFFull) pack(v, mn, 4, 0xFFFFFFFFull);
        else                             pack_wide(v);
    }

    void from_const(int64_t value, size_t n) { set_const(value, n); }

    Col() = default;
    // 値の並びから作れるようにする。診断用の経路が素の並びを渡してくる。
    Col(const std::vector<int64_t>& v) { from(v); }

    // 値の並びをそのまま代入できるようにする。呼ぶ側は幅を意識しない。
    Col& operator=(const std::vector<int64_t>& v) { from(v); return *this; }
    Col& operator=(std::vector<int64_t>&& v) {
        from(v);
        std::vector<int64_t>().swap(v);   // 元は要らない。詰めた直後に手放す
        return *this;
    }

    // 値の並びに戻す。幅を意識したくない所のため。
    std::vector<int64_t> to_vector() const {
        std::vector<int64_t> v(n_);
        for (size_t i = 0; i < n_; ++i) v[i] = (*this)[i];
        return v;
    }

    // 詰めたあとの実際の占有 byte 数。測るときに使う。
    size_t bytes() const { return b.capacity(); }

private:
    size_t n_ = 0;

    void set_const(int64_t value, size_t n) {
        base = value; mask = 0; w = 0; n_ = n;
        std::vector<uint8_t>(8, 0).swap(b);
    }
    void pack(const std::vector<int64_t>& v, int64_t mn, uint8_t width, uint64_t m) {
        base = mn; mask = m; w = width;
        std::vector<uint8_t>(n_ * width + 8, 0).swap(b);
        for (size_t i = 0; i < n_; ++i) {
            uint64_t d = (uint64_t)v[i] - (uint64_t)mn;
            std::memcpy(b.data() + i * width, &d, width);
        }
    }
    void pack_wide(const std::vector<int64_t>& v) {
        base = 0; mask = ~0ull; w = 8;
        std::vector<uint8_t>(n_ * 8 + 8, 0).swap(b);
        std::memcpy(b.data(), v.data(), n_ * 8);
    }
};

} // namespace pcc
