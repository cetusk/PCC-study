// Python 実装との一致を確かめる。
// 標準入力から int64 の列を読み、符号化した結果を標準出力へ出す。
// Python 側が同じ入力から作ったバイト列と **完全に一致** しなければ移植は失敗。
#include "pcc/rangecoder.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <cinttypes>

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "encode";
    // 入力: [int64 n][int64 v0][int64 v1]...
    int64_t n = 0;
    if (fread(&n, sizeof(int64_t), 1, stdin) != 1) { fprintf(stderr, "no input\n"); return 1; }
    std::vector<int64_t> v(static_cast<size_t>(n));
    if (n && fread(v.data(), sizeof(int64_t), static_cast<size_t>(n), stdin) != (size_t)n) {
        fprintf(stderr, "short input\n"); return 1;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto blob = pcc::encode_ints(v.data(), v.size());
    auto t1 = std::chrono::steady_clock::now();
    std::vector<int64_t> back(v.size());
    pcc::decode_ints(blob.data(), blob.size(), back.data(), back.size());
    auto t2 = std::chrono::steady_clock::now();
    bool ok = (back == v);
    double enc_s = std::chrono::duration<double>(t1 - t0).count();
    double dec_s = std::chrono::duration<double>(t2 - t1).count();
    if (mode == "bench") {
        fprintf(stderr, "n=%" PRId64 " bytes=%zu enc=%.3fs (%.0f k/s) dec=%.3fs (%.0f k/s) roundtrip=%s\n",
                n, blob.size(), enc_s, n / enc_s / 1000.0, dec_s, n / dec_s / 1000.0,
                ok ? "OK" : "FAIL");
        return ok ? 0 : 2;
    }
    fwrite(blob.data(), 1, blob.size(), stdout);
    fprintf(stderr, "roundtrip=%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 2;
}
