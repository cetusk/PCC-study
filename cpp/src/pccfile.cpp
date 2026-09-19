#include "pcc/pccfile.hpp"
#include "pcc/rangecoder.hpp"
#include <cstdio>
#include <cstring>

namespace pcc {

template <class T> static void put(std::vector<uint8_t>& o, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    o.insert(o.end(), p, p + sizeof(T));
}

bool write_container(const std::string& path, const std::string& spec,
                     const std::map<std::string, std::vector<int64_t>>& streams,
                     uint64_t& bytes_out, std::string& err, bool delta) {
    std::vector<uint8_t> out;
    out.insert(out.end(), {'P', 'C', 'C', '1'});
    put<uint32_t>(out, (uint32_t)spec.size());
    out.insert(out.end(), spec.begin(), spec.end());
    for (const auto& kv : streams) {
        // 差分を取ってから符号化する（先頭要素を失わないよう prepend 0）
        bool nd = kv.first.rfind("sp:", 0) == 0;
        std::vector<int64_t> d(kv.second.size());
        if (delta && !nd) {
            int64_t prev = 0;
            for (size_t i = 0; i < kv.second.size(); ++i) { d[i] = kv.second[i] - prev; prev = kv.second[i]; }
        } else d = kv.second;
        auto blob = encode_ints(d.data(), d.size());
        put<uint16_t>(out, (uint16_t)kv.first.size());
        out.insert(out.end(), kv.first.begin(), kv.first.end());
        put<uint8_t>(out, nd ? M_RANGE_NODELTA : M_RANGE);
        put<uint32_t>(out, (uint32_t)blob.size());
        out.insert(out.end(), blob.begin(), blob.end());
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { err = "書き込めない: " + path; return false; }
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    bytes_out = out.size();
    return true;
}

bool read_container(const std::string& path, std::string& spec,
                    std::map<std::string, std::vector<int64_t>>& streams,
                    size_t n, std::string& err, bool delta) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "開けない: " + path; return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { fclose(f); err = "短い"; return false; }
    fclose(f);
    if (sz < 8 || memcmp(buf.data(), "PCC1", 4)) { err = "PCC1 ではない"; return false; }
    size_t p = 4;
    uint32_t sl; memcpy(&sl, buf.data() + p, 4); p += 4;
    spec.assign((char*)buf.data() + p, sl); p += sl;
    while (p + 7 <= (size_t)sz) {
        uint16_t nl; memcpy(&nl, buf.data() + p, 2); p += 2;
        std::string nm((char*)buf.data() + p, nl); p += nl;
        uint8_t m = buf[p]; p += 1;
        uint32_t bl; memcpy(&bl, buf.data() + p, 4); p += 4;
        if (m != M_RANGE && m != M_RANGE_NODELTA) { err = "未知の符号化方式"; return false; }
        std::vector<int64_t> d(n);
        decode_ints(buf.data() + p, bl, d.data(), n);
        p += bl;
        if (delta && m == M_RANGE) { int64_t acc = 0; for (size_t i = 0; i < n; ++i) { acc += d[i]; d[i] = acc; } }
        streams[nm] = std::move(d);
    }
    return true;
}

} // namespace pcc
