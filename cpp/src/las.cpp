#include "pcc/las.hpp"
#include <set>
#include <laszip/laszip_api.h>
#include <cstring>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <sys/stat.h>
#include <algorithm>

namespace pcc {

int ftype_size(FType t) {
    switch (t) {
        case FType::U8: case FType::I8: return 1;
        case FType::U16: case FType::I16: return 2;
        case FType::U32: case FType::I32: case FType::F32: return 4;
        default: return 8;
    }
}

static int64_t read_raw(const uint8_t* p, FType t) {
    switch (t) {
        case FType::U8:  { uint8_t  v; memcpy(&v, p, 1); return v; }
        case FType::I8:  { int8_t   v; memcpy(&v, p, 1); return v; }
        case FType::U16: { uint16_t v; memcpy(&v, p, 2); return v; }
        case FType::I16: { int16_t  v; memcpy(&v, p, 2); return v; }
        case FType::U32: { uint32_t v; memcpy(&v, p, 4); return v; }
        case FType::I32: { int32_t  v; memcpy(&v, p, 4); return v; }
        case FType::F32: { uint32_t v; memcpy(&v, p, 4); return v; }   // ビットのまま
        case FType::U64: { uint64_t v; memcpy(&v, p, 8); return (int64_t)v; }
        case FType::I64: { int64_t  v; memcpy(&v, p, 8); return v; }
        case FType::F64: { uint64_t v; memcpy(&v, p, 8); return (int64_t)v; }
    }
    return 0;
}

static void write_raw(uint8_t* p, FType t, int64_t v) {
    switch (t) {
        case FType::U8:  { uint8_t  x = (uint8_t)v;  memcpy(p, &x, 1); break; }
        case FType::I8:  { int8_t   x = (int8_t)v;   memcpy(p, &x, 1); break; }
        case FType::U16: { uint16_t x = (uint16_t)v; memcpy(p, &x, 2); break; }
        case FType::I16: { int16_t  x = (int16_t)v;  memcpy(p, &x, 2); break; }
        case FType::U32: { uint32_t x = (uint32_t)v; memcpy(p, &x, 4); break; }
        case FType::I32: { int32_t  x = (int32_t)v;  memcpy(p, &x, 4); break; }
        case FType::F32: { uint32_t x = (uint32_t)v; memcpy(p, &x, 4); break; }
        default:         { uint64_t x = (uint64_t)v; memcpy(p, &x, 8); break; }
    }
}

// LAS の ExtraBytes の data_type からバイト数を出す。
// 0 は「未定義（options の size 指定）」、11-30 は 2要素/3要素の配列型（非推奨）。
// 扱えない型でもサイズだけは正しく進めなければならない。
// 進めないと後続フィールドの読み出し位置が全部ずれる。
static int eb_type_size(uint8_t dt) {
    static const int base[11] = {0, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8};
    if (dt <= 10) return base[dt];
    // dt 11-20 は型 1-10 の 2 要素配列、21-30 は 3 要素配列。
    // 添字は dt-10 / dt-20 である。(dt-11)/2+1 は dt=11,12 でしか合わず、
    // dt=23（ushort[3]）を 3 byte と数えて後続の次元を全部ずらしていた。
    if (dt <= 20) return base[dt - 10] * 2;
    if (dt <= 30) return base[dt - 20] * 3;
    return 0;
}

// ExtraBytes VLR（LASF_Spec / 4）は 192 バイトのレコードの並び
static void parse_extra_vlr(const uint8_t* data, size_t len, std::vector<ExtraDim>& out) {
    const size_t REC = 192;
    int off = 0;
    for (size_t i = 0; i + REC <= len; i += REC) {
        const uint8_t* r = data + i;
        ExtraDim e;
        uint8_t dt = r[2];
        uint8_t opts = r[3];
        if (dt < 1 || dt > 10) {
            // 値として解釈できない型（配列・未定義バイト列）。
            // **捨てずに生のバイト列として運ぶ。**捨てると往復で戻らない
            // （extra.laz の Colors / Reserved / Flags がそうだった）。
            // dt=0 は options がサイズ（未定義バイト列）。
            int sz = (dt == 0) ? (int)r[3] : eb_type_size(dt);
            e.type = FType::U8;
            e.nbytes = sz;
            e.raw_dt = dt; e.raw_opts = opts;
            char nm0[33] = {0}; memcpy(nm0, r + 4, 32);
            e.name = nm0;
            char ds0[33] = {0}; memcpy(ds0, r + 4 + 32 + 4 + 24 * 5, 32);
            e.description = ds0;
            e.byte_offset = off;
            off += sz;
            out.push_back(e);
            continue;
        }
        e.type = static_cast<FType>(dt);
        char nm[33] = {0}; memcpy(nm, r + 4, 32);
        e.name = nm;
        e.has_scale  = (opts & 0x08) != 0;
        e.has_offset = (opts & 0x10) != 0;
        memcpy(e.scale,  r + 4 + 32 + 4 + 24 * 3, 24);
        memcpy(e.offset, r + 4 + 32 + 4 + 24 * 4, 24);
        char ds[33] = {0}; memcpy(ds, r + 4 + 32 + 4 + 24 * 5, 32);
        e.description = ds;
        e.byte_offset = off;
        off += ftype_size(e.type);
        out.push_back(e);
    }
}

static void build_extra_vlr(const std::vector<ExtraDim>& dims, std::vector<uint8_t>& out) {
    const size_t REC = 192;
    out.assign(dims.size() * REC, 0);
    for (size_t i = 0; i < dims.size(); ++i) {
        uint8_t* r = out.data() + i * REC;
        if (dims[i].nbytes > 0) {          // 生のバイト列。元の型と options を戻す
            r[2] = dims[i].raw_dt; r[3] = dims[i].raw_opts;
        } else {
            r[2] = static_cast<uint8_t>(dims[i].type);
            r[3] = (dims[i].has_scale ? 0x08 : 0) | (dims[i].has_offset ? 0x10 : 0);
        }
        memcpy(r + 4, dims[i].name.c_str(),
               std::min<size_t>(31, dims[i].name.size()));
        memcpy(r + 4 + 32 + 4 + 24 * 3, dims[i].scale, 24);
        memcpy(r + 4 + 32 + 4 + 24 * 4, dims[i].offset, 24);
        memcpy(r + 4 + 32 + 4 + 24 * 5, dims[i].description.c_str(),
               std::min<size_t>(31, dims[i].description.size()));
    }
}

// EVLR を LAS/LAZ のファイルから直に読む。laszip の C API は配列で出さない。
// LAS 1.4 のヘッダ: 235 に最初の EVLR の位置（u64）、243 に本数（u32）。
// EVLR の頭は 60 byte（reserved 2 / user_id 16 / record_id 2 / 長さ 8 / 説明 32）。
static void read_evlrs(const std::string& path, std::vector<PointCloud::RawVlr>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;
    uint8_t h[375];
    if (fread(h, 1, sizeof h, f) != sizeof h) { fclose(f); return; }
    if (memcmp(h, "LASF", 4) || h[25] < 3) { fclose(f); return; }
    uint64_t start = 0; uint32_t cnt = 0;
    if (h[25] == 3) {
        // LAS 1.3 は EVLR の欄を持たず、波形データの位置だけが 1 本の EVLR を指す
        memcpy(&start, h + 227, 8);
        cnt = start ? 1 : 0;
    } else {
        memcpy(&start, h + 235, 8);
        memcpy(&cnt, h + 243, 4);
    }
    if (!cnt || !start) { fclose(f); return; }
    if (fseek(f, (long)start, SEEK_SET) != 0) { fclose(f); return; }
    for (uint32_t i = 0; i < cnt; ++i) {
        uint8_t r[60];
        if (fread(r, 1, 60, f) != 60) break;
        PointCloud::RawVlr v;
        char uid[17] = {0}; memcpy(uid, r + 2, 16); v.user_id = uid;
        memcpy(&v.record_id, r + 18, 2);
        uint64_t len = 0; memcpy(&len, r + 20, 8);
        char ds[33] = {0}; memcpy(ds, r + 28, 32); v.description = ds;
        if (len > (1ull << 32)) break;
        v.data.resize((size_t)len);
        if (len && fread(v.data.data(), 1, (size_t)len, f) != len) break;
        out.push_back(std::move(v));
    }
    fclose(f);
}

// 書いた LAS/LAZ の末尾に EVLR を足し、ヘッダの位置と本数を書き直す。
static bool append_evlrs(const std::string& path,
                         const std::vector<PointCloud::RawVlr>& ev) {
    if (ev.empty()) return true;
    FILE* f = fopen(path.c_str(), "r+b");
    if (!f) return false;
    uint8_t h[375];
    if (fread(h, 1, sizeof h, f) != sizeof h) { fclose(f); return false; }
    if (memcmp(h, "LASF", 4) || h[25] < 3) { fclose(f); return true; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    const uint64_t start = (uint64_t)ftell(f);
    for (const auto& v : ev) {
        uint8_t r[60] = {0};
        memcpy(r + 2, v.user_id.c_str(), std::min<size_t>(16, v.user_id.size()));
        memcpy(r + 18, &v.record_id, 2);
        uint64_t len = v.data.size();
        memcpy(r + 20, &len, 8);
        memcpy(r + 28, v.description.c_str(), std::min<size_t>(32, v.description.size()));
        if (fwrite(r, 1, 60, f) != 60) { fclose(f); return false; }
        if (len && fwrite(v.data.data(), 1, (size_t)len, f) != len)
            { fclose(f); return false; }
    }
    const uint32_t cnt = (uint32_t)ev.size();
    bool ok = true;
    if (h[25] >= 4) {                   // 1.3 には EVLR の欄が無い（位置は波形の欄で言う）
        ok = fseek(f, 235, SEEK_SET) == 0 && fwrite(&start, 8, 1, f) == 1 &&
             fwrite(&cnt, 4, 1, f) == 1;
    }
    if (fclose(f) != 0) ok = false;
    return ok;
}

// 書いたファイルで、指定の EVLR の頭が何 byte 目にあるか。
static bool evlr_position(const std::string& path, const char* uid, uint16_t rid, uint64_t& pos) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t h[375];
    bool found = false;
    if (fread(h, 1, sizeof h, f) == sizeof h && !memcmp(h, "LASF", 4) && h[25] >= 3) {
        uint64_t p = 0; uint32_t cnt = 0;
        if (h[25] == 3) { memcpy(&p, h + 227, 8); cnt = p ? 1 : 0; }
        else { memcpy(&p, h + 235, 8); memcpy(&cnt, h + 243, 4); }
        for (uint32_t i = 0; i < cnt && !found; ++i) {
            uint8_t r[60];
            if (fseek(f, (long)p, SEEK_SET) != 0 || fread(r, 1, 60, f) != 60) break;
            char u[17] = {0}; memcpy(u, r + 2, 16);
            uint16_t id = 0; memcpy(&id, r + 18, 2);
            uint64_t len = 0; memcpy(&len, r + 20, 8);
            if (!strcmp(u, uid) && id == rid) { pos = p; found = true; }
            p += 60 + len;
        }
    }
    fclose(f);
    return found;
}

void recount_header(const PointCloud& pc, double mm[6], uint32_t& legacy_n,
                    uint32_t by_ret[5], uint64_t ext_by_ret[15], uint64_t& ext_n) {
    for (int k = 0; k < 5; ++k) by_ret[k] = 0;
    for (int k = 0; k < 15; ++k) ext_by_ret[k] = 0;
    for (int k = 0; k < 6; ++k) mm[k] = 0;
    const size_t n = pc.n;
    if (n && pc.X.size() >= n && pc.Y.size() >= n && pc.Z.size() >= n) {
        int32_t lo[3] = {pc.X[0], pc.Y[0], pc.Z[0]}, hi[3] = {pc.X[0], pc.Y[0], pc.Z[0]};
        for (size_t i = 1; i < n; ++i) {
            const int32_t v[3] = {pc.X[i], pc.Y[i], pc.Z[i]};
            for (int c = 0; c < 3; ++c) { if (v[c] < lo[c]) lo[c] = v[c]; if (v[c] > hi[c]) hi[c] = v[c]; }
        }
        // LAS の並びは max_x, min_x, max_y, min_y, max_z, min_z
        for (int c = 0; c < 3; ++c) {
            mm[c * 2 + 0] = (double)hi[c] * pc.scale[c] + pc.offset[c];
            mm[c * 2 + 1] = (double)lo[c] * pc.scale[c] + pc.offset[c];
        }
    }
    const bool ext = (pc.point_format & 0x3F) >= 6;
    auto it = pc.fields.find("bit_fields");
    if (it != pc.fields.end()) {
        for (size_t i = 0; i < n; ++i) {
            const uint64_t bf = (uint64_t)it->second.at(i);
            const int rn = ext ? (int)(bf & 15) : (int)(bf & 7);
            if (rn >= 1 && rn <= 5) ++by_ret[rn - 1];
            if (rn >= 1 && rn <= 15) ++ext_by_ret[rn - 1];
        }
    }
    legacy_n = (ext || n >= (1ull << 32)) ? 0 : (uint32_t)n;
    // 点形式 6 以上では旧形式の欄は 0 と決まっている（LASzip もそう書く）
    if (ext) for (int k = 0; k < 5; ++k) by_ret[k] = 0;
    ext_n = pc.version_minor >= 4 ? (uint64_t)n : 0;
}

void decide_header_mask(PointCloud& pc) {
    pc.hdr_mask = 0;
    if (!pc.hdr_known) return;
    // 途中で切った読み込みには元の範囲・点数が当てはまらないので、上書きしない。
    const bool full = (pc.hdr_n == (uint64_t)pc.n);
    if (full) {
        double mm[6]; uint32_t ln = 0, br[5]; uint64_t eb[15], en = 0;
        recount_header(pc, mm, ln, br, eb, en);
        for (int k = 0; k < 6; ++k)
            if (memcmp(&mm[k], &pc.hdr_minmax[k], sizeof(double)) != 0) pc.hdr_mask |= (uint16_t)(1u << k);
        if (ln != pc.hdr_legacy_n) pc.hdr_mask |= 1u << 6;
        if (memcmp(br, pc.hdr_by_ret, sizeof br) != 0) pc.hdr_mask |= 1u << 7;
        // 拡張の点数の欄は LAS 1.4 にしか無い。それより前の版では読み込み値が 0 で、
        // 比べると必ず食い違う（1.2 の plane で 120 byte を無駄に載せていた）。
        if (pc.version_minor >= 4 && memcmp(eb, pc.hdr_ext_by_ret, sizeof eb) != 0)
            pc.hdr_mask |= 1u << 8;
        if (pc.version_minor >= 4 && en != pc.hdr_ext_n) pc.hdr_mask |= 1u << 11;
    }
    static const uint8_t zero16[16] = {0};
    if (memcmp(pc.hdr_guid, zero16, 16) != 0) pc.hdr_mask |= 1u << 9;
    // 波形データの位置は LAS 1.3 から
    if (pc.version_minor >= 3 && pc.hdr_waveform_start != 0) pc.hdr_mask |= 1u << 10;
}

bool pointclouds_equal(const PointCloud& a, const PointCloud& b, bool counts,
                       std::string& diff) {
    char m[256];
    auto bits_eq = [](double x, double y) { return memcmp(&x, &y, sizeof x) == 0; };
    if (a.n != b.n) { snprintf(m, sizeof m, "点数 %zu != %zu", a.n, b.n); diff = m; return false; }
    for (int c = 0; c < 3; ++c)
        if (!bits_eq(a.scale[c], b.scale[c]) || !bits_eq(a.offset[c], b.offset[c]))
            { diff = "scale / offset"; return false; }
    const std::vector<int32_t>* ga[3] = {&a.X, &a.Y, &a.Z};
    const std::vector<int32_t>* gb[3] = {&b.X, &b.Y, &b.Z};
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < a.n; ++i)
            if ((*ga[c])[i] != (*gb[c])[i]) {
                snprintf(m, sizeof m, "%c[%zu] %d != %d", "XYZ"[c], i, (*ga[c])[i], (*gb[c])[i]);
                diff = m; return false;
            }
    if (a.point_format != b.point_format || a.point_record_len != b.point_record_len ||
        a.version_minor != b.version_minor) { diff = "点形式・レコード長・版"; return false; }
    if (a.fields.size() != b.fields.size()) { diff = "列の数"; return false; }
    for (const auto& kv : a.fields) {
        auto it = b.fields.find(kv.first);
        if (it == b.fields.end()) { diff = "列が無い: " + kv.first; return false; }
        for (size_t i = 0; i < a.n; ++i)
            if (kv.second.at(i) != it->second.at(i)) {
                snprintf(m, sizeof m, "%s[%zu] %lld != %lld", kv.first.c_str(), i,
                         (long long)kv.second.at(i), (long long)it->second.at(i));
                diff = m; return false;
            }
    }
    if (a.extra.size() != b.extra.size()) { diff = "ExtraBytes の次元の数"; return false; }
    for (size_t k = 0; k < a.extra.size(); ++k) {
        const ExtraDim& x = a.extra[k]; const ExtraDim& y = b.extra[k];
        if (x.name != y.name || x.col != y.col || x.type != y.type || x.nbytes != y.nbytes ||
            x.raw_dt != y.raw_dt || x.raw_opts != y.raw_opts || x.byte_offset != y.byte_offset ||
            x.undocumented != y.undocumented || x.description != y.description)
            { diff = "ExtraBytes の次元: " + x.name; return false; }
    }
    auto vlr_eq = [](const std::vector<PointCloud::RawVlr>& x,
                     const std::vector<PointCloud::RawVlr>& y) {
        if (x.size() != y.size()) return false;
        for (size_t k = 0; k < x.size(); ++k)
            if (x[k].user_id != y[k].user_id || x[k].record_id != y[k].record_id ||
                x[k].description != y[k].description || x[k].data != y[k].data) return false;
        return true;
    };
    if (!vlr_eq(a.vlrs, b.vlrs)) { diff = "VLR"; return false; }
    if (!vlr_eq(a.evlrs, b.evlrs)) { diff = "EVLR"; return false; }
    if (a.extra_vlr != b.extra_vlr || a.extra_vlr_desc != b.extra_vlr_desc ||
        a.extra_vlr_pos != b.extra_vlr_pos) { diff = "ExtraBytes の VLR"; return false; }
    if (a.file_source_id != b.file_source_id || a.global_encoding != b.global_encoding ||
        a.creation_day != b.creation_day || a.creation_year != b.creation_year ||
        a.system_identifier != b.system_identifier ||
        a.generating_software != b.generating_software) { diff = "ヘッダの欄"; return false; }
    if (a.user_in_header != b.user_in_header || a.user_after_header != b.user_after_header)
        { diff = "ユーザーデータ"; return false; }
    if (memcmp(a.hdr_guid, b.hdr_guid, 16) != 0) { diff = "GUID"; return false; }
    // 波形データの位置は、EVLR を指しているなら書き直したファイルでの位置になる
    // （点データの長さが違えば値も違う）。そのときは「どちらも指している」を比べる。
    {
        bool wave_evlr = false;
        for (const auto& v : a.evlrs) if (v.user_id == "LASF_Spec" && v.record_id == 65535) wave_evlr = true;
        const bool same = wave_evlr ? ((a.hdr_waveform_start != 0) == (b.hdr_waveform_start != 0))
                                    : a.hdr_waveform_start == b.hdr_waveform_start;
        if (!same) { diff = "波形データの位置"; return false; }
    }
    if (counts) {
        for (int k = 0; k < 6; ++k)
            if (!bits_eq(a.hdr_minmax[k], b.hdr_minmax[k])) { diff = "ヘッダの範囲"; return false; }
        if (a.hdr_legacy_n != b.hdr_legacy_n || a.hdr_ext_n != b.hdr_ext_n ||
            memcmp(a.hdr_by_ret, b.hdr_by_ret, sizeof a.hdr_by_ret) != 0 ||
            memcmp(a.hdr_ext_by_ret, b.hdr_ext_by_ret, sizeof a.hdr_ext_by_ret) != 0)
            { diff = "ヘッダの点数の欄"; return false; }
    }
    return true;
}

void assign_extra_cols(PointCloud& pc) {
    // 標準の列名は点形式に依らず全部予約する（器の復元では点形式を見ずに決めるため）。
    std::set<std::string> used = {"X", "Y", "Z", "intensity", "bit_fields",
                                  "classification", "user_data", "scan_angle",
                                  "point_source_id", "gps_time", "red", "green",
                                  "blue", "nir"};
    for (int k = 0; k < 29; ++k) used.insert("wave_packet#" + std::to_string(k));
    auto names_of = [](const ExtraDim& e, const std::string& base) {
        std::vector<std::string> v;
        if (e.nbytes > 0) for (int k = 0; k < e.nbytes; ++k) v.push_back(base + "#" + std::to_string(k));
        else v.push_back(base);
        return v;
    };
    for (size_t i = 0; i < pc.extra.size(); ++i) {
        ExtraDim& e = pc.extra[i];
        std::string base = e.name;
        auto clash = [&](const std::string& b) {
            if (b.empty()) return true;
            for (const auto& nm : names_of(e, b)) if (used.count(nm)) return true;
            return false;
        };
        if (clash(base)) {
            base = "eb" + std::to_string(i) + ":" + e.name;
            for (int t = 0; clash(base); ++t) base = "eb" + std::to_string(i) + "_" + std::to_string(t) + ":" + e.name;
        }
        e.col = base;
        for (const auto& nm : names_of(e, base)) used.insert(nm);
    }
}

// LASzip の手綱。途中で return しても必ず破棄する（以前は失敗のたびに漏れていた）。
struct LzHandle {
    laszip_POINTER p = nullptr;
    LzHandle() = default;
    LzHandle(const LzHandle&) = delete;
    LzHandle& operator=(const LzHandle&) = delete;
    ~LzHandle() { if (p) laszip_destroy(p); }
};
// 失敗したら LASzip 自身の理由も添える。
#define LZ_CHECK(call, what) \
    if ((call) != 0) { \
        laszip_CHAR* m_ = nullptr; \
        if (lz) laszip_get_error(lz, &m_); \
        err = std::string("LASzip: ") + what + (m_ && *m_ ? std::string(": ") + m_ : ""); \
        return false; \
    }

bool read_las(const std::string& path, PointCloud& pc, std::string& err,
              size_t max_points) {
    LzHandle lzh;
    if (laszip_create(&lzh.p) != 0) { err = "LASzip: create"; return false; }
    laszip_POINTER lz = lzh.p;
    laszip_BOOL compressed = 0;
    LZ_CHECK(laszip_open_reader(lz, path.c_str(), &compressed), "open_reader");
    laszip_header* h = nullptr;
    LZ_CHECK(laszip_get_header_pointer(lz, &h), "get_header");

    struct stat st{};
    pc.src_bytes = (stat(path.c_str(), &st) == 0) ? (uint64_t)st.st_size : 0;
    pc.src_path = path;
    pc.point_format = h->point_data_format;
    pc.point_record_len = h->point_data_record_length;
    pc.version_minor = h->version_minor;
    // 書き戻せるのは 1.0〜1.4 まで（1.5 はヘッダが 393 byte で、書く側が作れない）
    if (h->version_major != 1 || h->version_minor > 4) {
        err = "未対応の LAS の版 " + std::to_string(h->version_major) + "." +
              std::to_string(h->version_minor);
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        pc.scale[i]  = (&h->x_scale_factor)[i];
        pc.offset[i] = (&h->x_offset)[i];
    }
    uint64_t total = h->extended_number_of_point_records ?
                     h->extended_number_of_point_records : h->number_of_point_records;
    size_t n = (max_points && max_points < total) ? max_points : (size_t)total;
    pc.n = n;

    read_evlrs(path, pc.evlrs);
    // 中身から決まるヘッダの欄を、数え直さずにそのまま持つ（書き戻しで使う）。
    pc.hdr_known = true;
    pc.hdr_n = total;
    {
        const double mm[6] = {h->max_x, h->min_x, h->max_y, h->min_y, h->max_z, h->min_z};
        for (int k = 0; k < 6; ++k) pc.hdr_minmax[k] = mm[k];
    }
    pc.hdr_legacy_n = h->number_of_point_records;
    pc.hdr_ext_n = h->extended_number_of_point_records;
    for (int k = 0; k < 5; ++k) pc.hdr_by_ret[k] = h->number_of_points_by_return[k];
    for (int k = 0; k < 15; ++k) pc.hdr_ext_by_ret[k] = h->extended_number_of_points_by_return[k];
    memcpy(pc.hdr_guid + 0, &h->project_ID_GUID_data_1, 4);
    memcpy(pc.hdr_guid + 4, &h->project_ID_GUID_data_2, 2);
    memcpy(pc.hdr_guid + 6, &h->project_ID_GUID_data_3, 2);
    memcpy(pc.hdr_guid + 8, h->project_ID_GUID_data_4, 8);
    pc.hdr_waveform_start = h->start_of_waveform_data_packet_record;
    if (h->user_data_in_header_size && h->user_data_in_header)
        pc.user_in_header.assign(h->user_data_in_header,
                                 h->user_data_in_header + h->user_data_in_header_size);
    if (h->user_data_after_header_size && h->user_data_after_header)
        pc.user_after_header.assign(h->user_data_after_header,
                                    h->user_data_after_header + h->user_data_after_header_size);
    pc.file_source_id = h->file_source_ID;
    pc.global_encoding = h->global_encoding;
    pc.creation_day = h->file_creation_day;
    pc.creation_year = h->file_creation_year;
    pc.system_identifier.assign(h->system_identifier,
                                strnlen(h->system_identifier, 32));
    pc.generating_software.assign(h->generating_software,
                                  strnlen(h->generating_software, 32));

    for (uint32_t i = 0; i < h->number_of_variable_length_records; ++i) {
        const laszip_vlr_struct& v = h->vlrs[i];
        const std::string uid = v.user_id;
        // ExtraBytes として読むのは 192 byte の記録が 1 つ以上ある VLR だけ。
        // 空や半端な長さのものは素の VLR として運ぶ（以前は黙って消えていた）。
        const bool eb_vlr = uid == "LASF_Spec" && v.record_id == 4 &&
                            v.record_length_after_header > 0 &&
                            v.record_length_after_header % 192 == 0 && pc.extra_vlr.empty();
        if (eb_vlr) {
            parse_extra_vlr(v.data, v.record_length_after_header, pc.extra);
            pc.extra_vlr.assign(v.data, v.data + v.record_length_after_header);
            pc.extra_vlr_desc = v.description;
            pc.extra_vlr_pos = (uint32_t)pc.vlrs.size();   // 元の並びでの位置
        }
        // 測地系などの VLR をそのまま持ち帰る。捨てると往復で消える。
        // laszip 自身の VLR は書くときに作り直されるので持たない。
        // ExtraBytes はこちらが列の構成から作り直すので持たない。
        if (uid == "laszip encoded") continue;
        if (eb_vlr) continue;
        PointCloud::RawVlr rv;
        rv.user_id = uid;
        rv.record_id = v.record_id;
        rv.description = v.description;
        rv.data.assign(v.data, v.data + v.record_length_after_header);
        pc.vlrs.push_back(std::move(rv));
    }

    // 取り出すフィールドを決める。
    //
    // 以前は全フィールドを resize(n) で確保していたので、定数の列（幾何だけの
    // PF6 では大半がそう）にも 8n バイトを払っていた。400 万点で 265 MB。
    // 代わりに「値が変わるまで確保しない」形にする。最初の値と違う値が来た
    // 時点で確保し、それまでを最初の値で埋める。
    // あわせて、点ごとの pc.fields["名前"] という map 検索もやめる
    // （400 万点 × 10 列で 4000 万回の文字列比較になっていた）。
    struct Lazy {
        std::vector<int64_t>* v = nullptr;   // 実体（確保されるまで nullptr）
        int64_t first = 0;
        bool started = false, varied = false;
        size_t n = 0;
        inline void put(size_t i, int64_t val) {
            if (!started) { first = val; started = true; return; }
            if (!varied) {
                if (val == first) return;
                varied = true;
                v->assign(n, first);
            }
            (*v)[i] = val;
        }
        void finish() {
            if (!varied) { v->clear(); v->shrink_to_fit(); }
        }
    };
    std::vector<Lazy> lz_f;
    auto add = [&](const std::string& nm, FType t, bool ex) -> size_t {
        pc.order.push_back(nm);
        Field f; f.ftype = t; f.is_extra = ex;
        pc.fields[nm] = std::move(f);
        Lazy L; L.v = &pc.fields[nm].v; L.n = n;
        lz_f.push_back(L);
        return lz_f.size() - 1;
    };
    bool has_gps = (pc.point_format == 1 || pc.point_format >= 3);
    bool has_rgb = (pc.point_format == 2 || pc.point_format == 3 ||
                    pc.point_format == 5 || pc.point_format == 7 ||
                    pc.point_format == 8 || pc.point_format == 10);
    bool has_nir = (pc.point_format == 8 || pc.point_format == 10);
    // 波形パケット（29 byte の生のバイト列）を持つ点形式。捨てると戻らない。
    bool has_wave = (pc.point_format == 4 || pc.point_format == 5 ||
                     pc.point_format == 9 || pc.point_format == 10);
    const size_t I_INT = add("intensity", FType::U16, false);
    const size_t I_BF  = add("bit_fields", FType::U16, false);
    const size_t I_CLS = add("classification", FType::U8, false);
    const size_t I_UD  = add("user_data", FType::U8, false);
    const size_t I_SA  = add("scan_angle", FType::I16, false);
    const size_t I_PSI = add("point_source_id", FType::U16, false);
    size_t I_GPS = 0, I_R = 0, I_G = 0, I_B = 0, I_NIR = 0;
    if (has_gps) I_GPS = add("gps_time", FType::F64, false);
    if (has_rgb) { I_R = add("red", FType::U16, false); I_G = add("green", FType::U16, false);
                   I_B = add("blue", FType::U16, false); }
    if (has_nir) I_NIR = add("nir", FType::U16, false);
    std::vector<size_t> I_WV;
    if (has_wave)
        for (int k = 0; k < 29; ++k)
            I_WV.push_back(add("wave_packet#" + std::to_string(k), FType::U8, false));
    // 生のバイト列の次元は 1 byte ずつ別の列にする（name#0, name#1, ...）。
    std::vector<std::vector<size_t>> I_EX;
    // **ExtraBytes の VLR が記述していない余りのバイト**を 1 つの生バイト列の次元にする。
    // 点レコードの長さから、点形式の基本の大きさと記述された次元を引いた残り。
    {
        static const int BASE[11] = {20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67};
        const int pf = pc.point_format & 0x3F;
        if (pf <= 10) {
            int described = 0;
            for (const auto& e : pc.extra)
                described = std::max(described, e.byte_offset + (e.nbytes > 0 ? e.nbytes : ftype_size(e.type)));
            const int total_ex = (int)pc.point_record_len - BASE[pf];
            if (total_ex > described) {
                ExtraDim u;
                u.undocumented = true;
                u.nbytes = total_ex - described;
                u.byte_offset = described;
                u.raw_dt = 0;
                pc.extra.push_back(u);
            }
        }
    }
    assign_extra_cols(pc);
    for (const auto& e : pc.extra) {
        std::vector<size_t> ids;
        if (e.nbytes > 0)
            for (int k = 0; k < e.nbytes; ++k)
                ids.push_back(add(e.col + "#" + std::to_string(k), FType::U8, true));
        else
            ids.push_back(add(e.col, e.type, true));
        I_EX.push_back(std::move(ids));
    }

    pc.X.resize(n); pc.Y.resize(n); pc.Z.resize(n);
    laszip_point* p = nullptr;
    LZ_CHECK(laszip_get_point_pointer(lz, &p), "get_point");
    for (size_t i = 0; i < n; ++i) {
        LZ_CHECK(laszip_read_point(lz), "read_point");
        pc.X[i] = p->X; pc.Y[i] = p->Y; pc.Z[i] = p->Z;
        lz_f[I_INT].put(i, p->intensity);
        // 詰め込みビットは 1 列にまとめる。**全部入れる。**
        // 以前は戻り番号 2 つしか入れておらず、走査の向き・飛行線の端・
        // 分類フラグ・スキャナ番号が容器に入らないまま落ちていた。
        //   bit 0-3 戻り番号 / 4-7 戻り総数 / 8 走査の向き / 9 飛行線の端
        //   bit 10-13 分類フラグ / 14-15 スキャナ番号
        // 旧形式（点形式 5 以下）は 3 bit の欄と個別のフラグから、
        // 拡張形式（6 以上）は extended_* から作る。
        uint16_t bf;
        if (pc.point_format >= 6) {
            bf = (uint16_t)(p->extended_return_number
                            | (p->extended_number_of_returns << 4)
                            | (p->scan_direction_flag << 8)
                            | (p->edge_of_flight_line << 9)
                            | (p->extended_classification_flags << 10)
                            | (p->extended_scanner_channel << 14));
        } else {
            bf = (uint16_t)(p->return_number
                            | (p->number_of_returns << 4)
                            | (p->scan_direction_flag << 8)
                            | (p->edge_of_flight_line << 9)
                            | (p->synthetic_flag << 10)
                            | (p->keypoint_flag << 11)
                            | (p->withheld_flag << 12));
        }
        lz_f[I_BF].put(i, bf);
        lz_f[I_CLS].put(i, p->extended_classification ? p->extended_classification
                                                      : p->classification);
        lz_f[I_UD].put(i, p->user_data);
        // 走査角は形式で欄が違う。旧形式（点形式 5 以下）は 1 度刻みの符号つき
        // 1 byte（scan_angle_rank）、拡張形式は 0.006 度刻みの 2 byte。
        // 拡張側だけを持つと、旧形式では往復で丸めて戻らない。
        lz_f[I_SA].put(i, pc.point_format >= 6 ? (int64_t)p->extended_scan_angle
                                               : (int64_t)p->scan_angle_rank);
        lz_f[I_PSI].put(i, p->point_source_ID);
        if (has_gps) { uint64_t b; memcpy(&b, &p->gps_time, 8);
                       lz_f[I_GPS].put(i, (int64_t)b); }
        if (has_rgb) { lz_f[I_R].put(i, p->rgb[0]);
                       lz_f[I_G].put(i, p->rgb[1]);
                       lz_f[I_B].put(i, p->rgb[2]); }
        if (has_wave)
            for (int k = 0; k < 29; ++k) lz_f[I_WV[k]].put(i, (int64_t)p->wave_packet[k]);
        if (has_nir) lz_f[I_NIR].put(i, p->rgb[3]);
        for (size_t k = 0; k < pc.extra.size(); ++k) {
            const auto& e = pc.extra[k];
            const int need = e.nbytes > 0 ? e.nbytes : ftype_size(e.type);
            if (p->num_extra_bytes >= e.byte_offset + need) {
                if (e.nbytes > 0)
                    for (int q = 0; q < e.nbytes; ++q)
                        lz_f[I_EX[k][q]].put(i, (int64_t)p->extra_bytes[e.byte_offset + q]);
                else
                    lz_f[I_EX[k][0]].put(i, read_raw(p->extra_bytes + e.byte_offset, e.type));
            }
        }
    }
    // 定数のままだった列は実体を持たせない。値だけを覚えておく。
    {
        size_t k = 0;
        for (const auto& nm : pc.order) {
            Field& fl = pc.fields[nm];
            if (k < lz_f.size() && !lz_f[k].varied) {
                fl.is_const = true;
                fl.cval = lz_f[k].first;
                fl.v.clear();
                fl.v.shrink_to_fit();
            }
            ++k;
        }
    }
    laszip_close_reader(lz);
    decide_header_mask(pc);
    return true;
}

bool write_las(const std::string& path, const PointCloud& pc,
               const std::vector<std::string>& keep, std::string& err,
               const std::vector<int32_t>* Xq, const std::vector<int32_t>* Yq,
               const std::vector<int32_t>* Zq, const double* scale_override) {
    auto kept = [&](const std::string& s) {
        for (const auto& k : keep) if (k == s) return true;
        return false;
    };
    // **扱えない型も含めて全部残す。**落とすと点レコードの並びが詰まり、
    // 後続の次元が別の位置に書かれる（extra.laz で 15 byte ずれていた）。
    std::vector<ExtraDim> ex;
    int off = 0;
    for (auto e : pc.extra) {
        if (e.col.empty()) e.col = e.name;          // 古い呼び出し元（列名＝名前）
        const bool keep_it = e.nbytes > 0 ? kept(e.col + "#0") : kept(e.col);
        if (!keep_it) continue;
        e.byte_offset = off;
        off += e.nbytes > 0 ? e.nbytes : ftype_size(e.type);
        ex.push_back(e);
    }
    LzHandle lzh;
    if (laszip_create(&lzh.p) != 0) { err = "LASzip: create"; return false; }
    laszip_POINTER lz = lzh.p;
    // 既定では LASzip が generating_software を自分の名前で上書きする。
    // 元のファイルの欄を残す。
    LZ_CHECK(laszip_preserve_generating_software(lz, 1), "preserve_gen");
    laszip_header h{};
    h.version_major = 1; h.version_minor = pc.version_minor;
    h.file_source_ID = pc.file_source_id;
    h.global_encoding = pc.global_encoding;
    h.file_creation_day = pc.creation_day;
    h.file_creation_year = pc.creation_year;
    memset(h.system_identifier, 0, 32);
    memcpy(h.system_identifier, pc.system_identifier.c_str(),
           std::min<size_t>(32, pc.system_identifier.size()));
    memset(h.generating_software, 0, 32);
    memcpy(h.generating_software, pc.generating_software.c_str(),
           std::min<size_t>(32, pc.generating_software.size()));
    // LAS 1.0-1.2 = 227, 1.3 = 235（波形情報の欄が増える）, 1.4 = 375
    h.header_size = (pc.version_minor >= 4) ? 375 : (pc.version_minor == 3 ? 235 : 227);
    // ヘッダの中のユーザーデータ（標準の欄より後ろ、header_size の内側）
    if (!pc.user_in_header.empty()) {
        h.header_size = (laszip_U16)(h.header_size + pc.user_in_header.size());
        h.user_data_in_header_size = (laszip_U32)pc.user_in_header.size();
        h.user_data_in_header = const_cast<laszip_U8*>(pc.user_in_header.data());
    }
    h.offset_to_point_data = h.header_size;
    // VLR の後ろ・点データの前のユーザーデータ
    if (!pc.user_after_header.empty()) {
        h.user_data_after_header_size = (laszip_U32)pc.user_after_header.size();
        h.user_data_after_header = const_cast<laszip_U8*>(pc.user_after_header.data());
        // LASzip は VLR を足すたびに点データの位置を進めるが、この分は足さない
        // （足さないと open_writer が「位置が合わない」で失敗する）。
        h.offset_to_point_data += h.user_data_after_header_size;
    }
    // GUID と LAS 1.3 の波形データの位置
    memcpy(&h.project_ID_GUID_data_1, pc.hdr_guid + 0, 4);
    memcpy(&h.project_ID_GUID_data_2, pc.hdr_guid + 4, 2);
    memcpy(&h.project_ID_GUID_data_3, pc.hdr_guid + 6, 2);
    memcpy(h.project_ID_GUID_data_4, pc.hdr_guid + 8, 8);
    // 波形データの位置は閉じた後に書く（LASzip は 0 を書く。下の後始末を見よ）
    h.point_data_format = pc.point_format;
    int base = pc.point_record_len;
    // 生のバイト列の次元は nbytes 分の場所を取る。ftype_size は 1 を返すので、
    // そのまま引くとレコード長が伸び、基準の LAZ に 1 点あたり数 byte の
    // ゼロ詰めが入る（extra.laz で 61 → 70 byte、基準が 6.9% 水増しされていた）。
    for (const auto& e : pc.extra) base -= (e.nbytes > 0 ? e.nbytes : ftype_size(e.type));
    h.point_data_record_length = (laszip_U16)(base + off);
    if (pc.version_minor >= 4) h.extended_number_of_point_records = pc.n;
    h.number_of_point_records = (pc.n < (1ull << 32)) ? (laszip_U32)pc.n : 0;
    // 中身から決まるヘッダの欄は、読み込みと同じ recount_header で数え直して書き、
    // 元のファイルと食い違っていた欄（hdr_mask）だけ元の値にする。
    // 食い違わない欄は封筒に載せなくてよいので、その分だけ小さくなる。
    // **点数の欄は、LASzip には数え直した（互いに整合した）値を渡し、元の値は閉じた後に
    // 書き戻す。**LASzip は 1.4 の点形式 5 以下で旧形式と拡張の点数が食い違うと
    // 書き始めを拒むので、元の値をそのまま渡すと書けないファイルがある。
    {
        double mm[6]; uint32_t ln = 0, br[5]; uint64_t eb[15], en = 0;
        recount_header(pc, mm, ln, br, eb, en);
        // 量子化した座標（Xq）や別の刻み（scale_override）で書くときは、
        // 実際に書く整数と刻みから範囲を数え直す
        if ((Xq || scale_override) && pc.n) {
            const std::vector<int32_t>* q[3] = {Xq ? Xq : &pc.X, Yq ? Yq : &pc.Y, Zq ? Zq : &pc.Z};
            for (int c = 0; c < 3; ++c) {
                const double sc = scale_override ? scale_override[c] : pc.scale[c];
                const auto mnmx = std::minmax_element(q[c]->begin(), q[c]->begin() + pc.n);
                mm[c * 2 + 0] = (double)*mnmx.second * sc + pc.offset[c];
                mm[c * 2 + 1] = (double)*mnmx.first * sc + pc.offset[c];
            }
        }
        const uint16_t m = pc.hdr_mask;
        double* dst[6] = {&h.max_x, &h.min_x, &h.max_y, &h.min_y, &h.max_z, &h.min_z};
        for (int k = 0; k < 6; ++k) *dst[k] = (m >> k & 1) ? pc.hdr_minmax[k] : mm[k];
        h.number_of_point_records = ln;
        for (int k = 0; k < 5; ++k) h.number_of_points_by_return[k] = br[k];
        for (int k = 0; k < 15; ++k) h.extended_number_of_points_by_return[k] = eb[k];
        if (pc.version_minor >= 4) h.extended_number_of_point_records = en;
    }
    for (int i = 0; i < 3; ++i) {
        (&h.x_scale_factor)[i] = scale_override ? scale_override[i] : pc.scale[i];
        (&h.x_offset)[i] = pc.offset[i];
    }
    // 元の欄が空のときだけ自分の名前を書く。上書きすると器が変わる。
    if (pc.generating_software.empty())
        strncpy(h.generating_software, "pccnorm", 31);
    LZ_CHECK(laszip_set_header(lz, &h), "set_header");
    // ExtraBytes の VLR を用意する。元の VLR をそのまま書き戻せるなら、そうする。
    // 作り直すと scale / offset / no_data / min / max / reserved が既定値になり、
    // 192 byte のレコードがバイト一致しない。
    std::vector<uint8_t> xvlr;
    std::string xdesc = "extra bytes";
    // VLR に載せるのは**記述された次元だけ**（記述されていない余りのバイトは載せない）。
    std::vector<ExtraDim> exd;
    for (const auto& e : ex) if (!e.undocumented) exd.push_back(e);
    size_t ndoc_all = 0;
    for (const auto& e : pc.extra) if (!e.undocumented) ++ndoc_all;
    if (!exd.empty()) {
        bool verbatim = (exd.size() == ndoc_all && !pc.extra_vlr.empty() &&
                         pc.extra_vlr.size() == ndoc_all * 192);
        if (verbatim) { xvlr = pc.extra_vlr; xdesc = pc.extra_vlr_desc; }
        else build_extra_vlr(exd, xvlr);
    }
    auto add_extra = [&]() {
        LZ_CHECK(laszip_add_vlr(lz, "LASF_Spec", 4, (laszip_U16)xvlr.size(),
                                xdesc.c_str(), xvlr.data()), "add_vlr");
        return true;
    };
    // 元のファイルにあった VLR を戻す。**ExtraBytes は元の並びの位置に入れる。**
    for (size_t k = 0; k < pc.vlrs.size(); ++k) {
        if (!exd.empty() && k == pc.extra_vlr_pos && !add_extra()) return false;
        const auto& rv = pc.vlrs[k];
        LZ_CHECK(laszip_add_vlr(lz, rv.user_id.c_str(), rv.record_id,
                                (laszip_U16)rv.data.size(), rv.description.c_str(),
                                rv.data.empty() ? nullptr
                                                : const_cast<laszip_U8*>(rv.data.data())),
                 "add_vlr");
    }
    if (!exd.empty() && pc.extra_vlr_pos >= pc.vlrs.size() && !add_extra()) return false;
    // **拡張子で圧縮するかを決める。**以前は常に LAZ で書いたので、`--las x.las` と
    // 指定しても中身は LAZ だった（拡張子と中身が食い違う）。".laz" なら圧縮、
    // それ以外は非圧縮の LAS で書く。
    auto ends_ci = [&](const char* suf) {
        const size_t m = strlen(suf);
        if (path.size() < m) return false;
        for (size_t k = 0; k < m; ++k)
            if (tolower((unsigned char)path[path.size() - m + k]) != suf[k]) return false;
        return true;
    };
    const laszip_BOOL compress = ends_ci(".laz") ? 1 : 0;
    LZ_CHECK(laszip_open_writer(lz, path.c_str(), compress), "open_writer");
    laszip_point* p = nullptr;
    LZ_CHECK(laszip_get_point_pointer(lz, &p), "get_point");
    auto get = [&](const std::string& nm, size_t i) -> int64_t {
        auto it = pc.fields.find(nm);
        if (it == pc.fields.end() || !kept(nm)) return 0;
        return it->second.at(i);
    };
    for (size_t i = 0; i < pc.n; ++i) {
        p->X = Xq ? (*Xq)[i] : pc.X[i];
        p->Y = Yq ? (*Yq)[i] : pc.Y[i];
        p->Z = Zq ? (*Zq)[i] : pc.Z[i];
        p->intensity = (laszip_U16)get("intensity", i);
        uint16_t bf = (uint16_t)get("bit_fields", i);
        p->extended_return_number = bf & 0x0F;
        p->extended_number_of_returns = (bf >> 4) & 0x0F;
        p->return_number = bf & 0x07;
        p->number_of_returns = (bf >> 4) & 0x07;
        p->scan_direction_flag = (bf >> 8) & 1;
        p->edge_of_flight_line = (bf >> 9) & 1;
        if (pc.point_format >= 6) {
            p->extended_classification_flags = (bf >> 10) & 0x0F;
            p->extended_scanner_channel = (bf >> 14) & 0x03;
            // LASzip は旧欄と拡張欄の下位 3 bit が食い違うと書き込みを拒む。
            // 立てないと、分類フラグを持つ点形式 6 以上のファイルが書けない。
            p->synthetic_flag = (bf >> 10) & 1;
            p->keypoint_flag  = (bf >> 11) & 1;
            p->withheld_flag  = (bf >> 12) & 1;
        } else {
            p->synthetic_flag = (bf >> 10) & 1;
            p->keypoint_flag  = (bf >> 11) & 1;
            p->withheld_flag  = (bf >> 12) & 1;
        }
        uint8_t cl = (uint8_t)get("classification", i);
        p->extended_classification = cl;
        p->classification = (cl < 32) ? cl : 0;
        p->user_data = (laszip_U8)get("user_data", i);
        if (pc.point_format >= 6) {
            p->extended_scan_angle = (laszip_I16)get("scan_angle", i);
            p->scan_angle_rank = (laszip_I8)(get("scan_angle", i) * 0.006);
        } else {
            p->scan_angle_rank = (laszip_I8)get("scan_angle", i);
            p->extended_scan_angle = (laszip_I16)(p->scan_angle_rank / 0.006);
        }
        p->point_source_ID = (laszip_U16)get("point_source_id", i);
        if (pc.fields.count("gps_time")) {
            uint64_t b = (uint64_t)get("gps_time", i);
            memcpy(&p->gps_time, &b, 8);
        }
        if (pc.fields.count("red")) {
            p->rgb[0] = (laszip_U16)get("red", i);
            p->rgb[1] = (laszip_U16)get("green", i);
            p->rgb[2] = (laszip_U16)get("blue", i);
        }
        if (pc.fields.count("nir")) p->rgb[3] = (laszip_U16)get("nir", i);
        if (pc.fields.count("wave_packet#0"))
            for (int k = 0; k < 29; ++k) {
                auto it = pc.fields.find("wave_packet#" + std::to_string(k));
                if (it != pc.fields.end()) p->wave_packet[k] = (uint8_t)it->second.at(i);
            }
        if (off) {
            // LASzip が確保済みのバッファに書き込む。
            // ポインタを差し替えると LASzip 側の解放と二重になって落ちる。
            if (p->extra_bytes && p->num_extra_bytes >= off) {
                memset(p->extra_bytes, 0, (size_t)p->num_extra_bytes);
                for (const auto& e : ex) {
                    if (e.nbytes > 0) {
                        for (int q = 0; q < e.nbytes; ++q) {
                            auto it = pc.fields.find(e.col + "#" + std::to_string(q));
                            if (it != pc.fields.end())
                                p->extra_bytes[e.byte_offset + q] = (uint8_t)it->second.at(i);
                        }
                    } else {
                        write_raw(p->extra_bytes + e.byte_offset, e.type,
                                  pc.fields.at(e.col).at(i));
                    }
                }
            }
        }
        LZ_CHECK(laszip_write_point(lz), "write_point");
    }
    // 閉じるときに残りの点を書き出すので、失敗を見落とすと欠けたファイルになる。
    LZ_CHECK(laszip_close_writer(lz), "close_writer");
    if (!append_evlrs(path, pc.evlrs)) { err = "EVLR を書けない"; return false; }
    // **閉じた後に、元のファイルと食い違っていた点数の欄と波形データの位置を書き戻す。**
    // LAZ でもヘッダは圧縮されないので、同じ位置を書けばよい。
    //   107 旧形式の点数 / 111 戻り番号ごと（旧）/ 227 波形データの位置（1.3〜）/
    //   247 拡張の点数 / 255 戻り番号ごと（拡張）（1.4）
    {
        const uint16_t m = pc.hdr_mask;
        // 波形データの位置: 元が EVLR（LASF_Spec 65535）を指していたなら、書き直した
        // ファイルでのその EVLR の位置にする（点データの長さが変わるので元の値は使えない）。
        // 指していなければ元の値をそのまま書く。
        uint64_t wave = pc.hdr_waveform_start;
        if (wave && pc.version_minor >= 3) {
            uint64_t pos = 0;
            if (evlr_position(path, "LASF_Spec", 65535, pos)) wave = pos;
        }
        const bool need = (m & ((1u << 6) | (1u << 7) | (1u << 8) | (1u << 11))) ||
                          (wave && pc.version_minor >= 3);
        if (need) {
            FILE* fp = fopen(path.c_str(), "r+b");
            bool ok = fp != nullptr;
            auto put_at = [&](long off, const void* p, size_t sz) {
                ok = ok && fseek(fp, off, SEEK_SET) == 0 && fwrite(p, 1, sz, fp) == sz;
            };
            if (m >> 6 & 1) put_at(107, &pc.hdr_legacy_n, 4);
            if (m >> 7 & 1) put_at(111, pc.hdr_by_ret, 20);
            if (wave && pc.version_minor >= 3) put_at(227, &wave, 8);
            if (pc.version_minor >= 4) {
                if (m >> 11 & 1) put_at(247, &pc.hdr_ext_n, 8);
                if (m >> 8 & 1) put_at(255, pc.hdr_ext_by_ret, 120);
            }
            if (fp && fclose(fp) != 0) ok = false;
            if (!ok) { err = "ヘッダの点数の欄を書き戻せない"; return false; }
        }
    }
    return true;
}

} // namespace pcc

namespace pcc {
// PLY / KITTI .bin も読めるようにする（解析系サブコマンド用。座標のみ）
bool read_points_any(const std::string& path, std::vector<double>& xyz,
                     size_t& n, std::string& err, size_t max_points,
                     int* src_bytes) {
    if (src_bytes) src_bytes[0] = src_bytes[1] = src_bytes[2] = 8;
    auto ends = [&](const char* s) {
        size_t L = strlen(s);
        return path.size() >= L && path.compare(path.size()-L, L, s) == 0;
    };
    if (ends(".las") || ends(".laz") || ends(".LAZ") || ends(".LAS")) {
        PointCloud pc;
        if (!read_las(path, pc, err, max_points)) return false;
        n = pc.n; xyz.resize(n * 3);
        for (size_t i = 0; i < n; ++i) {
            xyz[i*3]   = pc.X[i] * pc.scale[0] + pc.offset[0];
            xyz[i*3+1] = pc.Y[i] * pc.scale[1] + pc.offset[1];
            xyz[i*3+2] = pc.Z[i] * pc.scale[2] + pc.offset[2];
        }
        return true;
    }
    if (ends(".bin")) {                       // KITTI: float32 x,y,z,intensity
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { err = "開けない"; return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz < 0 || sz % 16 != 0) { fclose(f); err = "KITTI .bin の長さが 16 byte の倍数でない"; return false; }
        size_t cnt = (size_t)sz / 16;
        if (max_points && max_points < cnt) cnt = max_points;
        std::vector<float> buf(cnt * 4);
        if (fread(buf.data(), 16, cnt, f) != cnt) { fclose(f); err = "短い"; return false; }
        fclose(f);
        n = cnt; xyz.resize(n * 3);
        for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) xyz[i*3+d] = buf[i*4+d];
        if (src_bytes) src_bytes[0] = src_bytes[1] = src_bytes[2] = 4;
        return true;
    }
    if (ends(".ply")) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { err = "開けない"; return false; }
        char line[512]; bool ascii = false, bin_le = false, bin_be = false; size_t nv = 0;
        // 性質ごとに（名前, バイト数, 種類）。種類は 'f' 浮動小数 / 'i' 符号つき整数 / 'u' 符号なし整数
        struct Prop { std::string name; int size; char kind; };
        std::vector<Prop> props;
        auto tinfo = [](const std::string& t, int& sz, char& kd) -> bool {
            if (t=="char"||t=="int8")    { sz = 1; kd = 'i'; return true; }
            if (t=="uchar"||t=="uint8")  { sz = 1; kd = 'u'; return true; }
            if (t=="short"||t=="int16")  { sz = 2; kd = 'i'; return true; }
            if (t=="ushort"||t=="uint16"){ sz = 2; kd = 'u'; return true; }
            if (t=="int"||t=="int32")    { sz = 4; kd = 'i'; return true; }
            if (t=="uint"||t=="uint32")  { sz = 4; kd = 'u'; return true; }
            if (t=="float"||t=="float32"){ sz = 4; kd = 'f'; return true; }
            if (t=="double"||t=="float64"){ sz = 8; kd = 'f'; return true; }
            return false;
        };
        // 頂点の要素が最初でないと、ヘッダの直後から頂点を読む前提が崩れる。
        bool in_vertex = false, seen_element = false, vertex_first = false;
        std::string bad;
        while (fgets(line, sizeof(line), f)) {
            std::string s(line);
            if (!s.empty() && s.back() != '\n' && !feof(f)) { bad = "PLY のヘッダの行が長すぎる"; break; }
            while (!s.empty() && (s.back()=='\n'||s.back()=='\r')) s.pop_back();
            if (s.rfind("format ascii",0)==0) ascii = true;
            else if (s.rfind("format binary_little_endian",0)==0) bin_le = true;
            else if (s.rfind("format binary_big_endian",0)==0) bin_be = true;
            else if (s.rfind("element vertex",0)==0) {
                // 「element vertex」の後ろに数が無い行で文字列の外を読まない
                nv = s.size() > 15 ? strtoull(s.c_str() + 15, nullptr, 10) : 0;
                in_vertex = true;
                if (!seen_element) vertex_first = true;
                seen_element = true;
            }
            else if (s.rfind("element ",0)==0) { in_vertex = false; seen_element = true; }
            else if (in_vertex && s.rfind("property ",0)==0) {
                char t[64], nm[64];
                if (s.rfind("property list",0)==0) { bad = "PLY の頂点に list の性質がある（未対応）"; break; }
                Prop pr;
                if (sscanf(s.c_str(), "property %63s %63s", t, nm) != 2 || !tinfo(t, pr.size, pr.kind))
                    { bad = "PLY の性質の型が読めない: " + s; break; }
                pr.name = nm;
                props.push_back(pr);
            } else if (s.rfind("end_header",0)==0) break;
        }
        if (bad.empty() && !vertex_first) bad = "PLY の最初の要素が vertex でない（未対応）";
        if (!bad.empty()) { fclose(f); err = bad; return false; }
        size_t cnt = (max_points && max_points < nv) ? max_points : nv;
        int ix=-1, iy=-1, iz=-1, stride=0;
        std::vector<int> offs(props.size());
        for (size_t i = 0; i < props.size(); ++i) {
            offs[i] = stride; stride += props[i].size;
            if (props[i].name=="x") ix=(int)i;
            if (props[i].name=="y") iy=(int)i;
            if (props[i].name=="z") iz=(int)i;
        }
        if (ix<0||iy<0||iz<0) { fclose(f); err="PLY に x/y/z が無い"; return false; }
        n = cnt; xyz.resize(n * 3);
        // **器が宣言している型**を返す。ascii でも binary でも同じ規則にする。
        // 読み手が何に展開したかではなく、ファイルが何と書いているかが基準になる。
        // 整数の型は double に正確に入るので 8 を返す（格子は刻み 1 で見つかる）。
        // 以前は int32 を float のビット列として読み、short は初期化していない
        // バイトを混ぜていた。
        if (src_bytes) {
            const int ax[3] = {ix, iy, iz};
            for (int d = 0; d < 3; ++d)
                src_bytes[d] = props[ax[d]].kind == 'f' ? props[ax[d]].size : 8;
        }
        if (ascii) {
            // 行ではなく数を読む（長い行で fgets が行を分けて列がずれていた。
            // 値の置き場も性質の数だけ取る。以前は 32 個に固定であふれた）。
            std::vector<double> vals(props.size());
            for (size_t i = 0; i < cnt; ++i) {
                for (size_t k = 0; k < props.size(); ++k)
                    if (fscanf(f, " %lf", &vals[k]) != 1) { fclose(f); err="行が足りない"; return false; }
                xyz[i*3]=vals[ix]; xyz[i*3+1]=vals[iy]; xyz[i*3+2]=vals[iz];
            }
        } else if (bin_le || bin_be) {
            std::vector<uint8_t> rec(stride);
            for (size_t i = 0; i < cnt; ++i) {
                if (fread(rec.data(),1,stride,f) != (size_t)stride) { fclose(f); err="短い"; return false; }
                auto rd = [&](int k)->double {
                    const int sz2 = props[k].size;
                    // 大きい端から読むときは、並べ替えながら写す（sz2 は 1/2/4/8）
                    uint8_t b[8] = {0};
                    const uint8_t* src = rec.data() + offs[k];
                    for (int q = 0; q < sz2 && q < 8; ++q) b[q] = bin_be ? src[sz2 - 1 - q] : src[q];
                    switch (props[k].kind) {
                    case 'f': if (sz2==4) { float v; memcpy(&v, b, 4); return v; }
                              { double v; memcpy(&v, b, 8); return v; }
                    case 'i': if (sz2==1) { int8_t v;  memcpy(&v, b, 1); return v; }
                              if (sz2==2) { int16_t v; memcpy(&v, b, 2); return v; }
                              { int32_t v; memcpy(&v, b, 4); return v; }
                    default:  if (sz2==1) return b[0];
                              if (sz2==2) { uint16_t v; memcpy(&v, b, 2); return v; }
                              { uint32_t v; memcpy(&v, b, 4); return v; }
                    }
                };
                xyz[i*3]=rd(ix); xyz[i*3+1]=rd(iy); xyz[i*3+2]=rd(iz);
            }
        } else { fclose(f); err="PLY の形式が未対応"; return false; }
        fclose(f);
        return true;
    }
    err = "未対応の拡張子";
    return false;
}
} // namespace pcc
