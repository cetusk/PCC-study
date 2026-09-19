#include "pcc/las.hpp"
#include <laszip/laszip_api.h>
#include <cstring>
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
// 扱えない型でも**サイズだけは正しく進めなければならない**。
// 進めないと後続フィールドの読み出し位置が全部ずれる（実測: extrabytes.las で
// Time が blue と一致すると誤検出した）。
static int eb_type_size(uint8_t dt) {
    static const int base[11] = {0, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8};
    if (dt <= 10) return base[dt];
    if (dt <= 20) { int b = base[(dt - 11) / 2 + 1]; return b * 2; }   // 2 要素配列
    if (dt <= 30) { int b = base[(dt - 21) / 2 + 1]; return b * 3; }   // 3 要素配列
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
            // 扱えない型。読まないが、バイト位置は必ず進める。
            // dt=0 は options の下位 4bit + 1 がサイズ（未定義バイト列）
            int sz = (dt == 0) ? (int)r[3] : eb_type_size(dt);
            off += sz;
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
        r[2] = static_cast<uint8_t>(dims[i].type);
        r[3] = (dims[i].has_scale ? 0x08 : 0) | (dims[i].has_offset ? 0x10 : 0);
        memcpy(r + 4, dims[i].name.c_str(),
               std::min<size_t>(31, dims[i].name.size()));
        memcpy(r + 4 + 32 + 4 + 24 * 3, dims[i].scale, 24);
        memcpy(r + 4 + 32 + 4 + 24 * 4, dims[i].offset, 24);
        memcpy(r + 4 + 32 + 4 + 24 * 5, dims[i].description.c_str(),
               std::min<size_t>(31, dims[i].description.size()));
    }
}

#define LZ_CHECK(call, what) \
    if ((call) != 0) { err = std::string("LASzip: ") + what; return false; }

bool read_las(const std::string& path, PointCloud& pc, std::string& err,
              size_t max_points) {
    laszip_POINTER lz = nullptr;
    LZ_CHECK(laszip_create(&lz), "create");
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
    for (int i = 0; i < 3; ++i) {
        pc.scale[i]  = (&h->x_scale_factor)[i];
        pc.offset[i] = (&h->x_offset)[i];
    }
    uint64_t total = h->extended_number_of_point_records ?
                     h->extended_number_of_point_records : h->number_of_point_records;
    size_t n = (max_points && max_points < total) ? max_points : (size_t)total;
    pc.n = n;

    for (uint32_t i = 0; i < h->number_of_variable_length_records; ++i) {
        const laszip_vlr_struct& v = h->vlrs[i];
        if (std::string(v.user_id) == "LASF_Spec" && v.record_id == 4)
            parse_extra_vlr(v.data, v.record_length_after_header, pc.extra);
    }

    // 取り出すフィールドを決める
    auto add = [&](const std::string& nm, FType t, bool ex) {
        pc.order.push_back(nm);
        Field f; f.ftype = t; f.is_extra = ex; f.v.resize(n);
        pc.fields[nm] = std::move(f);
    };
    bool has_gps = (pc.point_format == 1 || pc.point_format >= 3);
    bool has_rgb = (pc.point_format == 2 || pc.point_format == 3 ||
                    pc.point_format == 5 || pc.point_format == 7 ||
                    pc.point_format == 8 || pc.point_format == 10);
    bool has_nir = (pc.point_format == 8 || pc.point_format == 10);
    add("intensity", FType::U16, false);
    add("bit_fields", FType::U8, false);
    add("classification", FType::U8, false);
    add("user_data", FType::U8, false);
    add("scan_angle", FType::I16, false);
    add("point_source_id", FType::U16, false);
    if (has_gps) add("gps_time", FType::F64, false);
    if (has_rgb) { add("red", FType::U16, false); add("green", FType::U16, false);
                   add("blue", FType::U16, false); }
    if (has_nir) add("nir", FType::U16, false);
    for (const auto& e : pc.extra) add(e.name, e.type, true);

    pc.X.resize(n); pc.Y.resize(n); pc.Z.resize(n);
    laszip_point* p = nullptr;
    LZ_CHECK(laszip_get_point_pointer(lz, &p), "get_point");
    for (size_t i = 0; i < n; ++i) {
        LZ_CHECK(laszip_read_point(lz), "read_point");
        pc.X[i] = p->X; pc.Y[i] = p->Y; pc.Z[i] = p->Z;
        pc.fields["intensity"].v[i] = p->intensity;
        // 戻り番号など詰め込みビットは 1 つにまとめて扱う
        uint8_t bf = (uint8_t)(p->extended_return_number |
                               (p->extended_number_of_returns << 4));
        pc.fields["bit_fields"].v[i] = bf;
        pc.fields["classification"].v[i] =
            p->extended_classification ? p->extended_classification : p->classification;
        pc.fields["user_data"].v[i] = p->user_data;
        pc.fields["scan_angle"].v[i] = p->extended_scan_angle;
        pc.fields["point_source_id"].v[i] = p->point_source_ID;
        if (has_gps) { uint64_t b; memcpy(&b, &p->gps_time, 8);
                       pc.fields["gps_time"].v[i] = (int64_t)b; }
        if (has_rgb) { pc.fields["red"].v[i] = p->rgb[0];
                       pc.fields["green"].v[i] = p->rgb[1];
                       pc.fields["blue"].v[i] = p->rgb[2]; }
        if (has_nir) pc.fields["nir"].v[i] = p->rgb[3];
        for (const auto& e : pc.extra)
            if (p->num_extra_bytes >= e.byte_offset + ftype_size(e.type))
                pc.fields[e.name].v[i] = read_raw(p->extra_bytes + e.byte_offset, e.type);
    }
    laszip_close_reader(lz);
    laszip_destroy(lz);
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
    std::vector<ExtraDim> ex;
    int off = 0;
    for (auto e : pc.extra) if (kept(e.name)) { e.byte_offset = off;
                                                off += ftype_size(e.type); ex.push_back(e); }
    laszip_POINTER lz = nullptr;
    LZ_CHECK(laszip_create(&lz), "create");
    laszip_header h{};
    h.version_major = 1; h.version_minor = pc.version_minor;
    // LAS 1.0-1.2 = 227, 1.3 = 235（波形情報の欄が増える）, 1.4 = 375
    h.header_size = (pc.version_minor >= 4) ? 375 : (pc.version_minor == 3 ? 235 : 227);
    h.offset_to_point_data = h.header_size;
    h.point_data_format = pc.point_format;
    int base = pc.point_record_len;
    for (const auto& e : pc.extra) base -= ftype_size(e.type);
    h.point_data_record_length = (laszip_U16)(base + off);
    if (pc.version_minor >= 4) h.extended_number_of_point_records = pc.n;
    h.number_of_point_records = (pc.n < (1ull << 32)) ? (laszip_U32)pc.n : 0;
    for (int i = 0; i < 3; ++i) {
        (&h.x_scale_factor)[i] = scale_override ? scale_override[i] : pc.scale[i];
        (&h.x_offset)[i] = pc.offset[i];
    }
    strncpy(h.generating_software, "pccnorm", 31);
    LZ_CHECK(laszip_set_header(lz, &h), "set_header");
    if (!ex.empty()) {
        std::vector<uint8_t> vlr;
        build_extra_vlr(ex, vlr);
        LZ_CHECK(laszip_add_vlr(lz, "LASF_Spec", 4, (laszip_U16)vlr.size(),
                                "extra bytes", vlr.data()), "add_vlr");
    }
    LZ_CHECK(laszip_open_writer(lz, path.c_str(), 1), "open_writer");
    laszip_point* p = nullptr;
    LZ_CHECK(laszip_get_point_pointer(lz, &p), "get_point");
    auto get = [&](const std::string& nm, size_t i) -> int64_t {
        auto it = pc.fields.find(nm);
        if (it == pc.fields.end() || !kept(nm)) return 0;
        return it->second.v[i];
    };
    for (size_t i = 0; i < pc.n; ++i) {
        p->X = Xq ? (*Xq)[i] : pc.X[i];
        p->Y = Yq ? (*Yq)[i] : pc.Y[i];
        p->Z = Zq ? (*Zq)[i] : pc.Z[i];
        p->intensity = (laszip_U16)get("intensity", i);
        uint8_t bf = (uint8_t)get("bit_fields", i);
        p->extended_return_number = bf & 0x0F;
        p->extended_number_of_returns = (bf >> 4) & 0x0F;
        p->return_number = bf & 0x07;
        p->number_of_returns = (bf >> 4) & 0x07;
        uint8_t cl = (uint8_t)get("classification", i);
        p->extended_classification = cl;
        p->classification = (cl < 32) ? cl : 0;
        p->user_data = (laszip_U8)get("user_data", i);
        p->extended_scan_angle = (laszip_I16)get("scan_angle", i);
        p->scan_angle_rank = (laszip_I8)(get("scan_angle", i) * 0.006);
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
        if (off) {
            // LASzip が確保済みのバッファに書き込む。
            // ポインタを差し替えると LASzip 側の解放と二重になって落ちる。
            if (p->extra_bytes && p->num_extra_bytes >= off) {
                memset(p->extra_bytes, 0, (size_t)p->num_extra_bytes);
                for (const auto& e : ex)
                    write_raw(p->extra_bytes + e.byte_offset, e.type, pc.fields.at(e.name).v[i]);
            }
        }
        LZ_CHECK(laszip_write_point(lz), "write_point");
    }
    laszip_close_writer(lz);
    laszip_destroy(lz);
    return true;
}

} // namespace pcc

namespace pcc {
// PLY / KITTI .bin も読めるようにする（解析系サブコマンド用。座標のみ）
bool read_points_any(const std::string& path, std::vector<double>& xyz,
                     size_t& n, std::string& err, size_t max_points) {
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
        size_t cnt = sz / 16;
        if (max_points && max_points < cnt) cnt = max_points;
        std::vector<float> buf(cnt * 4);
        if (fread(buf.data(), 16, cnt, f) != cnt) { fclose(f); err = "短い"; return false; }
        fclose(f);
        n = cnt; xyz.resize(n * 3);
        for (size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) xyz[i*3+d] = buf[i*4+d];
        return true;
    }
    if (ends(".ply")) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { err = "開けない"; return false; }
        char line[512]; bool ascii = false, bin_le = false, bin_be = false; size_t nv = 0;
        std::vector<std::pair<std::string,int>> props;  // 名前, バイト数
        int esz[16] = {0};
        auto tsize = [](const std::string& t)->int {
            if (t=="char"||t=="uchar"||t=="int8"||t=="uint8") return 1;
            if (t=="short"||t=="ushort"||t=="int16"||t=="uint16") return 2;
            if (t=="int"||t=="uint"||t=="int32"||t=="uint32"||t=="float"||t=="float32") return 4;
            return 8;
        };
        bool in_vertex = false;
        while (fgets(line, sizeof(line), f)) {
            std::string s(line);
            while (!s.empty() && (s.back()=='\n'||s.back()=='\r')) s.pop_back();
            if (s.rfind("format ascii",0)==0) ascii = true;
            else if (s.rfind("format binary_little_endian",0)==0) bin_le = true;
            else if (s.rfind("format binary_big_endian",0)==0) bin_be = true;
            else if (s.rfind("element vertex",0)==0) { nv = strtoull(s.c_str()+15,nullptr,10); in_vertex = true; }
            else if (s.rfind("element ",0)==0) in_vertex = false;
            else if (in_vertex && s.rfind("property ",0)==0) {
                char t[64], nm[64];
                if (sscanf(s.c_str(), "property %63s %63s", t, nm) == 2)
                    props.push_back({nm, tsize(t)});
            } else if (s.rfind("end_header",0)==0) break;
        }
        (void)esz;
        size_t cnt = (max_points && max_points < nv) ? max_points : nv;
        n = cnt; xyz.resize(n * 3);
        int ix=-1, iy=-1, iz=-1, stride=0;
        std::vector<int> offs(props.size());
        for (size_t i = 0; i < props.size(); ++i) {
            offs[i] = stride; stride += props[i].second;
            if (props[i].first=="x") ix=(int)i;
            if (props[i].first=="y") iy=(int)i;
            if (props[i].first=="z") iz=(int)i;
        }
        if (ix<0||iy<0||iz<0) { fclose(f); err="PLY に x/y/z が無い"; return false; }
        if (ascii) {
            for (size_t i = 0; i < cnt; ++i) {
                if (!fgets(line, sizeof(line), f)) { fclose(f); err="行が足りない"; return false; }
                double vals[32]; int k = 0;
                char* p = line;
                while (k < (int)props.size() && *p) { vals[k++] = strtod(p, &p); while (*p==' ') ++p; }
                xyz[i*3]=vals[ix]; xyz[i*3+1]=vals[iy]; xyz[i*3+2]=vals[iz];
            }
        } else if (bin_le || bin_be) {
            std::vector<uint8_t> rec(stride);
            for (size_t i = 0; i < cnt; ++i) {
                if (fread(rec.data(),1,stride,f) != (size_t)stride) { fclose(f); err="短い"; return false; }
                auto rd = [&](int k)->double {
                    int sz2 = props[k].second;
                    uint8_t b[8];
                    memcpy(b, rec.data()+offs[k], sz2);
                    if (bin_be) for (int a = 0; a < sz2/2; ++a) std::swap(b[a], b[sz2-1-a]);
                    if (sz2==4) { float v; memcpy(&v, b, 4); return v; }
                    double v; memcpy(&v, b, 8); return v;
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
