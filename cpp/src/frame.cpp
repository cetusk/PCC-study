#include "pcc/frame.hpp"
#include "pcc/grid.hpp"
#include "pcc/dettrig.hpp"
#include "pcc/normalize.hpp"
#include <sys/stat.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <set>

namespace pcc {

static void put_envelope(const PointCloud& pc, Frame& f);

// ---- 封筒（元の器を再生成するのに要る情報）の直列化
namespace {
template <class T> void put(std::vector<uint8_t>& o, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    o.insert(o.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t>& o, const std::string& s) {
    put<uint16_t>(o, (uint16_t)s.size());
    o.insert(o.end(), s.begin(), s.end());
}
template <class T> bool get(const std::vector<uint8_t>& b, size_t& p, T& v) {
    if (p + sizeof(T) > b.size()) return false;
    memcpy(&v, b.data() + p, sizeof(T)); p += sizeof(T); return true;
}
void put_uv(std::vector<uint8_t>& o, uint64_t v) {
    while (v >= 0x80) { o.push_back((uint8_t)(v | 0x80)); v >>= 7; }
    o.push_back((uint8_t)v);
}
bool get_uv(const std::vector<uint8_t>& b, size_t& p, uint64_t& v) {
    v = 0;
    for (int sh = 0; sh < 64; sh += 7) {
        if (p >= b.size()) return false;
        const uint8_t c = b[p++];
        v |= (uint64_t)(c & 0x7F) << sh;
        if (!(c & 0x80)) return true;
    }
    return false;
}
bool get_str(const std::vector<uint8_t>& b, size_t& p, std::string& s) {
    uint16_t l; if (!get(b, p, l)) return false;
    if (p + l > b.size()) return false;
    s.assign((const char*)b.data() + p, l); p += l; return true;
}
} // namespace

bool frame_from_las(const PointCloud& pc, Frame& f, std::string& err) {
    f.n = pc.n;
    f.source_kind = "las";
    f.source_bytes = pc.src_bytes;
    f.geom_repr = "int";
    for (int i = 0; i < 3; ++i) { f.scale[i] = pc.scale[i]; f.offset[i] = pc.offset[i]; }
    f.geom[0] = "X"; f.geom[1] = "Y"; f.geom[2] = "Z";

    auto add_geom = [&](const char* nm, const std::vector<int32_t>& s) {
        ColSpec c; c.name = nm; c.ftype = FType::I32; c.role = Role::Geometry;
        f.schema.push_back(c);
        std::vector<int64_t> v(pc.n);
        for (size_t i = 0; i < pc.n; ++i) v[i] = s[i];
        f.col[nm] = std::move(v);
    };
    add_geom("X", pc.X); add_geom("Y", pc.Y); add_geom("Z", pc.Z);

    for (const auto& nm : pc.order) {
        auto it = pc.fields.find(nm);
        if (it == pc.fields.end()) continue;
        ColSpec c; c.name = nm; c.ftype = it->second.ftype;
        c.role = (nm == "gps_time") ? Role::Time : Role::Attribute;
        c.is_extra = it->second.is_extra;
        f.schema.push_back(c);
        if (it->second.is_const) f.col[nm].from_const(it->second.cval, pc.n);
        else f.col[nm] = it->second.v;
    }

    put_envelope(pc, f);
    apply_column_grid(f);
    (void)err;
    return true;
}

static void put_envelope(const PointCloud& pc, Frame& f) {
    // 封筒: 点形式・レコード長・バージョン・ExtraBytes 定義
    std::vector<uint8_t>& e = f.envelope; e.clear();
    put<uint8_t>(e, pc.point_format);
    put<uint16_t>(e, pc.point_record_len);
    put<uint8_t>(e, pc.version_minor);
    put<uint32_t>(e, (uint32_t)pc.extra.size());
    for (const auto& x : pc.extra) {
        put_str(e, x.name);
        put(e, (int32_t)x.nbytes);
        put(e, x.raw_dt); put(e, x.raw_opts);
        put<int32_t>(e, (int32_t)x.type);
        put<uint8_t>(e, x.has_scale ? 1 : 0);
        put<uint8_t>(e, x.has_offset ? 1 : 0);
        for (int i = 0; i < 3; ++i) put<double>(e, x.scale[i]);
        for (int i = 0; i < 3; ++i) put<double>(e, x.offset[i]);
        put_str(e, x.description);
        put<int32_t>(e, x.byte_offset);
    }
    // 元のファイルの VLR（測地系・WKT など）。捨てると往復で消える。
    // ヘッダの欄
    put<uint16_t>(e, pc.file_source_id);
    put<uint16_t>(e, pc.global_encoding);
    put<uint16_t>(e, pc.creation_day);
    put<uint16_t>(e, pc.creation_year);
    put_str(e, pc.system_identifier);
    put_str(e, pc.generating_software);
    put_str(e, pc.extra_vlr_desc);
    put<uint32_t>(e, (uint32_t)pc.extra_vlr.size());
    e.insert(e.end(), pc.extra_vlr.begin(), pc.extra_vlr.end());
    auto put_vlrs = [&](const std::vector<PointCloud::RawVlr>& vs) {
        put<uint32_t>(e, (uint32_t)vs.size());
        for (const auto& v : vs) {
            put_str(e, v.user_id);
            put<uint16_t>(e, v.record_id);
            put_str(e, v.description);
            put<uint32_t>(e, (uint32_t)v.data.size());
            e.insert(e.end(), v.data.begin(), v.data.end());
        }
    };
    put_vlrs(pc.vlrs);
    put_vlrs(pc.evlrs);
    // ここから先は「札つきの拡張」の並び: u8 札 | 長さ（可変長整数）| 中身。
    // **既定の値と同じものは書かない。**大半のファイルでは 1 byte も増えない
    // （ExtraBytes の VLR の位置を常に 4 byte 書くと extra の 1065 点で 0.03 bpp 伸びた）。
    // 読み手は知らない札を読み飛ばすので、後から札を足しても古い読み手が壊れない。
    auto put_tag = [&](uint8_t tag, const std::vector<uint8_t>& body) {
        put<uint8_t>(e, tag);
        put_uv(e, body.size());
        e.insert(e.end(), body.begin(), body.end());
    };
    // 札 1: ExtraBytes の VLR の元の位置（AHN4 のように途中にあるファイルだけ）
    if (pc.extra_vlr_pos != 0) {
        std::vector<uint8_t> b; put<uint32_t>(b, pc.extra_vlr_pos); put_tag(1, b);
    }
    // 札 2: 中身から数え直した値と食い違っていたヘッダの欄（hdr_mask の立った欄だけ）
    if (pc.hdr_mask != 0) {
        const uint16_t m = pc.hdr_mask;
        std::vector<uint8_t> b; put<uint16_t>(b, m);
        for (int k = 0; k < 6; ++k) if (m >> k & 1) put<double>(b, pc.hdr_minmax[k]);
        // 点数は可変長整数で書く（0 が多いので 1 byte ずつで済む）
        if (m >> 6 & 1) put_uv(b, pc.hdr_legacy_n);
        if (m >> 7 & 1) for (int k = 0; k < 5; ++k) put_uv(b, pc.hdr_by_ret[k]);
        if (m >> 8 & 1) for (int k = 0; k < 15; ++k) put_uv(b, pc.hdr_ext_by_ret[k]);
        if (m >> 9 & 1) b.insert(b.end(), pc.hdr_guid, pc.hdr_guid + 16);
        if (m >> 10 & 1) put<uint64_t>(b, pc.hdr_waveform_start);
        if (m >> 11 & 1) put_uv(b, pc.hdr_ext_n);
        put_tag(2, b);
    }
    // 札 3 / 4: ヘッダの中・直後のユーザーデータ
    if (!pc.user_in_header.empty()) put_tag(3, pc.user_in_header);
    if (!pc.user_after_header.empty()) put_tag(4, pc.user_after_header);
    // 札 5: ExtraBytes の VLR に**記述されていない**次元の番号（pc.extra の添字）。
    // 印を運ばないと、復元した側はそれを記述された次元と見なして VLR に載せてしまう。
    {
        std::vector<uint8_t> b;
        for (size_t k = 0; k < pc.extra.size(); ++k)
            if (pc.extra[k].undocumented) put_uv(b, k);
        if (!b.empty()) put_tag(5, b);
    }
}

// 正規化の計画を先に立ててから Frame を作る。
//
// これまでは「全列をコピーして Frame を作る → 計画を立てる → 要らない列を消す」
// の順だったので、pc と f が同じ内容を同時に持つ瞬間があり、そこがピークだった
// （100 万点の LAS で 147 MB のうち 95 MB がこの二重持ち）。
// 計画を先に立てれば、残す列だけを pc から**移して**渡せる。
bool frame_from_las_normalized(PointCloud& pc, Frame& f, bool residual_ops,
                               std::string& err) {
    Plan plan = analyze(pc, residual_ops);
    // 走査モデルはこの 3 列を幾何より前に必要とする。落とすと復号の最後まで
    // 復元されず、候補として提示できなくなる。定数列でも符号長は 0.001 bpp 程度。
    {
        static const char* need[] = {"point_source_id", "gps_time", "bit_fields"};
        std::vector<Op> keep_ops;
        for (const auto& o : plan.ops) {
            bool skip = false;
            for (const char* nm : need) if (o.target == nm) skip = true;
            if (!skip) keep_ops.push_back(o);
        }
        plan.ops.swap(keep_ops);
    }
    std::vector<std::string> keep;
    std::map<std::string, std::vector<int64_t>> external;
    apply_plan(pc, plan, keep, external);
    std::set<std::string> kept(keep.begin(), keep.end());

    f.n = pc.n;
    f.source_kind = "las";
    f.source_bytes = pc.src_bytes;
    f.geom_repr = "int";
    for (int i = 0; i < 3; ++i) { f.scale[i] = pc.scale[i]; f.offset[i] = pc.offset[i]; }
    f.geom[0] = "X"; f.geom[1] = "Y"; f.geom[2] = "Z";

    auto add_geom = [&](const char* nm, std::vector<int32_t>& s) {
        ColSpec c; c.name = nm; c.ftype = FType::I32; c.role = Role::Geometry;
        f.schema.push_back(c);
        std::vector<int64_t> v(pc.n);
        for (size_t i = 0; i < pc.n; ++i) v[i] = s[i];
        s.clear(); s.shrink_to_fit();       // 変換が済んだら元は要らない
        f.col[nm] = std::move(v);
    };
    add_geom("X", pc.X); add_geom("Y", pc.Y); add_geom("Z", pc.Z);

    for (const auto& nm : pc.order) {
        auto it = pc.fields.find(nm);
        if (it == pc.fields.end()) continue;
        ColSpec c; c.name = nm; c.ftype = it->second.ftype;
        c.role = (nm == "gps_time") ? Role::Time : Role::Attribute;
        c.is_extra = it->second.is_extra;
        auto e = external.find(nm);
        if (e != external.end()) {
            c.storage = Storage::Residual;
            f.col[nm] = std::move(e->second);
        } else if (kept.count(nm)) {
            c.storage = Storage::Raw;
            it->second.materialize(pc.n);          // 定数のまま残す列はここで展開
            f.col[nm] = std::move(it->second.v);   // コピーせずに移す
        } else {
            c.storage = Storage::Derived;          // 計画から復元するので持たない
        }
        it->second.v.clear();
        it->second.v.shrink_to_fit();
        f.schema.push_back(c);
    }
    put_envelope(pc, f);
    apply_column_grid(f);
    f.plan = plan.to_json();
    (void)err;
    return true;
}

bool frame_to_las(const Frame& fin, PointCloud& pc, std::string& err) {
    if (fin.source_kind != "las") { err = "封筒が LAS ではない"; return false; }
    // 粗い格子で割ってあれば掛け戻す。元の LAS と同じ整数・同じ scale に戻る。
    Frame tmp;
    const Frame* fp = &fin;
    if (!fin.coldiv.empty()) { tmp = fin; restore_column_grid(tmp); fp = &tmp; }
    const Frame& f = *fp;
    pc.n = f.n;
    for (int i = 0; i < 3; ++i) { pc.scale[i] = f.scale[i]; pc.offset[i] = f.offset[i]; }
    const Col* g[3];
    for (int i = 0; i < 3; ++i) {
        g[i] = f.get(f.geom[i]);
        if (!g[i]) { err = "幾何の列がない: " + f.geom[i]; return false; }
    }
    pc.X.resize(f.n); pc.Y.resize(f.n); pc.Z.resize(f.n);
    for (size_t i = 0; i < f.n; ++i) {
        pc.X[i] = (int32_t)(*g[0])[i];
        pc.Y[i] = (int32_t)(*g[1])[i];
        pc.Z[i] = (int32_t)(*g[2])[i];
    }
    for (const auto& s : f.schema) {
        if (s.role == Role::Geometry) continue;
        Field fl; fl.ftype = s.ftype; fl.is_extra = s.is_extra;
        auto it = f.col.find(s.name);
        if (it == f.col.end()) { err = "列がない: " + s.name; return false; }
        fl.v = it->second.to_vector();
        pc.order.push_back(s.name);
        pc.fields[s.name] = std::move(fl);
    }
    size_t p = 0; uint32_t ne = 0;
    const std::vector<uint8_t>& e = f.envelope;
    if (!get(e, p, pc.point_format) || !get(e, p, pc.point_record_len) ||
        !get(e, p, pc.version_minor) || !get(e, p, ne)) { err = "封筒が短い"; return false; }
    for (uint32_t i = 0; i < ne; ++i) {
        ExtraDim x; int32_t t = 0; uint8_t hs = 0, ho = 0;
        int32_t nb = 0;
        if (!get_str(e, p, x.name) || !get(e, p, nb) || !get(e, p, x.raw_dt)
            || !get(e, p, x.raw_opts) || !get(e, p, t) || !get(e, p, hs) || !get(e, p, ho))
            { err = "封筒の ExtraBytes が壊れている"; return false; }
        x.type = (FType)t; x.has_scale = hs; x.has_offset = ho; x.nbytes = nb;
        for (int k = 0; k < 3; ++k) if (!get(e, p, x.scale[k])) { err = "封筒 scale"; return false; }
        for (int k = 0; k < 3; ++k) if (!get(e, p, x.offset[k])) { err = "封筒 offset"; return false; }
        if (!get_str(e, p, x.description) || !get(e, p, x.byte_offset))
            { err = "封筒 description"; return false; }
        pc.extra.push_back(x);
    }
    assign_extra_cols(pc);                  // 読み込みと同じ規則で列の名前を作り直す
    if (!get(e, p, pc.file_source_id) || !get(e, p, pc.global_encoding) ||
        !get(e, p, pc.creation_day) || !get(e, p, pc.creation_year) ||
        !get_str(e, p, pc.system_identifier) ||
        !get_str(e, p, pc.generating_software))
        return true;                        // ヘッダの欄を持たない古い封筒
    {
        uint32_t el = 0;
        if (!get_str(e, p, pc.extra_vlr_desc) || !get(e, p, el)) return true;
        if (p + el > e.size()) { err = "封筒の ExtraBytes VLR が短い"; return false; }
        pc.extra_vlr.assign(e.begin() + p, e.begin() + p + el); p += el;
    }
    auto get_vlrs = [&](std::vector<PointCloud::RawVlr>& out) -> int {
        uint32_t nv = 0;
        if (!get(e, p, nv)) return 0;       // ここを持たない古い封筒
        for (uint32_t i = 0; i < nv; ++i) {
            PointCloud::RawVlr v; uint32_t dl = 0;
            if (!get_str(e, p, v.user_id) || !get(e, p, v.record_id) ||
                !get_str(e, p, v.description) || !get(e, p, dl)) return -1;
            if (p + dl > e.size()) return -1;
            v.data.assign(e.begin() + p, e.begin() + p + dl); p += dl;
            out.push_back(std::move(v));
        }
        return 1;
    };
    int r1 = get_vlrs(pc.vlrs);
    if (r1 < 0) { err = "封筒の VLR が壊れている"; return false; }
    if (r1 == 0) return true;
    if (get_vlrs(pc.evlrs) < 0) { err = "封筒の EVLR が壊れている"; return false; }
    pc.extra_vlr_pos = 0;
    pc.hdr_mask = 0;
    while (p < e.size()) {
        uint8_t tag = 0; uint64_t len = 0;
        if (!get(e, p, tag) || !get_uv(e, p, len) || len > e.size() - p)
            { err = "封筒の拡張が壊れている"; return false; }
        const std::vector<uint8_t> b(e.begin() + p, e.begin() + p + len);
        p += len;
        size_t q = 0;
        bool ok = true;
        if (tag == 1) {
            ok = get(b, q, pc.extra_vlr_pos);
        } else if (tag == 2) {
            uint16_t m = 0;
            ok = get(b, q, m);
            for (int k = 0; ok && k < 6; ++k) if (m >> k & 1) ok = get(b, q, pc.hdr_minmax[k]);
            uint64_t u = 0;
            auto get32 = [&](uint32_t& v) { bool r = get_uv(b, q, u) && u <= UINT32_MAX; v = (uint32_t)u; return r; };
            if (ok && (m >> 6 & 1)) ok = get32(pc.hdr_legacy_n);
            if (m >> 7 & 1) for (int k = 0; ok && k < 5; ++k) ok = get32(pc.hdr_by_ret[k]);
            if (m >> 8 & 1) for (int k = 0; ok && k < 15; ++k) ok = get_uv(b, q, pc.hdr_ext_by_ret[k]);
            if (ok && (m >> 9 & 1)) {
                ok = q + 16 <= b.size();
                if (ok) { memcpy(pc.hdr_guid, b.data() + q, 16); q += 16; }
            }
            if (ok && (m >> 10 & 1)) ok = get(b, q, pc.hdr_waveform_start);
            if (ok && (m >> 11 & 1)) ok = get_uv(b, q, pc.hdr_ext_n);
            pc.hdr_mask = m;
            pc.hdr_known = true;
        } else if (tag == 3) {
            pc.user_in_header = b; q = b.size();
        } else if (tag == 4) {
            pc.user_after_header = b; q = b.size();
        } else if (tag == 5) {
            while (ok && q < b.size()) {
                uint64_t k = 0;
                ok = get_uv(b, q, k) && k < pc.extra.size();
                if (ok) pc.extra[(size_t)k].undocumented = true;
            }
        } else {
            continue;                       // 知らない札は読み飛ばす
        }
        if (!ok || q != b.size()) { err = "封筒の拡張 " + std::to_string(tag) + " が壊れている"; return false; }
    }
    return true;
}

// ---- 浮動小数の列を、格子に乗っていれば整数に直す
//
// 点群を float で持つ器は、中身が整数のことがある（KITTI の xyz は 1 mm 刻み）。
// ビットパターンのまま符号化すると値の構造が使えないので、刻みを見つけて直す。
// 採るのは**全点でビット列を再生できると証明できたときだけ**である。
//
// 負のゼロは整数を経ると符号が消える。値は等しいがビット列は違うので、
// 位置を封筒に載せて元の器を再生できるようにする。
namespace {

struct GridCol {
    std::string name;
    double step = 0;
    // 十進の文字列から来た列は「10^dec_exp で割る」でないと元のビット列に戻らない。
    // 刻みだけを載せても足りないので、道順も封筒に入れる。
    int32_t dec_exp = -1;
    std::vector<uint64_t> neg_zero;
};

// 封筒に格子の情報を書く。
// 'FGRD' | u8 列数 | 列ごとに（名前・刻み・道順・-0.0 の位置）
void put_grid_envelope(const std::vector<GridCol>& gc, Frame& f) {
    std::vector<uint8_t>& e = f.envelope;
    e.clear();
    if (gc.empty()) return;
    const char m[4] = {'F', 'G', 'R', 'D'};
    e.insert(e.end(), m, m + 4);
    put(e, (uint8_t)gc.size());
    for (const auto& c : gc) {
        put_str(e, c.name);
        put(e, c.step);
        put(e, c.dec_exp);
        put(e, (uint32_t)c.neg_zero.size());
        for (uint64_t i : c.neg_zero) put(e, i);
    }
}

}  // namespace

// ---- KITTI .bin（x,y,z,intensity の float32）
static bool load_kitti(const std::string& path, Frame& f, std::string& err, size_t maxp) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { err = "開けない: " + path; return false; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    // KITTI の .bin は 1 点 16 byte（float32 × 4）が並ぶだけで、ヘッダも尾も無い。
    // 16 の倍数でなければ KITTI ではない。黙って端数を落とすと可逆でなくなる。
    if (sz < 0 || sz % 16 != 0) {
        fclose(fp);
        err = "KITTI .bin の長さが 16 byte の倍数でない（" + std::to_string(sz) + " byte）";
        return false;
    }
    size_t n = (size_t)sz / 16;
    if (maxp && n > maxp) n = maxp;
    std::vector<float> buf(n * 4);
    if (fread(buf.data(), 4, n * 4, fp) != n * 4) { fclose(fp); err = "短い"; return false; }
    fclose(fp);
    f.n = n; f.source_kind = "kitti"; f.source_bytes = (uint64_t)sz;
    const char* nm[4] = {"X", "Y", "Z", "intensity"};

    // 列ごとに 1 本に取り出してから格子を調べる（.bin は交互に並んでいる）。
    std::vector<std::vector<float>> cols(4, std::vector<float>(n));
    for (int c = 0; c < 4; ++c)
        for (size_t i = 0; i < n; ++i) cols[c][i] = buf[i * 4 + c];
    std::vector<float>().swap(buf);

    std::vector<GridFit> fit(4);
    for (int c = 0; c < 4; ++c) fit[c] = fit_grid_f32(cols[c].data(), n);
    // 幾何は 3 軸そろって整数に直せるときだけ直す。1 軸だけ整数では
    // 座標系が混ざり、幾何の候補が扱えない。
    const bool geom_int = fit[0].ok && fit[1].ok && fit[2].ok;
    f.geom_repr = geom_int ? "int" : "f32bits";

    std::vector<GridCol> gc;
    for (int c = 0; c < 4; ++c) {
        const bool as_int = (c < 3) ? geom_int : fit[c].ok;
        ColSpec s; s.name = nm[c]; s.ftype = FType::F32;
        s.role = (c < 3) ? Role::Geometry : Role::Attribute;
        f.schema.push_back(s);
        std::vector<int64_t> v(n);
        if (as_int) {
            const double st = fit[c].step;
            const double p10 = fit[c].dec_exp >= 0 ? exact_pow10(fit[c].dec_exp) : 0.0;
            for (size_t i = 0; i < n; ++i)
                v[i] = (int64_t)llround(fit[c].dec_exp >= 0 ? (double)cols[c][i] * p10
                                                            : (double)cols[c][i] / st);
            if (c < 3) { f.scale[c] = st; f.offset[c] = 0.0; }
            gc.push_back({nm[c], st, (int32_t)fit[c].dec_exp, fit[c].neg_zero});
        } else {
            for (size_t i = 0; i < n; ++i) {
                uint32_t b; memcpy(&b, &cols[c][i], 4); v[i] = (int64_t)b;
            }
        }
        std::vector<float>().swap(cols[c]);
        f.col[nm[c]] = std::move(v);
    }
    put_grid_envelope(gc, f);
    return true;
}

// PLY が ascii かどうか。読み手が double に展開する経路かを分けるために要る。
static bool is_ascii_ply(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    char line[256];
    bool a = false;
    for (int i = 0; i < 8 && fgets(line, sizeof(line), fp); ++i)
        if (strncmp(line, "format ascii", 12) == 0) { a = true; break; }
    fclose(fp);
    return a;
}

bool load_frame(const std::string& path, Frame& f, std::string& err, size_t max_points) {
    auto ends = [&](const char* s) {
        size_t l = strlen(s);
        return path.size() >= l && strcasecmp(path.c_str() + path.size() - l, s) == 0;
    };
    if (ends(".las") || ends(".laz")) {
        PointCloud pc;
        if (!read_las(path, pc, err, max_points)) return false;
        return frame_from_las(pc, f, err);
    }
    if (ends(".bin")) return load_kitti(path, f, err, max_points);
    if (ends(".ply")) {
        std::vector<double> xyz; size_t n = 0;
        int fb[3] = {8, 8, 8};
        if (!read_points_any(path, xyz, n, err, max_points, fb)) return false;
        f.n = n; f.source_kind = "ply";
        // 基準は**同じ列だけ**を、**器が宣言している型**で並べたときの大きさ。
        // ファイル全体と比べると、法線や信頼度も持つ PLY では x/y/z しか
        // 符号化していない側が不当に良く見える（既報の同じ誤りを繰り返さない）。
        // ascii でも宣言型を使うので、行ごとに規則が変わらない。
        // ascii の元の文字列はこれより大きいことが多いが、それは器の冗長であって
        // 値そのものの大きさではない。
        f.source_bytes = (uint64_t)n * (uint64_t)(fb[0] + fb[1] + fb[2]);
        const char* nm[3] = {"X", "Y", "Z"};
        std::vector<std::vector<double>> cols(3, std::vector<double>(n));
        for (int c = 0; c < 3; ++c)
            for (size_t i = 0; i < n; ++i) cols[c][i] = xyz[i * 3 + c];
        std::vector<double>().swap(xyz);

        // 格子を調べる幅は、**読み手が持っている幅**であって宣言型ではない。
        // ascii は strtod で double に展開しているので、宣言が float でも
        // 再現すべきビット列は double のものである。
        int rb[3] = {fb[0], fb[1], fb[2]};
        if (is_ascii_ply(path)) rb[0] = rb[1] = rb[2] = 8;
        std::vector<GridFit> fit(3);
        std::vector<std::vector<float>> f32(3);
        for (int c = 0; c < 3; ++c) {
            if (rb[c] == 4) {
                f32[c].resize(n);
                for (size_t i = 0; i < n; ++i) f32[c][i] = (float)cols[c][i];
                fit[c] = fit_grid_f32(f32[c].data(), n);
            } else {
                fit[c] = fit_grid_f64(cols[c].data(), n);
            }
        }
        const bool geom_int = fit[0].ok && fit[1].ok && fit[2].ok;
        const bool all_f32 = (rb[0] == 4 && rb[1] == 4 && rb[2] == 4);
        f.geom_repr = geom_int ? "int" : (all_f32 ? "f32bits" : "f64bits");

        std::vector<GridCol> gc;
        for (int c = 0; c < 3; ++c) {
            ColSpec s; s.name = nm[c];
            s.ftype = (rb[c] == 4) ? FType::F32 : FType::F64;
            s.role = Role::Geometry;
            f.schema.push_back(s);
            std::vector<int64_t> v(n);
            if (geom_int) {
                const double st = fit[c].step;
                const double p10 = fit[c].dec_exp >= 0 ? exact_pow10(fit[c].dec_exp) : 0.0;
                for (size_t i = 0; i < n; ++i)
                    v[i] = (int64_t)llround(fit[c].dec_exp >= 0 ? cols[c][i] * p10
                                                                : cols[c][i] / st);
                f.scale[c] = st; f.offset[c] = 0.0;
                gc.push_back({nm[c], st, (int32_t)fit[c].dec_exp, fit[c].neg_zero});
            } else if (all_f32) {
                for (size_t i = 0; i < n; ++i) {
                    float ff = (float)cols[c][i]; uint32_t b;
                    memcpy(&b, &ff, 4); v[i] = (int64_t)b;
                }
            } else {
                for (size_t i = 0; i < n; ++i) { uint64_t b;
                    memcpy(&b, &cols[c][i], 8); v[i] = (int64_t)b; }
            }
            std::vector<float>().swap(f32[c]);
            std::vector<double>().swap(cols[c]);
            f.col[nm[c]] = std::move(v);
        }
        put_grid_envelope(gc, f);
        return true;
    }
    err = "対応していない拡張子: " + path;
    return false;
}

// 封筒から格子の情報を取り出す。
static bool get_grid_cols(const Frame& f, std::vector<GridCol>& out);

void drop_grid_cols(Frame& f, const std::vector<std::string>& names) {
    std::vector<GridCol> gc;
    if (!get_grid_cols(f, gc)) return;
    std::vector<GridCol> keep;
    for (auto& c : gc)
        if (std::find(names.begin(), names.end(), c.name) == names.end()) keep.push_back(std::move(c));
    put_grid_envelope(keep, f);
}

static bool get_grid_cols(const Frame& f, std::vector<GridCol>& out) {
    const std::vector<uint8_t>& e = f.envelope;
    if (e.size() < 5 || memcmp(e.data(), "FGRD", 4) != 0) return false;
    size_t p = 4;
    uint8_t nc = 0;
    if (!get(e, p, nc)) return false;
    for (int i = 0; i < (int)nc; ++i) {
        GridCol c; uint32_t nz = 0;
        if (!get_str(e, p, c.name) || !get(e, p, c.step) || !get(e, p, c.dec_exp)
            || !get(e, p, nz)) return false;
        if (nz > (e.size() - p) / 8) return false;   // 壊れた個数で巨大な確保をしない
        if (c.dec_exp > 22) return false;             // 符号化側は 1〜15 しか作らない（exact_pow10）
        c.neg_zero.resize(nz);
        for (uint32_t j = 0; j < nz; ++j)
            if (!get(e, p, c.neg_zero[j])) return false;
        out.push_back(std::move(c));
    }
    return true;
}

bool frame_to_kitti_bin(const Frame& f, const std::string& path, std::string& err) {
    if (f.source_kind != "kitti") { err = "出所が kitti ではない"; return false; }
    // 格子に乗った列が 1 本も無いとき、封筒は空である（それで正しい）。
    // 空でないのに読めないときだけ壊れている。
    std::vector<GridCol> gc;
    if (!f.envelope.empty() && !get_grid_cols(f, gc)) { err = "封筒の格子の情報が壊れている"; return false; }
    if (f.geom_repr == "polar") { err = "極座標のまま書こうとした（先に denormalize_frame を通す）"; return false; }
    const char* nm[4] = {"X", "Y", "Z", "intensity"};
    const size_t n = (size_t)f.n;
    std::vector<float> buf(n * 4, 0.0f);
    for (int c = 0; c < 4; ++c) {
        const Col* v = f.get(nm[c]);
        if (!v) { err = std::string("列が無い: ") + nm[c]; return false; }
        const GridCol* g = nullptr;
        for (const auto& q : gc) if (q.name == nm[c]) g = &q;
        if (g) {
            const double p10 = g->dec_exp >= 0 ? exact_pow10(g->dec_exp) : 0.0;
            for (size_t i = 0; i < n; ++i) {
                double k = (double)(*v)[i];
                buf[i * 4 + c] = (float)(g->dec_exp >= 0 ? k / p10 : k * g->step);
            }
            // 整数を経ると消える負のゼロを戻す。
            const float mz = -0.0f;
            // 非可逆のときは値そのものが変わっているので戻さない（検証した値と食い違う）
            if (f.fid.exact) for (uint64_t i : g->neg_zero) if (i < n) buf[i * 4 + c] = mz;
        } else if (c < 3 && f.geom_repr == "int") {
            // 非可逆の粗い格子（--eps）で入った幾何。刻みと原点は scale/offset にある。
            for (size_t i = 0; i < n; ++i)
                buf[i * 4 + c] = (float)((double)(*v)[i] * f.scale[c] + f.offset[c]);
        } else {
            // 格子に乗らなかった列は、ビットパターンをそのまま持っている。
            for (size_t i = 0; i < n; ++i) {
                uint32_t b = (uint32_t)(*v)[i];
                memcpy(&buf[i * 4 + c], &b, 4);
            }
        }
    }
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { err = "書けない: " + path; return false; }
    bool ok = fwrite(buf.data(), 4, n * 4, fp) == n * 4;
    fclose(fp);
    if (!ok) { err = "書き込みが短い"; return false; }
    return true;
}

std::string grid_summary(const Frame& f) {
    const std::vector<uint8_t>& e = f.envelope;
    if (e.size() < 5 || memcmp(e.data(), "FGRD", 4) != 0) return std::string();
    size_t p = 4;
    uint8_t nc = 0;
    if (!get(e, p, nc)) return std::string();
    std::string out;
    char b[160];
    for (int i = 0; i < (int)nc; ++i) {
        std::string nm; double st = 0; int32_t de = -1; uint32_t nz = 0;
        if (!get_str(e, p, nm) || !get(e, p, st) || !get(e, p, de) || !get(e, p, nz))
            return out;
        p += (size_t)nz * 8;                       // 位置そのものはここでは読まない
        snprintf(b, sizeof b, "%s%s 刻み %g%s", out.empty() ? "" : " / ", nm.c_str(), st,
                 de >= 0 ? "（十進）" : "");
        out += b;
        if (nz) { snprintf(b, sizeof b, "（-0.0 が %u 点）", nz); out += b; }
    }
    return out;
}

bool frames_equal(const Frame& a, const Frame& b, std::string& diff) {
    if (a.n != b.n) { diff = "点数が違う"; return false; }
    if (a.schema.size() != b.schema.size()) { diff = "列数が違う"; return false; }
    for (size_t i = 0; i < a.schema.size(); ++i)
        if (a.schema[i].name != b.schema[i].name) { diff = "列名が違う: " + a.schema[i].name; return false; }
    for (int i = 0; i < 3; ++i) {
        if (a.scale[i] != b.scale[i]) { diff = "scale が違う"; return false; }
        if (a.offset[i] != b.offset[i]) { diff = "offset が違う"; return false; }
    }
    for (const auto& s : a.schema) {
        const auto* x = a.get(s.name); const auto* y = b.get(s.name);
        if (!x || !y) { diff = "列がない: " + s.name; return false; }
        if (x->size() != y->size()) { diff = "長さが違う: " + s.name; return false; }
        for (size_t i = 0; i < x->size(); ++i)
            if ((*x)[i] != (*y)[i]) {
                char m[256];
                snprintf(m, sizeof m, "%s[%zu]: %lld != %lld", s.name.c_str(), i,
                         (long long)(*x)[i], (long long)(*y)[i]);
                diff = m; return false;
            }
    }
    return true;
}

void frame_world(const Frame& f, std::vector<double>& xyz) {
    xyz.assign(f.n * 3, 0.0);
    if (f.geom_repr == "polar") {
        // 逆変換は geom.cpp の polar_inverse と同じ polar_point（dettrig.hpp）を通る。
        const Col* qr = f.get(f.geom[0]);
        const Col* qa = f.get(f.geom[1]);
        const Col* qe = f.get(f.geom[2]);
        if (!qr || !qa || !qe) return;
        const double dr = f.polar_r_step, da = f.polar_ang_step;
        for (size_t i = 0; i < f.n; ++i) {
            const double r = (double)(*qr)[i] * dr;
            const double th = (double)(*qa)[i] * da;
            const double ph = (double)(*qe)[i] * da;
            polar_point(r, th, ph, f.polar_origin, f.polar_libm, &xyz[i * 3]);
        }
        return;
    }
    for (int c = 0; c < 3; ++c) {
        const auto* v = f.get(f.geom[c]);
        if (!v) continue;
        for (size_t i = 0; i < f.n; ++i) {
            double d;
            if (f.geom_repr == "f32bits") { uint32_t b = (uint32_t)(*v)[i]; float ff;
                memcpy(&ff, &b, 4); d = ff; }
            else if (f.geom_repr == "f64bits") { uint64_t b = (uint64_t)(*v)[i];
                memcpy(&d, &b, 8); }
            else d = (double)(*v)[i] * f.scale[c] + f.offset[c];
            xyz[i * 3 + c] = d;
        }
    }
}


// 先頭から n 点を切り出す。候補の順位付けに使うと、先頭が全体を代表して
// いないファイルで誤る（AHN3 の gps_time と nir で実際に起きた）。
// 代わりに sample_frame を使うこと。truncate_frame は互換のために残す。
Frame sample_frame(const Frame& f, size_t n, int chunks) {
    Frame g;
    g.n = std::min<uint64_t>(f.n, n);
    g.schema = f.schema;
    for (int i = 0; i < 3; ++i) { g.scale[i] = f.scale[i]; g.offset[i] = f.offset[i];
                                  g.geom[i] = f.geom[i]; }
    g.geom_repr = f.geom_repr; g.source_kind = f.source_kind;
    g.source_bytes = f.source_bytes; g.plan = f.plan; g.fid = f.fid;
    if (chunks < 1) chunks = 1;
    if ((uint64_t)chunks * 2 > g.n) chunks = 1;
    const size_t per = (size_t)(g.n / chunks);
    // 塊は等間隔に置く。差分符号器のために塊の中は連続させる
    // （塊の境目では差分が 1 点ぶん乱れるが、10 万点中 数点なので影響しない）。
    std::vector<size_t> beg(chunks);
    const size_t span = (size_t)f.n;
    for (int c = 0; c < chunks; ++c)
        beg[c] = (size_t)((double)c * (span - per) / (chunks > 1 ? chunks - 1 : 1));
    for (const auto& kv : f.col) {
        std::vector<int64_t> v;
        v.reserve(g.n);
        for (int c = 0; c < chunks; ++c) {
            size_t b = beg[c], e = std::min(span, b + per);
            for (size_t i = b; i < e; ++i) v.push_back(kv.second[i]);
        }
        v.resize(g.n);
        g.col[kv.first] = std::move(v);
    }
    return g;
}

Frame truncate_frame(const Frame& f, size_t n) {
    Frame g;
    g.n = std::min<uint64_t>(f.n, n);
    g.schema = f.schema;
    for (int i = 0; i < 3; ++i) { g.scale[i] = f.scale[i]; g.offset[i] = f.offset[i];
                                  g.geom[i] = f.geom[i]; }
    g.geom_repr = f.geom_repr; g.source_kind = f.source_kind;
    g.source_bytes = f.source_bytes; g.plan = f.plan; g.fid = f.fid;
    for (const auto& kv : f.col) {
        std::vector<int64_t> v((size_t)g.n);
        for (size_t i = 0; i < (size_t)g.n; ++i) v[i] = kv.second[i];
        g.col[kv.first] = std::move(v);
    }
    return g;
}

// ---- 正規化の計画を Frame に適用する / 復元する
bool normalize_frame(Frame& f, const PointCloud& pc, std::string& err, bool residual_ops) {
    // residual_ops=false: 「他フィールドとの残差」は符号器 C_ATTR_XREF が担当するので
    // 計画からは外し、厳密な構造の操作（定数・完全な複製・整数アフィン）だけ残す。
    Plan plan = analyze(pc, residual_ops);
    // 走査モデルはこの 3 列を幾何より前に必要とする。落とすと復号の最後まで
    // 復元されず、候補として提示できなくなる。定数列でも符号長は 0.001 bpp 程度。
    {
        static const char* need[] = {"point_source_id", "gps_time", "bit_fields"};
        std::vector<Op> keep_ops;
        for (const auto& o : plan.ops) {
            bool skip = false;
            for (const char* nm : need) if (o.target == nm) skip = true;
            if (!skip) keep_ops.push_back(o);
        }
        plan.ops.swap(keep_ops);
    }
    std::vector<std::string> keep;
    std::map<std::string, std::vector<int64_t>> external;
    apply_plan(pc, plan, keep, external);
    std::vector<std::string> kept(keep.begin(), keep.end());
    for (auto& c : f.schema) {
        if (c.role == Role::Geometry) continue;
        auto e = external.find(c.name);
        if (e != external.end()) {
            c.storage = Storage::Residual;
            f.col[c.name] = std::move(e->second);
            continue;
        }
        bool in_keep = false;
        for (const auto& k : kept) if (k == c.name) { in_keep = true; break; }
        if (!in_keep) { c.storage = Storage::Derived; f.col.erase(c.name); }
        else c.storage = Storage::Raw;
    }
    f.plan = plan.to_json();
    (void)err;
    return true;
}

bool denormalize_frame(Frame& f, std::string& err) {
    // 極座標で入っているなら、まず座標を戻して元の器のビット列に直す。
    // 計画の有無とは独立なので、早期に返る前に行う。
    if (f.geom_repr == "polar") {
        std::vector<double> w;
        frame_world(f, w);
        for (int c = 0; c < 3; ++c) {
            std::vector<int64_t> v(f.n);
            for (size_t i = 0; i < f.n; ++i) {
                const double d = w[i * 3 + c];
                if (f.polar_base == "f32bits") {
                    float ff = (float)d; uint32_t bb;
                    std::memcpy(&bb, &ff, 4); v[i] = (int64_t)(uint64_t)bb;
                } else if (f.polar_base == "f64bits") {
                    uint64_t bb; std::memcpy(&bb, &d, 8); v[i] = (int64_t)bb;
                } else {
                    // 整数格子に戻す。刻みは取り込み時に見つけたもので、
                    // 量子化の誤差はこの丸めより粗い。
                    const double sc = f.scale[c] != 0 ? f.scale[c] : 1.0;
                    v[i] = (int64_t)std::llround((d - f.offset[c]) / sc);
                }
            }
            f.col[f.geom[c]] = std::move(v);
        }
        f.geom_repr = f.polar_base;
    }
    if (f.plan.empty()) return true;
    Plan plan = Plan::from_json(f.plan);
    std::map<std::string, std::vector<int64_t>> fields, ext;
    for (const auto& c : f.schema) {
        if (c.role == Role::Geometry) continue;
        auto it = f.col.find(c.name);
        if (c.storage == Storage::Residual) {
            if (it == f.col.end()) { err = "残差の列が無い: " + c.name; return false; }
            ext[c.name] = it->second.to_vector();
            f.col.erase(it);
        } else if (c.storage == Storage::Raw) {
            if (it == f.col.end()) { err = "列が無い: " + c.name; return false; }
            fields[c.name] = it->second.to_vector();
        }
    }
    if (!invert_plan(fields, ext, plan, f.n, err)) return false;
    for (auto& c : f.schema) {
        if (c.role == Role::Geometry) continue;
        auto it = fields.find(c.name);
        if (it == fields.end()) { err = "復元できなかった列: " + c.name; return false; }
        f.col[c.name] = std::move(it->second);
        c.storage = Storage::Raw;
    }
    return true;
}


namespace {
uint64_t gcdu(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = a % b; a = b; b = t; }
    return a;
}
}  // namespace

void apply_column_grid(Frame& f) {
    // PCC_COLDIV=0 で切れる。入れた前後を同じ二値で比べるために要る。
    static const bool ON = [] {
        const char* e = getenv("PCC_COLDIV");
        return !e || atoi(e) != 0;
    }();
    f.coldiv.clear();
    if (!ON) {
        for (int i = 0; i < 3; ++i) { f.coldiv_scale[i] = f.scale[i];
                                      f.coldiv_offset[i] = f.offset[i]; }
        return;
    }
    for (const auto& sp : f.schema) {
        auto it = f.col.find(sp.name);
        if (it == f.col.end()) continue;
        const Col& c = it->second;
        const size_t n = c.size();
        if (n < 3) continue;
        // 基準の取り方は列の性質で変える。
        //   整数の列   … 基準 0。最小値を引くとバイト境界がずれ、バイトごとに
        //                模型を持つ記号版の符号器が読めなくなる
        //                （extra.laz の U32 の ExtraBytes で +3.7 bpp 伸びた）
        //   実数の列   … 基準は最小値。値はビット列であって数ではないので、
        //                「値が割り切れる」に意味がない。意味があるのは
        //                仮数部の下位が揃って 0 という差の構造のほう
        const bool bits = (sp.ftype == FType::F32 || sp.ftype == FType::F64);
        int64_t mn = 0;
        if (bits) {
            mn = c[0];
            for (size_t i = 1; i < n; ++i) if (c[i] < mn) mn = c[i];
        }
        // 実数の列はビット列を int64 に入れてあるので、正負が混ざると
        // c[i] - mn が符号付きの桁あふれ（未定義）になりうる。符号なしで引く。
        // 整数の列は**絶対値**の最大公約数を取り、商も符号つきで持つ。
        // 以前は負の値を符号なしに読み替えていたので、刻みが 2 の冪でないと
        // 見つからず（-3 と 6 の公約数が 1 になる）、2 の冪なら見つかっても
        // 商が 2^62 近くに飛んで、負の値を持つ列が割る前より長くなっていた。
        auto mag = [&](int64_t v) -> uint64_t {
            return bits ? (uint64_t)v - (uint64_t)mn
                        : (v < 0 ? 0 - (uint64_t)v : (uint64_t)v);
        };
        uint64_t gu = 0;
        for (size_t i = 0; i < n; ++i) {
            gu = gcdu(gu, mag(c[i]));
            if (gu == 1) break;                // これ以上は縮まない
        }
        if (gu <= 1 || gu > (uint64_t)INT64_MAX) continue;
        const int64_t g = (int64_t)gu;
        // 割った後の幅が 1 以下（値が 2 種類まで）の列は、割っても縮まない。
        // 刻みを容器に書くぶんだけ損をする。
        {
            int64_t mx2 = c[0];
            for (size_t i = 1; i < n; ++i) if (c[i] > mx2) mx2 = c[i];
            int64_t mn2 = c[0];
            for (size_t i = 1; i < n; ++i) if (c[i] < mn2) mn2 = c[i];
            // 幅は符号なしで引く（正負の端が離れていると符号つきでは桁あふれする）
            const uint64_t wq = ((uint64_t)mx2 - (uint64_t)mn2) / gu;
            if (wq <= 1) {
                if (getenv("PCC_COLDIV_DEBUG"))
                    fprintf(stderr, "[粗い格子] %-16s 刻み %lld だが幅が %llu なので外す\n",
                            sp.name.c_str(), (long long)g, (unsigned long long)wq);
                continue;
            }
        }
        if (getenv("PCC_COLDIV_DEBUG")) {
            int64_t mx = c[0];
            for (size_t i = 1; i < n; ++i) if (c[i] > mx) mx = c[i];
            fprintf(stderr, "[粗い格子] %-16s ftype=%d base=%lld step=%lld 幅 %lld → %lld\n",
                    sp.name.c_str(), (int)sp.ftype, (long long)mn, (long long)g,
                    (long long)((uint64_t)mx - (uint64_t)mn),
                    (long long)(((uint64_t)mx - (uint64_t)mn) / gu));
        }
        std::vector<int64_t> q(n);
        // 整数の列は符号つきで割る（割り切れることは上で確かめてある）。
        // 戻すのは base + g*q を 2^64 を法として計算するので、どちらでも元に戻る。
        for (size_t i = 0; i < n; ++i)
            q[i] = bits ? (int64_t)(((uint64_t)c[i] - (uint64_t)mn) / gu) : c[i] / g;
        f.col[sp.name] = std::move(q);
        f.coldiv[sp.name] = {mn, g};
    }
    // 幾何は世界座標が動かないように scale/offset を合わせる。
    //   world = scale*v + offset = scale*(mn + g*q) + offset
    //         = (scale*g)*q + (offset + scale*mn)
    for (int i = 0; i < 3; ++i) {
        f.coldiv_scale[i] = f.scale[i];
        f.coldiv_offset[i] = f.offset[i];
        auto d = f.coldiv.find(f.geom[i]);
        if (d == f.coldiv.end()) continue;
        f.offset[i] += f.scale[i] * (double)d->second.first;
        f.scale[i] *= (double)d->second.second;
    }
}

void restore_column_grid(Frame& f) {
    if (f.coldiv.empty()) return;
    // 計算で戻すと丸めで漂う。覚えておいた値をそのまま書き戻す。
    for (int i = 0; i < 3; ++i) {
        f.scale[i] = f.coldiv_scale[i];
        f.offset[i] = f.coldiv_offset[i];
    }
    for (const auto& kv : f.coldiv) {
        auto it = f.col.find(kv.first);
        if (it == f.col.end()) continue;
        const Col& c = it->second;
        std::vector<int64_t> v(c.size());
        for (size_t i = 0; i < c.size(); ++i)
            v[i] = (int64_t)((uint64_t)kv.second.first
                             + (uint64_t)kv.second.second * (uint64_t)c[i]);
        f.col[kv.first] = std::move(v);
    }
    f.coldiv.clear();
}

std::string coldiv_summary(const Frame& f) {
    if (f.coldiv.empty()) return std::string();
    std::string out;
    char b[96];
    for (const auto& kv : f.coldiv) {
        snprintf(b, sizeof b, "%s%s %lld", out.empty() ? "" : " / ",
                 kv.first.c_str(), (long long)kv.second.second);
        out += b;
    }
    return out;
}

} // namespace pcc
