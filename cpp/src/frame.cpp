#include "pcc/frame.hpp"
#include "pcc/normalize.hpp"
#include <cstring>
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
        if (it->second.is_const) f.col[nm].assign(pc.n, it->second.cval);
        else f.col[nm] = it->second.v;
    }

    put_envelope(pc, f);
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
        put<int32_t>(e, (int32_t)x.type);
        put<uint8_t>(e, x.has_scale ? 1 : 0);
        put<uint8_t>(e, x.has_offset ? 1 : 0);
        for (int i = 0; i < 3; ++i) put<double>(e, x.scale[i]);
        for (int i = 0; i < 3; ++i) put<double>(e, x.offset[i]);
        put_str(e, x.description);
        put<int32_t>(e, x.byte_offset);
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
    f.plan = plan.to_json();
    (void)err;
    return true;
}

bool frame_to_las(const Frame& f, PointCloud& pc, std::string& err) {
    if (f.source_kind != "las") { err = "封筒が LAS ではない"; return false; }
    pc.n = f.n;
    for (int i = 0; i < 3; ++i) { pc.scale[i] = f.scale[i]; pc.offset[i] = f.offset[i]; }
    const std::vector<int64_t>* g[3];
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
        fl.v = it->second;
        pc.order.push_back(s.name);
        pc.fields[s.name] = std::move(fl);
    }
    size_t p = 0; uint32_t ne = 0;
    const std::vector<uint8_t>& e = f.envelope;
    if (!get(e, p, pc.point_format) || !get(e, p, pc.point_record_len) ||
        !get(e, p, pc.version_minor) || !get(e, p, ne)) { err = "封筒が短い"; return false; }
    for (uint32_t i = 0; i < ne; ++i) {
        ExtraDim x; int32_t t = 0; uint8_t hs = 0, ho = 0;
        if (!get_str(e, p, x.name) || !get(e, p, t) || !get(e, p, hs) || !get(e, p, ho))
            { err = "封筒の ExtraBytes が壊れている"; return false; }
        x.type = (FType)t; x.has_scale = hs; x.has_offset = ho;
        for (int k = 0; k < 3; ++k) if (!get(e, p, x.scale[k])) { err = "封筒 scale"; return false; }
        for (int k = 0; k < 3; ++k) if (!get(e, p, x.offset[k])) { err = "封筒 offset"; return false; }
        if (!get_str(e, p, x.description) || !get(e, p, x.byte_offset))
            { err = "封筒 description"; return false; }
        pc.extra.push_back(x);
    }
    return true;
}

// ---- KITTI .bin（x,y,z,intensity の float32）
static bool load_kitti(const std::string& path, Frame& f, std::string& err, size_t maxp) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { err = "開けない: " + path; return false; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    size_t n = sz / 16;
    if (maxp && n > maxp) n = maxp;
    std::vector<float> buf(n * 4);
    if (fread(buf.data(), 4, n * 4, fp) != n * 4) { fclose(fp); err = "短い"; return false; }
    fclose(fp);
    f.n = n; f.source_kind = "kitti"; f.source_bytes = (uint64_t)sz;
    f.geom_repr = "f32bits";
    const char* nm[4] = {"X", "Y", "Z", "intensity"};
    for (int c = 0; c < 4; ++c) {
        ColSpec s; s.name = nm[c]; s.ftype = FType::F32;
        s.role = (c < 3) ? Role::Geometry : Role::Attribute;
        f.schema.push_back(s);
        std::vector<int64_t> v(n);
        for (size_t i = 0; i < n; ++i) {
            uint32_t b; memcpy(&b, &buf[i * 4 + c], 4); v[i] = (int64_t)b;
        }
        f.col[nm[c]] = std::move(v);
    }
    return true;
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
        if (!read_points_any(path, xyz, n, err, max_points)) return false;
        f.n = n; f.source_kind = "ply"; f.geom_repr = "f64bits";
        const char* nm[3] = {"X", "Y", "Z"};
        for (int c = 0; c < 3; ++c) {
            ColSpec s; s.name = nm[c]; s.ftype = FType::F64; s.role = Role::Geometry;
            f.schema.push_back(s);
            std::vector<int64_t> v(n);
            for (size_t i = 0; i < n; ++i) { double d = xyz[i * 3 + c]; uint64_t b;
                memcpy(&b, &d, 8); v[i] = (int64_t)b; }
            f.col[nm[c]] = std::move(v);
        }
        return true;
    }
    err = "対応していない拡張子: " + path;
    return false;
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
            v.insert(v.end(), kv.second.begin() + b, kv.second.begin() + e);
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
    for (const auto& kv : f.col)
        g.col[kv.first] = std::vector<int64_t>(kv.second.begin(), kv.second.begin() + g.n);
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
    if (f.plan.empty()) return true;
    Plan plan = Plan::from_json(f.plan);
    std::map<std::string, std::vector<int64_t>> fields, ext;
    for (const auto& c : f.schema) {
        if (c.role == Role::Geometry) continue;
        auto it = f.col.find(c.name);
        if (c.storage == Storage::Residual) {
            if (it == f.col.end()) { err = "残差の列が無い: " + c.name; return false; }
            ext[c.name] = std::move(it->second);
            f.col.erase(it);
        } else if (c.storage == Storage::Raw) {
            if (it == f.col.end()) { err = "列が無い: " + c.name; return false; }
            fields[c.name] = it->second;
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

} // namespace pcc
