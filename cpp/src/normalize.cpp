#include "pcc/normalize.hpp"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>

namespace pcc {

std::string Op::describe() const {
    std::ostringstream o;
    if (kind == "drop_constant")  o << target << " 定数 " << value << " → 送らない";
    else if (kind == "drop_duplicate") o << target << " " << source << " と完全に同一 → 送らない";
    else if (kind == "drop_affine") o << target << " = (" << num << "*" << source
                                      << " + " << b << "*" << den << ")/" << den << " → 送らない";
    else if (kind == "residual_code") o << target << " " << source << " との残差を外部符号化";
    else o << target << ": " << kind;
    return o.str();
}

double entropy0(const std::vector<int64_t>& v) {
    std::unordered_map<int64_t, uint64_t> c;
    c.reserve(v.size() / 4 + 16);
    for (int64_t x : v) ++c[x];
    double n = (double)v.size(), h = 0;
    for (auto& kv : c) { double p = kv.second / n; h -= p * std::log2(p); }
    return h;
}

// 標本だけで差分のエントロピーを見積もる。相手選びの順位づけにしか使わないので、
// 採用の可否は選ばれた 1 組について全点で測り直す。
static double entropy0_diff_sampled(const std::vector<int64_t>& a,
                                    const std::vector<int64_t>& b,
                                    const std::vector<size_t>& idx) {
    std::unordered_map<int64_t, uint64_t> c;
    c.reserve(idx.size() / 4 + 16);
    for (size_t i : idx) ++c[a[i] - b[i]];
    double n = (double)idx.size(), h = 0;
    for (auto& kv : c) { double p = kv.second / n; h -= p * std::log2(p); }
    return h;
}

double entropy0_diff(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    std::unordered_map<int64_t, uint64_t> c;
    c.reserve(a.size() / 4 + 16);
    for (size_t i = 0; i < a.size(); ++i) ++c[a[i] - b[i]];
    double n = (double)a.size(), h = 0;
    for (auto& kv : c) { double p = kv.second / n; h -= p * std::log2(p); }
    return h;
}

static bool all_equal(const std::vector<int64_t>& v, int64_t& out) {
    if (v.empty()) return false;
    out = v[0];
    for (int64_t x : v) if (x != out) return false;
    return true;
}

// 標本から傾きを推定し、分母の小さい有理数に丸めてから全点で厳密に検証する
static bool fit_affine(const std::vector<int64_t>& a, const std::vector<int64_t>& s,
                       int64_t& num, int64_t& den, int64_t& b) {
    size_t n = a.size();
    size_t step = std::max<size_t>(1, n / 50000);
    double sx = 0, sy = 0, sxx = 0, sxy = 0; size_t m = 0;
    int64_t xmin = INT64_MAX, xmax = INT64_MIN;
    for (size_t i = 0; i < n; i += step) {
        double x = (double)s[i], y = (double)a[i];
        sx += x; sy += y; sxx += x * x; sxy += x * y; ++m;
        xmin = std::min(xmin, s[i]); xmax = std::max(xmax, s[i]);
    }
    if (m < 4 || xmin == xmax) return false;
    double denom = m * sxx - sx * sx;
    if (std::abs(denom) < 1e-12) return false;
    double A0 = (m * sxy - sx * sy) / denom;
    static const int64_t DENS[] = {1, 2, 4, 5, 8, 10, 100};
    for (int64_t d : DENS) {
        int64_t nu = (int64_t)std::llround(A0 * (double)d);
        if (nu == 0) continue;
        if (std::abs((double)nu / (double)d - A0) > 1e-6) continue;
        // rem = a*d - nu*s が全点で同じ定数か
        int64_t rem0 = a[0] * d - nu * s[0];
        bool ok = true;
        for (size_t i = 1; i < n; ++i)
            if (a[i] * d - nu * s[i] != rem0) { ok = false; break; }
        if (!ok) continue;
        if (rem0 % d != 0) continue;
        int64_t bb = rem0 / d;
        for (size_t i = 0; i < n; ++i)
            if (a[i] != (nu * s[i] + bb * d) / d) { ok = false; break; }
        if (!ok) continue;
        num = nu; den = d; b = bb;
        return true;
    }
    return false;
}

Plan analyze(const PointCloud& pc, bool enable_residual) {
    Plan plan;
    static const std::set<std::string> PROTECTED = {"bit_fields"};
    std::vector<std::string> names;
    std::map<std::string, int> pos;
    for (const auto& nm : pc.order)
        if (!PROTECTED.count(nm)) { pos[nm] = (int)names.size(); names.push_back(nm); }
    std::set<std::string> dropped;

    // 1) 定数
    for (const auto& nm : names) {
        int64_t v;
        if (all_equal(pc.fields.at(nm).v, v)) {
            Op o; o.kind = "drop_constant"; o.target = nm; o.value = v;
            plan.ops.push_back(o); dropped.insert(nm);
        }
    }
    // 2) 完全重複 / アフィン（参照元は「自分より前」かつ落とさないもの）
    for (const auto& nm : names) {
        if (dropped.count(nm)) continue;
        const auto& a = pc.fields.at(nm).v;
        bool found = false;
        for (const auto& s : names) {
            if (found || dropped.count(s) || pos[s] >= pos[nm]) continue;
            const auto& b = pc.fields.at(s).v;
            if (a == b) {
                Op o; o.kind = "drop_duplicate"; o.target = nm; o.source = s;
                plan.ops.push_back(o); dropped.insert(nm); found = true;
            }
        }
        for (const auto& s : names) {
            if (found || dropped.count(s) || pos[s] >= pos[nm]) continue;
            int64_t nu, de, bb;
            if (fit_affine(a, pc.fields.at(s).v, nu, de, bb)) {
                Op o; o.kind = "drop_affine"; o.target = nm; o.source = s;
                o.num = nu; o.den = de; o.b = bb;
                plan.ops.push_back(o); dropped.insert(nm); found = true;
            }
        }
    }
    // 3) 残差符号化（参照元は容器に残るフィールドに限る）
    //
    // 相手選びは標本で行う。全フィールド対を全点で回すと O(M^2 N) になり、
    // 4,929万点では 232 秒を要していた。順位づけに必要な精度は標本で足りる。
    // 採用の可否だけは、選ばれた 1 組について全点で測り直す。
    if (enable_residual) {
        const size_t SAMPLE = 250000;
        std::vector<size_t> idx;
        if (pc.n > SAMPLE * 2) {
            idx.reserve(SAMPLE);
            // 一様に間引く。連続した塊を避けるため素数の歩幅を使う
            size_t step = pc.n / SAMPLE;
            for (size_t i = 0, k = 0; k < SAMPLE && i < pc.n; i += step, ++k) idx.push_back(i);
        } else {
            idx.resize(pc.n);
            for (size_t i = 0; i < pc.n; ++i) idx[i] = i;
        }
        std::set<std::string> resid;
        for (const auto& nm : names) {
            if (dropped.count(nm)) continue;
            const auto& a = pc.fields.at(nm).v;
            std::string bs; double best_s = entropy0_diff_sampled(a, a, idx);
            // 自分との差分は 0 なので、基準は標本での h0 にする
            {
                std::unordered_map<int64_t, uint64_t> c;
                for (size_t i : idx) ++c[a[i]];
                double n2 = (double)idx.size(); best_s = 0;
                for (auto& kv : c) { double p = kv.second / n2; best_s -= p * std::log2(p); }
            }
            for (const auto& s : names) {
                if (s == nm || dropped.count(s) || resid.count(s)) continue;
                double h = entropy0_diff_sampled(a, pc.fields.at(s).v, idx);
                if (h < best_s) { best_s = h; bs = s; }
            }
            double h0 = 0, best = 0;
            if (!bs.empty()) {
                h0 = entropy0(a);
                best = entropy0_diff(a, pc.fields.at(bs).v);
                if (best >= h0) bs.clear();
            }
            if (!bs.empty() && h0 - best >= 1.0) {
                Op o; o.kind = "residual_code"; o.target = nm; o.source = bs;
                std::ostringstream ss; ss << "H " << h0 << " -> " << best << " bit";
                o.note = ss.str();
                plan.ops.push_back(o); resid.insert(nm);
            }
        }
    }
    return plan;
}

void apply_plan(const PointCloud& pc, const Plan& plan,
                std::vector<std::string>& keep,
                std::map<std::string, std::vector<int64_t>>& external) {
    std::set<std::string> drop;
    for (const auto& o : plan.ops) {
        if (o.kind == "residual_code") {
            const auto& a = pc.fields.at(o.target).v;
            const auto& s = pc.fields.at(o.source).v;
            std::vector<int64_t> r(a.size());
            for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - s[i];
            external[o.target] = std::move(r);
            drop.insert(o.target);
        } else drop.insert(o.target);
    }
    keep.clear();
    for (const auto& nm : pc.order) if (!drop.count(nm)) keep.push_back(nm);
}

bool invert_plan(std::map<std::string, std::vector<int64_t>>& f,
                 const std::map<std::string, std::vector<int64_t>>& ext,
                 const Plan& plan, size_t n, std::string& err) {
    std::vector<const Op*> pending;
    for (const auto& o : plan.ops) pending.push_back(&o);
    while (!pending.empty()) {
        std::vector<const Op*> still;
        bool progressed = false;
        for (const Op* o : pending) {
            if (!o->source.empty() && !f.count(o->source)) { still.push_back(o); continue; }
            if (o->kind == "drop_constant") f[o->target] = std::vector<int64_t>(n, o->value);
            else if (o->kind == "drop_duplicate") f[o->target] = f[o->source];
            else if (o->kind == "drop_affine") {
                const auto& s = f[o->source];
                std::vector<int64_t> a(n);
                for (size_t i = 0; i < n; ++i) a[i] = (o->num * s[i] + o->b * o->den) / o->den;
                f[o->target] = std::move(a);
            } else if (o->kind == "residual_code") {
                auto it = ext.find(o->target);
                if (it == ext.end()) { err = "外部ストリームが無い: " + o->target; return false; }
                const auto& s = f[o->source];
                std::vector<int64_t> a(n);
                for (size_t i = 0; i < n; ++i) a[i] = it->second[i] + s[i];
                f[o->target] = std::move(a);
            }
            progressed = true;
        }
        if (!progressed) { err = "正規化仕様の依存が解決できない"; return false; }
        pending.swap(still);
    }
    return true;
}

// --- 仕様の直列化（依存のない最小限の JSON）
static void esc(std::ostringstream& o, const std::string& s) {
    o << '"';
    for (char c : s) { if (c == '"' || c == '\\') o << '\\'; o << c; }
    o << '"';
}

std::string Plan::to_json() const {
    std::ostringstream o;
    o << "{\"grid\":" << (grid ? 1 : 0) << ",\"grid_bits\":" << grid_bits
      << ",\"max_err_m\":" << max_err_m << ",\"ops\":[";
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto& p = ops[i];
        if (i) o << ",";
        o << "{\"kind\":"; esc(o, p.kind);
        o << ",\"target\":"; esc(o, p.target);
        o << ",\"source\":"; esc(o, p.source);
        o << ",\"value\":" << p.value << ",\"num\":" << p.num
          << ",\"den\":" << p.den << ",\"b\":" << p.b << "}";
    }
    o << "]}";
    return o.str();
}

static std::string jstr(const std::string& s, size_t& i) {
    while (i < s.size() && s[i] != '"') ++i;
    ++i; std::string r;
    while (i < s.size() && s[i] != '"') { if (s[i] == '\\') ++i; r += s[i++]; }
    ++i; return r;
}
static int64_t jint(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ':' || s[i] == ' ')) ++i;
    size_t st = i;
    while (i < s.size() && (isdigit((unsigned char)s[i]) || s[i] == '-' || s[i] == '+')) ++i;
    return std::stoll(s.substr(st, i - st));
}

Plan Plan::from_json(const std::string& s) {
    Plan p;
    size_t i = s.find("\"grid\"");
    if (i != std::string::npos) { i += 6; p.grid = jint(s, i) != 0; }
    i = s.find("\"grid_bits\"");
    if (i != std::string::npos) { i += 11; p.grid_bits = (int)jint(s, i); }
    i = s.find("\"ops\"");
    if (i == std::string::npos) return p;
    while ((i = s.find("{\"kind\"", i)) != std::string::npos) {
        size_t j = i + 7;
        Op o;
        o.kind = jstr(s, j);
        j = s.find("\"target\"", j) + 8; o.target = jstr(s, j);
        j = s.find("\"source\"", j) + 8; o.source = jstr(s, j);
        j = s.find("\"value\"", j) + 7;  o.value = jint(s, j);
        j = s.find("\"num\"", j) + 5;    o.num = jint(s, j);
        j = s.find("\"den\"", j) + 5;    o.den = jint(s, j);
        j = s.find("\"b\"", j) + 3;      o.b = jint(s, j);
        p.ops.push_back(o);
        i = j;
    }
    return p;
}

std::string Plan::report() const {
    std::ostringstream o;
    if (ops.empty() && !grid) { o << "（正規化できる冗長は見つからなかった）"; return o.str(); }
    o << "検出した冗長 " << ops.size() << " 件\n";
    if (grid) o << "  [誤差有] xyz: 下位 " << grid_bits << " ビットを落とす\n";
    for (const auto& p : ops) o << "  [可逆] " << p.describe() << "\n";
    return o.str();
}

} // namespace pcc
