// pccnorm — 表現の正規化レイヤー（C++ 実装）
//
// 使い方:
//   pccnorm bench <file.laz> [--max-points N] [--no-residual] [--grid-bits K]
//
// Python 版 (python/bench_normalize.py) と同じ測定を行い、
// 毎回復元検証を通す。通らなければ結果を表示しない。
#include "pcc/las.hpp"
#include "pcc/normalize.hpp"
#include "pcc/pccfile.hpp"
#include "pcc/geom.hpp"
#include "pcc/analysis.hpp"
#include "pcc/budget.hpp"
#include "pcc/persist.hpp"
#include "pcc/diff.hpp"
#include "pcc/attr.hpp"
#include "pcc/distort.hpp"
#include "pcc/surfcode.hpp"
#include "pcc/pcc2.hpp"
#include "pcc/dettrig.hpp"

#include "pcc/rangecoder.hpp"
#include <cstdio>
#include <thread>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <random>
#include <algorithm>

using namespace pcc;

static double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double peak_gb() {
    rusage r{}; getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1024.0 / 1024.0;
}

// いま実際に使っている常駐量。ピークだけ見てもどこで積み上がったか判らない。
static double rss_mb() {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long total = 0, res = 0;
    if (fscanf(f, "%ld %ld", &total, &res) != 2) res = 0;
    fclose(f);
    return res * (double)sysconf(_SC_PAGESIZE) / 1048576.0;
}

static bool g_mem_trace = false;

// 幾何の軸 d の器の型が float32 なら、書き出すときに float に丸まる。
// 誤差は**利用者が受け取る値**で測る（元の値も float32 なので同じく丸めて比べる）。
// 以前はこの丸めを数えず、代わりに上限に 2% の余裕を持たせていた。KITTI の
// 1 frame で、戻した .bin の誤差が上限 0.002 m を 0.000003 m 超えていた。
static bool geom_is_f32(const Frame& f, int d) {
    for (const auto& c : f.schema) if (c.name == f.geom[d]) return c.ftype == FType::F32;
    return false;
}
static inline double as_output(const Frame& f, int d, double v) {
    return geom_is_f32(f, d) ? (double)(float)v : v;
}

// 非可逆の経路の検証。**属性はビット完全**、幾何だけ誤差上限の中に入っているか。
// 幾何も一致を求めると通らないし、逆に全部を誤差で見ると属性の壊れを見逃す。
static bool frames_close(const Frame& a, const Frame& b, double eps,
                         double& worst, std::string& diff) {
    if (a.n != b.n) { diff = "点数が違う"; return false; }
    if (a.schema.size() != b.schema.size()) { diff = "列数が違う"; return false; }
    for (const auto& c : a.schema) {
        bool isgeo = false;
        for (int i = 0; i < 3; ++i) if (c.name == a.geom[i]) isgeo = true;
        if (isgeo) continue;
        const Col* x = a.get(c.name);
        const Col* y = b.get(c.name);
        if (!x || !y) { diff = "列が無い: " + c.name; return false; }
        for (size_t i = 0; i < a.n; ++i)
            if ((*x)[i] != (*y)[i]) {
                char m[192];
                snprintf(m, sizeof m, "%s[%zu]: %lld != %lld", c.name.c_str(), i,
                         (long long)(*x)[i], (long long)(*y)[i]);
                diff = m; return false;
            }
    }
    std::vector<double> wa, wb;
    frame_world(a, wa);
    frame_world(b, wb);
    for (int d = 0; d < 3; ++d)
        if (geom_is_f32(a, d))
            for (size_t i = 0; i < a.n; ++i) {
                wa[i * 3 + d] = (double)(float)wa[i * 3 + d];
                wb[i * 3 + d] = (double)(float)wb[i * 3 + d];
            }
    worst = 0; size_t at = 0;
    for (size_t i = 0; i < a.n; ++i) {
        const double dx = wa[i * 3] - wb[i * 3];
        const double dy = wa[i * 3 + 1] - wb[i * 3 + 1];
        const double dz = wa[i * 3 + 2] - wb[i * 3 + 2];
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d > worst) { worst = d; at = i; }
    }
    // 余裕は倍精度の計算誤差の分だけ（以前は 2%）
    if (worst > eps * (1 + 1e-12)) {
        char m[192];
        snprintf(m, sizeof m, "点 %zu で誤差 %.6g m が上限 %.6g m を超えた", at, worst, eps);
        diff = m; return false;
    }
    return true;
}

// **ru_maxrss は exec をまたいで引き継がれる。**太った親から起動されると、
// 子のピークに親の分が乗ったまま出る（起動直後に 1.4 GB と出た）。
// 本当のピークは自分で /proc/self/statm を刻んで取る。
static std::atomic<double> g_rss_peak{0.0};
static std::atomic<bool> g_rss_stop{false};
static std::thread g_rss_thread;
static void rss_watch_start() {
    if (!g_mem_trace) return;
    g_rss_thread = std::thread([] {
        while (!g_rss_stop.load(std::memory_order_relaxed)) {
            double v = rss_mb();
            double p = g_rss_peak.load(std::memory_order_relaxed);
            while (v > p && !g_rss_peak.compare_exchange_weak(p, v)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    });
}
static void rss_watch_stop() {
    if (!g_rss_thread.joinable()) return;
    g_rss_stop.store(true);
    g_rss_thread.join();
    fprintf(stderr, "  [メモリ] %-28s %7.1f MB\n", "実測ピーク（3 ms 刻み）",
            g_rss_peak.load());
}
static void mem_mark(const char* what) {
    if (g_mem_trace)
        fprintf(stderr, "  [メモリ] %-28s 常駐 %7.1f MB\n", what, rss_mb());
}
static uint64_t fsize(const std::string& p) {
    struct stat st{}; return stat(p.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
}

// 解析用のコマンドは列を .v で直に読む。読み込みは値が 1 種類の列を配列にせず 1 値だけ
// 持つので、そのままでは範囲外を読む（`combine` に色だけの LAS を渡すと落ちた）。
// pack / unpack の本流は定数のまま扱えるので、ここは解析用のコマンドだけで呼ぶ。
static void materialize_all(PointCloud& pc) {
    for (auto& kv : pc.fields) kv.second.materialize(pc.n);
}

static int main_impl(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "使い方:\n"
            "  pccnorm bench   <file.laz> [--max-points N] [--no-residual] [--grid-bits K] [--no-delta] [--spatial] [--full-search]\n"
            "  pccnorm score   <file.laz> [--max-points N]      取得構造スコア\n"
            "  pccnorm octant  <file.laz> [--max-points N]      オクタント予測\n"
            "  pccnorm geom    <file.laz> [--max-points N] [--eps E]  幾何表現の選択\n"
            "  pccnorm lfs     <file.laz> [--max-points N] [--voxel V]  TSDF 経由の局所特徴サイズ\n"
            "  pccnorm persist <file.laz> [--max-points N]  パーシステントホモロジー（特徴と寿命）\n"
            "  pccnorm sphere  <R> <N> <sigma>              正解つき形状で lfs を検証\n"
            "  pccnorm residue <file.laz> [--max-points N]  正規化後に何が残っているか\n"
            "  pccnorm extern  <file.laz> [--max-points N]  各フィールドを外部に出すと得か\n"
            "  pccnorm diff <target> <reference> [--max-points N]  反復測量の差分符号化\n"
            "  pccnorm attr <file.laz> [--max-points N]  属性の空間予測と格納順予測の比較\n"
            "  pccnorm distort <A> <B>   対応を仮定しない歪み（Chamfer / Hausdorff / 面距離）\n"
            "  pccnorm surf <file> [--voxel V] [--eps E] [--nbits B] [--out P.ply]\n"
            "        面を符号化して点を引き直す方式と、点をそのまま送る方式の比較\n"
            "  pccnorm pack   <in> <out.pcc2> [--max-points N] [--split-geom]\n"
            "                 [--force-geom <候補名>] [--fast-attr] [--no-fallback]\n"
            "                 [--eps <m>] [--sample-select N] [--no-verify] [--trace]\n"
            "                 [--no-normalize] [--no-spatial]\n"
            "        PCC2 コンテナへ符号化し、決定性・往復・LAS 書き戻しを検証して 5 軸の計測を出す\n"
            "        （--eps は float の器だけの非可逆。--no-verify は検証を全部飛ばす）\n"
            "  pccnorm unpack <in.pcc2> [--las <out.laz>] [--bin <out.bin>]\n"
            "        PCC2 を復号して中身を出す（--bin は KITTI .bin として書き戻す）\n"
            "  pccnorm combine <orig.laz> <geom_decoded.ply> <geom_stream.bin> [--max-points N]\n"
            "        幾何を外部符号器に任せ，属性のみ本手法で符号化したときの合計を測る\n");
        return 1;
    }
    std::string cmd = argv[1], path = argv[2];

    // ---------------------------------------------------------------- PCC2
    if (cmd == "pack") {
        // **復号に効くのに器へ記録されない実験用の口**が設定されていたら拒む。
        // 復号側は常に既定値を使うので、既定以外で符号化した器は正しく戻らない
        // （pack の自己検証で食い違って分かるが、--no-verify だと黙って壊れた器が残る）。
        // 実験で測るときだけ PCC_ALLOW_EXPERIMENT=1 を付ける。
        {
            static const char* EXP[] = {"PCC_LMS_KIND", "PCC_LMS_ORD", "PCC_RAY_PRED",
                                        "PCC_RAY_HIST", "PCC_RAWKEEP", "PCC_BM_SCHED", "PCC_BM_SCHED2"};
            const char* allow = getenv("PCC_ALLOW_EXPERIMENT");
            std::string set;
            for (const char* v : EXP) if (getenv(v)) set += std::string(set.empty() ? "" : ", ") + v;
            if (!set.empty() && !(allow && allow[0] == '1')) {
                fprintf(stderr, "実験用の環境変数が設定されている（%s）。これらは復号に効くのに"
                                "器へ記録されないので、既定の復号器では正しく戻らない器になる。"
                                "測るためだけに使うなら PCC_ALLOW_EXPERIMENT=1 を付けること。\n",
                        set.c_str());
                return 1;
            }
        }
        if (argc < 4) { fprintf(stderr, "pack <in> <out.pcc2>\n"); return 1; }
        // **作業用の名前に書き、検証が通ってから本来の名前に移す。**
        // 以前は検証に落ちても（戻り値 2）壊れた器が本来の名前で残り、
        // 途中で失敗したときも書きかけが残っていた。
        const std::string final_out = argv[3];
        std::string outp = final_out + ".part";
        struct PartGuard {
            const std::string& p; bool keep = false;
            ~PartGuard() { if (!keep) remove(p.c_str()); }
        } part_guard{outp};
        // rename は既存の出力を置き換える（先に消すと、改名に失敗したとき古い出力まで失う）
        auto commit = [&]() -> bool {
            if (rename(outp.c_str(), final_out.c_str()) != 0) {
                fprintf(stderr, "書き込み失敗: %s に移せない\n", final_out.c_str());
                return false;
            }
            part_guard.keep = true;
            return true;
        };
        size_t mp = 0, samp = 0;
        bool joint = true, do_norm = true, do_spatial = true, trace_all = false;
        std::string force_geom;
        bool fast_attr = false, no_fallback = false, no_verify = false;
        double eps_lossy = 0;
        std::string lossy_kind;
        for (int i = 4; i < argc; ++i) {
            if (!strcmp(argv[i], "--max-points") && i + 1 < argc) mp = atol(argv[++i]);
            else if (!strcmp(argv[i], "--split-geom")) joint = false;
            else if (!strcmp(argv[i], "--no-normalize")) do_norm = false;
            else if (!strcmp(argv[i], "--no-spatial")) do_spatial = false;
            else if (!strcmp(argv[i], "--trace")) trace_all = true;
            else if (!strcmp(argv[i], "--sample-select") && i + 1 < argc) samp = atol(argv[++i]);
            else if (!strcmp(argv[i], "--force-geom") && i + 1 < argc) force_geom = argv[++i];
            else if (!strcmp(argv[i], "--fast-attr")) fast_attr = true;
            else if (!strcmp(argv[i], "--no-fallback")) no_fallback = true;
            // 符号化だけを測るための指定。復号検証・元ファイルの読み直し・
            // 決定性の二度書きを飛ばす。時間とメモリを他の符号器と同じ土俵で
            // 比べるときに使う（それらは符号化の費用ではない）。
            else if (!strcmp(argv[i], "--no-verify")) no_verify = true;
            else if (!strcmp(argv[i], "--mem-trace")) g_mem_trace = true;
            // **非可逆の経路**（既定では通らない）。誤差上限 [m] を渡すと、
            // 浮動小数の座標を極座標格子に量子化してから符号化する。
            else if (!strcmp(argv[i], "--eps") && i + 1 < argc) {
                // 数として読めない値・0 以下・NaN を黙って可逆に落とさない
                char* end = nullptr;
                eps_lossy = strtod(argv[++i], &end);
                if (!end || *end || !(eps_lossy > 0) || !std::isfinite(eps_lossy)) {
                    fprintf(stderr, "--eps には正の数（m）を渡すこと: %s\n", argv[i]);
                    return 1;
                }
            }
        }
        std::string err;
        double t0 = now();
        rss_watch_start();
        mem_mark("開始");
        Frame f;
        bool is_las = false;
        {
            std::string lp = path;
            for (auto& ch : lp) ch = tolower(ch);
            is_las = lp.size() > 4 && (lp.substr(lp.size() - 4) == ".las" ||
                                       lp.substr(lp.size() - 4) == ".laz");
        }
        // 基準 ℓ_C(P): 同じ点を一度書き直した LAZ。元の配布ファイルとは比べない
        // （切り出して測るときに点数が違ってしまうため）。
        uint64_t base_bytes = 0;
        std::string basep;              // 基準に書いた LAZ。退避に使うので消さずに残す
        // 途中で返っても基準の LAZ が残らないように、消すのは後始末に任せる
        struct BaseGuard { std::string& p; ~BaseGuard() { if (!p.empty()) remove(p.c_str()); } }
            base_guard{basep};
        PointCloud pc;
        if (is_las) {
            if (!read_las(path, pc, err, mp)) { fprintf(stderr, "読み込み失敗: %s\n", err.c_str()); return 1; }
            mem_mark("LAS 読み込み後");
            std::string bp = outp + ".base.laz";
            std::vector<std::string> all;
            for (const auto& nm : pc.order) all.push_back(nm);
            if (write_las(bp, pc, all, err)) { base_bytes = fsize(bp); basep = bp; }
            else remove(bp.c_str());
            mem_mark("基準 LAZ を書いた後");
            // 正規化するなら、計画を先に立てて残す列だけを移す。
            // 全列をコピーしてから落とすと、pc と f の二重持ちがピークになる。
            bool built = do_norm ? frame_from_las_normalized(pc, f, false, err)
                                 : frame_from_las(pc, f, err);
            if (!built) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
            mem_mark("Frame を作った後");
        } else {
            if (!load_frame(path, f, err, mp)) { fprintf(stderr, "読み込み失敗: %s\n", err.c_str()); return 1; }
            base_bytes = f.source_bytes;
        }
        double t_read = now() - t0;
        // 点が 0 個の器は符号化する中身が無い。通すと「元の点数 == Frame の点数」
        // が 0 == 0 で真になり、正規化が空の PointCloud を材料に全列を落とす。
        if (f.n == 0) { fprintf(stderr, "点が 0 個なので符号化しない: %s\n", path.c_str()); return 1; }

        double t1 = now();
        // LAS 経路の正規化は frame_from_las_normalized が済ませている。
        // 正規化レイヤーは PointCloud の列を解析して計画を立てるので、
        // LAS 以外の器（.bin / .ply）では材料が無い。空の PointCloud を渡すと
        // 「どの列も計画から復元できる」と読まれて全属性が落ちるため、渡さない。
        if (do_norm && !is_las && pc.n == f.n && !normalize_frame(f, pc, err)) {
            fprintf(stderr, "正規化に失敗: %s\n", err.c_str()); return 1;
        }
        mem_mark("正規化後");
        // **非可逆の経路。**誤差上限を渡されたときだけ通る。座標が浮動小数で
        // 入っている器（.bin / .ply）に限る。LAS は既に整数格子なので対象外。
        if (eps_lossy > 0) {
            // LAS は器の側が整数格子と scale/offset を宣言しているので対象外。
            // .bin / .ply は幾何が "f32bits" か、可逆に整数化された "int" で入る。
            if (is_las) {
                fprintf(stderr, "--eps は .bin / .ply にしか使えない（LAS は対象外）\n");
                return 1;
            }
            std::vector<double> w;
            frame_world(f, w);
            // NaN / 無限大は量子化できない（格子の範囲も誤差も測れない）。
            // 通すと格子の選択が NaN を比べて、黙って壊れた器ができる。
            for (double v : w)
                if (!std::isfinite(v)) {
                    fprintf(stderr, "--eps: 座標に NaN か無限大がある。非可逆の量子化は使えない\n");
                    return 1;
                }
            // **復号と同じ道を通して測る。**極座標に丸めた誤差だけでは足りない。
            // 復号は世界座標を元の器（整数格子 / float32）に書き戻すので、
            // そこでもう一度丸めが入る。1 mm 格子なら最大 0.866 mm 増える。
            // 符号化時の申告がこれを数えていないと、宣言した上限を超えて出る。
            auto end_err = [&](const GeomCandidate& g) {
                double worst = 0;
                const bool pol = (g.kind == "polar");
                const double dr = g.r_step, da = g.ang_step;
                for (size_t i = 0; i < f.n; ++i) {
                    double v[3];
                    if (pol) {
                        const double r = (double)g.streams[0][i] * dr;
                        const double th = (double)g.streams[1][i] * da;
                        const double ph = (double)g.streams[2][i] * da;
                        const double oo[3] = {g.origin[0], g.origin[1], g.origin[2]};
                        polar_point(r, th, ph, oo, false, v);   // 復号（frame_world）と同じ式
                    } else {
                        for (int d = 0; d < 3; ++d)
                            v[d] = (double)g.streams[d][i] * g.step + g.origin[d];
                    }
                    for (int d = 0; d < 3; ++d) {
                        if (f.geom_repr == "f32bits")      v[d] = (double)(float)v[d];
                        else if (f.geom_repr != "f64bits") {
                            const double sc = f.scale[d] != 0 ? f.scale[d] : 1.0;
                            v[d] = std::llround((v[d] - f.offset[d]) / sc) * sc + f.offset[d];
                        }
                        v[d] = as_output(f, d, v[d]);   // 器が float32 なら最後にもう一度丸まる
                    }
                    const double dx = v[0] - as_output(f, 0, w[i * 3]),
                                 dy = v[1] - as_output(f, 1, w[i * 3 + 1]),
                                 dz = v[2] - as_output(f, 2, w[i * 3 + 2]);
                    const double e = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (e > worst) worst = e;
                }
                return worst;
            };
            // 種類を 1 つ決めて、誤差上限に収まるまで刻みを詰める（kind が空なら代理の符号長で選ぶ）。
            auto fit_kind = [&](const std::string& kind, GeomCandidate& g, double& got_out) -> bool {
                double target = eps_lossy;
                for (int tries = 0; tries < 6; ++tries) {
                    // PCC_GEOM_VERBOSE=1 で候補ごとの代理の符号長を出す（符号化側の診断だけ。器は変わらない）
                    static const bool GV = getenv("PCC_GEOM_VERBOSE") != nullptr;
                    g = choose_geometry(w, f.n, target, true, 200000, GV, kind);
                    if (g.kind != "polar" && g.kind != "grid") return false;
                    got_out = end_err(g);
                    if (got_out <= eps_lossy) return true;
                    // 超えたぶんだけ刻みを詰めて測り直す。
                    target *= eps_lossy / got_out * 0.98;
                }
                return false;
            };
            // **格子の種類は、幾何の流れを実際に符号化した長さで選ぶ。**代理の符号長（先頭 20 万点の
            // 差分を汎用の圧縮器に通した長さ）で選ぶと、KITTI の 2 mm で 68 件中 3 件、直交格子のほうが
            // 0.4〜1.2% 短いのに極座標を選んでいた（lattice_loss_v22.log）。3 種類それぞれで幾何だけの
            // 計画を立てて比べる（比べる 3 回と本番を合わせて、非可逆の経路の幾何の計画が最大 4 回になる）。PCC_LATTICE_MEASURE=0 で
            // 代理の選択に戻す。PCC_GEOM_KIND で種類を固定したときは測らない。
            static const bool LAT_MEASURE = [] {
                const char* e = getenv("PCC_LATTICE_MEASURE"); return !e || e[0] != '0'; }();
            const bool measure = LAT_MEASURE && !getenv("PCC_GEOM_KIND");
            auto geom_bytes = [&](const GeomCandidate& g) -> size_t {
                Frame g2;
                g2.n = f.n;
                for (int c = 0; c < 3; ++c) g2.geom[c] = f.geom[c];
                for (const auto& cs : f.schema)
                    for (int c = 0; c < 3; ++c)
                        if (cs.name == f.geom[c]) g2.schema.push_back(cs);
                for (int c = 0; c < 3; ++c) g2.col[f.geom[c]] = Col(g.streams[c]);
                CodecCtx c2; c2.fr = &g2; c2.force_geom = force_geom;   // 本番と同じ固定
                std::string lg;
                auto st2 = plan_streams(g2, joint, &lg, &c2);   // 本番と同じ幾何のまとめ方で
                size_t b = 0;
                for (const auto& x : st2) b += x.data.size() + x.param.size();
                return b;
            };
            GeomCandidate gc;
            double got = 0;
            bool fit = false;
            if (!measure) {
                fit = fit_kind("", gc, got);
            } else {
                size_t best_b = (size_t)-1;
                for (const char* kind : {"grid", "polar/origin", "polar/centroid"}) {
                    GeomCandidate g; double gt = 0;
                    if (!fit_kind(kind, g, gt)) continue;
                    const size_t b = geom_bytes(g);
                    if (getenv("PCC_GEOM_VERBOSE"))
                        printf("    格子 %-16s 幾何の流れ %zu byte（誤差 %.4g mm）\n", kind, b, gt * 1000);
                    if (b < best_b) { best_b = b; gc = std::move(g); got = gt; fit = true; }
                }
            }
            std::vector<double>().swap(w);
            if (!fit) {
                if (gc.kind.empty())
                    fprintf(stderr, "量子化が誤差上限に収まらなかった（どの種類の格子も上限を守れなかった）\n");
                else
                    fprintf(stderr, "量子化が誤差上限に収まらなかった"
                                    "（種類 %s / 端から端までの実測誤差 %.6g m）\n",
                            gc.kind.c_str(), got);
                return 1;
            }
            if (gc.kind == "polar") {
                f.polar_base = f.geom_repr;
                for (int c = 0; c < 3; ++c) f.polar_origin[c] = gc.origin[c];
                f.polar_r_step = gc.r_step;
                f.polar_ang_step = gc.ang_step;
                f.geom_repr = "polar";
            } else {
                // 粗いデカルト格子は、既にある整数格子の表し方でそのまま言える。
                // 値 = scale*q + offset。復号側に足すものは無い。
                for (int c = 0; c < 3; ++c) { f.scale[c] = gc.step; f.offset[c] = gc.origin[c]; }
                f.geom_repr = "int";
                // 取り込み時に見つけた可逆の格子（封筒の FGRD）は、もう幾何に当てはまらない。
                // 残すと .bin に戻すときに古い刻みで割り戻して、座標が数百 m ずれる。
                drop_grid_cols(f, {f.geom[0], f.geom[1], f.geom[2]});
            }
            for (int c = 0; c < 3; ++c) f.col[f.geom[c]] = std::move(gc.streams[c]);
            f.fid.exact = false;
            f.fid.declared_eps = eps_lossy;
            f.fid.measured_max = got;
            lossy_kind = gc.kind;
        }
        pc = PointCloud();                    // 正規化が済んだら元の列は要らない
        mem_mark("pc を解放した後");
        // 幾何は属性より先に復号されるので、座標は副情報なしで使える
        CodecCtx ctx;
        std::vector<double> world;
        ctx.fr = &f;
        ctx.force_geom = force_geom;
        ctx.bitfields_first = joint;         // 走査モデル用の前置き列（bit_fields を含む）が先に出る
        ctx.fast_attr = fast_attr;
        // 実際に空間予測の候補が試されるまで作らない
        if (do_spatial) ctx.want_world = true;
        mem_mark("world は遅延構築にした");

        std::string log;
        std::vector<Stream> st;
        if (samp && samp < f.n) {
            // 候補選択だけ標本で行い、選ばれた符号器で全点を符号化し直す。
            int nchunk = 8;
            if (const char* e = getenv("PCC_SAMPLE_CHUNKS")) nchunk = atoi(e);
            Frame fs = sample_frame(f, samp, nchunk);
            CodecCtx sctx; std::vector<double> sworld;
            sctx.fr = &fs;
            sctx.full = &f;
            sctx.force_geom = force_geom;
            sctx.bitfields_first = joint;
            sctx.fast_attr = fast_attr;
            if (do_spatial) { frame_world(fs, sworld); sctx.world = &sworld; }
            auto sel = plan_streams(fs, joint, &log, &sctx, trace_all);
            // 旗「類」の相手の列は plan_streams が文脈に書く。全点での後置検査は本番の ctx で回すので写す
            // （写さないと --sample-select の経路では類が一度も試されない）。
            ctx.ctx_cols = sctx.ctx_cols;
            // 標本の 1 位が全点でも 1 位とは限らない。上位 2 つを全点で測り、
            // 短い方を採る。全候補を全点で測るより速く、1 位だけを信じるより安全。
            for (auto& s0 : sel) {
                std::vector<const Col*> cv;
                for (const auto& c : s0.cols) {
                    const Col* col = f.get(c);
                    if (!col) { fprintf(stderr, "列が無い: %s\n", c.c_str()); return 1; }
                    cv.push_back(col);
                }
                Stream s1; s1.cols = s0.cols;
                bool got = false;
                std::vector<Cand> tryv{{s0.codec, s0.param}};
                for (const auto& a : s0.alt) tryv.push_back(a);
                for (const auto& cd : tryv) {
                    std::vector<uint8_t> blob; std::string e2;
                    if (!codec_encode(cd.codec, cv, cd.param, blob, e2, &ctx)) continue;
                    if (!got || blob.size() < s1.data.size()) {
                        s1.codec = cd.codec; s1.param = cd.param;
                        s1.data = std::move(blob); got = true;
                    }
                }
                if (!got) {
                    fprintf(stderr, "全点での符号化に失敗\n"); return 1;
                }
                // **勝者に旗を重ねた版も試す。**best_stream は標本の上では後追いを
                // しない（全点で測り直すため）ので、ここで足さないと
                // `--sample-select` の経路では旗が一度も付かない（以前は素通しだけを
                // 試していて、autzen_trim の X+Y+Z で 0.111 bpp 取り逃していた。
                // 符号・光線・束ね・曲面は試していなかった）。
                {
                    std::string pt;
                    apply_post_flags(s1, cv, &ctx, trace_all ? &pt : nullptr);
                    log += pt;
                }
                {   // 標本での値ではなく、全点で実測した値を出す
                    std::string nm;
                    for (size_t i2 = 0; i2 < s1.cols.size(); ++i2)
                        nm += (i2 ? "+" : "") + s1.cols[i2];
                    char m[256];
                    snprintf(m, sizeof m, "  全点 %-17s %-8s %8.3f bpp\n", nm.c_str(),
                             cand_name(s1.codec, s1.param).c_str(),
                             f.n ? s1.data.size() * 8.0 / f.n : 0.0);
                    log += m;
                }
                st.push_back(std::move(s1));
            }
            log += "  （符号器の選択は先頭 " + std::to_string(samp) +
                   " 点で行った。上の bpp は標本での値。下の合計は全点の実測）\n";
        } else {
            st = plan_streams(f, joint, &log, &ctx, trace_all);
        }
        mem_mark("候補選択と符号化の後");
        // 回帰試験用: 最初の流れの最後のバイトを反転させ、決定性の検査が
        // 食い違いを捕まえて書かずに終わることを確かめる（regress_fixes.py）。
        // 検証を飛ばす経路では効かせない（壊れた器が確定してしまう）。
        if (!no_verify && getenv("PCC_TEST_DET_FLIP") && !st.empty() && !st[0].data.empty())
            st[0].data.back() ^= 0x01;
        uint64_t bytes = 0;
        if (!write_pcc2(outp, f, st, bytes, err)) { fprintf(stderr, "書き込み失敗: %s\n", err.c_str()); return 1; }
        mem_mark("書き出した後");
        rss_watch_stop();

        // 自前の符号器が元の器に負けることがある。両方書いて短い方を残し、
        // どちらを使ったかを容器に書いておく。符号器そのものは独立のままで、
        // 「決して悪化しない」という運用上の保証だけを取り戻す。
        Frame fe;
        bool used_embed = false;
        // 測定では退避路を切る。落ちると検証の対象が埋め込んだ器になり、
        // 報告している符号器の値と検証対象が食い違う。
        if (!no_fallback && !basep.empty() && base_bytes) {
            std::vector<uint8_t> lz(base_bytes);
            FILE* lf = fopen(basep.c_str(), "rb");
            bool got = lf && fread(lz.data(), 1, base_bytes, lf) == base_bytes;
            if (lf) fclose(lf);
            if (got) {
                fe.n = f.n; fe.source_kind = f.source_kind;
                fe.embed_kind = "laz"; fe.embed = std::move(lz);
                std::string ep = outp + ".embed";
                uint64_t be = 0;
                if (write_pcc2(ep, fe, {}, be, err) && be < bytes) {
                    remove(outp.c_str());
                    if (rename(ep.c_str(), outp.c_str()) == 0) { bytes = be; used_embed = true; }
                } else {
                    remove(ep.c_str());
                }
            }
        }
        if (!basep.empty()) { remove(basep.c_str()); basep.clear(); }
        double t_enc = now() - t1;

        if (no_verify) {
            double bl0 = f.n ? base_bytes * 8.0 / f.n : 0.0;
            double ml0 = f.n ? bytes * 8.0 / f.n : 0.0;
            printf("入力        %s\n            %zu 点 / 列 %zu\n",
                   path.c_str(), (size_t)f.n, f.schema.size());
            printf("ストリーム選択（候補を実際に符号化して最短を採る）\n%s", log.c_str());
            printf("基準 %-7s %10.1f MB  %8.3f bpp\n",
                   is_las ? "LASzip" : "元", base_bytes / 1e6, bl0);
            printf("PCC2        %10.1f MB  %8.3f bpp\n", bytes / 1e6, ml0);
            printf("5 軸        enc %.2fs / dec ---- / 読込 %.2fs / ピーク %.2f GB / 検証なし\n",
                   t_enc, t_read, peak_gb());
            return commit() ? 0 : 1;
        }

        // **決定性の確認を先に済ませる。**これには符号化側の Frame と流れが要る。
        // 先にやっておけば、復号と読み直しを始める前に符号化側を手放せる。
        // 後回しにすると、符号化側・復号したもの・読み直したものの 3 つを
        // 同時に抱えることになる（200 万点・17 列で 1 つ 250 MB）。
        //
        // 確かめることは 2 つ。(a) 選ばれた流れを、選択で温まった表を使わずに符号化し直すと
        // 同じバイト列になるか（符号器の決定性）。(b) 同じ流れを器に書き直すと同じバイト列に
        // なるか（器の書き出しの決定性）。以前は (b) だけで、符号器を走らせ直していなかった。
        // **どちらも食い違ったら書かない。**候補の選択そのものの決定性は、スレッド数を
        // 変えた md5 の比較（verify_suite の threads）で見る。
        // 選択で温まった表（近傍表・順序表・走査の文脈・字母・world）はもう要らない。
        // 抱えたまま新しい文脈で同じ表を作り直すと、200 万点で 100 MB を超えて二重に持つ。
        ctx.perm_by_n.clear();
        ctx.cls_cache.clear();
        ctx.pred_by_np.clear();
        ctx.scan_cache.reset();
        ctx.sym_cache.reset();
        std::vector<double>().swap(ctx.world_own);
        const double t_det0 = now();
        std::string det_diff;
        bool det = used_embed || reencode_matches(f, st, ctx, det_diff);
        const double t_det = now() - t_det0;
        std::string p2 = outp + ".det";
        uint64_t b2 = 0;
        det = det && (used_embed ? write_pcc2(p2, fe, {}, b2, err)
                                 : write_pcc2(p2, f, st, b2, err)) && b2 == bytes;
        if (det) {
            FILE* a = fopen(outp.c_str(), "rb"); FILE* b = fopen(p2.c_str(), "rb");
            std::vector<uint8_t> ba(bytes), bb(bytes);
            det = a && b && fread(ba.data(), 1, bytes, a) == bytes &&
                  fread(bb.data(), 1, bytes, b) == bytes && ba == bb;
            if (a) fclose(a);
            if (b) fclose(b);
            if (!det) det_diff = "器に書き直すとバイト列が違う";
        } else if (det_diff.empty()) {
            det_diff = b2 != bytes && b2 ? "器に書き直すと長さが違う" : "器を書き直せない";
        }
        remove(p2.c_str());
        if (!det) {
            fprintf(stderr, "決定性に落ちた: %s\n検証に落ちたので %s は書かない\n",
                    det_diff.c_str(), final_out.c_str());
            return 2;
        }

        // 以降で使う覚え書きだけ残して、列の実体と流れを手放す。
        const size_t rep_n = (size_t)f.n, rep_ncol = f.schema.size();
        const std::string rep_src = f.source_kind, rep_geom = f.geom_repr, rep_plan = f.plan;
        const Fidelity rep_fid = f.fid;
        const std::string rep_grid = grid_summary(f);
        const std::string rep_cdiv = coldiv_summary(f);
        std::map<std::string, Col>().swap(f.col);
        std::vector<Stream>().swap(st);
        std::vector<uint8_t>().swap(fe.embed);
        mem_mark("符号化側を手放した後");

        double t2 = now();
        Frame g;
        // 包んだ器を書き出した一時ファイル。下の LAS の照合でも使うので、そこまで残す。
        // 途中で返っても消えるように、消すのは後始末に任せる。
        struct TmpFile { std::string p; ~TmpFile() { if (!p.empty()) remove(p.c_str()); } } embed_tmp;
        if (!read_pcc2(outp, g, err)) { fprintf(stderr, "復号失敗: %s\n", err.c_str()); return 1; }
        if (!g.embed.empty()) {
            // 包んであるのは元の器そのもの。書き出して読み直せば列が揃う。
            std::string tp = outp + ".tmp.laz";
            FILE* tf = fopen(tp.c_str(), "wb");
            if (!tf) { fprintf(stderr, "復号失敗: 一時ファイルを作れない\n"); return 1; }
            fwrite(g.embed.data(), 1, g.embed.size(), tf);
            fclose(tf);
            Frame ge;
            bool okr = load_frame(tp, ge, err, 0);
            if (!okr) { remove(tp.c_str()); fprintf(stderr, "復号失敗: %s\n", err.c_str()); return 1; }
            embed_tmp.p = tp;
            g = std::move(ge);
            // **読み直した側も粗い格子で割られている。**load_frame は入力側で
            // apply_column_grid を通すので、ここで戻さないと比較相手（orig、下で
            // restore_column_grid を掛ける）と空間が揃わず、正しい出力に対して
            // 検証が落ちる（fullwave の red が 49 対 25186 = 514×49 になった）。
            restore_column_grid(g);
        } else {
            restore_column_grid(g);          // 計画より先に戻す（上と同じ理由）
            if (!denormalize_frame(g, err)) {
                fprintf(stderr, "復元失敗: %s\n", err.c_str()); return 1;
            }
        }
        double t_dec = now() - t2;

        // 検証は元ファイルを読み直して全列を突き合わせる。
        // 正規化前の Frame を保持し続けるとピークメモリが 2 倍になるため。
        std::string diff;
        Frame orig;
        if (!load_frame(path, orig, err, mp)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        // **粗い格子で割ってあれば、両方を戻してから比べる。**割った空間どうしで
        // 比べると、割り算が情報を落としていても、刻みの書き出しが壊れていても
        // 検証が通ってしまう（どちらの辺も同じ割り算を通っているため）。
        restore_column_grid(orig);
        double worst_err = 0;
        const bool ok = rep_fid.exact
                      ? frames_equal(orig, g, diff)
                      : frames_close(orig, g, rep_fid.declared_eps, worst_err, diff);
        orig = Frame();
        // **unpack --las と同じ道で書き戻し、元のファイルと器として比べる。**
        // Frame どうしの比較は列しか見ないので、封筒（ヘッダの欄・未記述のバイト・
        // ユーザーデータ・VLR）の欠落を見逃す。実際に見逃していた（verify_unpack.py で発覚）。
        bool las_ok = true;
        std::string las_diff;
        // 包んだ器（基準の LAZ そのもの）も write_las が書いたものなので、同じく照合する。
        const bool las_check = ok && is_las && rep_fid.exact;
        if (las_check && !embed_tmp.p.empty()) {
            g = Frame();
            PointCloud pa, pb;
            if (!read_las(embed_tmp.p, pb, err) || !read_las(path, pa, err, mp))
                { las_ok = false; las_diff = err; }
            else las_ok = pointclouds_equal(pa, pb, pa.hdr_n == (uint64_t)pa.n, las_diff);
        } else if (las_check) {
            const std::string tp = outp + ".chk.las";
            PointCloud pw;
            if (!frame_to_las(g, pw, err)) { las_ok = false; las_diff = err; }
            else {
                std::vector<std::string> keep;
                for (const auto& s : g.schema) if (s.role != Role::Geometry) keep.push_back(s.name);
                g = Frame();
                if (!write_las(tp, pw, keep, err)) { las_ok = false; las_diff = err; }
                pw = PointCloud();
                PointCloud pa, pb;
                if (las_ok) {
                    if (!read_las(tp, pb, err) || !read_las(path, pa, err, mp))
                        { las_ok = false; las_diff = err; }
                    else las_ok = pointclouds_equal(pa, pb, pa.hdr_n == (uint64_t)pa.n, las_diff);
                }
            }
            remove(tp.c_str());
        }

        printf("入力        %s\n            %zu 点 / 出所 %s / 幾何 %s / 列 %zu\n",
               path.c_str(), rep_n, rep_src.c_str(), rep_geom.c_str(), rep_ncol);
        // 浮動小数の器を整数に直したなら、何を見つけたかを出す。
        // 刻みはデータから見つけたもので、仮定ではない。
        if (!rep_grid.empty()) printf("格子        %s\n", rep_grid.c_str());
        // 宣言された分解能より粗い格子に乗っていた列。割ってから符号化している。
        if (!rep_cdiv.empty()) printf("粗い格子    %s\n", rep_cdiv.c_str());
        if (do_norm && !rep_plan.empty()) {
            Plan pl = Plan::from_json(rep_plan);
            printf("正規化      %s", pl.report().c_str());
            if (pl.ops.empty()) printf("\n");
        }
        printf("ストリーム選択（候補を実際に符号化して最短を採る）\n%s", log.c_str());
        double bl = rep_n ? base_bytes * 8.0 / rep_n : 0.0;
        double ml = rep_n ? bytes * 8.0 / rep_n : 0.0;
        printf("基準 %-7s %10.1f MB  %8.3f bpp\n", is_las ? "LASzip" : "元", base_bytes / 1e6, bl);
        printf("PCC2        %10.1f MB  %8.3f bpp", bytes / 1e6, ml);
        if (base_bytes) printf("   %+.1f%%", 100.0 * (ml / bl - 1.0));
        printf("\n");
        printf("中身        %s\n", used_embed ? "元の器を包んだ（自前の符号器より短かった）"
                                              : "自前の符号器");
        if (rep_fid.exact) {
            printf("検証        全列一致 = %s%s\n", ok ? "true" : "false",
                   ok ? "" : ("  差異: " + diff).c_str());
            if (las_check)
                printf("            LAS に書き戻して元と一致 = %s%s\n", las_ok ? "true" : "false",
                       las_ok ? "" : ("  差異: " + las_diff).c_str());
        } else {
            printf("**非可逆**  幾何を%sに量子化した。"
                   "宣言 %.6g m / 符号化時の実測 %.6g m\n",
                   lossy_kind == "polar" ? "極座標格子" : "粗いデカルト格子",
                   rep_fid.declared_eps, rep_fid.measured_max);
            printf("検証        属性は全列一致・幾何は誤差上限の中 = %s"
                   "（復号後の実測 %.6g m）%s\n", ok ? "true" : "false", worst_err,
                   ok ? "" : ("  差異: " + diff).c_str());
        }
        printf("5 軸        enc %.2fs / dec %.2fs / 読込 %.2fs / ピーク %.2f GB / 決定性 %s\n",
               t_enc, t_dec, t_read, peak_gb(), det ? "バイト一致" : "不一致");
        // 見出しに「決定性」を使わない。集計スクリプト（regress_pcc2.py など）は「決定性」を
        // 含む最後の行で「バイト一致」を探すので、上の 5 軸の行の判定を上書きしてしまう。
        if (used_embed)
            printf("再符号化    包んだ器なので省いた（器の書き直しはバイト一致）\n");
        else
            printf("再符号化    流れを符号化し直してバイト一致・器の書き直しもバイト一致（%.2fs）\n",
                   t_det);
        if (!ok || !las_ok) {
            fprintf(stderr, "検証に落ちたので %s は書かない\n", final_out.c_str());
            return 2;
        }
        return commit() ? 0 : 1;
    }

    if (cmd == "unpack") {
        std::string err; Frame f;
        if (!read_pcc2(path, f, err)) { fprintf(stderr, "復号失敗: %s\n", err.c_str()); return 1; }
        // **包んであるなら中身は LASzip が書いた器そのものである。**列は 1 本も
        // 入っていないので、復号して組み立て直す道は無い。バイト列をそのまま出す。
        // 元ファイルとバイト一致するとは限らない（元が .las なら .laz になり、
        // 元が .laz でも LASzip の版が違えば並びが変わる）。**中身は一致する。**
        if (!f.embed.empty()) {
            printf("中身 元の器を包んだもの（%s、%zu byte）\n",
                   f.embed_kind.c_str(), f.embed.size());
            bool wrote = false;
            for (int i = 3; i < argc; ++i)
                if ((!strcmp(argv[i], "--las") || !strcmp(argv[i], "--bin")) && i + 1 < argc) {
                    const char* op = argv[++i];
                    FILE* of = fopen(op, "wb");
                    if (!of) { fprintf(stderr, "書き出せない: %s\n", op); return 1; }
                    size_t w = fwrite(f.embed.data(), 1, f.embed.size(), of);
                    if (fclose(of) != 0 || w != f.embed.size()) {
                        fprintf(stderr, "書き出せない: %s\n", op); return 1;
                    }
                    printf("包んだ器（%s）をそのまま書いた: %s\n",
                           f.embed_kind.c_str(), op);
                    wrote = true;
                }
            if (!wrote)
                printf("（出力先を指定していない。--las <出力.laz> で器が出る）\n");
            return 0;
        }
        // **粗い格子を先に戻す。**正規化の計画は「割る前」の単位で立てられている
        // ので、割ったままで逆正規化すると、複製を指す操作が別の値を書き戻す。
        restore_column_grid(f);
        if (!denormalize_frame(f, err)) { fprintf(stderr, "復元失敗: %s\n", err.c_str()); return 1; }
        printf("点数 %zu / 出所 %s / 幾何 %s / 列 %zu\n", (size_t)f.n, f.source_kind.c_str(),
               f.geom_repr.c_str(), f.schema.size());

        // 浮動小数の器を整数に直したなら、何を見つけたかを出す。
        // 刻みはデータから見つけたもので、仮定ではない。
        printf("精度 %s", f.fid.exact ? "可逆" : "誤差上限つき");
        if (!f.fid.exact) printf("（宣言 %.4g m / 実測 %.4g m）", f.fid.declared_eps, f.fid.measured_max);
        printf("\n計画 %s\n", f.plan.empty() ? "（なし）" : f.plan.c_str());
        for (int i = 3; i < argc; ++i)
            if (!strcmp(argv[i], "--las") && i + 1 < argc) {
                PointCloud pc;
                if (!frame_to_las(f, pc, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
                std::vector<std::string> keep;
                for (const auto& s : f.schema) if (s.role != Role::Geometry) keep.push_back(s.name);
                if (!write_las(argv[++i], pc, keep, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
                printf("LAS を書いた: %s\n", argv[i]);
            }
        for (int i = 3; i < argc; ++i)
            if (!strcmp(argv[i], "--bin") && i + 1 < argc) {
                if (!frame_to_kitti_bin(f, argv[++i], err))
                    { fprintf(stderr, "%s\n", err.c_str()); return 1; }
                printf(".bin を書いた: %s\n", argv[i]);
            }
        return 0;
    }
    if (cmd == "surf") {
        SurfOpt o; std::string outp; size_t mp = 0; double vmul = 8;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--voxel" && i+1 < argc) o.voxel = atof(argv[++i]);
            if (a == "--vmul"  && i+1 < argc) vmul = atof(argv[++i]);
            if (a == "--eps"   && i+1 < argc) o.eps = atof(argv[++i]);
            if (a == "--nbits" && i+1 < argc) o.nbits = atoi(argv[++i]);
            if (a == "--tau"   && i+1 < argc) { o.tau = atof(argv[++i]); o.adaptive = true; }
            if (a == "--minpts"&& i+1 < argc) o.min_pts = atoi(argv[++i]);
            if (a == "--jitter") o.jitter = true;
            if (a == "--strat")  o.strat = true;
            if (a == "--predict") o.predict = true;
            if (a == "--out"   && i+1 < argc) outp = argv[++i];
            if (a == "--max-points" && i+1 < argc) mp = strtoull(argv[++i], nullptr, 10);
        }
        // LAS/LAZ なら属性も読み、面上の場として符号化する
        PointCloud pcA; bool with_attr = false;
        std::vector<double> P; size_t n = 0; std::string e;
        {
            std::string ext = path.size() > 4 ? path.substr(path.size()-4) : "";
            for (auto& ch : ext) ch = (char)tolower(ch);
            if (ext == ".laz" || ext == ".las") {
                if (!read_las(path, pcA, e, mp)) { fprintf(stderr, "%s\n", e.c_str()); return 1; }
                materialize_all(pcA);   // 解析用: 定数の列も配列にして .v で読めるようにする
                n = pcA.n; with_attr = true; P.resize(n * 3);
                for (size_t i = 0; i < n; ++i) {
                    P[i*3]   = pcA.X[i] * pcA.scale[0];
                    P[i*3+1] = pcA.Y[i] * pcA.scale[1];
                    P[i*3+2] = pcA.Z[i] * pcA.scale[2];
                }
            } else if (!read_points_any(path, P, n, e, mp)) {
                fprintf(stderr, "%s\n", e.c_str()); return 1;
            }
        }
        double sp = point_spacing(P);
        if (o.voxel <= 0) o.voxel = sp * vmul;
        printf("# 面の符号化  N=%zu  点間隔 %.4f m  %s\n", n, sp,
               o.adaptive ? "適応分割" : "固定体素");
        if (o.adaptive) printf("  分割の閾値 %.4f m / 最小点数 %d", o.tau, o.min_pts);
        else            printf("  体素 %.4f m", o.voxel);
        printf(" / 法線 %d bit×2 / 刻み %.4f m%s%s\n\n", o.nbits, o.eps,
               o.jitter ? " / 粗さ復元" : "", o.predict ? " / 近傍から予測" : "");
        std::vector<double> out;
        auto R = surface_code(P, n, o, out);
        uint64_t tot = R.total();
        auto row = [&](const char* nm, uint64_t b){
            if (b) printf("    %-18s%12llu B  %7.3f bit/点\n", nm, (unsigned long long)b, b*8.0/n); };
        row("節点の占有", R.b_occ); row("分割の判断", R.b_split);
        row("法線", R.b_normal); row("面までの距離", R.b_offset);
        row("葉あたりの点数", R.b_count); row("粗さ", R.b_sigma);
        printf("    %-18s%12llu B  %7.3f bit/点   （葉 %zu 個・平均 %.1f 点/葉・復元 %zu 点）\n",
               "合計", (unsigned long long)tot, tot*8.0/n, R.n_leaf, R.mean_pts_per_leaf, R.n_out);
        if (with_attr) {
            // 属性を葉ごとの代表値（平均）にして、葉の順（幾何由来）で差分符号化する
            printf("\n  属性を面上の場にする\n");
            printf("    %-14s%12s%10s%12s%12s\n", "フィールド", "bytes", "bit/点", "誤差平均", "誤差95%");
            uint64_t ab = 0; double worst = 0;
            for (const auto& nm : pcA.order) {
                auto it = pcA.fields.find(nm);
                if (it == pcA.fields.end()) continue;
                const_cast<Field&>(it->second).materialize(n);
                if (it->second.v.size() != n) continue;
                const auto& v = it->second.v;
                std::vector<int64_t> mean(R.n_leaf);
                for (size_t t = 0; t < R.n_leaf; ++t) {
                    const auto& ids = R.leaf_ids[t];
                    if (ids.empty()) { mean[t] = 0; continue; }
                    long double s2 = 0;
                    for (int64_t i : ids) s2 += (long double)v[(size_t)i];
                    mean[t] = (int64_t)llroundl(s2 / (long double)ids.size());
                }
                std::vector<int64_t> d(R.n_leaf);
                for (size_t t = 0; t < R.n_leaf; ++t) d[t] = mean[t] - (t ? mean[t-1] : 0);
                uint64_t b = encode_ints(d.data(), d.size()).size();
                ab += b;
                std::vector<double> err; err.reserve(n);
                for (size_t t = 0; t < R.n_leaf; ++t)
                    for (int64_t i : R.leaf_ids[t])
                        err.push_back(std::fabs((double)(v[(size_t)i] - mean[t])));
                double s3 = 0; for (double x : err) s3 += x;
                double em = err.empty() ? 0 : s3 / err.size();
                size_t k = (size_t)(err.size() * 0.95);
                if (k >= err.size() && !err.empty()) k = err.size()-1;
                std::nth_element(err.begin(), err.begin()+k, err.end());
                double e95 = err.empty() ? 0 : err[k];
                printf("    %-14s%12llu%10.3f%12.2f%12.2f\n", nm.c_str(),
                       (unsigned long long)b, b*8.0/n, em, e95);
                worst = std::max(worst, em);
            }
            printf("    %-14s%12llu%10.3f\n", "属性 合計", (unsigned long long)ab, ab*8.0/n);
            printf("\n  幾何 %.3f + 属性 %.3f = %.3f bit/点\n",
                   tot*8.0/n, ab*8.0/n, (tot+ab)*8.0/n);
        }
        if (!outp.empty()) {
            FILE* f = fopen(outp.c_str(), "wb");
            fprintf(f, "ply\nformat binary_little_endian 1.0\nelement vertex %zu\n"
                       "property float x\nproperty float y\nproperty float z\nend_header\n", R.n_out);
            std::vector<float> buf(R.n_out*3);
            for (size_t i = 0; i < R.n_out*3; ++i) buf[i] = (float)out[i];
            fwrite(buf.data(), 4, buf.size(), f); fclose(f);
        }
        return 0;
    }
    if (cmd == "distort") {
        if (argc < 4) { fprintf(stderr, "distort には 2 つの点群が要る\n"); return 1; }
        std::vector<double> A, B; size_t na = 0, nb = 0; std::string e;
        if (!read_points_any(path, A, na, e)) { fprintf(stderr, "A: %s\n", e.c_str()); return 1; }
        if (!read_points_any(argv[3], B, nb, e)) { fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
        printf("# 対応を仮定しない歪み\n  A %s  %zu 点\n  B %s  %zu 点\n\n",
               path.c_str(), na, argv[3], nb);
        auto D = distortion(A, na, B, nb);
        printf("  点間の距離\n");
        printf("    Chamfer（双方向平均）   %10.4f m\n", D.chamfer);
        printf("    95 パーセンタイル       %10.4f m\n", D.p95);
        printf("    Hausdorff（双方向最大） %10.4f m\n", D.hausdorff);
        printf("      A→B 平均 %.4f / 最大 %.4f\n", D.a2b_mean, D.a2b_max);
        printf("      B→A 平均 %.4f / 最大 %.4f\n", D.b2a_mean, D.b2a_max);
        printf("\n  面までの距離（B の点を A の局所平面へ射影）\n");
        printf("    平均 %10.4f m / 95%%  %10.4f m / 最大 %10.4f m\n",
               D.plane_mean, D.plane_p95, D.plane_max);
        printf("\n  点の数の比 B/A = %.4f\n", (double)nb / (double)na);
        return 0;
    }
    if (cmd == "combine") {
        // 幾何は外部の符号器（G-PCC など）が運ぶ。本手法は属性だけを担当する。
        //
        // 順序の扱いが要点である。外部符号器は点の順序を保存しないので、
        // 復号器は座標から正準順序を作れなければならない。両側で Morton 順を使う。
        // 座標が完全に同じ点どうしの並びは幾何からは決まらないが、
        // 座標が同じである以上どちらにどの属性が付くかは幾何由来の情報ではない。
        // よって検証は (座標, 属性) の多重集合の一致で行う。
        if (argc < 5) { fprintf(stderr, "combine には 元LAZ・復号済み幾何PLY・幾何ストリーム の 3 つが要る\n"); return 1; }
        std::string plydec = argv[3], gstream = argv[4];
        size_t mp = 0;
        for (int i = 5; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--max-points" && i + 1 < argc) mp = strtoull(argv[++i], nullptr, 10);
        }
        PointCloud pc; std::string e;
        double t0 = now();
        if (!read_las(path, pc, e, mp)) { fprintf(stderr, "%s\n", e.c_str()); return 1; }
        materialize_all(pc);   // 解析用: 定数の列も配列にして .v で読めるようにする
        size_t n = pc.n;
        std::vector<double> Xd; size_t nd = 0;
        if (!read_points_any(plydec, Xd, nd, e, 0)) { fprintf(stderr, "%s\n", e.c_str()); return 1; }
        printf("# 幾何を外部に任せた構成  %s  N=%zu  読み込み %.1fs\n", path.c_str(), n, now() - t0);
        if (nd != n) { fprintf(stderr, "点数が違う: 元 %zu / 復号 %zu\n", n, nd); return 1; }

        // --- 正準順序（Morton）を両側で作る
        auto mort = [&](const std::vector<double>& xyz, size_t m) {
            double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
            for (size_t i = 0; i < m; ++i) for (int d = 0; d < 3; ++d) {
                lo[d] = std::min(lo[d], xyz[i*3+d]); hi[d] = std::max(hi[d], xyz[i*3+d]); }
            (void)hi;
            std::vector<int32_t> pm = coding_order(xyz, m, "morton");
            return pm;
        };
        // 座標は格納されている整数のまま扱う。PLY 側も同じ整数を持つので
        // 浮動小数の丸めを挟まずに厳密な照合ができる。
        std::vector<double> Xe(n * 3);
        for (size_t i = 0; i < n; ++i) {
            Xe[i*3] = (double)pc.X[i]; Xe[i*3+1] = (double)pc.Y[i]; Xe[i*3+2] = (double)pc.Z[i];
        }
        // 外部符号器は原点を自分で決めるので、両側とも軸ごとの最小値を引いて揃える
        auto rebase = [](std::vector<double>& v, size_t m) {
            for (int d = 0; d < 3; ++d) {
                double lo = 1e300;
                for (size_t i = 0; i < m; ++i) lo = std::min(lo, v[i*3+d]);
                for (size_t i = 0; i < m; ++i) v[i*3+d] -= lo;
            }
        };
        rebase(Xe, n); rebase(Xd, nd);
        auto pe = mort(Xe, n);       // 符号化側
        auto pd = mort(Xd, nd);      // 復号側
        // 同じ点集合かを Morton 順の座標列で確認する
        size_t dup = 0, mism = 0;
        for (size_t t = 0; t < n; ++t) {
            for (int d = 0; d < 3; ++d) {
                double a = Xe[(size_t)pe[t]*3+d], b = Xd[(size_t)pd[t]*3+d];
                if (std::fabs(a - b) > 1e-9) { ++mism; break; }
            }
            if (t && std::fabs(Xe[(size_t)pe[t]*3]   - Xe[(size_t)pe[t-1]*3])   < 1e-12
                  && std::fabs(Xe[(size_t)pe[t]*3+1] - Xe[(size_t)pe[t-1]*3+1]) < 1e-12
                  && std::fabs(Xe[(size_t)pe[t]*3+2] - Xe[(size_t)pe[t-1]*3+2]) < 1e-12) ++dup;
        }
        printf("  正準順序の照合: 座標のずれ %zu 点 / 完全重複 %zu 点\n", mism, dup);
        if (mism) { fprintf(stderr, "幾何が一致しない。比較にならない\n"); return 1; }

        // --- 属性を正準順序に並べ替え、幾何は 0 にした点群を作る
        PointCloud pa = pc;
        for (size_t t = 0; t < n; ++t) { pa.X[t] = 0; pa.Y[t] = 0; pa.Z[t] = 0; }
        for (auto& kv : pa.fields) {
            kv.second.materialize(n);
            if (kv.second.v.size() != n) continue;
            // 元の列は定数なら値の配列を持たない（1 値だけ覚える）。at() で読む。
            // 以前は .v を直に読み、定数の列がある入力（色だけの LAS など）で範囲外を読んで落ちた。
            const Field& src = pc.fields.at(kv.first);
            for (size_t t = 0; t < n; ++t) kv.second.v[t] = src.at((size_t)pe[t]);
        }
        // 予測子は実際の座標から作る（正準順序に並べ替えた幾何）
        std::vector<double> Xs(n * 3);
        for (size_t t = 0; t < n; ++t) for (int d = 0; d < 3; ++d) Xs[t*3+d] = Xe[(size_t)pe[t]*3+d];
        std::vector<int32_t> perm(n), pred;
        for (size_t t = 0; t < n; ++t) perm[t] = (int32_t)t;   // 既に正準順序
        build_causal_predictors(Xs, n, perm, 1, 16, pred);

        // --- 属性の正規化と空間予測
        Plan plan = analyze(pa, true);
        std::vector<std::string> keep;
        std::map<std::string, std::vector<int64_t>> ext;
        apply_plan(pa, plan, keep, ext);
        auto has = [&](const char* nm){ auto it = pa.fields.find(nm);
                                        return it != pa.fields.end() && it->second.v.size() == n; };
        // 既に定数や複製として落とした色に、さらに空間予測を掛けない
        auto already_dropped = [&](const char* nm) {
            for (const auto& o : plan.ops)
                if (o.target == nm && (o.kind == "drop_constant" ||
                    o.kind == "drop_duplicate" || o.kind == "drop_affine")) return true;
            return false;
        };
        if (has("red") && has("green") && has("blue") &&
            !already_dropped("red") && !already_dropped("green") && !already_dropped("blue")) {
            for (const char* nm : {"red","green","blue"}) {
                plan.ops.erase(std::remove_if(plan.ops.begin(), plan.ops.end(),
                    [&](const Op& o){ return o.kind == "residual_code" && o.target == nm; }), plan.ops.end());
                ext.erase(nm);
                keep.erase(std::remove(keep.begin(), keep.end(), std::string(nm)), keep.end());
            }
            const auto& R = pa.fields.at("red").v; const auto& G = pa.fields.at("green").v;
            const auto& B = pa.fields.at("blue").v;
            std::vector<int64_t> Y(n), Co(n), Cg(n);
            for (size_t i = 0; i < n; ++i) {
                int64_t co = R[i] - B[i], t = B[i] + (co >> 1), cg = G[i] - t;
                Y[i] = t + (cg >> 1); Co[i] = co; Cg[i] = cg;
            }
            spatial_residual(Y,  perm, pred, 1, n, ext["sp:red"]);
            spatial_residual(Co, perm, pred, 1, n, ext["sp:green"]);
            spatial_residual(Cg, perm, pred, 1, n, ext["sp:blue"]);
            for (const char* nm : {"red","green","blue"})
                plan.ops.push_back(Op{"spatial_code", nm, "", 0,1,1,0,true,""});
            plan.ops.push_back(Op{"ycocg", "", "", 0,1,1,0,true,""});
        }
        std::vector<std::string> nk;
        for (const auto& nm : keep) {
            auto it = pa.fields.find(nm);
            if (it == pa.fields.end() || it->second.v.size() != n) { nk.push_back(nm); continue; }
            const auto& v = it->second.v;
            std::vector<int64_t> rd(n), rs;
            for (size_t i = 0; i < n; ++i) rd[i] = v[i] - (i ? v[i-1] : 0);
            spatial_residual(v, perm, pred, 1, n, rs);
            if (encode_ints(rs.data(), n).size() * 20 < encode_ints(rd.data(), n).size() * 19) {
                ext["sp:" + nm] = rs;
                plan.ops.push_back(Op{"spatial_code", nm, "", 0,1,1,0,true,""});
            } else nk.push_back(nm);
        }
        keep = nk;

        std::string tmpL = "/tmp/pcc_comb.laz", tmpC = "/tmp/pcc_comb.pcc";
        if (!write_las(tmpL, pa, keep, e)) { fprintf(stderr, "属性容器の書き込み失敗: %s\n", e.c_str()); return 1; }
        uint64_t cb = 0;
        if (!write_container(tmpC, plan.to_json(), ext, cb, e, true)) {
            fprintf(stderr, "コンテナ失敗: %s\n", e.c_str()); return 1; }
        uint64_t G = fsize(gstream), L = fsize(tmpL), C = cb;

        // --- 検証: 属性を復元して元と照合する
        PointCloud rb;
        if (!read_las(tmpL, rb, e)) { fprintf(stderr, "読み戻し失敗\n"); return 1; }
        materialize_all(rb);   // 解析用: 定数の列も配列にして .v で読めるようにする
        std::string spec2; std::map<std::string, std::vector<int64_t>> ext2;
        if (!read_container(tmpC, spec2, ext2, n, e, true)) { fprintf(stderr, "コンテナ読み失敗\n"); return 1; }
        Plan p2 = Plan::from_json(spec2);
        std::map<std::string, std::vector<int64_t>> rec;
        for (const auto& nm : keep) if (rb.fields.count(nm)) rec[nm] = rb.fields.at(nm).v;
        if (rb.fields.count("bit_fields")) rec["bit_fields"] = rb.fields.at("bit_fields").v;
        bool yc = false;
        for (const auto& o : p2.ops) {
            if (o.kind == "spatial_code") {
                auto it = ext2.find("sp:" + o.target);
                if (it == ext2.end()) { fprintf(stderr, "sp:%s が無い\n", o.target.c_str()); return 1; }
                std::vector<int64_t> v; spatial_restore(it->second, perm, pred, 1, n, v);
                rec[o.target] = std::move(v);
            } else if (o.kind == "ycocg") yc = true;
        }
        if (yc) {
            auto& Y = rec["red"]; auto& Co = rec["green"]; auto& Cg = rec["blue"];
            for (size_t i = 0; i < n; ++i) {
                int64_t t = Y[i] - (Cg[i] >> 1), g = Cg[i] + t, b = t - (Co[i] >> 1), r = Co[i] + b;
                Y[i] = r; Co[i] = g; Cg[i] = b;
            }
        }
        p2.ops.erase(std::remove_if(p2.ops.begin(), p2.ops.end(),
            [](const Op& o){ return o.kind == "spatial_code" || o.kind == "ycocg"; }), p2.ops.end());
        if (!invert_plan(rec, ext2, p2, n, e)) { fprintf(stderr, "逆適用失敗: %s\n", e.c_str()); return 1; }
        std::vector<std::string> bad;
        for (const auto& nm : pc.order) {
            auto it = rec.find(nm);
            if (it == rec.end()) { bad.push_back(nm); continue; }
            const auto& want = pa.fields.at(nm).v;
            if (it->second != want) bad.push_back(nm);
        }

        printf("\n%-40s%14s%10s\n", "", "bytes", "bpp");
        printf("%-40s%14llu%10.3f\n", "幾何ストリーム（外部符号器）", (unsigned long long)G, G * 8.0 / n);
        printf("%-40s%14llu%10.3f\n", "属性容器（幾何を 0 にした LAZ）", (unsigned long long)L, L * 8.0 / n);
        printf("%-40s%14llu%10.3f\n", "属性の残差 + 仕様", (unsigned long long)C, C * 8.0 / n);
        printf("%-40s%14llu%10.3f\n", "合計", (unsigned long long)(G + L + C), (G + L + C) * 8.0 / n);
        printf("\n検証: 属性 完全一致=%s", bad.empty() ? "true" : "false");
        if (!bad.empty()) { printf("  ★不一致:"); for (auto& b : bad) printf(" %s", b.c_str()); }
        printf("\n");
        return 0;
    }
    if (cmd == "geomres") {
        // 幾何に残る余地を測る。取得順での各種予測の残差を同じ符号器に通す。
        size_t mp = 0;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--max-points" && i+1 < argc) mp = strtoull(argv[++i], nullptr, 10);
        }
        PointCloud pc; std::string e;
        if (!read_las(path, pc, e, mp)) { fprintf(stderr, "%s\n", e.c_str()); return 1; }
        materialize_all(pc);   // 解析用: 定数の列も配列にして .v で読めるようにする
        size_t n = pc.n;
        printf("# 幾何の予測  %s  N=%zu\n\n", path.c_str(), n);
        std::vector<std::vector<int64_t>> V(3, std::vector<int64_t>(n));
        for (size_t i = 0; i < n; ++i) { V[0][i]=pc.X[i]; V[1][i]=pc.Y[i]; V[2][i]=pc.Z[i]; }
        auto bits = [&](const std::vector<int64_t>& v){ return encode_ints(v.data(), n).size()*8.0/n; };
        const char* ax[3] = {"X","Y","Z"};
        double t_raw=0,t_d1=0,t_d2=0,t_d3=0;
        printf("  %-6s%10s%10s%10s%10s\n","軸","そのまま","1次差分","2次差分","3次差分");
        printf("  %s\n", std::string(46,'-').c_str());
        for (int d = 0; d < 3; ++d) {
            std::vector<int64_t> a1(n), a2(n), a3(n);
            for (size_t i=0;i<n;++i) a1[i]=V[d][i]-(i?V[d][i-1]:0);
            for (size_t i=0;i<n;++i) a2[i]=a1[i]-(i?a1[i-1]:0);
            for (size_t i=0;i<n;++i) a3[i]=a2[i]-(i?a2[i-1]:0);
            double b0=bits(V[d]), b1=bits(a1), b2=bits(a2), b3=bits(a3);
            t_raw+=b0; t_d1+=b1; t_d2+=b2; t_d3+=b3;
            printf("  %-6s%9.3fb%9.3fb%9.3fb%9.3fb\n", ax[d], b0,b1,b2,b3);
        }
        printf("  %s\n", std::string(46,'-').c_str());
        printf("  %-6s%9.3fb%9.3fb%9.3fb%9.3fb\n","合計",t_raw,t_d1,t_d2,t_d3);

        // オクトツリーで符号化した場合。幾何は空間予測が使えない
        //（位置そのものを符号化するので近傍探索が循環する）ので、
        // 使える構造は「取得順」か「オクトツリー」の二つしかない。
        {
            std::vector<double> W(n * 3);
            for (size_t i = 0; i < n; ++i) {
                W[i*3]   = pc.X[i] * pc.scale[0];
                W[i*3+1] = pc.Y[i] * pc.scale[1];
                W[i*3+2] = pc.Z[i] * pc.scale[2];
            }
            printf("\n  オクトツリーで符号化した場合\n");
            printf("  %10s%14s\n", "葉", "幾何の符号長");
            for (double leaf : {0.001, 0.002, 0.004, 0.008}) {
                uint64_t b = octree_bytes(W, n, leaf);
                printf("  %9.4fm%13.3fb\n", leaf, b * 8.0 / n);
            }
        }

        // Z を平面位置の関数とみなして、XY 近傍から予測する。
        // 復号器が XY を先に持てば、Z の予測に副情報は要らない。
        std::vector<double> XY(n*3);
        for (size_t i=0;i<n;++i){ XY[i*3]=pc.X[i]*pc.scale[0]; XY[i*3+1]=pc.Y[i]*pc.scale[1]; XY[i*3+2]=0; }
        auto perm = coding_order(XY, n, "morton");
        printf("\n  Z を XY 近傍から予測する（復号器は XY を先に持つ）\n");
        printf("  %-10s%12s\n","予測子の数","Z の符号長");
        for (int P : {1,2,3,5}) {
            std::vector<int32_t> pr; build_causal_predictors(XY, n, perm, P, 16, pr);
            std::vector<int64_t> rs; spatial_residual(V[2], perm, pr, P, n, rs);
            printf("  %-10d%11.3fb\n", P, encode_ints(rs.data(), n).size()*8.0/n);
        }
        return 0;
    }
    if (cmd == "attr") {
        size_t mp = 0; int ks = 16, P = 3; std::string ord = "morton"; bool ycocg = false;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--max-points" && i + 1 < argc) mp = strtoull(argv[++i], nullptr, 10);
            if (a == "--k" && i + 1 < argc) ks = atoi(argv[++i]);
            if (a == "--pred" && i + 1 < argc) P = atoi(argv[++i]);
            if (a == "--order" && i + 1 < argc) ord = argv[++i];
            if (a == "--ycocg") ycocg = true;
        }
        PointCloud pc; std::string e;
        double t0 = now();
        if (!read_las(path, pc, e, mp)) { fprintf(stderr, "%s\n", e.c_str()); return 1; }
        materialize_all(pc);   // 解析用: 定数の列も配列にして .v で読めるようにする
        size_t n = pc.n;
        printf("# 属性の空間予測  %s  N=%zu  読み込み %.1fs\n", path.c_str(), n, now() - t0);
        std::vector<double> xyz(n * 3);
        for (size_t i = 0; i < n; ++i) {
            xyz[i*3] = pc.X[i] * pc.scale[0]; xyz[i*3+1] = pc.Y[i] * pc.scale[1];
            xyz[i*3+2] = pc.Z[i] * pc.scale[2];
        }
        t0 = now();
        auto perm = coding_order(xyz, n, ord);
        std::vector<int32_t> pred;
        build_causal_predictors(xyz, n, perm, P, ks, pred);
        printf("  符号化順 %s / 予測子 %d 個 / k=%d 探索  構築 %.1fs\n\n",
               ord.c_str(), P, ks, now() - t0);

        printf("  %-14s%11s%11s%11s   %s\n", "フィールド", "そのまま", "格納順差分", "空間予測", "最良");
        printf("  %s\n", std::string(62, '-').c_str());
        double sr = 0, so = 0, ss = 0, sbest = 0; bool allok = true;
        // 座標も同じ土俵で測る（幾何は別経路だが参考に出す）
        std::vector<std::pair<std::string, const std::vector<int64_t>*>> list;
        std::vector<int64_t> zs(n);
        for (size_t i = 0; i < n; ++i) zs[i] = pc.Z[i];
        // 可逆の色変換 YCoCg-R。R,G,B を Y,Co,Cg に置き換える（完全に戻せる）。
        if (ycocg && pc.fields.count("red") && pc.fields.count("green") && pc.fields.count("blue")) {
            auto& r = pc.fields["red"].v; auto& g = pc.fields["green"].v; auto& b = pc.fields["blue"].v;
            std::vector<int64_t> Y(n), Co(n), Cg(n);
            for (size_t i = 0; i < n; ++i) {
                int64_t co = r[i] - b[i];
                int64_t t  = b[i] + (co >> 1);
                int64_t cg = g[i] - t;
                Y[i] = t + (cg >> 1); Co[i] = co; Cg[i] = cg;
            }
            // 戻せることをその場で確かめる
            bool inv_ok = true;
            for (size_t i = 0; i < n && inv_ok; ++i) {
                int64_t t = Y[i] - (Cg[i] >> 1);
                int64_t gg = Cg[i] + t, bb = t - (Co[i] >> 1), rr = Co[i] + bb;
                if (rr != r[i] || gg != g[i] || bb != b[i]) inv_ok = false;
            }
            printf("  YCoCg-R 変換  可逆性 %s\n\n", inv_ok ? "OK" : "NG");
            r = Y; g = Co; b = Cg;
            pc.fields["Y"]  = pc.fields["red"];   pc.fields["Co"] = pc.fields["green"];
            pc.fields["Cg"] = pc.fields["blue"];
            for (auto& nm : pc.order) { if (nm=="red") nm="Y"; else if (nm=="green") nm="Co";
                                        else if (nm=="blue") nm="Cg"; }
        }
        for (const auto& name : pc.order) {
            auto it = pc.fields.find(name);
            if (it == pc.fields.end() || it->second.v.size() != n) continue;
            list.emplace_back(name, &it->second.v);
        }
        for (auto& kv : list) {
            auto R = compare_field(kv.first, *kv.second, perm, pred, P, n);
            if (!R.roundtrip_ok) allok = false;
            sr += R.bpp_raw; so += R.bpp_order; ss += R.bpp_spatial;
            sbest += std::min(R.bpp_raw, std::min(R.bpp_order, R.bpp_spatial));
            const char* best = "そのまま";
            double bm = R.bpp_raw;
            if (R.bpp_order < bm)   { bm = R.bpp_order;   best = "格納順差分"; }
            if (R.bpp_spatial < bm) { bm = R.bpp_spatial; best = "空間予測"; }
            printf("  %-14s%10.3fb%10.3fb%10.3fb   %s\n", R.name.c_str(),
                   R.bpp_raw, R.bpp_order, R.bpp_spatial, best);
        }
        printf("  %s\n", std::string(60, '-').c_str());
        printf("  %-14s%10.3fb%10.3fb%10.3fb   最良の合計 %.3fb（検証 %s）\n", "合計",
               sr, so, ss, sbest, allok ? "OK" : "NG");
        printf("\n  元ファイル %.3f bpp（%zu bytes）\n",
               pc.src_bytes * 8.0 / n, (size_t)pc.src_bytes);
        return 0;
    }
    if (cmd == "diff") {
        if (argc < 4) { fprintf(stderr, "diff には対象と参照の 2 ファイルが要る\n"); return 1; }
        std::string refpath = argv[3];
        size_t mp = 0; int force_depth = 0;
        for (int i = 4; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--max-points" && i + 1 < argc) mp = strtoull(argv[++i], nullptr, 10);
            if (a == "--depth" && i + 1 < argc) force_depth = atoi(argv[++i]);
        }
        std::vector<double> T, Rf; size_t nt = 0, nr = 0; std::string e;
        if (!read_points_any(path, T, nt, e, mp)) { fprintf(stderr, "対象: %s\n", e.c_str()); return 1; }
        if (!read_points_any(refpath, Rf, nr, e, mp)) { fprintf(stderr, "参照: %s\n", e.c_str()); return 1; }
        printf("# 反復測量の差分符号化\n  対象 %s  %zu 点\n  参照 %s  %zu 点\n\n",
               path.c_str(), nt, refpath.c_str(), nr);
        double sp = point_spacing(T);
        printf("  対象の点間隔 %.4f m\n\n", sp);
        int md = 1; { double sp2 = sp; (void)sp2; }
        {   // 葉が点間隔の 1/4 程度になるまで降りる
            double side = 0; double lo[3]={1e300,1e300,1e300},hi[3]={-1e300,-1e300,-1e300};
            for (size_t i=0;i<nt;++i) for(int d=0;d<3;++d){lo[d]=std::min(lo[d],T[i*3+d]);hi[d]=std::max(hi[d],T[i*3+d]);}
            for (int d=0;d<3;++d) side=std::max(side,hi[d]-lo[d]);
            while (md < 20 && side/(double)(1ull<<md) > sp*0.25) ++md;
            if (force_depth > 0) md = force_depth;
        }
        printf("  オクトツリー占有バイトの条件付きエントロピー（深さ %d まで）\n", md);
        printf("  %7s%10s%12s%11s%11s%9s%11s\n", "深さ", "葉", "ノード数",
               "H(c)", "H(c|ref)", "削減", "参照命中");
        printf("  %s\n", std::string(72, '-').c_str());
        double ba = 0, bg = 0;
        for (auto& R : octree_diff(T, nt, Rf, nr, md)) {
            printf("  %6d%9.4fm%12zu%10.3fb%10.3fb%8.1f%%%10.1f%%\n",
                   R.depth, R.leaf, R.nodes, R.H, R.H_given,
                   100.0*(R.H-R.H_given)/std::max(R.H,1e-9), R.ref_hit*100);
            ba += R.bits_alone; bg += R.bits_given;
        }
        printf("  %s\n", std::string(72, '-').c_str());
        printf("  幾何の合計    参照なし %.3f bit/点   参照あり %.3f bit/点   削減 %.1f%%\n\n",
               ba, bg, 100.0*(ba-bg)/std::max(ba,1e-9));

        printf("\n  最近傍残差での符号化（座標そのものを送る場合）\n");
        printf("  %10s%14s%14s%10s%12s%12s\n", "格子", "参照なし", "残差", "削減", "最近傍中央", "同 P95");
        printf("  %s\n", std::string(74, '-').c_str());
        for (double m : {1.0, 0.25, 0.0625}) {
            double v = sp * m;
            auto R = nearest_residual(T, nt, Rf, nr, v);
            printf("  %9.4fm%13.3fb%13.3fb%9.1f%%%11.4fm%11.4fm\n",
                   v, R.bits_alone, R.bits_resid,
                   100.0*(R.bits_alone-R.bits_resid)/std::max(R.bits_alone,1e-9),
                   R.median_dist, R.p95_dist);
        }
        return 0;
    }
    size_t max_points = 0; bool residual = true; int grid_bits = 0; bool delta = true;
    bool spatial = false;   // 属性を幾何由来の順序＋空間予測で符号化する
    bool full_search = false;  // 貪欲ではなく、収束するまで全方式を試す
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max-points" && i + 1 < argc) max_points = strtoull(argv[++i], nullptr, 10);
        else if (a == "--no-residual") residual = false;
        else if (a == "--grid-bits" && i + 1 < argc) grid_bits = atoi(argv[++i]);
        else if (a == "--no-delta") delta = false;   // Python の bench と揃えるため
        else if (a == "--spatial") spatial = true;
        else if (a == "--full-search") full_search = true;
    }

    if (cmd == "sphere") {
        double R = atof(argv[2]);
        size_t N = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 20000;
        double sig = (argc > 4) ? atof(argv[4]) : 0.0;
        std::mt19937_64 rng(0); std::normal_distribution<double> g(0, 1);
        std::vector<double> S(N * 3);
        for (size_t i = 0; i < N; ++i) {
            double u[3] = {g(rng), g(rng), g(rng)};
            double nn = std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
            double rr = R * (1.0 + (sig > 0 ? sig * g(rng) : 0.0));
            for (int d = 0; d < 3; ++d) S[i*3+d] = u[d] / nn * rr;
        }
        double ssp = point_spacing(S);
        double vox = std::max(4 * ssp, 2 * sig * R);
        std::vector<double> NN; estimate_normals(S, N, 24, nullptr, NN);
        Tsdf T; std::string e2;
        if (!build_tsdf(S, NN, N, vox, T, e2)) { fprintf(stderr, "TSDF 失敗: %s\n", e2.c_str()); return 1; }
        std::vector<double> P, PN; extract_surface(T, P, PN);
        auto lfs = shrinking_ball_lfs(P, PN, 4 * R);
        std::sort(lfs.begin(), lfs.end());
        double med = lfs.empty() ? 0 : lfs[lfs.size()/2];
        printf("球 R=%g  N=%zu  σ=%g  ボクセル=%.5g  頂点 %zu  lfs中央 %.5g  誤差 %.1f%%\n",
               R, N, sig, vox, P.size()/3, med, std::fabs(med - R) / R * 100);
        return 0;
    }

    std::string err;
    double eps_opt = -1;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--eps" && i + 1 < argc) eps_opt = atof(argv[++i]);
    }
    double t0 = now();
    if (cmd == "score" || cmd == "octant" || cmd == "geom" ||
        cmd == "lfs" || cmd == "persist") {
        std::vector<double> W; size_t NP = 0;
        if (!read_points_any(path, W, NP, err, max_points)) {
            fprintf(stderr, "読み込み失敗: %s\n", err.c_str()); return 1; }
        struct { size_t n; } pc{NP};
        if (cmd == "score") {
            AcqScore S = acquisition_score(W, pc.n);
            printf("取得構造スコア %.2f   (%.1fs, %zu点)\n", S.score, S.seconds,
                   std::min<size_t>(60000, pc.n));
            for (const auto& kv : S.comp) {
                printf("  %-14s%5.2f  ", kv.first.c_str(), kv.second);
                for (int b = 0; b < (int)std::lround(kv.second * 24); ++b) printf("#");
                printf("\n");
            }
            for (const auto& nt : S.notes) printf("  ・%s\n", nt.c_str());
            printf("  → 格納順維持=%d  Morton候補=%d  極座標候補=%d  回転探索=%d\n",
                   S.keep_storage_order, S.consider_morton, S.consider_polar, S.consider_rotation);
            return 0;
        }
        // 重心を引いてから測る。**geom では引かない。**極座標の候補は座標の原点（地上型や
        // 回転式ではスキャナの位置）を中心にするものを含むので、重心に寄せると原点の候補が
        // 重心の候補と同じになり、スキャナの位置が失われる。TLS の講堂で「極座標は 34〜57% 長い」
        // と出たのはこのためだった（pack は寄せないので 38〜43% 短い。results/tls_polar_finding.md）。
        if (cmd != "geom") {
            double c[3] = {0,0,0};
            for (size_t i = 0; i < pc.n; ++i) for (int d = 0; d < 3; ++d) c[d] += W[i*3+d];
            for (int d = 0; d < 3; ++d) c[d] /= (double)pc.n;
            for (size_t i = 0; i < pc.n; ++i) for (int d = 0; d < 3; ++d) W[i*3+d] -= c[d];
        }
        double sp = point_spacing(W);
        if (cmd == "octant") {
            printf("点間隔 %.6g\n", sp);
            printf("  %12s%8s%9s%12s%9s%9s\n", "格子/点間隔", "被覆", "H(子)", "H(子|予測)", "情報量", "的中率");
            for (double r : {1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0, 2.0}) {
                auto R = octant_experiment(W, pc.n, sp * r);
                if (!R.n) { printf("  %12.4f%7.0f%%%9s%12s%9s%9s\n", r, 0.0, "—","—","—","—"); continue; }
                printf("  %12.4f%7.0f%%%9.3f%12.3f%9.3f%8.1f%%\n",
                       r, R.coverage*100, R.H, R.H_cond, R.mi(), R.acc*100);
            }
            return 0;
        }
        if (cmd == "lfs" || cmd == "persist") {
            // 標本を絞る（位相と lfs は点数に敏感なので、明示して測る）
            size_t m = std::min<size_t>(pc.n, (cmd == "persist") ? 20000 : 200000);
            std::vector<double> S(m * 3);
            size_t stride = std::max<size_t>(1, pc.n / m);
            for (size_t i = 0, j = 0; i < pc.n && j < m; i += stride, ++j)
                for (int d = 0; d < 3; ++d) S[j*3+d] = W[i*3+d];
            double ssp = point_spacing(S);
            if (cmd == "persist") {
                auto F = persistence_features(S, m);
                printf("標本 %zu 点  点間隔 %.6g  有限寿命の特徴 %zu 個\n", m, ssp, F.size());
                printf("  %4s%12s%12s%12s%11s%20s\n", "次元","誕生","消滅","寿命 L","L/点間隔","保存に必要な ε=L/2");
                for (size_t i = 0; i < std::min<size_t>(6, F.size()); ++i)
                    printf("  H%d%12.5g%12.5g%12.5g%11.2f%20.5g\n", F[i].dim,
                           F[i].birth, F[i].death, F[i].life, F[i].life/ssp, F[i].life/2);
                return 0;
            }
            double vox = eps_opt > 0 ? eps_opt : 4 * ssp;
            std::vector<double> NN;
            estimate_normals(S, m, 24, nullptr, NN);
            Tsdf T; std::string e2;
            if (!build_tsdf(S, NN, m, vox, T, e2)) { fprintf(stderr, "TSDF 失敗: %s\n", e2.c_str()); return 1; }
            std::vector<double> P, PN;
            extract_surface(T, P, PN);
            double lo2[3]={1e300,1e300,1e300}, hi2[3]={-1e300,-1e300,-1e300};
            for (size_t i=0;i<m;++i) for(int d=0;d<3;++d){lo2[d]=std::min(lo2[d],S[i*3+d]);hi2[d]=std::max(hi2[d],S[i*3+d]);}
            double ext=std::sqrt((hi2[0]-lo2[0])*(hi2[0]-lo2[0])+(hi2[1]-lo2[1])*(hi2[1]-lo2[1])+(hi2[2]-lo2[2])*(hi2[2]-lo2[2]));
            auto lfs = shrinking_ball_lfs(P, PN, ext / 4);
            std::sort(lfs.begin(), lfs.end());
            auto q = [&](double f){ return lfs.empty() ? 0.0 : lfs[(size_t)(f*(lfs.size()-1))]; };
            printf("標本 %zu 点  点間隔 %.6g  ボクセル %.6g（点間隔の %.1f 倍）  曲面頂点 %zu\n",
                   m, ssp, vox, vox/ssp, P.size()/3);
            printf("  lfs  P1 %.6g  P25 %.6g  中央 %.6g  P75 %.6g\n", q(0.01), q(0.25), q(0.5), q(0.75));
            printf("  ε=P1/2 %.6g   ε=P25/2 %.6g\n", q(0.01)/2, q(0.25)/2);
            return 0;
        }
        double eps = eps_opt > 0 ? eps_opt : sp / 16;
        printf("誤差上限 %.6g（点間隔 %.6g の 1/16）\n", eps, sp);
        auto g = choose_geometry(W, pc.n, eps, true, 200000, true);
        printf("  選択 %s%s  実誤差 %.4f mm\n", g.kind.c_str(),
               g.anchor.empty() ? "" : ("/" + g.anchor).c_str(), g.max_err * 1000);
        return 0;
    }

    // --- bench / residue 経路は LAS/LAZ のみ（属性まで扱うため）
    PointCloud pc;
    if (!read_las(path, pc, err, max_points)) {
        fprintf(stderr, "読み込み失敗: %s\n", err.c_str()); return 1; }
    materialize_all(pc);   // 解析用: 定数の列も配列にして .v で読めるようにする
    double t_read = now() - t0;

    if (cmd == "extern") {
        // 各フィールドについて「容器に残す」と「外部に出す（差分+レンジコーダ）」を
        // 実際に符号化して比べる。残差の相手が無いフィールドでも、
        // 差分が滑らかなら外部の方が安くなりうる。
        Plan plan = analyze(pc, true);
        std::vector<std::string> keep;
        std::map<std::string, std::vector<int64_t>> ext;
        apply_plan(pc, plan, keep, ext);
        std::string tmp = "/tmp/pccnorm_x.laz";
        if (!write_las(tmp, pc, keep, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        uint64_t base = fsize(tmp);
        printf("# 各フィールドを外部に出すと得か  %s  N=%zu\n", path.c_str(), pc.n);
        printf("  基準（正規化後の容器）= %.3f bpp\n\n", base * 8.0 / pc.n);
        printf("  %-20s%12s%12s%12s   判定\n", "フィールド", "容器での費用", "外部での費用", "差");
        printf("  %s\n", std::string(72, '-').c_str());
        double gain_total = 0;
        std::vector<std::string> win;
        for (const auto& nm : keep) {
            if (nm == "bit_fields") continue;          // 位置決定に関わるので触らない
            std::vector<std::string> k2;
            for (const auto& x : keep) if (x != nm) k2.push_back(x);
            if (!write_las(tmp, pc, k2, err)) continue;
            double in_container = (double)((int64_t)base - (int64_t)fsize(tmp)) * 8.0 / pc.n;
            std::map<std::string, std::vector<int64_t>> one{{nm, pc.fields.at(nm).v}};
            uint64_t b1 = 0;
            write_container("/tmp/pccnorm_x.pcc", "{}", one, b1, err, true);
            double outside = b1 * 8.0 / pc.n;
            double d = in_container - outside;
            if (d > 0) { gain_total += d; win.push_back(nm); }
            printf("  %-20s%11.3f %11.3f %11.3f   %s\n", nm.c_str(),
                   in_container, outside, d, d > 0 ? "外部が得" : "容器が得");
        }
        printf("\n  外部に出すと得なフィールドの合計利得 = %.3f bpp\n", gain_total);
        printf("  対象: ");
        for (const auto& w : win) printf("%s ", w.c_str());
        printf("\n");
        unlink(tmp.c_str()); unlink("/tmp/pccnorm_x.pcc");
        return 0;
    }

    if (cmd == "residue") {
        // 各フィールドを 1 つずつ定数で潰し、容器の縮み方から
        // 既存コーデックがそのフィールドに使っているビット数を出す。
        Plan plan = analyze(pc, true);
        std::vector<std::string> keep;
        std::map<std::string, std::vector<int64_t>> ext;
        apply_plan(pc, plan, keep, ext);
        std::string tmp = "/tmp/pccnorm_res.laz";
        if (!write_las(tmp, pc, keep, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        uint64_t base = fsize(tmp);
        uint64_t cb = 0;
        write_container("/tmp/pccnorm_res.pcc", plan.to_json(), ext, cb, err, true);
        printf("# 正規化後に何が残っているか  %s  N=%zu\n", path.c_str(), pc.n);
        printf("  容器 %.3f bpp + 外部 %.3f bpp = %.3f bpp\n",
               base * 8.0 / pc.n, cb * 8.0 / pc.n, (base + cb) * 8.0 / pc.n);
        printf("\n  %-22s%12s%10s   内訳\n", "フィールド", "bpp", "割合");
        printf("  %s\n", std::string(66, '-').c_str());
        // 幾何だけを残した場合
        std::vector<std::string> geo_only;
        uint64_t gonly = 0;
        if (write_las(tmp, pc, geo_only, err)) gonly = fsize(tmp);
        struct Row { std::string nm; double bpp; };
        std::vector<Row> rows;
        rows.push_back({"幾何 (XYZ)", gonly * 8.0 / pc.n});
        for (const auto& nm : keep) {
            std::vector<std::string> k2;
            for (const auto& x : keep) if (x != nm) k2.push_back(x);
            if (!write_las(tmp, pc, k2, err)) continue;
            rows.push_back({nm, (double)((int64_t)base - (int64_t)fsize(tmp)) * 8.0 / pc.n});
        }
        for (const auto& kv : ext) {
            std::map<std::string, std::vector<int64_t>> one{{kv.first, kv.second}};
            uint64_t b1 = 0;
            write_container("/tmp/pccnorm_res1.pcc", "{}", one, b1, err, true);
            rows.push_back({kv.first + " (外部)", b1 * 8.0 / pc.n});
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.bpp > b.bpp; });
        double tot = (base + cb) * 8.0 / pc.n;
        for (const auto& r : rows) {
            printf("  %-22s%11.3f%9.1f%%   ", r.nm.c_str(), r.bpp, 100.0 * r.bpp / tot);
            for (int i = 0; i < (int)std::lround(40 * r.bpp / std::max(rows[0].bpp, 1e-9)); ++i) printf("#");
            printf("\n");
        }
        unlink(tmp.c_str()); unlink("/tmp/pccnorm_res.pcc"); unlink("/tmp/pccnorm_res1.pcc");
        return 0;
    }

    printf("# 正規化レイヤー(C++)  %s  N=%zu  読み込み %.1fs\n", path.c_str(), pc.n, t_read);

    // A: 元のまま書き直す
    std::string tmpA = "/tmp/pccnorm_A.laz", tmpB = "/tmp/pccnorm_B.laz", tmpC = "/tmp/pccnorm.pcc";
    std::vector<std::string> all = pc.order;
    double ta = now();
    if (!write_las(tmpA, pc, all, err)) { fprintf(stderr, "A 書き込み失敗: %s\n", err.c_str()); return 1; }
    double t_a = now() - ta;
    uint64_t A = fsize(tmpA);
    if (A == 0) { fprintf(stderr, "A の書き込みに失敗した（空き容量を確認）\n"); return 2; }

    // 残差符号化を入れるかは、両方書いて短い方を採る。
    // エントロピーの閾値で決めると、既存コーデックが回収済みのフィールドを
    // 外に出して逆に太ることがある。
    double tan = now();
    Plan plan = analyze(pc, residual);
    double t_an = now() - tan;

    // 幾何の量子化
    std::vector<int32_t> Xq, Yq, Zq;
    double sc[3] = {pc.scale[0], pc.scale[1], pc.scale[2]};
    if (grid_bits > 0) {
        plan.grid = true; plan.grid_bits = grid_bits;
        int32_t half = 1 << (grid_bits - 1);
        Xq.resize(pc.n); Yq.resize(pc.n); Zq.resize(pc.n);
        for (size_t i = 0; i < pc.n; ++i) {
            Xq[i] = (int32_t)((pc.X[i] + half) >> grid_bits);
            Yq[i] = (int32_t)((pc.Y[i] + half) >> grid_bits);
            Zq[i] = (int32_t)((pc.Z[i] + half) >> grid_bits);
        }
        for (int i = 0; i < 3; ++i) sc[i] = pc.scale[i] * (double)(1 << grid_bits);
        double s0 = pc.scale[0] * (1 << grid_bits);
        plan.max_err_m = 0.5 * std::sqrt(3.0) * s0;
    }

    auto measure = [&](const Plan& pl, std::vector<std::string>& kp,
                       std::map<std::string, std::vector<int64_t>>& ex,
                       uint64_t& container, uint64_t& side) -> bool {
        apply_plan(pc, pl, kp, ex);
        std::string tmp = "/tmp/pccnorm_try.laz";
        if (!write_las(tmp, pc, kp, err, grid_bits ? &Xq : nullptr,
                       grid_bits ? &Yq : nullptr, grid_bits ? &Zq : nullptr,
                       grid_bits ? sc : nullptr)) return false;
        container = fsize(tmp);
        uint64_t cb = 0;
        if (!write_container("/tmp/pccnorm_try.pcc", pl.to_json(), ex, cb, err, delta))
            return false;
        side = cb;
        unlink(tmp.c_str()); unlink("/tmp/pccnorm_try.pcc");
        return true;
    };

    std::vector<std::string> keep;
    std::map<std::string, std::vector<int64_t>> external;
    double t_sel0 = now();
    if (residual) {
        // 残差ありと残差なしを実際に符号化して短い方を採る
        Plan p_res = plan, p_drop = plan;
        p_drop.ops.erase(std::remove_if(p_drop.ops.begin(), p_drop.ops.end(),
            [](const Op& o){ return o.kind == "residual_code"; }), p_drop.ops.end());
        std::vector<std::string> k1, k2;
        std::map<std::string, std::vector<int64_t>> e1, e2;
        uint64_t c1 = 0, s1 = 0, c2 = 0, s2 = 0;
        bool ok1 = measure(p_res, k1, e1, c1, s1);
        bool ok2 = measure(p_drop, k2, e2, c2, s2);
        if (ok1 && ok2) {
            printf("  残差符号化の採否: あり %.3f bpp / なし %.3f bpp → %s\n",
                   (c1 + s1) * 8.0 / pc.n, (c2 + s2) * 8.0 / pc.n,
                   (c1 + s1 <= c2 + s2) ? "あり" : "なし");
            if (c1 + s1 <= c2 + s2) { plan = p_res; keep = k1; external = e1; }
            else                    { plan = p_drop; keep = k2; external = e2; }
        } else apply_plan(pc, plan, keep, external);
    } else apply_plan(pc, plan, keep, external);

    // --spatial: 幾何は先に復号されるので、属性の符号化順を幾何から導いてよい。
    // Morton 順に並べ替え、その順で先行する最近傍から予測して残差を送る。
    // pc.fields には一切触らない。容器に入れない値は keep から外すだけでよい。
    // 採否は「空間予測ありの合計」と「なしの合計」を両方書いて短い方を採る。
    std::vector<int32_t> perm, pred;
    if (spatial) {
        std::vector<double> gxyz(pc.n * 3);
        for (size_t i = 0; i < pc.n; ++i) {
            int64_t x = grid_bits ? ((int64_t)Xq[i] << grid_bits) : (int64_t)pc.X[i];
            int64_t y = grid_bits ? ((int64_t)Yq[i] << grid_bits) : (int64_t)pc.Y[i];
            int64_t z = grid_bits ? ((int64_t)Zq[i] << grid_bits) : (int64_t)pc.Z[i];
            gxyz[i*3] = x * pc.scale[0]; gxyz[i*3+1] = y * pc.scale[1]; gxyz[i*3+2] = z * pc.scale[2];
        }
        perm = coding_order(gxyz, pc.n, "morton");
        build_causal_predictors(gxyz, pc.n, perm, 1, 16, pred);

        Plan plan_s = plan;
        auto keep_s = keep;
        auto ext_s = external;

        auto has = [&](const char* nm) {
            auto it = pc.fields.find(nm);
            return it != pc.fields.end() && it->second.v.size() == pc.n;
        };
        // 既に定数や複製として落とした色に、さらに空間予測を掛けない
        auto already_dropped = [&](const char* nm) {
            for (const auto& o : plan_s.ops)
                if (o.target == nm && (o.kind == "drop_constant" ||
                    o.kind == "drop_duplicate" || o.kind == "drop_affine")) return true;
            return false;
        };
        if (has("red") && has("green") && has("blue") &&
            !already_dropped("red") && !already_dropped("green") && !already_dropped("blue")) {
            // 色は 3 チャネルまとめて扱いたいので、解析が付けた残差の指示は取り消す
            for (const char* nm : {"red", "green", "blue"}) {
                plan_s.ops.erase(std::remove_if(plan_s.ops.begin(), plan_s.ops.end(),
                    [&](const Op& o){ return o.kind == "residual_code" && o.target == nm; }),
                    plan_s.ops.end());
                ext_s.erase(nm);
                keep_s.erase(std::remove(keep_s.begin(), keep_s.end(), std::string(nm)), keep_s.end());
            }
            const auto& R = pc.fields.at("red").v;
            const auto& G = pc.fields.at("green").v;
            const auto& Bb = pc.fields.at("blue").v;
            std::vector<int64_t> Y(pc.n), Co(pc.n), Cg(pc.n);
            for (size_t i = 0; i < pc.n; ++i) {
                int64_t co = R[i] - Bb[i];
                int64_t t  = Bb[i] + (co >> 1);
                int64_t cg = G[i] - t;
                Y[i] = t + (cg >> 1); Co[i] = co; Cg[i] = cg;
            }
            spatial_residual(Y,  perm, pred, 1, pc.n, ext_s["sp:red"]);
            spatial_residual(Co, perm, pred, 1, pc.n, ext_s["sp:green"]);
            spatial_residual(Cg, perm, pred, 1, pc.n, ext_s["sp:blue"]);
            for (const char* nm : {"red", "green", "blue"}) {
                Op op{"spatial_code", nm, "", 0,1,1,0,true,""};
                op.value = 1;                          // 画像由来は最近傍 1 個が最良
                plan_s.ops.push_back(op);
            }
            plan_s.ops.push_back(Op{"ycocg", "", "", 0,1,1,0,true,""});
            // （YCoCg を採ったことは色の流れの有無で分かる）
        }
        // 残りのフィールドは候補を並べて選ぶ。
        //   (1) 容器に残す
        //   (2) そのまま外部へ
        //   (3) 空間予測で外部へ、予測子の個数 P を選ぶ
        //
        // (1) の費用は「そのフィールドを抜いた容器」を実際に書いて測る。
        // 内側のコーデックは列をまたぐ文脈を使うので、単体の差分符号長を
        // 代理指標にすると判断を誤る。
        // 外部の候補どうしは同じ符号器なので、符号長を直接比べればよい。
        // 周辺費用は点あたりで見ればほぼスケール不変なので、標本で測る。
        // 全点で測ると属性の数だけ巨大な LAS を書くことになり、
        // 実行時間の大半を占めてしまう。
        const size_t MC_SAMPLE = 2000000;
        PointCloud pcs;                      // 標本の点群（全点なら pc をそのまま指す）
        const PointCloud* pmc = &pc;
        if (pc.n > MC_SAMPLE * 2) {
            pcs = pc; pcs.n = MC_SAMPLE;
            size_t stride = pc.n / MC_SAMPLE;
            pcs.X.resize(MC_SAMPLE); pcs.Y.resize(MC_SAMPLE); pcs.Z.resize(MC_SAMPLE);
            for (size_t t = 0; t < MC_SAMPLE; ++t) {
                size_t i = t * stride;
                pcs.X[t] = pc.X[i]; pcs.Y[t] = pc.Y[i]; pcs.Z[t] = pc.Z[i];
            }
            for (auto& kv : pcs.fields) {
                if (kv.second.v.size() != pc.n) continue;
                const auto& src = pc.fields.at(kv.first).v;
                std::vector<int64_t> w(MC_SAMPLE);
                for (size_t t = 0; t < MC_SAMPLE; ++t) w[t] = src[t * stride];
                kv.second.v = std::move(w);
            }
            pmc = &pcs;
        }
        auto las_bytes = [&](const std::vector<std::string>& kp) -> uint64_t {
            std::string e2;
            if (!write_las("/tmp/pccnorm_mc.laz", *pmc, kp, e2)) return UINT64_MAX;
            uint64_t b = fsize("/tmp/pccnorm_mc.laz");
            unlink("/tmp/pccnorm_mc.laz");
            // 0 バイトは書き込み失敗（多くはディスク不足）。黙って通すと
            // 費用が 0 に見えて判断が壊れるので、必ず落とす。
            if (b == 0) { fprintf(stderr, "一時ファイルの書き込みに失敗した（空き容量を確認）\n"); exit(2); }
            return b;
        };
        const double mc_scale = (double)pc.n / (double)pmc->n;
        uint64_t base_las = las_bytes(keep_s);
        std::vector<std::string> nk;
        const int PCAND[] = {1, 3, 5};
        for (const auto& nm : keep_s) {
            auto it = pc.fields.find(nm);
            if (it == pc.fields.end() || it->second.v.size() != pc.n) { nk.push_back(nm); continue; }
            const auto& v = it->second.v;
            // 容器から抜いたときに実際に減る量
            std::vector<std::string> without;
            for (const auto& x : keep_s) if (x != nm) without.push_back(x);
            uint64_t bwo = las_bytes(without);
            uint64_t marginal = (bwo == UINT64_MAX || bwo > base_las) ? 0
                              : (uint64_t)((base_las - bwo) * mc_scale);

            uint64_t best = marginal; int bestP = 0; bool best_raw = false;
            std::vector<int64_t> bestres;
            uint64_t braw = encode_ints(v.data(), pc.n).size();
            if (braw < best) { best = braw; best_raw = true; }
            for (int P : PCAND) {
                std::vector<int32_t> pr;
                if (P == 1) pr = pred;
                else build_causal_predictors(gxyz, pc.n, perm, P, 16, pr);
                std::vector<int64_t> rs;
                spatial_residual(v, perm, pr, P, pc.n, rs);
                uint64_t bs = encode_ints(rs.data(), pc.n).size();
                if (bs < best) { best = bs; bestP = P; best_raw = false; bestres = std::move(rs); }
            }
            if (bestP > 0) {
                ext_s["sp:" + nm] = std::move(bestres);
                Op op{"spatial_code", nm, "", 0,1,1,0,true,""};
                op.value = bestP;
                plan_s.ops.push_back(op);
            } else if (best_raw) {
                ext_s[nm] = v;
                plan_s.ops.push_back(Op{"plain_code", nm, "", 0,1,1,0,true,""});
            } else nk.push_back(nm);
        }
        keep_s = nk;

        // 代理指標で「外に出す」と判断したフィールドを、実際に書いて検証する。
        // 容器から 1 列外したときの LASzip の減り分は、その列の単体費用と一致しない。
        // 色は負値を取りうるので容器に入れられず、検証の対象外とする。
        auto total_of0 = [&](const Plan& pl, const std::vector<std::string>& kp,
                             const std::map<std::string, std::vector<int64_t>>& ex) -> uint64_t {
            std::string e2;
            if (!write_las("/tmp/pccnorm_vf.laz", pc, kp, e2, grid_bits ? &Xq : nullptr,
                           grid_bits ? &Yq : nullptr, grid_bits ? &Zq : nullptr,
                           grid_bits ? sc : nullptr)) return UINT64_MAX;
            uint64_t cb = 0;
            if (!write_container("/tmp/pccnorm_vf.pcc", pl.to_json(), ex, cb, e2, delta))
                return UINT64_MAX;
            uint64_t lb = fsize("/tmp/pccnorm_vf.laz");
            unlink("/tmp/pccnorm_vf.laz"); unlink("/tmp/pccnorm_vf.pcc");
            if (lb == 0) { fprintf(stderr, "一時ファイルの書き込みに失敗した（空き容量を確認）\n"); exit(2); }
            return lb + cb;
        };
        // --full-search: 貪欲な「戻すだけ」ではなく、各属性について全方式を
        // 端から端まで測って最良を採る。これを変化が止まるまで繰り返す。
        // 貪欲がどれだけ取りこぼすかを測るための参照実装であり、
        // 属性数×方式数の書き込みを毎巡行うので大きなファイルでは重い。
        if (full_search) {
            std::vector<std::string> cand;
            for (const auto& nm : pc.order) {
                auto it = pc.fields.find(nm);
                if (it == pc.fields.end() || it->second.v.size() != pc.n) continue;
                if (nm == "red" || nm == "green" || nm == "blue") continue;
                bool used = false;
                for (const auto& o : plan_s.ops)
                    if (o.target == nm && o.kind != "spatial_code" && o.kind != "plain_code")
                        used = true;     // 定数・複製・アフィン・残差は触らない
                if (!used) cand.push_back(nm);
            }
            bool changed = true; int rounds = 0;
            uint64_t cur = total_of0(plan_s, keep_s, ext_s);
            printf("    全方式探索を開始（候補 %zu 個、現在 %.3f bpp）\n",
                   cand.size(), cur * 8.0 / pc.n);
            while (changed && rounds < 6) {
                changed = false; ++rounds;
                for (const auto& nm : cand) {
                    // いまの方式を外した構成を土台にする
                    Plan p0 = plan_s; auto k0 = keep_s; auto e0 = ext_s;
                    for (size_t oi = 0; oi < p0.ops.size(); ++oi)
                        if (p0.ops[oi].target == nm &&
                            (p0.ops[oi].kind == "spatial_code" || p0.ops[oi].kind == "plain_code")) {
                            e0.erase(p0.ops[oi].kind == "spatial_code" ? ("sp:"+nm) : nm);
                            p0.ops.erase(p0.ops.begin() + (long)oi);
                            break;
                        }
                    if (std::find(k0.begin(), k0.end(), nm) == k0.end()) k0.push_back(nm);
                    const auto& v = pc.fields.at(nm).v;
                    uint64_t bestT = total_of0(p0, k0, e0);      // 容器に残す
                    Plan bp = p0; auto bk = k0; auto be = e0;
                    // そのまま外部へ
                    {
                        Plan p1 = p0; auto k1 = k0; auto e1 = e0;
                        k1.erase(std::remove(k1.begin(), k1.end(), nm), k1.end());
                        e1[nm] = v;
                        p1.ops.push_back(Op{"plain_code", nm, "", 0,1,1,0,true,""});
                        uint64_t t = total_of0(p1, k1, e1);
                        if (t < bestT) { bestT = t; bp = std::move(p1); bk = std::move(k1); be = std::move(e1); }
                    }
                    for (int P : {1, 3, 5}) {
                        std::vector<int32_t> pr;
                        if (P == 1) pr = pred;
                        else build_causal_predictors(gxyz, pc.n, perm, P, 16, pr);
                        std::vector<int64_t> rs; spatial_residual(v, perm, pr, P, pc.n, rs);
                        Plan p1 = p0; auto k1 = k0; auto e1 = e0;
                        k1.erase(std::remove(k1.begin(), k1.end(), nm), k1.end());
                        e1["sp:" + nm] = std::move(rs);
                        Op op{"spatial_code", nm, "", 0,1,1,0,true,""}; op.value = P;
                        p1.ops.push_back(op);
                        uint64_t t = total_of0(p1, k1, e1);
                        if (t < bestT) { bestT = t; bp = std::move(p1); bk = std::move(k1); be = std::move(e1); }
                    }
                    if (bestT + 1 < cur) {
                        printf("    %s を替えて %.3f → %.3f bpp\n", nm.c_str(),
                               cur * 8.0 / pc.n, bestT * 8.0 / pc.n);
                        plan_s = std::move(bp); keep_s = std::move(bk); ext_s = std::move(be);
                        cur = bestT; changed = true;
                    }
                }
            }
            printf("    全方式探索を終了（%d 巡、%.3f bpp）\n", rounds, cur * 8.0 / pc.n);
        }
        {
            bool changed = true;
            int rounds = 0;
            while (changed && rounds < 3) {
                changed = false; ++rounds;
                uint64_t cur = total_of0(plan_s, keep_s, ext_s);
                for (size_t oi = 0; oi < plan_s.ops.size(); ++oi) {
                    const Op& op = plan_s.ops[oi];
                    bool is_sp = (op.kind == "spatial_code" || op.kind == "plain_code");
                    if (!is_sp) continue;
                    if (op.target == "red" || op.target == "green" || op.target == "blue") continue;
                    // この 1 件を取りやめた構成を作って比べる
                    Plan p2 = plan_s; auto k2 = keep_s; auto e2m = ext_s;
                    p2.ops.erase(p2.ops.begin() + (long)oi);
                    e2m.erase(op.kind == "spatial_code" ? ("sp:" + op.target) : op.target);
                    k2.push_back(op.target);
                    uint64_t alt = total_of0(p2, k2, e2m);
                    if (alt < cur) {
                        printf("    %s を外に出すのは損だった（%.3f → %.3f bpp）。容器に戻す\n",
                               op.target.c_str(), cur * 8.0 / pc.n, alt * 8.0 / pc.n);
                        plan_s = std::move(p2); keep_s = std::move(k2); ext_s = std::move(e2m);
                        cur = alt; changed = true; break;
                    }
                }
            }
        }

        // 両方を実際に書いて短い方を採る
        auto total_of = [&](const Plan& pl, const std::vector<std::string>& kp,
                            const std::map<std::string, std::vector<int64_t>>& ex) -> uint64_t {
            std::string e2;
            if (!write_las("/tmp/pccnorm_sp.laz", pc, kp, e2, grid_bits ? &Xq : nullptr,
                           grid_bits ? &Yq : nullptr, grid_bits ? &Zq : nullptr,
                           grid_bits ? sc : nullptr)) return UINT64_MAX;
            uint64_t cb = 0;
            if (!write_container("/tmp/pccnorm_sp.pcc", pl.to_json(), ex, cb, e2, delta))
                return UINT64_MAX;
            uint64_t t = fsize("/tmp/pccnorm_sp.laz") + cb;
            unlink("/tmp/pccnorm_sp.laz"); unlink("/tmp/pccnorm_sp.pcc");
            return t;
        };
        uint64_t t_plain = total_of(plan, keep, external);
        uint64_t t_sp    = total_of(plan_s, keep_s, ext_s);
        printf("  空間予測の採否: あり %.3f bpp / なし %.3f bpp → %s\n",
               t_sp * 8.0 / pc.n, t_plain * 8.0 / pc.n, t_sp < t_plain ? "あり" : "なし");
        if (t_sp < t_plain) { plan = plan_s; keep = keep_s; external = ext_s; }
        else { spatial = false; }
    }

    double t_sel = now() - t_sel0;
    double tb = now();
    if (!write_las(tmpB, pc, keep, err, grid_bits ? &Xq : nullptr,
                   grid_bits ? &Yq : nullptr, grid_bits ? &Zq : nullptr,
                   grid_bits ? sc : nullptr)) {
        fprintf(stderr, "B 書き込み失敗: %s\n", err.c_str()); return 1;
    }
    double t_b = now() - tb;
    uint64_t Bc = fsize(tmpB);

    std::string spec = plan.to_json();
    uint64_t cbytes = 0;
    double te = now();
    if (!write_container(tmpC, spec, external, cbytes, err, delta)) {
        fprintf(stderr, "コンテナ失敗: %s\n", err.c_str()); return 1;
    }
    double t_ext = now() - te;
    uint64_t B = Bc + cbytes;

    // 正規化して得にならないなら、しない。仕様には空の計画を書く。
    // 小さいファイルでは仕様とコンテナのヘッダが固定費として効くので、
    // 「効くと分かったときだけ適用する」が唯一の安全な既定になる。
    bool applied = true;
    if (grid_bits == 0 && B >= A) {
        applied = false;
        printf("  正規化は得にならなかった（%.3f → %.3f bpp）。適用しない\n",
               A * 8.0 / pc.n, B * 8.0 / pc.n);
        plan = Plan();
        keep = pc.order;
        external.clear();
        if (!write_las(tmpB, pc, keep, err)) { fprintf(stderr, "B 再書き込み失敗\n"); return 1; }
        Bc = fsize(tmpB);
        spec = plan.to_json();
        if (!write_container(tmpC, spec, external, cbytes, err, delta)) return 1;
        B = Bc + cbytes;
    }
    (void)applied;

    // --- 検証: B から元データを復元する
    double tv = now();
    PointCloud rb;
    if (!read_las(tmpB, rb, err)) { fprintf(stderr, "B 読み戻し失敗: %s\n", err.c_str()); return 1; }
    materialize_all(rb);   // 解析用: 定数の列も配列にして .v で読めるようにする
    std::string spec2;
    std::map<std::string, std::vector<int64_t>> ext2;
    if (!read_container(tmpC, spec2, ext2, pc.n, err, delta)) {
        fprintf(stderr, "コンテナ読み失敗: %s\n", err.c_str()); return 1;
    }
    Plan plan2 = Plan::from_json(spec2);
    std::map<std::string, std::vector<int64_t>> rec;
    for (const auto& nm : keep) if (rb.fields.count(nm)) rec[nm] = rb.fields.at(nm).v;
    if (rb.fields.count("bit_fields")) rec["bit_fields"] = rb.fields.at("bit_fields").v;
    // 空間予測と YCoCg-R を先に戻す。復号器は幾何（rb）を持っているので、
    // 符号化側とまったく同じ順序表・近傍表を組める。
    if (spatial) {
        std::vector<double> gxyz2(pc.n * 3);
        for (size_t i = 0; i < pc.n; ++i) {
            int64_t x = grid_bits ? ((int64_t)rb.X[i] << grid_bits) : (int64_t)rb.X[i];
            int64_t y = grid_bits ? ((int64_t)rb.Y[i] << grid_bits) : (int64_t)rb.Y[i];
            int64_t z = grid_bits ? ((int64_t)rb.Z[i] << grid_bits) : (int64_t)rb.Z[i];
            gxyz2[i*3] = x * pc.scale[0]; gxyz2[i*3+1] = y * pc.scale[1]; gxyz2[i*3+2] = z * pc.scale[2];
        }
        auto perm2 = coding_order(gxyz2, pc.n, "morton");
        std::vector<int32_t> pred2;
        build_causal_predictors(gxyz2, pc.n, perm2, 1, 16, pred2);
        bool has_ycocg = false;
        for (const auto& o : plan2.ops) {
            if (o.kind == "spatial_code") {
                auto it = ext2.find("sp:" + o.target);
                if (it == ext2.end()) { fprintf(stderr, "sp:%s が無い\n", o.target.c_str()); return 1; }
                int P = o.value > 0 ? (int)o.value : 1;
                std::vector<int32_t> pr;
                if (P == 1) pr = pred2;
                else build_causal_predictors(gxyz2, pc.n, perm2, P, 16, pr);
                std::vector<int64_t> v;
                spatial_restore(it->second, perm2, pr, P, pc.n, v);
                rec[o.target] = std::move(v);
            } else if (o.kind == "plain_code") {
                auto it = ext2.find(o.target);
                if (it == ext2.end()) { fprintf(stderr, "%s が無い\n", o.target.c_str()); return 1; }
                rec[o.target] = it->second;
            } else if (o.kind == "ycocg") has_ycocg = true;
        }
        if (has_ycocg) {
            auto& Y = rec["red"]; auto& Co = rec["green"]; auto& Cg = rec["blue"];
            for (size_t i = 0; i < pc.n; ++i) {
                int64_t t = Y[i] - (Cg[i] >> 1);
                int64_t g = Cg[i] + t, b = t - (Co[i] >> 1), r = Co[i] + b;
                Y[i] = r; Co[i] = g; Cg[i] = b;
            }
        }
        // 自前で戻した分は invert_plan に渡さない
        plan2.ops.erase(std::remove_if(plan2.ops.begin(), plan2.ops.end(),
            [](const Op& o){ return o.kind == "spatial_code" || o.kind == "plain_code"
                                 || o.kind == "ycocg"; }), plan2.ops.end());
    }
    if (!invert_plan(rec, ext2, plan2, pc.n, err)) {
        fprintf(stderr, "逆適用失敗: %s\n", err.c_str()); return 1;
    }
    std::vector<std::string> bad;
    for (const auto& nm : pc.order) {
        auto it = rec.find(nm);
        if (it == rec.end() || it->second != pc.fields.at(nm).v) bad.push_back(nm);
    }
    bool geom_ok = true; double emax = 0;
    if (grid_bits > 0) {
        for (size_t i = 0; i < pc.n; ++i) {
            double dx = ((double)((int64_t)rb.X[i] << grid_bits) - pc.X[i]) * pc.scale[0];
            double dy = ((double)((int64_t)rb.Y[i] << grid_bits) - pc.Y[i]) * pc.scale[1];
            double dz = ((double)((int64_t)rb.Z[i] << grid_bits) - pc.Z[i]) * pc.scale[2];
            emax = std::max(emax, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        geom_ok = emax <= plan.max_err_m + 1e-12;
    } else {
        for (size_t i = 0; i < pc.n; ++i)
            if (rb.X[i] != pc.X[i] || rb.Y[i] != pc.Y[i] || rb.Z[i] != pc.Z[i]) { geom_ok = false; break; }
    }
    double t_ver = now() - tv;

    printf("\n%s\n", plan.report().c_str());
    printf("%-38s%14s%10s\n", "", "bytes", "bpp");
    printf("%-38s%14llu%10.3f\n", "A. 元のまま LAZ", (unsigned long long)A, A * 8.0 / pc.n);
    printf("%-38s%14llu%10.3f\n", "B. 正規化後 容器", (unsigned long long)Bc, Bc * 8.0 / pc.n);
    printf("%-38s%14llu%10.3f\n", "   + 外部 + 仕様", (unsigned long long)cbytes, cbytes * 8.0 / pc.n);
    printf("%-38s%14llu%10.3f\n", "B. 合計", (unsigned long long)B, B * 8.0 / pc.n);
    int64_t diff = (int64_t)A - (int64_t)B;     // 符号なし同士の引き算は破綻する
    printf("%-38s%14lld%10.3f   = %+.1f%%\n", "削減", (long long)diff,
           diff * 8.0 / (double)pc.n, 100.0 * (double)diff / (double)A);
    if (grid_bits > 0)
        printf("\n検証: 幾何 誤差上限 %.2fmm 以内（実測 max %.2fmm） → OK=%s   属性 完全一致=%s\n",
               plan.max_err_m * 1000, emax * 1000, geom_ok ? "true" : "false",
               bad.empty() ? "true" : "false");
    else
        printf("\n検証: 幾何 ビット完全 → OK=%s   属性 完全一致=%s\n",
               geom_ok ? "true" : "false", bad.empty() ? "true" : "false");
    if (!bad.empty()) { printf("  ★不一致: "); for (auto& s : bad) printf("%s ", s.c_str()); printf("\n"); }
    printf("時間: 読込 %.1fs / A書き %.1fs / 解析 %.1fs / 選択 %.1fs / B書き %.1fs / 外部 %.1fs / 検証 %.1fs / 合計 %.1fs\n",
           t_read, t_a, t_an, t_sel, t_b, t_ext, t_ver, now() - t0);
    printf("ピークメモリ %.2f GB\n", peak_gb());
    unlink(tmpA.c_str()); unlink(tmpB.c_str()); unlink(tmpC.c_str());
    return (geom_ok && bad.empty()) ? 0 : 2;
}

// 例外は最後にここで受ける。受けないと std::terminate で落ち、後始末（作業用の
// 器を消すなど）が走らない。巻き戻しの途中で各所の後始末が走ってから、ここに来る。
int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "失敗: %s\n", e.what());
        return 1;
    }
}
