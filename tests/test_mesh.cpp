// The three oracles must produce bit-identical meshes, and the meshes must be closed.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <string>

#include "lpm/adaptive.h"
#include "lpm/dc.h"
#include "lpm/synthetic.h"

using namespace lpm;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static Mesh run(const Scene& s, Oracle o, int depth, bool analytic, bool inherit) {
    OctreeParams p;
    p.oracle = o;
    p.max_depth = depth;
    p.inherit = inherit;
    Counters c;
    Octree oct = build_octree(s, p, c);
    DCParams dp;
    dp.analytic_leaves = analytic;
    DCStats st;
    return dual_contour(oct, dp, c, st);
}

static void check_closed(const Mesh& m, const char* label) {
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (size_t i = 0; i < m.F.size(); i += 3) {
        for (int k = 0; k < 3; k++) {
            const uint32_t a = m.F[i + k], b = m.F[i + (k + 1) % 3];
            edges[{a, b}]++;
        }
    }
    // Closed: every half-edge is matched by its opposite, and manifold: no edge with more
    // than two triangles (one vertex per component of crossing edges).
    int boundary = 0, nonmanifold = 0;
    for (const auto& [e, n] : edges) {
        auto it = edges.find({e.second, e.first});
        const int m = it == edges.end() ? 0 : it->second;
        if (n != m) boundary++;
        if (n > 1 && e.first < e.second) nonmanifold++;
    }
    CHECK(boundary == 0, "%s: %d boundary half-edges", label, boundary);
    CHECK(nonmanifold == 0, "%s: %d non-manifold edges", label, nonmanifold);
    printf("%s: %d non-manifold edges of %zu\n", label, nonmanifold, edges.size() / 2);
    // Signed volume must be positive for an outward orientation.
    double vol = 0.0;
    for (size_t i = 0; i < m.F.size(); i += 3) {
        const double* a = &m.V[3 * m.F[i]];
        const double* b = &m.V[3 * m.F[i + 1]];
        const double* c = &m.V[3 * m.F[i + 2]];
        vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    CHECK(vol > 0.0, "%s: negative signed volume %g", label, vol);
}

// The OBJ text of write_obj is that of printf("%.17g") and "%u", and write_ply stores the mesh
// exactly.
static void check_writers(const Mesh& m) {
    std::mt19937_64 rng(7);
    int bad = 0;
    for (int i = 0; i < 200000; i++) {
        uint64_t u = rng();
        double v;
        memcpy(&v, &u, 8);
        if (!std::isfinite(v)) continue;
        if (i % 2) v = std::ldexp((double)(u >> 11) / 9007199254740992.0 - 0.5, (int)(u % 8) - 4);
        char ref[64];
        snprintf(ref, sizeof ref, "%.17g", v);
        std::string t;
        detail::append_double(t, v);
        bad += t != ref;
    }
    CHECK(bad == 0, "append_double differs from %%.17g on %d values", bad);
    const std::string path = "test_mesh_writer.ply";
    CHECK(write_ply(m, path), "cannot write %s", path.c_str());
    FILE* f = fopen(path.c_str(), "rb");
    CHECK(f != nullptr, "cannot read %s", path.c_str());
    if (!f) return;
    char line[128];
    size_t nv = 0, nf = 0;
    while (fgets(line, sizeof line, f) && strcmp(line, "end_header\n") != 0) {
        sscanf(line, "element vertex %zu", &nv);
        sscanf(line, "element face %zu", &nf);
    }
    std::vector<double> V(3 * nv);
    const size_t got_v = fread(V.data(), 8, V.size(), f);
    std::vector<uint32_t> F;
    for (size_t i = 0; i < nf; i++) {
        unsigned char n = 0;
        uint32_t tri[3];
        if (fread(&n, 1, 1, f) != 1 || fread(tri, 4, 3, f) != 3) break;
        CHECK(n == 3, "face %zu has %d vertices", i, n);
        F.insert(F.end(), tri, tri + 3);
    }
    fclose(f);
    remove(path.c_str());
    CHECK(got_v == m.V.size() && V == m.V && F == m.F, "PLY round trip differs");
    printf("writers: OBJ text as printf, PLY round trip exact (%zu vertices, %zu triangles)\n", nv, nf);
}

int main() {
    bool writers_checked = false;
    for (const char* shapes : {"sphere", "mixed"}) {
        for (double k : {0.0, 0.03}) {
            SyntheticParams p;
            p.n = 40; p.shapes = shapes; p.ops = "mixed"; p.k = k; p.seed = 21; p.radius = 0.15; p.extent = 0.6;
            Scene s = make_synthetic(p);
            const Mesh full = run(s, ORACLE_FULL, 6, false, true);
            const Mesh lip = run(s, ORACLE_LIPSCHITZ, 6, false, true);
            const Mesh itv = run(s, ORACLE_INTERVAL, 6, false, true);
            const Mesh noinh = run(s, ORACLE_LIPSCHITZ, 6, false, false);
            CHECK(full.V == lip.V && full.F == lip.F, "%s k=%g: lipschitz mesh differs (%zu vs %zu vertices)", shapes, k, full.V.size() / 3, lip.V.size() / 3);
            CHECK(full.V == itv.V && full.F == itv.F, "%s k=%g: interval mesh differs", shapes, k);
            CHECK(full.V == noinh.V && full.F == noinh.F, "%s k=%g: non-inherited mesh differs", shapes, k);
            check_closed(full, shapes);
            if (std::string(shapes) == "sphere") {
                const Mesh ana = run(s, ORACLE_LIPSCHITZ, 6, true, true);
                CHECK(ana.F == full.F, "analytic: connectivity differs");
                double maxd = 0.0;
                for (size_t i = 0; i < full.V.size(); i++) maxd = std::max(maxd, std::fabs(full.V[i] - ana.V[i]));
                CHECK(maxd < 1e-6, "analytic: vertices moved by %g", maxd);
                printf("analytic variant: max vertex deviation %.3g\n", maxd);
            }
            printf("%s k=%g: %zu vertices, %zu triangles, identical for the three oracles\n", shapes, k, full.V.size() / 3, full.F.size() / 3);
            if (!writers_checked) { check_writers(full); writers_checked = true; }
            // Adaptive contouring, bottom-up and top-down: closed and manifold with leaves of
            // different sizes, the top-down one the same in two runs.
            for (bool top_down : {false, true})
                for (bool all : {false, true}) {
                    AdaptiveParams ap;
                    ap.max_depth = 6; ap.tol = 0.05; ap.single_primitive_only = !all; ap.top_down = top_down;
                    DCParams dp;
                    Counters c;
                    AdaptiveStats ast;
                    const Mesh am = adaptive_mesh(s, ap, dp, c, ast);
                    char label[96];
                    snprintf(label, sizeof label, "adaptive %s %s (%s)", top_down ? "top-down" : "bottom-up", shapes, all ? "any cell" : "single primitive");
                    check_closed(am, label);
                    if (top_down) {
                        AdaptiveStats ast2;
                        const Mesh am2 = adaptive_mesh(s, ap, dp, c, ast2);
                        CHECK(am.V == am2.V && am.F == am2.F, "%s: two runs differ", label);
                    }
                    printf("%s: %llu leaves (%llu coarse), %zu triangles\n", label, (unsigned long long)ast.leaves, (unsigned long long)ast.collapsed, am.F.size() / 3);
                }
        }
    }
    printf("%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
