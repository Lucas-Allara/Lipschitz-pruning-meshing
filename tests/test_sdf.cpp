// Checks the 1-Lipschitz property of the scenes and the soundness of the interval bounds.
#include <cstdio>
#include <cstdlib>
#include <random>

#include "lpm/synthetic.h"
#include "lpm/tree.h"

using namespace lpm;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void run(const Scene& s, unsigned seed) {
    std::mt19937_64 rng(seed);
    auto uni = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
    const std::vector<TreeEntry> full = full_tree(s);
    const TreeView t{full.data(), (uint32_t)full.size()};
    Scratch sc;
    Counters c;
    // Lipschitz: |f(p) - f(q)| <= |p - q|.
    for (int it = 0; it < 20000; it++) {
        double p[3], q[3], d2 = 0.0;
        for (int a = 0; a < 3; a++) {
            p[a] = uni(s.aabb_min[a], s.aabb_max[a]);
            q[a] = p[a] + uni(-0.1, 0.1) * (s.aabb_max[a] - s.aabb_min[a]);
            d2 += (p[a] - q[a]) * (p[a] - q[a]);
        }
        const double fp = eval_tree(s, t, p, sc, c), fq = eval_tree(s, t, q, sc, c);
        CHECK(std::fabs(fp - fq) <= std::sqrt(d2) * (1.0 + 1e-9) + 1e-12, "Lipschitz violated: |%g - %g| > %g", fp, fq, std::sqrt(d2));
    }
    // Interval soundness: f(p) in F(box) for p in box, for random boxes of random sizes.
    for (int it = 0; it < 3000; it++) {
        Interval box[3];
        const double frac = std::pow(2.0, -uni(0.0, 8.0));
        for (int a = 0; a < 3; a++) {
            const double w = frac * (s.aabb_max[a] - s.aabb_min[a]);
            const double lo = uni(s.aabb_min[a], s.aabb_max[a] - w);
            box[a] = Interval(lo, lo + w);
        }
        const Interval F = eval_tree_interval(s, t, box, sc, c);
        for (int k = 0; k < 20; k++) {
            double p[3];
            for (int a = 0; a < 3; a++) p[a] = uni(box[a].lo, box[a].hi);
            const double fp = eval_tree(s, t, p, sc, c);
            CHECK(F.contains(fp), "interval [%g, %g] misses %g", F.lo, F.hi, fp);
        }
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        Scene s = load_scene_json(argv[i]);
        run(s, 7);
        printf("%s: %zu nodes checked\n", argv[i], s.nodes.size());
    }
    for (const char* shapes : {"sphere", "mixed"}) {
        for (const char* ops : {"union", "mixed"}) {
            for (double k : {0.0, 0.03}) {
                SyntheticParams p;
                p.n = 60; p.shapes = shapes; p.ops = ops; p.k = k; p.seed = 11; p.radius = 0.15; p.extent = 0.6;
                Scene s = make_synthetic(p);
                run(s, 3);
                printf("synthetic %s/%s k=%g: ok\n", shapes, ops, k);
            }
        }
    }
    printf("%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
