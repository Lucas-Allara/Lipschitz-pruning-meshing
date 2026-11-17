// The pruned tree of a cell must evaluate exactly like the full tree inside the cell,
// whether it is pruned from the parent chain or directly from the full tree, with both
// oracles.
#include <cstdio>
#include <cstdlib>
#include <random>

#include "lpm/synthetic.h"
#include "lpm/tree.h"

using namespace lpm;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; if (failures < 30) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

static void run(const Scene& s, int max_depth, int ncells, unsigned seed, double margin) {
    std::mt19937_64 rng(seed);
    auto uni = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
    const std::vector<TreeEntry> full = full_tree(s);
    const TreeView tfull{full.data(), (uint32_t)full.size()};
    Scratch sc;
    Counters c;
    uint64_t sum_sizes = 0, sum_cells = 0;
    for (int it = 0; it < ncells; it++) {
        const int depth = 1 + (int)(uni(0, 1) * max_depth);
        const uint64_t N = 1ull << depth;
        uint64_t idx[3];
        for (int a = 0; a < 3; a++) idx[a] = (uint64_t)(uni(0, 1) * N);
        // Chain of pruned trees from the root (Lipschitz and interval), plus direct pruning.
        std::vector<TreeEntry> chainL = full, chainI = full, direct, directI;
        for (int d = 0; d <= depth; d++) {
            double size[3], c0[3], lo[3], hi[3], R = 0.0;
            for (int a = 0; a < 3; a++) {
                size[a] = (s.aabb_max[a] - s.aabb_min[a]) / (double)(1ull << d);
                lo[a] = s.aabb_min[a] + (idx[a] >> (depth - d)) * size[a];
                hi[a] = lo[a] + size[a];
                c0[a] = 0.5 * (lo[a] + hi[a]);
                R += 0.25 * size[a] * size[a];
            }
            R = std::sqrt(R);
            Interval box[3];
            for (int a = 0; a < 3; a++) box[a] = Interval(lo[a] - margin * 0.5 * size[a], hi[a] + margin * 0.5 * size[a]);
            std::vector<TreeEntry> out, outI;
            uint32_t n = 0, nI = 0;
            TreeView inL{chainL.data(), (uint32_t)chainL.size()};
            TreeView inI{chainI.data(), (uint32_t)chainI.size()};
            prune_lipschitz(s, inL, c0, R * (1.0 + margin), out, n, sc, c);
            prune_interval(s, inI, box, outI, nI, sc, c);
            chainL = out;
            chainI = outI;
            if (d == depth) {
                uint32_t nd = 0;
                prune_lipschitz(s, tfull, c0, R * (1.0 + margin), direct, nd, sc, c);
                prune_interval(s, tfull, box, directI, nd, sc, c);
                const TreeView tL{chainL.data(), (uint32_t)chainL.size()};
                const TreeView tI{chainI.data(), (uint32_t)chainI.size()};
                const TreeView tD{direct.data(), (uint32_t)direct.size()};
                const TreeView tDI{directI.data(), (uint32_t)directI.size()};
                const Interval F = eval_tree_interval(s, tfull, box, sc, c);
                sum_sizes += chainL.size();
                sum_cells++;
                for (int k = 0; k < 64; k++) {
                    // Uniform point in the ball of radius (1 + margin) R, intersected with the box
                    // enlarged by the margin for the interval oracle.
                    double p[3];
                    if (k < 32) {
                        double v[3], n2 = 0.0;
                        for (int a = 0; a < 3; a++) { v[a] = std::normal_distribution<double>(0, 1)(rng); n2 += v[a] * v[a]; }
                        const double r = R * (1.0 + margin) * std::cbrt(uni(0, 1)) / std::sqrt(n2);
                        for (int a = 0; a < 3; a++) p[a] = c0[a] + r * v[a];
                    } else {
                        for (int a = 0; a < 3; a++) p[a] = uni(lo[a], hi[a]);
                    }
                    const double f = eval_tree(s, tfull, p, sc, c);
                    const double fL = eval_tree(s, tL, p, sc, c);
                    const double fD = eval_tree(s, tD, p, sc, c);
                    CHECK(f == fL, "lipschitz chain: %.17g != %.17g (depth %d, tree %zu)", f, fL, depth, chainL.size());
                    CHECK(f == fD, "lipschitz direct: %.17g != %.17g", f, fD);
                    bool inside_box = true;
                    for (int a = 0; a < 3; a++) inside_box = inside_box && box[a].contains(p[a]);
                    if (inside_box) {
                        const double fI = eval_tree(s, tI, p, sc, c);
                        const double fDI = eval_tree(s, tDI, p, sc, c);
                        CHECK(f == fI, "interval chain: %.17g != %.17g", f, fI);
                        CHECK(f == fDI, "interval direct: %.17g != %.17g", f, fDI);
                        CHECK(F.contains(f), "root interval misses the value");
                    }
                }
            }
        }
    }
    printf("  %llu cells, average pruned size %.2f of %zu nodes\n", (unsigned long long)sum_cells,
           (double)sum_sizes / (double)sum_cells, s.nodes.size());
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        Scene s = load_scene_json(argv[i]);
        printf("%s\n", argv[i]);
        run(s, 8, 150, 5, 1e-3);
    }
    for (const char* shapes : {"sphere", "mixed"}) {
        for (const char* ops : {"union", "mixed"}) {
            for (double k : {0.0, 0.02}) {
                for (bool chain : {false, true}) {
                    SyntheticParams p;
                    p.n = 200; p.shapes = shapes; p.ops = ops; p.k = k; p.seed = 3; p.chain = chain; p.radius = 0.08; p.extent = 0.8;
                    Scene s = make_synthetic(p);
                    printf("synthetic %s/%s k=%g chain=%d\n", shapes, ops, k, (int)chain);
                    run(s, 8, 200, 9, 1e-3);
                }
            }
        }
    }
    printf("%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
