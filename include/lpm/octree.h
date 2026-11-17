// Adaptive octree over the scene bounding box. A cell is subdivided while it may contain
// the surface, which the 1-Lipschitz property decides with one evaluation at the center:
// |f(c)| > R, with R the radius of the bounding sphere of the cell, proves the cell empty.
// Each cell stores the tree pruned from the tree of its parent (or from the full tree).
#pragma once
#include <omp.h>

#include <algorithm>
#include <parallel/algorithm>
#include <cstdint>
#include <vector>

#include "lpm/tree.h"

namespace lpm {

enum Oracle { ORACLE_FULL = 0, ORACLE_LIPSCHITZ = 1, ORACLE_INTERVAL = 2, ORACLE_HYBRID = 3 };  // hybrid: Lipschitz pruning, interval culling

struct OctreeParams {
    Oracle oracle = ORACLE_LIPSCHITZ;
    int max_depth = 7;
    // Cells with |f(c)| > cull_factor * R are leaves. With 1.0 they are discarded (they
    // contain no surface); with a larger value they are kept as "far" leaves holding the
    // conservative constant sign(d) (|d| - R) of Barbier et al., for dense sampling.
    double cull_factor = 1.0;
    // The tree of a cell is pruned for the ball of radius (1 + margin) R, so that finite
    // differences slightly outside the cell remain exact.
    double margin = 1e-3;
    bool inherit = true;  // prune from the parent tree (true) or from the full tree (false)
    int threads = 0;      // 0 = OpenMP default
    // Keep the cells proven empty as leaves, with their pruned tree, instead of discarding
    // them: the dynamic update descends into them when a moving primitive enters.
    bool keep_empty_leaves = false;
};

struct Cell {
    uint32_t x, y, z;
    uint32_t tree_off = 0, tree_n = 0;
    double value = 0.0;  // f(center), or the constant of a far leaf
    uint8_t far = 0;
};

struct Level {
    std::vector<Cell> cells;
    std::vector<TreeEntry> pool;
    double size[3];
    // statistics
    uint64_t sum_tree_n = 0, max_tree_n = 0, single_prim = 0, far_cells = 0, tested = 0;
    // Number of cells of a GPU build, which keeps `cells` for the last level only (and of a dense
    // octree only the near ones); 0 in a CPU build, where `cells` holds them all.
    uint64_t num_cells = 0;
    double seconds = 0.0;
};

struct Octree {
    const Scene* scene = nullptr;
    OctreeParams params;
    std::vector<Level> levels;
    std::vector<TreeEntry> full;
    double build_seconds = 0.0;

    TreeView tree_of(int level, const Cell& c) const {
        if (params.oracle == ORACLE_FULL) return TreeView{full.data(), (uint32_t)full.size()};
        return TreeView{levels[level].pool.data() + c.tree_off, c.tree_n};
    }
    void cell_bounds(int level, const Cell& c, double lo[3], double hi[3]) const {
        const Level& L = levels[level];
        const uint32_t idx[3] = {c.x, c.y, c.z};
        for (int a = 0; a < 3; a++) {
            lo[a] = scene->aabb_min[a] + idx[a] * L.size[a];
            hi[a] = lo[a] + L.size[a];
        }
    }
    uint64_t pool_entries() const {
        uint64_t n = 0;
        for (const Level& L : levels) n += L.pool.size();
        return n;
    }
};

// cull_factor of the octree of dense sampling. Far leaves hold a constant, and marching cubes
// interpolates between the two ends of a crossing edge, which lie within one cell edge of the
// surface: those vertices keep their exact value when the far threshold is R + (longest edge),
// that is 1 + h_max / R, the same at every level.
inline double dense_cull_factor(const Scene& scene) {
    double R2 = 0.0, hmax = 0.0;
    for (int a = 0; a < 3; a++) {
        const double size = scene.aabb_max[a] - scene.aabb_min[a];
        R2 += 0.25 * size * size;
        hmax = std::max(hmax, size);
    }
    return 1.0 + hmax / std::sqrt(R2) + 1e-6;
}

inline uint64_t cell_key(uint32_t x, uint32_t y, uint32_t z) {
    return ((uint64_t)x << 42) | ((uint64_t)y << 21) | (uint64_t)z;
}

// Deepest octree the keys allow: vertex_key packs 20 bits per coordinate, and the edge keys of the
// dual contouring, 3 (2^depth + 1)^3, fit in 64 bits up to that depth too.
constexpr int kMaxDepth = 20;

inline Octree build_octree(const Scene& scene, const OctreeParams& params, Counters& counters) {
    Timer total;
    Octree oct;
    oct.scene = &scene;
    oct.params = params;
    oct.full = full_tree(scene);
    if (params.threads > 0) omp_set_num_threads(params.threads);
    const int nthreads = omp_get_max_threads();

    // Level 0: the bounding box, pruned from the full tree.
    {
        Level L0;
        for (int a = 0; a < 3; a++) L0.size[a] = scene.aabb_max[a] - scene.aabb_min[a];
        Cell root{0, 0, 0};
        double c[3], R = 0.0;
        for (int a = 0; a < 3; a++) { c[a] = scene.aabb_min[a] + 0.5 * L0.size[a]; R += 0.25 * L0.size[a] * L0.size[a]; }
        R = std::sqrt(R);
        Scratch sc;
        TreeView in{oct.full.data(), (uint32_t)oct.full.size()};
        if (params.oracle == ORACLE_LIPSCHITZ || params.oracle == ORACLE_HYBRID) {
            root.value = prune_lipschitz(scene, in, c, R * (1.0 + params.margin), L0.pool, root.tree_n, sc, counters);
        } else if (params.oracle == ORACLE_INTERVAL) {
            Interval box[3];
            for (int a = 0; a < 3; a++) {
                const double m = params.margin * 0.5 * L0.size[a];
                box[a] = Interval(scene.aabb_min[a] - m, scene.aabb_max[a] + m);
            }
            Interval F = prune_interval(scene, in, box, L0.pool, root.tree_n, sc, counters);
            root.value = 0.5 * (F.lo + F.hi);
        } else {
            root.value = eval_tree(scene, in, c, sc, counters);
            root.tree_n = (uint32_t)oct.full.size();
        }
        L0.tested = 1;
        L0.sum_tree_n = root.tree_n;
        L0.max_tree_n = root.tree_n;
        L0.single_prim = root.tree_n == 1;
        L0.cells.push_back(root);
        oct.levels.push_back(std::move(L0));
    }

    for (int depth = 1; depth <= params.max_depth; depth++) {
        Timer lt;
        const Level& parent = oct.levels[depth - 1];
        Level L;
        for (int a = 0; a < 3; a++) L.size[a] = parent.size[a] * 0.5;
        const double R = 0.5 * std::sqrt(L.size[0] * L.size[0] + L.size[1] * L.size[1] + L.size[2] * L.size[2]);
        const double R_prune = R * (1.0 + params.margin);
        std::vector<std::vector<Cell>> cells_t(nthreads);
        std::vector<std::vector<TreeEntry>> pool_t(nthreads);
        std::vector<Counters> cnt_t(nthreads);
        std::vector<uint64_t> tested_t(nthreads, 0);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            Scratch sc;
            std::vector<Cell>& cells = cells_t[tid];
            std::vector<TreeEntry>& pool = pool_t[tid];
            Counters& cnt = cnt_t[tid];
            // One task per child, in chunks that give every thread work even on the first levels,
            // where few parents carry large trees.
            const size_t tasks = 8 * parent.cells.size();
            const int chunk = (int)std::min<size_t>(16, std::max<size_t>(1, tasks / (64 * (size_t)nthreads)));
#pragma omp for schedule(dynamic, chunk)
            for (size_t task = 0; task < tasks; task++) {
                const Cell& pc = parent.cells[task >> 3];
                if (pc.far) continue;
                TreeView in = params.inherit ? oct.tree_of(depth - 1, pc)
                                             : TreeView{oct.full.data(), (uint32_t)oct.full.size()};
                {
                    const int child = (int)(task & 7);
                    Cell cc;
                    cc.x = 2 * pc.x + (child & 1);
                    cc.y = 2 * pc.y + ((child >> 1) & 1);
                    cc.z = 2 * pc.z + ((child >> 2) & 1);
                    const uint32_t idx[3] = {cc.x, cc.y, cc.z};
                    double c[3], lo[3], hi[3];
                    for (int a = 0; a < 3; a++) {
                        lo[a] = scene.aabb_min[a] + idx[a] * L.size[a];
                        hi[a] = lo[a] + L.size[a];
                        c[a] = 0.5 * (lo[a] + hi[a]);
                    }
                    tested_t[tid]++;
                    const size_t pool_base = pool.size();
                    bool keep = false, far = false;
                    double d = 0.0;
                    if (params.oracle == ORACLE_LIPSCHITZ || params.oracle == ORACLE_HYBRID) {
                        d = prune_lipschitz(scene, in, c, R_prune, pool, cc.tree_n, sc, cnt);
                        const double ad = std::fabs(d);
                        if (ad <= R * (1.0 + 1e-9)) keep = true;
                        if (keep && params.oracle == ORACLE_HYBRID && params.cull_factor <= 1.0) {
                            // The box bound of the (small) pruned tree can prove the cell empty where
                            // the ball test of the center cannot.
                            Interval box[3];
                            for (int a = 0; a < 3; a++) box[a] = Interval(lo[a], hi[a]);
                            const Interval F = eval_tree_interval(scene, TreeView{pool.data() + pool_base, cc.tree_n}, box, sc, cnt);
                            if (!F.contains_zero()) keep = false;
                        }
                        else if (ad <= params.cull_factor * R) keep = true;   // near, no surface: still refined in dense mode
                        else if (params.cull_factor > 1.0) { far = true; cc.value = (d > 0 ? 1.0 : -1.0) * (ad - R); }
                        else if (params.keep_empty_leaves) { far = true; cc.value = d; }
                        if (!far || params.keep_empty_leaves) cc.value = d;
                    } else if (params.oracle == ORACLE_INTERVAL) {
                        Interval box[3];
                        for (int a = 0; a < 3; a++) {
                            const double m = params.margin * 0.5 * L.size[a];
                            box[a] = Interval(lo[a] - m, hi[a] + m);
                        }
                        Interval F = prune_interval(scene, in, box, pool, cc.tree_n, sc, cnt);
                        cc.value = 0.5 * (F.lo + F.hi);
                        if (F.contains_zero()) keep = true;
                        else if (params.cull_factor > 1.0) { far = true; cc.value = F.lo > 0.0 ? F.lo : F.hi; }
                    } else {
                        d = eval_tree(scene, in, c, sc, cnt);
                        cc.tree_n = (uint32_t)oct.full.size();
                        const double ad = std::fabs(d);
                        if (ad <= R * (1.0 + 1e-9)) keep = true;
                        else if (ad <= params.cull_factor * R) keep = true;
                        else if (params.cull_factor > 1.0) { far = true; cc.value = (d > 0 ? 1.0 : -1.0) * (ad - R); }
                        if (!far) cc.value = d;
                    }
                    if (far) {
                        cc.far = 1;
                        if (params.keep_empty_leaves && params.cull_factor <= 1.0) {
                            cc.tree_off = (uint32_t)pool_base;  // the tree of an empty leaf is kept
                        } else {
                            pool.resize(pool_base);
                            cc.tree_n = 0;
                            cc.tree_off = 0;
                        }
                        cells.push_back(cc);
                    } else if (keep) {
                        cc.tree_off = (uint32_t)pool_base;
                        cells.push_back(cc);
                    } else {
                        pool.resize(pool_base);
                    }
                }
            }
        }
        // Merge the per-thread outputs (each thread copies its own, at offsets of the ones before)
        // and sort the cells, in parallel, to make the result deterministic.
        std::vector<size_t> coff(nthreads + 1, 0), poff(nthreads + 1, 0);
        for (int t = 0; t < nthreads; t++) {
            coff[t + 1] = coff[t] + cells_t[t].size();
            poff[t + 1] = poff[t] + pool_t[t].size();
            counters.add(cnt_t[t]);
            L.tested += tested_t[t];
        }
        L.cells.resize(coff[nthreads]);
        L.pool.resize(poff[nthreads]);
#pragma omp parallel for schedule(static, 1)
        for (int t = 0; t < nthreads; t++) {
            const uint32_t off = (uint32_t)poff[t];
            for (size_t k = 0; k < cells_t[t].size(); k++) {
                Cell c = cells_t[t][k];
                if (!c.far || c.tree_n > 0) c.tree_off += off;
                L.cells[coff[t] + k] = c;
            }
            std::copy(pool_t[t].begin(), pool_t[t].end(), L.pool.begin() + poff[t]);
        }
        __gnu_parallel::sort(L.cells.begin(), L.cells.end(), [](const Cell& a, const Cell& b) {
            return cell_key(a.x, a.y, a.z) < cell_key(b.x, b.y, b.z);
        });
        uint64_t far_cells = 0, sum_tree_n = 0, max_tree_n = 0, single_prim = 0;
#pragma omp parallel for schedule(static) reduction(+ : far_cells, sum_tree_n, single_prim) reduction(max : max_tree_n)
        for (size_t i = 0; i < L.cells.size(); i++) {
            const Cell& c = L.cells[i];
            if (c.far) { far_cells++; continue; }
            sum_tree_n += c.tree_n;
            max_tree_n = std::max<uint64_t>(max_tree_n, c.tree_n);
            if (c.tree_n == 1) single_prim++;
        }
        L.far_cells = far_cells;
        L.sum_tree_n = sum_tree_n;
        L.max_tree_n = max_tree_n;
        L.single_prim = single_prim;
        L.seconds = lt.seconds();
        oct.levels.push_back(std::move(L));
    }
    oct.build_seconds = total.seconds();
    return oct;
}

}  // namespace lpm
