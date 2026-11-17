// Dense sampling of the SDF on the (N+1)^3 vertices of the finest grid, either with the
// full tree (naive discretization) or through the octree built in dense mode, where every
// vertex is evaluated with the pruned tree of the finest cell that contains it.
#pragma once
#include <omp.h>

#include <cstdio>
#include <unordered_map>
#include <vector>

#include "lpm/octree.h"

namespace lpm {

inline std::vector<float> dense_sample_full(const Scene& scene, int depth, Counters& counters) {
    const uint64_t N = 1ull << depth, N1 = N + 1;
    std::vector<float> out(N1 * N1 * N1);
    const std::vector<TreeEntry> full = full_tree(scene);
    const TreeView t{full.data(), (uint32_t)full.size()};
    double size[3];
    for (int a = 0; a < 3; a++) size[a] = (scene.aabb_max[a] - scene.aabb_min[a]) / (double)N;
    const int nthreads = omp_get_max_threads();
    std::vector<Counters> cnt_t(nthreads);
#pragma omp parallel
    {
        Scratch sc;
        Counters& cnt = cnt_t[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 1)
        for (uint64_t i = 0; i < N1; i++) {
            for (uint64_t j = 0; j < N1; j++) {
                for (uint64_t k = 0; k < N1; k++) {
                    const double p[3] = {scene.aabb_min[0] + i * size[0], scene.aabb_min[1] + j * size[1], scene.aabb_min[2] + k * size[2]};
                    out[(i * N1 + j) * N1 + k] = (float)eval_tree(scene, t, p, sc, cnt);
                }
            }
        }
    }
    for (int t = 0; t < nthreads; t++) counters.add(cnt_t[t]);
    return out;
}

// Requires an octree built with cull_factor >= 1 + h_max / R (2/sqrt(3) for cubic cells) so
// that every vertex within one cell edge of the surface is evaluated exactly.
inline std::vector<float> dense_sample_octree(const Octree& oct, Counters& counters) {
    const Scene& scene = *oct.scene;
    const int depth = (int)oct.levels.size() - 1;
    const uint64_t N = 1ull << depth, N1 = N + 1;
    std::vector<std::unordered_map<uint64_t, uint32_t>> maps(depth + 1);
    for (int l = 0; l <= depth; l++) {
        maps[l].reserve(oct.levels[l].cells.size() * 2);
        for (uint32_t i = 0; i < oct.levels[l].cells.size(); i++) {
            const Cell& c = oct.levels[l].cells[i];
            maps[l].emplace(cell_key(c.x, c.y, c.z), i);
        }
    }
    std::vector<float> out(N1 * N1 * N1);
    double size[3];
    for (int a = 0; a < 3; a++) size[a] = oct.levels[depth].size[a];
    const int nthreads = omp_get_max_threads();
    std::vector<Counters> cnt_t(nthreads);
#pragma omp parallel
    {
        Scratch sc;
        Counters& cnt = cnt_t[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 1)
        for (uint64_t i = 0; i < N1; i++) {
            for (uint64_t j = 0; j < N1; j++) {
                for (uint64_t k = 0; k < N1; k++) {
                    const double p[3] = {scene.aabb_min[0] + i * size[0], scene.aabb_min[1] + j * size[1], scene.aabb_min[2] + k * size[2]};
                    float v = 0.0f;
                    for (int l = depth; l >= 0; l--) {
                        const uint64_t M = 1ull << l;
                        const uint32_t cx = (uint32_t)std::min(i >> (depth - l), M - 1);
                        const uint32_t cy = (uint32_t)std::min(j >> (depth - l), M - 1);
                        const uint32_t cz = (uint32_t)std::min(k >> (depth - l), M - 1);
                        auto it = maps[l].find(cell_key(cx, cy, cz));
                        if (it == maps[l].end()) continue;
                        const Cell& c = oct.levels[l].cells[it->second];
                        if (c.far) v = (float)c.value;
                        else v = (float)eval_tree(scene, oct.tree_of(l, c), p, sc, cnt);
                        break;
                    }
                    out[(i * N1 + j) * N1 + k] = v;
                }
            }
        }
    }
    for (int t = 0; t < nthreads; t++) counters.add(cnt_t[t]);
    return out;
}

// Returns false when the file cannot be opened or written in full.
[[nodiscard]] inline bool write_f32(const std::vector<float>& v, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    return fclose(f) == 0 && ok;
}

}  // namespace lpm
