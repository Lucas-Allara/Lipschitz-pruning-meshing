// Compares the GPU octree build (include/lpm/gpu.h) with the CPU one: same leaves, same pruned
// trees, and with --mesh the same mesh after the CPU dual contouring. With --gpu-mesh it also runs
// the dual contouring on the GPU and compares that mesh with the CPU one (same topology, distance
// of the vertices) and, with --full, with the GPU mesh of the full tree, which must be identical.
// With --interval both builds use the interval pruning instead of the Lipschitz one. With --coarse
// it compares the GPU mesh with coarse leaves against adaptive_mesh with top_down on the CPU: the
// same coarse leaves per level, the same vertices (up to the solver of the quadric), the same
// triangles, and a closed mesh. With --dense both builds are the octree of dense sampling
// (cull_factor > 1, lpm_mesh --dense-trees), compared level by level. The vertices of the two
// meshes differ by rounding and by where the root search of each crossing stops, since the normal
// there takes f = 0; the projection onto the surface, which would amplify both near creases, is off
// in the comparison. More than --max-vertex-distance (in cells) is a failure.
// Prints the times of everything.
//
//   lpm_gpu --scene S.json [--depth 8] [--repeat 5] [--threads T] [--mesh] [--gpu-mesh] [--full] [--interval] [--dense]
//           [--coarse [--adaptive-tol T] [--adaptive-min-depth L]] [--max-vertex-distance D] [--stats out.json]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <array>
#include <map>

#include "lpm/adaptive.h"
#include "lpm/dc.h"
#include "lpm/gpu.h"
#include "lpm/threads.h"

using namespace lpm;

static void usage() {
    fprintf(stderr, "usage: lpm_gpu --scene S.json [--depth D] [--repeat N] [--threads T] [--mesh] [--gpu-mesh] [--full] [--interval] [--dense]\n"
                    "               [--coarse [--adaptive-tol T] [--adaptive-min-depth L]] [--max-vertex-distance D] [--stats out.json]\n");
    exit(2);
}

// Leaves of the last level with their trees, sorted by cell, to compare two builds. The far leaves
// of a dense octree hold no tree, and the GPU build does not keep them.
struct LeafTree {
    uint64_t key;
    std::vector<TreeEntry> tree;
};

static std::vector<LeafTree> leaves_of(const Octree& oct) {
    const int depth = (int)oct.levels.size() - 1;
    const Level& L = oct.levels[depth];
    std::vector<LeafTree> v;
    v.reserve(L.cells.size());
    for (const Cell& c : L.cells) {
        if (c.far) continue;
        TreeView t = oct.tree_of(depth, c);
        v.push_back(LeafTree{cell_key(c.x, c.y, c.z), std::vector<TreeEntry>(t.e, t.e + t.n)});
    }
    std::sort(v.begin(), v.end(), [](const LeafTree& a, const LeafTree& b) { return a.key < b.key; });
    return v;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// Triangles with their smallest index first (the orientation kept), sorted.
static std::vector<std::array<uint32_t, 3>> triangle_set(const Mesh& m) {
    std::vector<std::array<uint32_t, 3>> t(m.F.size() / 3);
    for (size_t i = 0; i < t.size(); i++) {
        const uint32_t* f = &m.F[3 * i];
        const int k = f[0] <= f[1] && f[0] <= f[2] ? 0 : f[1] <= f[2] ? 1 : 2;
        t[i] = {f[k], f[(k + 1) % 3], f[(k + 2) % 3]};
    }
    std::sort(t.begin(), t.end());
    return t;
}

// Half-edges without their opposite, and edges with more than two triangles.
static void closedness(const Mesh& m, uint64_t& boundary, uint64_t& nonmanifold) {
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (size_t i = 0; i < m.F.size(); i += 3)
        for (int k = 0; k < 3; k++) edges[{m.F[i + k], m.F[i + (k + 1) % 3]}]++;
    boundary = nonmanifold = 0;
    for (const auto& [e, n] : edges) {
        auto it = edges.find({e.second, e.first});
        if (n != (it == edges.end() ? 0 : it->second)) boundary++;
        if (n > 1 && e.first < e.second) nonmanifold++;
    }
}

static int compare_coarse(GpuPruner& gpu, const Scene& scene, const std::string& scene_path, const OctreeParams& op, double tol,
                          int min_depth, int repeat, double max_distance, const std::string& stats_path) {
    const int depth = op.max_depth;
    DCParams dp;
    CoarseParams cp;
    cp.tol = tol;
    cp.min_depth = min_depth;
    AdaptiveParams ap;
    ap.max_depth = depth;
    ap.tol = tol;
    ap.min_depth = min_depth;
    ap.top_down = true;
    ap.single_primitive_only = true;
    ap.margin = op.margin;
    Mesh gm, um, cm;
    GpuMeshStats gms, ums;
    AdaptiveStats ast;
    std::vector<double> g_t, u_t, c_t;
    // The first run of each kind is not timed.
    for (int r = 0; r <= repeat; r++) {
        Counters c1, c2;
        GpuStats s;
        gm = gpu.mesh(scene, op, dp, c1, c2, s, gms, nullptr, &cp);
        if (r > 0) g_t.push_back(gms.t_total);
    }
    for (int r = 0; r <= repeat; r++) {
        Counters c1, c2;
        GpuStats s;
        um = gpu.mesh(scene, op, dp, c1, c2, s, ums);
        if (r > 0) u_t.push_back(ums.t_total);
    }
    for (int r = 0; r <= repeat; r++) {
        Counters c;
        cm = adaptive_mesh(scene, ap, dp, c, ast);
        if (r > 0) c_t.push_back(ast.total);
    }
    // Coarse leaves per level.
    uint64_t level_diff = 0;
    for (int l = 0; l < depth; l++) {
        const uint64_t g = l < (int)gms.coarse_per_level.size() ? gms.coarse_per_level[l] : 0;
        const uint64_t c = l < (int)ast.leaves_per_level.size() ? ast.leaves_per_level[l] : 0;
        level_diff += g > c ? g - c : c - g;
    }
    // Vertices (the same numbering when the leaves agree) and triangles.
    const bool same_vertices = gm.V.size() == cm.V.size();
    double max_dv = 0.0;
    if (same_vertices)
        for (size_t k = 0; k < gm.V.size(); k += 3) {
            double d2 = 0.0;
            for (int a = 0; a < 3; a++) d2 += (gm.V[k + a] - cm.V[k + a]) * (gm.V[k + a] - cm.V[k + a]);
            max_dv = std::max(max_dv, std::sqrt(d2));
        }
    const auto gt = triangle_set(gm), ct = triangle_set(cm);
    std::vector<std::array<uint32_t, 3>> only_g, only_c;
    std::set_difference(gt.begin(), gt.end(), ct.begin(), ct.end(), std::back_inserter(only_g));
    std::set_difference(ct.begin(), ct.end(), gt.begin(), gt.end(), std::back_inserter(only_c));
    uint64_t boundary, nonmanifold;
    closedness(gm, boundary, nonmanifold);
    double cell = scene.aabb_max[0] - scene.aabb_min[0];
    for (int a = 1; a < 3; a++) cell = std::min(cell, scene.aabb_max[a] - scene.aabb_min[a]);
    cell = std::ldexp(cell, -depth);

    FILE* f = stats_path.empty() ? stdout : fopen(stats_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", stats_path.c_str());
        return 1;
    }
    fprintf(f, "{\n  \"scene\": \"%s\", \"depth\": %d, \"tol\": %g, \"min_depth\": %d, \"repeat\": %d,\n", scene_path.c_str(), depth, tol,
            min_depth, repeat);
    fprintf(f, "  \"gpu_coarse_total\": %.6f, \"gpu_uniform_total\": %.6f, \"cpu_coarse_total\": %.6f,\n", median(g_t), median(u_t),
            median(c_t));
    fprintf(f, "  \"gpu_coarse_triangles\": %zu, \"gpu_uniform_triangles\": %zu, \"cpu_coarse_triangles\": %zu,\n", gm.F.size() / 3,
            um.F.size() / 3, cm.F.size() / 3);
    fprintf(f, "  \"coarse_per_level_gpu\": [");
    for (int l = 0; l < depth; l++) fprintf(f, "%s%llu", l ? ", " : "", (unsigned long long)(l < (int)gms.coarse_per_level.size() ? gms.coarse_per_level[l] : 0));
    fprintf(f, "], \"coarse_per_level_cpu\": [");
    for (int l = 0; l < depth; l++) fprintf(f, "%s%llu", l ? ", " : "", (unsigned long long)(l < (int)ast.leaves_per_level.size() ? ast.leaves_per_level[l] : 0));
    fprintf(f, "],\n  \"fine_leaves_gpu\": %llu, \"fine_leaves_cpu\": %llu, \"vertices_gpu\": %zu, \"vertices_cpu\": %zu,\n",
            (unsigned long long)gms.leaves, (unsigned long long)ast.fine_leaves, gm.V.size() / 3, cm.V.size() / 3);
    fprintf(f, "  \"max_vertex_distance_cells\": %.3g, \"triangles_only_gpu\": %zu, \"triangles_only_cpu\": %zu,\n", max_dv / cell,
            only_g.size(), only_c.size());
    fprintf(f, "  \"boundary_half_edges\": %llu, \"nonmanifold_edges\": %llu\n}\n", (unsigned long long)boundary,
            (unsigned long long)nonmanifold);
    if (f != stdout) fclose(f);
    const bool near = max_dv / cell <= max_distance;
    if (!near) fprintf(stderr, "vertices up to %.3g cells apart (--max-vertex-distance %g)\n", max_dv / cell, max_distance);
    const bool ok = level_diff == 0 && same_vertices && only_g.empty() && only_c.empty() && boundary == 0 && near;
    return ok ? 0 : 1;
}

static int run(int argc, char** argv) {
    std::string scene_path, stats_path;
    int depth = 8, repeat = 5, threads = 0;
    bool mesh = false, gpu_mesh = false, full = false, interval = false, coarse = false, dense = false;
    // The GPU and the CPU round differently (Jacobi and Eigen, contraction of products), and each
    // crossing takes f = 0 where its root search stops: up to 6e-5 cells on the scenes of the
    // repository (Monument at depth 7). A real difference is orders of magnitude larger.
    double coarse_tol = 0.02, max_distance = 1e-3;
    int coarse_min_depth = 3;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--scene") scene_path = next();
        else if (a == "--depth") depth = atoi(next());
        else if (a == "--repeat") repeat = std::max(1, atoi(next()));
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--mesh") mesh = true;
        else if (a == "--gpu-mesh") gpu_mesh = true;
        else if (a == "--full") full = true;
        else if (a == "--interval") interval = true;
        else if (a == "--coarse") coarse = true;
        else if (a == "--dense") dense = true;
        else if (a == "--max-vertex-distance") max_distance = atof(next());
        else if (a == "--adaptive-tol") coarse_tol = atof(next());
        else if (a == "--adaptive-min-depth") coarse_min_depth = atoi(next());
        else if (a == "--stats") stats_path = next();
        else usage();
    }
    if (scene_path.empty()) usage();
    if (depth < 1 || depth > kMaxDepth) {
        fprintf(stderr, "--depth must be between 1 and %d\n", kMaxDepth);
        return 2;
    }
    if (dense && (interval || mesh || gpu_mesh || full || coarse)) {
        fprintf(stderr, "--dense compares the octrees of the Lipschitz pruning only\n");
        return 2;
    }
    Scene scene = load_scene_json(scene_path);
    OctreeParams op;
    op.oracle = interval ? ORACLE_INTERVAL : ORACLE_LIPSCHITZ;
    op.max_depth = depth;
    op.threads = threads;
    if (dense) op.cull_factor = dense_cull_factor(scene);

    GpuPruner gpu;
    if (coarse) return compare_coarse(gpu, scene, scene_path, op, coarse_tol, coarse_min_depth, repeat, max_distance, stats_path);
    std::vector<double> g_total, g_levels, g_gpu, g_down, c_total;
    std::vector<std::vector<double>> g_level_gpu;
    Octree goct, coct;
    Counters gcnt, ccnt;
    GpuStats gs;
    // The first build of each kind warms up the caches and the pipeline; it is not timed.
    for (int r = 0; r <= repeat; r++) {
        Counters c;
        goct = gpu.build(scene, op, c, gs);
        if (r == 0) { gcnt = c; continue; }
        g_total.push_back(gs.t_total); g_levels.push_back(gs.t_levels); g_gpu.push_back(gs.t_gpu); g_down.push_back(gs.t_download);
        g_level_gpu.push_back(gs.level_gpu);
    }
    for (int r = 0; r <= repeat; r++) {
        Counters c;
        coct = build_octree(scene, op, c);
        if (r == 0) { ccnt = c; continue; }
        c_total.push_back(coct.build_seconds);
    }

    // Same leaves and the same pruned trees, entry by entry.
    const std::vector<LeafTree> gl = leaves_of(goct), cl = leaves_of(coct);
    uint64_t only_gpu = 0, only_cpu = 0, different = 0, same = 0;
    size_t i = 0, j = 0;
    while (i < gl.size() || j < cl.size()) {
        if (j >= cl.size() || (i < gl.size() && gl[i].key < cl[j].key)) { only_gpu++; i++; continue; }
        if (i >= gl.size() || cl[j].key < gl[i].key) { only_cpu++; j++; continue; }
        const auto& a = gl[i].tree;
        const auto& b = cl[j].tree;
        bool eq = a.size() == b.size();
        for (size_t k = 0; eq && k < a.size(); k++)
            eq = a[k].node == b[k].node && a[k].parent == b[k].parent && a[k].sign == b[k].sign;
        (eq ? same : different)++;
        i++; j++;
    }
    // And the statistics of every level, which is all the GPU build keeps of the levels above the
    // last: cells, far leaves among them, entries of the trees, largest tree, children tested and
    // single-primitive cells.
    uint64_t levels_different = goct.levels.size() != coct.levels.size();
    for (size_t l = 0; l < std::min(goct.levels.size(), coct.levels.size()); l++) {
        const Level& G = goct.levels[l];
        const Level& C = coct.levels[l];
        const uint64_t g_cells = G.num_cells ? G.num_cells : G.cells.size();
        if (g_cells != C.cells.size() || G.far_cells != C.far_cells || G.sum_tree_n != C.sum_tree_n || G.max_tree_n != C.max_tree_n ||
            G.tested != C.tested || G.single_prim != C.single_prim)
            levels_different++;
    }

    int mesh_identical = -1;
    uint64_t triangles = 0;
    double g_mesh_s = 0.0, c_mesh_s = 0.0;
    if (mesh) {
        DCParams dp;
        DCStats gst, cst;
        Counters c1, c2;
        Mesh gm = dual_contour(goct, dp, c1, gst);
        Mesh cm = dual_contour(coct, dp, c2, cst);
        g_mesh_s = gst.total;
        c_mesh_s = cst.total;
        triangles = cm.F.size() / 3;
        mesh_identical = gm.V.size() == cm.V.size() && gm.F == cm.F &&
                         memcmp(gm.V.data(), cm.V.data(), gm.V.size() * sizeof(double)) == 0;
    }

    // Dual contouring on the GPU against the CPU pipeline (CPU octree, CPU dual contouring).
    GpuMeshStats gms;
    Mesh gm, cpu_mesh;
    std::vector<double> gm_total, gm_octree, gm_index, gm_mesh, gm_down, cpu_pipeline;
    int topology_same = -1, full_identical = -1;
    double max_dv = 0.0;
    uint64_t moved = 0;
    if (gpu_mesh) {
        // Without the projection onto the surface: near creases its Newton steps turn rounding into
        // hundredths of a cell, so the comparison checks the pipeline up to the quadric vertices.
        DCParams dp;
        dp.project_vertices = false;
        for (int r = 0; r <= repeat; r++) {
            Counters c1, c2;
            GpuStats s2;
            gm = gpu.mesh(scene, op, dp, c1, c2, s2, gms);
            if (r == 0) continue;
            gm_total.push_back(gms.t_total); gm_octree.push_back(gms.t_octree); gm_index.push_back(gms.t_index);
            gm_mesh.push_back(gms.t_mesh); gm_down.push_back(gms.t_download);
        }
        for (int r = 0; r <= repeat; r++) {
            Counters c1, c2;
            DCStats dst;
            Timer t;
            Octree o = build_octree(scene, op, c1);
            cpu_mesh = dual_contour(o, dp, c2, dst);
            if (r > 0) cpu_pipeline.push_back(t.seconds());
        }
        topology_same = gm.V.size() == cpu_mesh.V.size() && gm.F == cpu_mesh.F;
        if (gm.V.size() == cpu_mesh.V.size())
            for (size_t k = 0; k < gm.V.size(); k += 3) {
                double d2 = 0.0;
                for (int a = 0; a < 3; a++) d2 += (gm.V[k + a] - cpu_mesh.V[k + a]) * (gm.V[k + a] - cpu_mesh.V[k + a]);
                const double d = std::sqrt(d2);
                if (d > 0.0) moved++;
                max_dv = std::max(max_dv, d);
            }
        if (full) {
            OctreeParams fp = op;
            fp.oracle = ORACLE_FULL;
            Counters c1, c2;
            GpuStats s2;
            GpuMeshStats fms;
            Mesh fm = gpu.mesh(scene, fp, dp, c1, c2, s2, fms);
            full_identical = fm.V.size() == gm.V.size() && fm.F == gm.F &&
                             memcmp(fm.V.data(), gm.V.data(), gm.V.size() * sizeof(double)) == 0;
            fprintf(stderr, "GPU full tree: %.3f s, %llu primitive evaluations\n", fms.t_total,
                    (unsigned long long)(c1.prim_evals + c2.prim_evals));
        }
    }

    FILE* f = stats_path.empty() ? stdout : fopen(stats_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", stats_path.c_str());
        return 1;
    }
    fprintf(f, "{\n  \"scene\": \"%s\", \"oracle\": \"%s\", \"dense\": %s, \"depth\": %d, \"nodes\": %zu, \"device\": \"%s\", \"cpu_threads\": %d, \"repeat\": %d,\n",
            scene_path.c_str(), interval ? "interval" : "lipschitz", dense ? "true" : "false", depth, scene.nodes.size(), gs.device.c_str(),
            omp_get_max_threads(), repeat);
    fprintf(f, "  \"gpu_init\": %.6f, \"gpu_setup\": %.6f,\n", gpu.init_seconds(), gs.t_setup);
    fprintf(f, "  \"gpu_total\": %.6f, \"gpu_levels\": %.6f, \"gpu_dispatch\": %.6f, \"gpu_download\": %.6f, \"cpu_total\": %.6f,\n",
            median(g_total), median(g_levels), median(g_gpu), median(g_down), median(c_total));
    fprintf(f, "  \"gpu_level_dispatch\": [");
    for (size_t l = 0; l < g_level_gpu.back().size(); l++) {
        std::vector<double> v;
        for (const auto& run : g_level_gpu) v.push_back(run[l]);
        fprintf(f, "%s%.6f", l ? ", " : "", median(v));
    }
    fprintf(f, "],\n  \"cpu_level_seconds\": [");
    for (size_t l = 0; l < coct.levels.size(); l++) fprintf(f, "%s%.6f", l ? ", " : "", coct.levels[l].seconds);
    fprintf(f, "],\n");
    fprintf(f, "  \"prim_evals_gpu\": %llu, \"prim_evals_cpu\": %llu,\n", (unsigned long long)(gcnt.prim_evals + gcnt.prim_ivals),
            (unsigned long long)(ccnt.prim_evals + ccnt.prim_ivals));
    fprintf(f, "  \"leaves_gpu\": %zu, \"leaves_cpu\": %zu, \"same_trees\": %llu, \"different_trees\": %llu, \"only_gpu\": %llu, \"only_cpu\": %llu, "
               "\"levels_different\": %llu",
            gl.size(), cl.size(), (unsigned long long)same, (unsigned long long)different, (unsigned long long)only_gpu,
            (unsigned long long)only_cpu, (unsigned long long)levels_different);
    if (mesh)
        fprintf(f, ",\n  \"triangles\": %llu, \"mesh_identical\": %s, \"dc_seconds_gpu_octree\": %.6f, \"dc_seconds_cpu_octree\": %.6f",
                (unsigned long long)triangles, mesh_identical ? "true" : "false", g_mesh_s, c_mesh_s);
    if (gpu_mesh) {
        const double cell = std::min(goct.levels.back().size[0], std::min(goct.levels.back().size[1], goct.levels.back().size[2]));
        fprintf(f, ",\n  \"gpu_mesh_total\": %.6f, \"gpu_mesh_octree\": %.6f, \"gpu_mesh_index\": %.6f, \"gpu_mesh_rest\": %.6f, \"gpu_mesh_unpack\": %.6f,",
                median(gm_total), median(gm_octree), median(gm_index), median(gm_mesh), median(gm_down));
        fprintf(f, "\n  \"cpu_pipeline\": %.6f, \"gpu_vertices\": %zu, \"cpu_vertices\": %zu, \"gpu_triangles\": %zu, \"cpu_triangles\": %zu,",
                median(cpu_pipeline), gm.V.size() / 3, cpu_mesh.V.size() / 3, gm.F.size() / 3, cpu_mesh.F.size() / 3);
        fprintf(f, "\n  \"same_topology\": %s, \"max_vertex_distance_cells\": %.3g, \"moved_vertices\": %llu, \"ambiguous_faces\": %llu",
                topology_same ? "true" : "false", max_dv / cell, (unsigned long long)moved, (unsigned long long)gms.ambiguous);
        if (full) fprintf(f, ",\n  \"gpu_full_identical\": %s", full_identical ? "true" : "false");
        fprintf(f, ",\n  \"gpu_mesh_stages\": {");
        for (size_t k = 0; k < gms.stage_gpu.size(); k++)
            fprintf(f, "%s\"%s\": %.6f", k ? ", " : "", gms.stage_gpu[k].first.c_str(), gms.stage_gpu[k].second);
        fprintf(f, "}");
    }
    fprintf(f, "\n}\n");
    if (f != stdout) fclose(f);
    bool near = true;
    if (gpu_mesh) {
        const double cell = std::min(goct.levels.back().size[0], std::min(goct.levels.back().size[1], goct.levels.back().size[2]));
        near = max_dv / cell <= max_distance;
        if (!near) fprintf(stderr, "vertices up to %.3g cells apart (--max-vertex-distance %g)\n", max_dv / cell, max_distance);
    }
    const bool ok = different == 0 && only_gpu == 0 && only_cpu == 0 && levels_different == 0 && mesh_identical != 0 && topology_same != 0 &&
                    full_identical != 0 && near;
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    lpm::passive_omp_wait(argv);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {  // a scene that cannot be read, no suitable GPU, or out of memory
        fprintf(stderr, "lpm_gpu: %s\n", e.what());
        return 1;
    }
}
