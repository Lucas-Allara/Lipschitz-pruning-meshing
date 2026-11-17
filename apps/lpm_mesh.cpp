// Meshes a smooth CSG scene: octree subdivision with one of the three oracles (full tree,
// Lipschitz pruning, interval pruning) and dual contouring of the leaves, or a dense
// sampling of the finest grid for marching cubes. With --gpu --repeat N the mesh is made N more
// times with the GPU already prepared and without copying it to the host, and time_resident is the
// median of those runs. --dense-trees builds only the octree of the dense sampling and reports its
// pruned trees (on the GPU with --gpu).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "lpm/adaptive.h"
#include "lpm/dc.h"
#include "lpm/dense.h"
#include "lpm/scene.h"
#include "lpm/synthetic.h"
#include "lpm/threads.h"
#ifdef LPM_HAVE_GPU
#include "lpm/gpu.h"
#endif

using namespace lpm;

static void usage() {
    fprintf(stderr,
            "usage: lpm_mesh --scene FILE --method full|lipschitz|interval|hybrid --depth D\n"
            "       [--threads T] [--out mesh.obj|mesh.ply] [--attr tree_n.txt] [--stats stats.json]\n"
            "       [--no-inherit] [--analytic] [--margin M] [--dense values.f32] [--dense-trees] [--gpu] [--repeat N] [--mc]\n"
            "       [--no-project]   (dual contouring without moving its vertices onto the surface, as before 2026-11-16)\n"
            "       [--adaptive] [--adaptive-tol T] [--adaptive-all] [--adaptive-min-depth L] [--adaptive-topdown]\n");
    exit(1);
}

static int run(int argc, char** argv) {
    std::string scene_path, method = "lipschitz", out_path, attr_path, stats_path, dense_path;
    int depth = 7, threads = 0, repeat = 0;
    bool inherit = true, analytic = false, adaptive = false, adaptive_all = false, adaptive_topdown = false, gpu = false;
    bool dense_trees = false, mc = false;  // mc: marching cubes on the leaves (GPU)
    bool project = true;                   // dual contouring moves its vertices onto the surface
    double margin = 1e-3, adaptive_tol = 0.02;
    int adaptive_min_depth = 3;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--scene") scene_path = next();
        else if (a == "--method") method = next();
        else if (a == "--depth") depth = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--out") out_path = next();
        else if (a == "--attr") attr_path = next();
        else if (a == "--stats") stats_path = next();
        else if (a == "--dense") dense_path = next();
        else if (a == "--dense-trees") dense_trees = true;
        else if (a == "--mc") mc = true;
        else if (a == "--no-project") project = false;
        else if (a == "--repeat") repeat = atoi(next().c_str());
        else if (a == "--no-inherit") inherit = false;
        else if (a == "--analytic") analytic = true;
        else if (a == "--gpu") gpu = true;  // octree and dual contouring on the GPU (full, lipschitz or interval, depth <= 9; coarse leaves with --adaptive-topdown)
        else if (a == "--margin") margin = atof(next().c_str());
        else if (a == "--adaptive") adaptive = true;
        else if (a == "--adaptive-all") adaptive_all = true;
        else if (a == "--adaptive-topdown") { adaptive = true; adaptive_topdown = true; }
        else if (a == "--adaptive-tol") adaptive_tol = atof(next().c_str());
        else if (a == "--adaptive-min-depth") adaptive_min_depth = atoi(next().c_str());
        else usage();
    }
    if (scene_path.empty()) usage();
    if (depth < 1 || depth > kMaxDepth) {
        fprintf(stderr, "--depth must be between 1 and %d\n", kMaxDepth);
        return 2;
    }
    if (threads > 0) omp_set_num_threads(threads);

    Timer t_all;
    Timer t_load;
    Scene scene = load_scene_json(scene_path);
    const double load_s = t_load.seconds();

    OctreeParams op;
    op.max_depth = depth;
    op.margin = margin;
    op.inherit = inherit;
    op.threads = threads;
    if (method == "full") op.oracle = ORACLE_FULL;
    else if (method == "lipschitz") op.oracle = ORACLE_LIPSCHITZ;
    else if (method == "interval") op.oracle = ORACLE_INTERVAL;
    else if (method == "hybrid") op.oracle = ORACLE_HYBRID;
    else usage();
    if (mc && (!gpu || adaptive || dense_trees || !dense_path.empty())) {
        fprintf(stderr, "--mc (marching cubes on the leaves) needs --gpu, without --adaptive or --dense\n");
        return 2;
    }
    if (repeat > 0 && (!gpu || dense_trees)) {
        fprintf(stderr, "--repeat times the GPU mesh only\n");
        return 2;
    }
    const bool dense = !dense_path.empty() || dense_trees;
    if (dense && method == "interval") {
        // The interval oracle makes far every box without a zero, whatever cull_factor says: the grid
        // vertices next to the surface would hold bounds, not values.
        fprintf(stderr, "--dense and --dense-trees need --method full, lipschitz or hybrid\n");
        return 2;
    }
    if (dense) op.cull_factor = dense_cull_factor(scene);

    Counters c_oct, c_mesh;
    DCStats dcs;
    AdaptiveStats ast;
    Mesh mesh;
    double dense_s = 0.0, octree_s = 0.0, gpu_init_s = 0.0;
    std::string gpu_device;
    uint64_t pool_entries = 0;
    std::vector<double> resident_runs;  // --repeat: the GPU mesh kept on the GPU, seconds per run
    std::vector<Level>* levels_ptr = nullptr;
    Octree oct;
    // A file that cannot be written is an error, reported once everything is written.
    bool written = true;
    auto save_mesh = [&](const Mesh& m) {
        if (!out_path.empty() && !write_mesh(m, out_path)) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); written = false; }
        if (!attr_path.empty() && !write_vertex_attribute(m.tree_n, attr_path)) { fprintf(stderr, "cannot write %s\n", attr_path.c_str()); written = false; }
    };
    auto save_dense = [&](const std::vector<float>& v) {
        if (!write_f32(v, dense_path)) { fprintf(stderr, "cannot write %s\n", dense_path.c_str()); written = false; }
    };
    if (adaptive && gpu && (!adaptive_topdown || adaptive_all)) {
        fprintf(stderr, "--gpu supports --adaptive-topdown only (single-primitive coarse leaves)\n");
        return 2;
    }
    if (adaptive && !gpu) {
        AdaptiveParams ap;
        ap.max_depth = depth; ap.tol = adaptive_tol; ap.single_primitive_only = !adaptive_all; ap.min_depth = adaptive_min_depth; ap.margin = margin;
        ap.top_down = adaptive_topdown;
        DCParams dp;
        mesh = adaptive_mesh(scene, ap, dp, c_mesh, ast);
        octree_s = ast.t_octree;
        dcs.total = ast.t_hermite + ast.t_collapse + ast.t_faces;
        dcs.vertices = ast.vertices; dcs.triangles = ast.triangles; dcs.cells = ast.leaves;
        save_mesh(mesh);
    } else if (!dense_path.empty() && method == "full") {
        Timer td;
        std::vector<float> v = dense_sample_full(scene, depth, c_mesh);
        dense_s = td.seconds();
        save_dense(v);
    } else {
        if (gpu) {
#ifdef LPM_HAVE_GPU
            // Octree and dual contouring on the GPU. The creation of the Vulkan device is not
            // counted, as the load of the scene is not; it is reported as time_gpu_init.
            if (!dense_path.empty()) { fprintf(stderr, "--gpu meshes with dual contouring only, not --dense\n"); return 2; }
            static GpuPruner pruner;
            GpuStats gs;
            GpuMeshStats gms;
            DCParams dp;
            dp.marching_cubes = mc;
            dp.project_vertices = project;
            // With --adaptive-topdown, the coarse leaves of adaptive_mesh, decided on the GPU too.
            CoarseParams cp;
            cp.tol = adaptive_tol;
            cp.min_depth = adaptive_min_depth;
            if (dense_trees) oct = pruner.build(scene, op, c_oct, gs);
            else mesh = pruner.mesh(scene, op, dp, c_oct, c_mesh, gs, gms, &oct, adaptive ? &cp : nullptr);
            // The entries of the pruned trees of every level, as counted by the GPU (without
            // pruning the cells share the full tree).
            if (method != "full") for (const Level& L : oct.levels) pool_entries += L.sum_tree_n;
            for (int r = 0; r < repeat; r++) {
                Counters c1, c2;
                GpuStats gs1;
                GpuMeshStats gms1;
                Timer tr;
                pruner.mesh(scene, op, dp, c1, c2, gs1, gms1, nullptr, adaptive ? &cp : nullptr, false);
                resident_runs.push_back(tr.seconds());
                // A warm run does the work of the first one: anything else is an error of the GPU
                // pipeline (state left over from the previous run), not a timing to report.
                if (gms1.leaves != gms.leaves || gms1.coarse_leaves != gms.coarse_leaves || gms1.crossings != gms.crossings ||
                    gms1.vertices != gms.vertices || c1.node_evals() != c_oct.node_evals() || c2.node_evals() != c_mesh.node_evals()) {
                    fprintf(stderr, "--repeat: run %d differs from the first (leaves %llu and %llu, crossings %llu and %llu, vertices %llu and %llu, "
                            "evaluations %llu and %llu)\n", r + 1, (unsigned long long)gms1.leaves, (unsigned long long)gms.leaves,
                            (unsigned long long)gms1.crossings, (unsigned long long)gms.crossings, (unsigned long long)gms1.vertices,
                            (unsigned long long)gms.vertices, (unsigned long long)(c1.node_evals() + c2.node_evals()),
                            (unsigned long long)(c_oct.node_evals() + c_mesh.node_evals()));
                    return 1;
                }
            }
            if (adaptive) {
                ast.fine_leaves = gms.leaves;
                ast.collapsed = gms.coarse_leaves;
                ast.leaves = gms.leaves + gms.coarse_leaves;
                ast.leaves_per_level = gms.coarse_per_level;
                ast.leaves_per_level.resize(depth + 1, 0);
                ast.leaves_per_level[depth] = gms.leaves;
            }
            gpu_init_s = pruner.init_seconds();
            gpu_device = gs.device;
            octree_s = dense_trees ? gs.t_total : gms.t_octree;
            dcs.total = gms.t_total - gms.t_octree;
            dcs.t_index = gms.t_index;
            dcs.cells = gms.leaves + gms.coarse_leaves; dcs.crossings = gms.crossings; dcs.vertices = gms.vertices; dcs.triangles = gms.triangles;
            dcs.ambiguous_faces = gms.ambiguous;
            levels_ptr = &oct.levels;
            if (!dense_trees) save_mesh(mesh);
#else
            fprintf(stderr, "lpm_mesh was built without Vulkan\n");
            return 2;
#endif
        } else {
            oct = build_octree(scene, op, c_oct);
            octree_s = oct.build_seconds;
            pool_entries = oct.pool_entries();
            levels_ptr = &oct.levels;
            if (dense_trees) {
                // Only the octree and its pruned trees.
            } else if (dense) {
                Timer td;
                std::vector<float> v = dense_sample_octree(oct, c_mesh);
                dense_s = td.seconds();
                save_dense(v);
            } else {
                DCParams dp;
                dp.analytic_leaves = analytic;
                dp.project_vertices = project;
                mesh = dual_contour(oct, dp, c_mesh, dcs);
                save_mesh(mesh);
            }
        }
    }
    const double total_s = t_all.seconds() - gpu_init_s;  // without the creation of the Vulkan device

    FILE* f = stats_path.empty() ? stdout : fopen(stats_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", stats_path.c_str());
        return 1;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"scene\": \"%s\", \"method\": \"%s\", \"depth\": %d, \"threads\": %d, \"inherit\": %s, \"analytic\": %s, \"dense\": %s,\n",
            scene_path.c_str(), method.c_str(), depth, omp_get_max_threads(), inherit ? "true" : "false",
            analytic ? "true" : "false", dense ? "true" : "false");
    if (gpu) fprintf(f, "  \"gpu\": \"%s\", \"time_gpu_init\": %.6f,\n", gpu_device.c_str(), gpu_init_s);
    fprintf(f, "  \"nodes\": %zu, \"prims\": %zu, \"ops\": %zu, \"tree_height\": %d,\n", scene.nodes.size(), scene.prims.size(), scene.ops.size(), scene.depth);
    fprintf(f, "  \"time_load\": %.6f, \"time_octree\": %.6f, \"time_mesh\": %.6f, \"time_dense\": %.6f, \"time_total\": %.6f,\n",
            load_s, octree_s, dcs.total, dense_s, total_s);
    if (!resident_runs.empty()) {
        std::vector<double> s = resident_runs;
        std::sort(s.begin(), s.end());
        const size_t m = s.size() / 2;
        fprintf(f, "  \"time_resident\": %.6f, \"time_resident_runs\": [", s.size() % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]));
        for (size_t r = 0; r < resident_runs.size(); r++) fprintf(f, "%s%.6f", r ? ", " : "", resident_runs[r]);
        fprintf(f, "],\n");
    }
    fprintf(f, "  \"time_dc_index\": %.6f, \"time_dc_corners\": %.6f, \"time_dc_edges\": %.6f, \"time_dc_qef\": %.6f, \"time_dc_faces\": %.6f,\n",
            dcs.t_index, dcs.t_corners, dcs.t_edges, dcs.t_qef, dcs.t_faces);
    fprintf(f, "  \"octree_prim_evals\": %llu, \"octree_op_evals\": %llu, \"octree_prim_ivals\": %llu, \"octree_op_ivals\": %llu,\n",
            (unsigned long long)c_oct.prim_evals, (unsigned long long)c_oct.op_evals, (unsigned long long)c_oct.prim_ivals, (unsigned long long)c_oct.op_ivals);
    fprintf(f, "  \"mesh_prim_evals\": %llu, \"mesh_op_evals\": %llu,\n", (unsigned long long)c_mesh.prim_evals, (unsigned long long)c_mesh.op_evals);
    fprintf(f, "  \"prim_evals\": %llu, \"node_evals\": %llu,\n",
            (unsigned long long)(c_oct.prim_evals + c_oct.prim_ivals + c_mesh.prim_evals),
            (unsigned long long)(c_oct.node_evals() + c_mesh.node_evals()));
    fprintf(f, "  \"cells\": %llu, \"corners\": %llu, \"crossings\": %llu, \"vertices\": %llu, \"triangles\": %llu, \"analytic_hits\": %llu,\n",
            (unsigned long long)dcs.cells, (unsigned long long)dcs.corners, (unsigned long long)dcs.crossings,
            (unsigned long long)dcs.vertices, (unsigned long long)dcs.triangles, (unsigned long long)dcs.analytic_hits);
    fprintf(f, "  \"pool_entries\": %llu, \"pool_bytes\": %llu, \"peak_rss_mb\": %.1f,\n",
            (unsigned long long)pool_entries, (unsigned long long)(pool_entries * sizeof(TreeEntry)), peak_rss_mb());
    if (adaptive) {
        fprintf(f, "  \"adaptive\": {\"tol\": %g, \"single_primitive_only\": %s, \"min_depth\": %d, \"fine_leaves\": %llu, \"leaves\": %llu, \"collapsed\": %llu, "
                "\"time_hermite\": %.6f, \"time_collapse\": %.6f, \"time_faces\": %.6f, \"leaves_per_level\": [",
                adaptive_tol, adaptive_all ? "false" : "true", adaptive_min_depth, (unsigned long long)ast.fine_leaves, (unsigned long long)ast.leaves,
                (unsigned long long)ast.collapsed, ast.t_hermite, ast.t_collapse, ast.t_faces);
        for (size_t l = 0; l < ast.leaves_per_level.size(); l++) fprintf(f, "%s%llu", l ? ", " : "", (unsigned long long)ast.leaves_per_level[l]);
        fprintf(f, "]},\n");
    }
    fprintf(f, "  \"levels\": [");
    if (levels_ptr) {
        for (size_t l = 0; l < levels_ptr->size(); l++) {
            const Level& L = (*levels_ptr)[l];
            // A GPU build counts its cells, a CPU one keeps them (see Level::num_cells).
            const uint64_t cells = L.num_cells ? L.num_cells : L.cells.size(), near = cells - L.far_cells;
            fprintf(f, "%s\n    {\"depth\": %zu, \"cells\": %zu, \"far\": %llu, \"tested\": %llu, \"avg_tree\": %.4f, \"max_tree\": %llu, \"single_prim\": %llu, \"seconds\": %.6f}",
                    l ? "," : "", l, (size_t)cells, (unsigned long long)L.far_cells, (unsigned long long)L.tested,
                    near ? (double)L.sum_tree_n / (double)near : 0.0, (unsigned long long)L.max_tree_n,
                    (unsigned long long)L.single_prim, L.seconds);
        }
        fprintf(f, "\n  ");
    }
    fprintf(f, "]\n}\n");
    if (f != stdout) fclose(f);
    return written ? 0 : 1;
}

int main(int argc, char** argv) {
    lpm::passive_omp_wait(argv);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {  // a scene that cannot be read, or the GPU out of memory
        fprintf(stderr, "lpm_mesh: %s\n", e.what());
        return 1;
    }
}
