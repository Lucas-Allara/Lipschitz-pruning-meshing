// The animation of the official viewer, for meshing: a sphere of radius 0.2 travels through
// the scene, joined by a smooth union of radius 0.1, and the mesh is rebuilt every frame
// from scratch (octree + Lipschitz pruning + dual contouring), with the full tree, or with
// the dynamic update that reuses everything the sphere cannot affect. With --orbit the sphere
// follows a closed orbit instead, so that the frames loop. The methods gpu (from scratch) and
// gpu-patch (the dynamic update) run on the GPU; with --check, gpu-patch is compared with the GPU
// mesh rebuilt from scratch (the same triangles) and with the CPU patch (the same faces, the
// vertices within rounding). With --coarse no mesh from scratch has the faces of a frame (its
// coarse leaves come from the static scene), and --check asks every frame to be closed and
// consistently oriented instead.
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

#include "lpm/dynamic.h"
#include "lpm/threads.h"
#ifdef LPM_HAVE_GPU
#include "lpm/gpu.h"
#endif

using namespace lpm;

// Triangles as sorted position triples, so that meshes with different vertex orders compare.
static std::vector<std::array<double, 9>> signature(const Mesh& m) {
    std::vector<std::array<double, 9>> out(m.F.size() / 3);
    for (size_t t = 0; t < out.size(); t++) {
        std::array<std::array<double, 3>, 3> p;
        for (int k = 0; k < 3; k++) for (int a = 0; a < 3; a++) p[k][a] = m.V[3 * m.F[3 * t + k] + a];
        int first = 0;
        for (int k = 1; k < 3; k++) if (p[k] < p[first]) first = k;
        for (int k = 0; k < 3; k++) for (int a = 0; a < 3; a++) out[t][3 * k + a] = p[(first + k) % 3][a];
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Every directed edge of a triangle is matched by the reverse edge of another one, as many times:
// the mesh is closed and consistently oriented (an edge of two sheets, with four faces, passes).
static bool closed_and_oriented(const Mesh& m) {
    std::vector<uint64_t> fwd, rev;
    fwd.reserve(m.F.size());
    rev.reserve(m.F.size());
    for (size_t t = 0; t + 2 < m.F.size(); t += 3)
        for (int k = 0; k < 3; k++) {
            const uint64_t a = m.F[t + k], b = m.F[t + (k + 1) % 3];
            if (a < b) fwd.push_back(a << 32 | b);
            else if (b < a) rev.push_back(b << 32 | a);
        }
    std::sort(fwd.begin(), fwd.end());
    std::sort(rev.begin(), rev.end());
    return fwd == rev;
}

static void usage() {
    fprintf(stderr, "usage: lpm_anim --scene FILE [--frames F] [--depth D] [--radius R] [--k K] [--method scratch|incremental|patch|full|gpu|gpu-patch]\n"
                    "                [--check] [--out-dir DIR] [--stats FILE] [--threads T] [--orbit R_FRAC,Y_LO,Y_HI,PHASE_DEG]\n"
                    "                [--keep-on-gpu]   (gpu and gpu-patch: the mesh of each frame stays on the GPU, as in lpm_view;\n"
                    "                                   with gpu-patch --check the triangles of each frame are counted)\n"
                    "                [--mc]            (gpu and gpu-patch: marching cubes on the leaves instead of dual contouring)\n"
                    "                [--coarse]        (gpu-patch: coarse leaves where the sphere does not reach, dual contouring;\n"
                    "                                   --check then checks that every frame is closed and consistently oriented)\n");
    exit(1);
}

static int run(int argc, char** argv) {
    std::string scene_path, method = "incremental", out_dir, stats_path;
    int frames = 48, depth = 8, threads = 0;
    double radius = 0.2, k = 0.1;
    double orbit[4] = {0, 0, 0, 0};
    bool check = false, use_orbit = false, keep_on_gpu = false, mc = false, coarse = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--scene") scene_path = next();
        else if (a == "--frames") frames = atoi(next().c_str());
        else if (a == "--depth") depth = atoi(next().c_str());
        else if (a == "--radius") radius = atof(next().c_str());
        else if (a == "--k") k = atof(next().c_str());
        else if (a == "--method") method = next();
        else if (a == "--check") check = true;
        else if (a == "--keep-on-gpu") keep_on_gpu = true;
        else if (a == "--mc") mc = true;
        else if (a == "--coarse") coarse = true;
        else if (a == "--out-dir") out_dir = next();
        else if (a == "--stats") stats_path = next();
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--orbit") {
            if (sscanf(next().c_str(), "%lf,%lf,%lf,%lf", &orbit[0], &orbit[1], &orbit[2], &orbit[3]) != 4) usage();
            use_orbit = true;
        }
        else usage();
    }
    if (scene_path.empty()) usage();
    static const char* const methods[] = {"scratch", "incremental", "patch", "full", "gpu", "gpu-patch"};
    if (std::find(std::begin(methods), std::end(methods), method) == std::end(methods)) usage();
    if (coarse && (method != "gpu-patch" || mc || (check && keep_on_gpu))) {
        fprintf(stderr, "--coarse is for --method gpu-patch, without --mc; its --check needs the frames on the host (no --keep-on-gpu)\n");
        return 2;
    }
    if (mc && method != "gpu" && method != "gpu-patch") {
        fprintf(stderr, "--mc (marching cubes) is for --method gpu or gpu-patch\n");
        return 2;
    }
    if (frames < 1 || depth < 1 || depth > kMaxDepth) {
        fprintf(stderr, "--frames must be at least 1 and --depth between 1 and %d\n", kMaxDepth);
        return 2;
    }
    if (keep_on_gpu && ((method != "gpu" && method != "gpu-patch") || (check && method != "gpu-patch") || !out_dir.empty())) {
        fprintf(stderr, "--keep-on-gpu is for --method gpu or gpu-patch, without --out-dir; --check only with gpu-patch (it counts the triangles)\n");
        return 2;
    }
    if (threads > 0) omp_set_num_threads(threads);
    const double margin = 1e-3;
    if (!out_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec) {
            fprintf(stderr, "cannot create %s: %s\n", out_dir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    Scene base = load_scene_json(scene_path);
    DynamicScene dyn = make_dynamic_scene(base, k, radius);
    DCParams dp;
    dp.marching_cubes = mc;

    // Static structures of the dynamic update.
    double setup_s = 0.0;
    StaticOctree st;
    DCCache cache;
    StaticMesh sm;
    Counters c_setup;
    const bool on_gpu = method == "gpu" || method == "gpu-patch";
    // The CPU patch is the reference of gpu-patch --check, except with marching cubes (GPU only).
    const bool cpu_patch = method == "patch" || (method == "gpu-patch" && check && !mc && !coarse);
    if (method == "incremental" || cpu_patch) {
        Timer t;
        st = build_static(base, depth, margin, c_setup);
        if (cpu_patch) sm = build_static_mesh(st, dp, c_setup);
        else { DCStats s0; dual_contour(st.oct, dp, c_setup, s0, &cache); }
        if (method != "gpu-patch") setup_s = t.seconds();
    }
#ifdef LPM_HAVE_GPU
    std::unique_ptr<GpuPruner> gpu, gpu_ref;  // gpu_ref: the reference of gpu-patch --check
    if (on_gpu) gpu.reset(new GpuPruner());
    if (method == "gpu-patch") {
        Timer t;
        CoarseParams cp;
        cp.tol = 0.005;  // as the CAD table of the poster
        gpu->dynamic_setup(dyn, depth, margin, dp, c_setup, coarse ? &cp : nullptr);
        setup_s = t.seconds();
    }
#else
    if (on_gpu) { fprintf(stderr, "lpm_anim was built without Vulkan\n"); return 2; }
#endif
    int cpu_face_mismatches = 0, skipped_refs = 0;
    double cpu_max_dv = 0.0;
    bool written = true;

    FILE* f = stats_path.empty() ? stdout : fopen(stats_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", stats_path.c_str());
        return 1;
    }
    fprintf(f, "{\"scene\": \"%s\", \"method\": \"%s\", \"frames\": %d, \"depth\": %d, \"radius\": %g, \"k\": %g, \"threads\": %d,\n",
            scene_path.c_str(), method.c_str(), frames, depth, radius, k, omp_get_max_threads());
    fprintf(f, " \"setup_seconds\": %.6f, \"setup_prim_evals\": %llu, \"static_cells\": %zu,\n", setup_s,
            (unsigned long long)c_setup.prim_evals, (method == "incremental" || cpu_patch) ? st.oct.levels.back().cells.size() : 0);
    fprintf(f, " \"frames_data\": [");
    double sum_total = 0.0, max_total = 0.0;
    int mismatches = 0;
    Mesh mesh_buf;  // gpu-patch: the memory of the mesh, reused from frame to frame
    for (int fr = 0; fr < frames; fr++) {
        const double t = 2.0 * M_PI * fr / frames;
        double center[3];
        if (use_orbit) orbit_path(base, t, orbit[0], orbit[1], orbit[2], orbit[3] * M_PI / 180.0, center);
        else sphere_path(base, radius, t, center);
        set_sphere_center(dyn, center);
        Counters c_oct, c_mesh;
        DCStats ds;
        Mesh mesh;
        double oct_s = 0.0;
        uint64_t cells = 0, static_leaves = 0, dynamic_cells = 0;
        PatchStats pst;
        uint64_t gpu_triangles = 0;  // --keep-on-gpu: the triangles of the frame, on the GPU
        Timer tf;
        if (method == "gpu-patch") {
#ifdef LPM_HAVE_GPU
            GpuFrameStats gfs;
            if (keep_on_gpu) {
                const GpuMeshBuffers g = gpu->dynamic_frame_gpu(center, c_oct, gfs);
                gpu_triangles = g.n_triangles - g.n_empty;
            } else {
                gpu->dynamic_frame(center, c_oct, gfs, mesh_buf);
                mesh = std::move(mesh_buf);
            }
            oct_s = gfs.t_descent;
            dynamic_cells = gfs.dynamic_leaves;
            cells = gfs.region;
            pst.kept_quads = gfs.kept_quads;
            pst.touched = gfs.culled;
            pst.ring = gfs.region - gfs.dynamic_leaves;
            pst.new_quads = gfs.new_quads;
            pst.t_region = gfs.t_region;
            pst.t_dc = gfs.t_dc;
            pst.t_assemble = gfs.t_download;
#endif
        } else if (method == "gpu") {
#ifdef LPM_HAVE_GPU
            OctreeParams op;
            op.oracle = ORACLE_LIPSCHITZ;
            op.max_depth = depth;
            op.margin = margin;
            GpuStats gs;
            GpuMeshStats gms;
            mesh = gpu->mesh(dyn.scene, op, dp, c_oct, c_mesh, gs, gms, nullptr, nullptr, !keep_on_gpu);
            oct_s = gms.t_octree;
            ds.total = gms.t_total - gms.t_octree;
            cells = gms.leaves;
#endif
        } else if (method == "patch") {
            DynamicUpdate du = build_dynamic(dyn, st, margin, c_oct, false);
            oct_s = du.seconds;
            dynamic_cells = du.dynamic_cells;
            mesh = patch_mesh(dyn, st, sm, du, dp, c_mesh, pst, ds);
            cells = pst.dynamic_leaves + pst.ring;
        } else if (method == "incremental") {
            DynamicUpdate du = build_dynamic(dyn, st, margin, c_oct);
            oct_s = du.seconds;
            cells = du.leaves.size();
            static_leaves = du.static_leaves;
            dynamic_cells = du.dynamic_cells;
            mesh = dual_contour_leaves(dyn.scene, depth, st.oct.levels.back().size, du.leaves, dp, &cache, nullptr, c_mesh, ds);
        } else {
            OctreeParams op;
            op.oracle = method == "full" ? ORACLE_FULL : ORACLE_LIPSCHITZ;
            op.max_depth = depth;
            op.margin = margin;
            Octree oct = build_octree(dyn.scene, op, c_oct);
            oct_s = oct.build_seconds;
            cells = oct.levels.back().cells.size();
            mesh = dual_contour(oct, dp, c_mesh, ds);
        }
        const double total = tf.seconds();
        sum_total += total;
        max_total = std::max(max_total, total);
        bool same = true;
        if (check && coarse) {
            same = closed_and_oriented(mesh);
            if (!same) mismatches++;
        } else if (check) {
            OctreeParams op;
            op.oracle = ORACLE_LIPSCHITZ;
            op.max_depth = depth;
            op.margin = margin;
            Counters cc;
            DCStats ss;
            Mesh ref;
            if (on_gpu) {
#ifdef LPM_HAVE_GPU
                // The GPU reference: the same arithmetic, so the same values. For gpu-patch it runs
                // on a pruner of its own, so that the frames keep their state from one to the next
                // as without --check; when both do not fit in device memory, only the CPU patch is
                // compared.
                if (skipped_refs == 0 || method != "gpu-patch") {
                    try {
                        if (method == "gpu-patch" && !gpu_ref) gpu_ref.reset(new GpuPruner());
                        GpuPruner& g = method == "gpu-patch" ? *gpu_ref : *gpu;
                        Counters c2;
                        GpuStats gs;
                        GpuMeshStats gms;
                        ref = g.mesh(dyn.scene, op, dp, cc, c2, gs, gms);
                    } catch (const std::runtime_error& e) {
                        if (method != "gpu-patch" || mc) throw;  // marching cubes has no CPU patch to compare with
                        fprintf(stderr, "no GPU reference (%s); compared with the CPU patch only\n", e.what());
                        gpu_ref.reset();
                        skipped_refs = 1;
                    }
                }
#endif
            } else {
                Octree oct = build_octree(dyn.scene, op, cc);
                ref = dual_contour(oct, dp, cc, ss);
            }
            // A frame kept on the GPU has no mesh on the host: its number of triangles is compared.
            if (skipped_refs == 0)
                same = keep_on_gpu ? ref.F.size() / 3 == gpu_triangles
                       : (method == "patch" || method == "gpu-patch") ? signature(ref) == signature(mesh) : (ref.V == mesh.V && ref.F == mesh.F);
            if (method == "gpu-patch" && !mc) {
                // The CPU patch: the same faces in the same order, the vertices within rounding.
                Counters c1, c2;
                DynamicUpdate du = build_dynamic(dyn, st, margin, c1, false);
                PatchStats ps2;
                DCStats ds2;
                const Mesh cm = patch_mesh(dyn, st, sm, du, dp, c2, ps2, ds2);
                if (keep_on_gpu ? cm.F.size() / 3 != gpu_triangles : (cm.F != mesh.F || cm.V.size() != mesh.V.size())) {
                    cpu_face_mismatches++;
                    if (skipped_refs) same = false;
                } else if (!keep_on_gpu) for (size_t i = 0; i < cm.V.size(); i++) cpu_max_dv = std::max(cpu_max_dv, std::fabs(cm.V[i] - mesh.V[i]));
            }
            if (!same) mismatches++;
        }
        if (!out_dir.empty()) {
            char name[64];
            snprintf(name, sizeof name, "/frame_%03d.obj", fr);
            if (!write_mesh(mesh, out_dir + name)) {
                if (written) fprintf(stderr, "cannot write %s\n", (out_dir + name).c_str());
                written = false;
            }
        }
        // cells: the leaves contoured in the frame (all of them, or the region of a patch).
        // static_leaves: the leaves that incremental takes from the static octree.
        // patch_kept_quads: the quads of the static mesh that patch and gpu-patch keep (triangles with --mc).
        // triangles: null when --method gpu keeps the mesh on the GPU, where it is not counted.
        char triangles[24];
        if (keep_on_gpu && method == "gpu") snprintf(triangles, sizeof triangles, "null");
        else snprintf(triangles, sizeof triangles, "%llu", (unsigned long long)(keep_on_gpu ? gpu_triangles : mesh.F.size() / 3));
        fprintf(f, "%s\n  {\"frame\": %d, \"center\": [%.5f, %.5f, %.5f], \"time_octree\": %.6f, \"time_dc\": %.6f, \"time_total\": %.6f, "
                "\"cells\": %llu, \"static_leaves\": %llu, \"dynamic_cells\": %llu, \"reused_corners\": %llu, \"reused_crossings\": %llu, "
                "\"reused_vertices\": %llu, \"corners\": %llu, \"crossings\": %llu, \"prim_evals\": %llu, \"triangles\": %s, "
                "\"patch_touched\": %llu, \"patch_ring\": %llu, \"patch_kept_quads\": %llu, \"patch_new_quads\": %llu, \"patch_t_region\": %.6f, "
                "\"patch_t_assemble\": %.6f%s}",
                fr ? "," : "", fr, center[0], center[1], center[2], oct_s, (method == "patch" || method == "gpu-patch") ? pst.t_dc : ds.total, total,
                (unsigned long long)cells, (unsigned long long)static_leaves, (unsigned long long)dynamic_cells,
                (unsigned long long)ds.reused_corners, (unsigned long long)ds.reused_crossings, (unsigned long long)ds.reused_vertices,
                (unsigned long long)ds.corners, (unsigned long long)ds.crossings,
                (unsigned long long)(c_oct.prim_evals + c_mesh.prim_evals), triangles,
                (unsigned long long)pst.touched, (unsigned long long)pst.ring, (unsigned long long)pst.kept_quads, (unsigned long long)pst.new_quads,
                pst.t_region, pst.t_assemble,
                !check ? "" : coarse ? (same ? ", \"closed\": true" : ", \"closed\": false") : (same ? ", \"identical\": true" : ", \"identical\": false"));
        fflush(f);
        if (method == "gpu-patch") mesh_buf = std::move(mesh);
    }
    fprintf(f, "\n ],\n \"mean_frame_seconds\": %.6f, \"max_frame_seconds\": %.6f, \"peak_rss_mb\": %.1f%s}\n",
            sum_total / frames, max_total, peak_rss_mb(),
            !check ? "" : coarse ? (mismatches ? ", \"all_closed\": false" : ", \"all_closed\": true")
                                 : (mismatches ? ", \"all_identical\": false" : ", \"all_identical\": true"));
    if (f != stdout) fclose(f);
    if (check && coarse) fprintf(stderr, "%d of %d frames not closed or not consistently oriented\n", mismatches, frames);
    else if (check && skipped_refs == 0) fprintf(stderr, "%d of %d frames differ from the mesh rebuilt from scratch\n", mismatches, frames);
    if (check && cpu_patch && method == "gpu-patch") {
        if (keep_on_gpu)
            fprintf(stderr, "against the CPU patch: %d of %d frames with another number of triangles\n", cpu_face_mismatches, frames);
        else
            fprintf(stderr, "against the CPU patch: %d of %d frames with other faces, vertices within %.3g\n", cpu_face_mismatches,
                    frames, cpu_max_dv);
    }
    return (check && (mismatches || cpu_face_mismatches)) || !written ? 1 : 0;
}

int main(int argc, char** argv) {
    lpm::passive_omp_wait(argv);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {  // a scene that cannot be read, or the GPU out of memory
        fprintf(stderr, "lpm_anim: %s\n", e.what());
        return 1;
    }
}
