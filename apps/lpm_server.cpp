// Interactive server: keeps the static octree and the static mesh of a scene and, for every
// sphere center read from stdin ("x y z" per line), writes the updated mesh to stdout as a binary
// frame. With --gpu the whole update runs on the GPU (GpuPruner::dynamic_frame), otherwise on the
// CPU (build_dynamic and patch_mesh). scripts/interactive.py drives it from a PyVista window.
//
// Frame: "LPM1", uint32 nv, uint32 nf, double seconds, uint32 region_cells, uint32 kept_quads,
//        float32[3 nv] vertices, uint32[3 nf] triangles.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "lpm/dynamic.h"
#include "lpm/threads.h"
#ifdef LPM_HAVE_GPU
#include "lpm/gpu.h"
#endif

using namespace lpm;

static int run(int argc, char** argv) {
    std::string scene_path;
    int depth = 7, threads = 8;  // few threads: the per-frame work is small and 32 threads add scheduling hiccups
    double radius = 0.2, k = 0.1;
    bool use_gpu = false;
    const char* usage = "usage: lpm_server --scene FILE [--depth D] [--radius R] [--k K] [--threads T] [--gpu]\n";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) exit(1); return argv[++i]; };
        if (a == "--scene") scene_path = next();
        else if (a == "--depth") depth = atoi(next().c_str());
        else if (a == "--radius") radius = atof(next().c_str());
        else if (a == "--k") k = atof(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--gpu") use_gpu = true;
        else { fputs(usage, stderr); return 1; }
    }
    if (scene_path.empty()) { fputs(usage, stderr); return 1; }
    if (depth < 1 || depth > kMaxDepth) {
        fprintf(stderr, "--depth must be between 1 and %d\n", kMaxDepth);
        return 2;
    }
#ifndef LPM_HAVE_GPU
    if (use_gpu) { fprintf(stderr, "lpm_server was built without Vulkan\n"); return 2; }
#endif
    if (threads > 0) omp_set_num_threads(threads);
    const double margin = 1e-3;

    Timer t_setup;
    Scene base = load_scene_json(scene_path);
    DynamicScene dyn = make_dynamic_scene(base, k, radius);
    Counters c_setup;
    DCParams dp;
    StaticOctree st;
    StaticMesh sm;
    std::vector<float> static_v;  // GPU: the static vertices, sent in front of the new ones
#ifdef LPM_HAVE_GPU
    std::unique_ptr<GpuPruner> gpu;
    if (use_gpu) {
        gpu.reset(new GpuPruner());
        gpu->dynamic_setup(dyn, depth, margin, dp, c_setup);
        static_v.assign(gpu->static_mesh().V.begin(), gpu->static_mesh().V.end());
    }
#endif
    if (!use_gpu) {
        st = build_static(base, depth, margin, c_setup);
        sm = build_static_mesh(st, dp, c_setup);
    }
    fprintf(stderr, "ready (%s): %zu nodes, depth %d, setup %.2f s, aabb [%g %g %g] [%g %g %g]\n",
            use_gpu ? "GPU" : "CPU", base.nodes.size(), depth, t_setup.seconds(),
            base.aabb_min[0], base.aabb_min[1], base.aabb_min[2], base.aabb_max[0], base.aabb_max[1], base.aabb_max[2]);
    fflush(stderr);

    Mesh mesh;  // its memory is reused from frame to frame
    std::vector<float> vf;
    char line[256];
    while (fgets(line, sizeof line, stdin)) {
        if (strncmp(line, "quit", 4) == 0) break;
        double c[3];
        if (sscanf(line, "%lf %lf %lf", &c[0], &c[1], &c[2]) != 3) continue;
        Timer tf;
        Counters cnt;
        uint32_t cells = 0, statics = 0;
#ifdef LPM_HAVE_GPU
        if (use_gpu) {
            GpuFrameStats fs;
            gpu->dynamic_frame(c, cnt, fs, mesh, true);
            cells = (uint32_t)fs.region;
            statics = (uint32_t)fs.kept_quads;
        }
#endif
        if (!use_gpu) {
            set_sphere_center(dyn, c);
            DynamicUpdate du = build_dynamic(dyn, st, margin, cnt, false);
            DCStats ds;
            PatchStats ps;
            mesh = patch_mesh(dyn, st, sm, du, dp, cnt, ps, ds);
            cells = (uint32_t)(ps.dynamic_leaves + ps.ring);
            statics = (uint32_t)ps.kept_quads;
        }
        const double seconds = tf.seconds();
        vf.assign(mesh.V.begin(), mesh.V.end());
        const uint32_t nv = (uint32_t)((static_v.size() + vf.size()) / 3), nf = (uint32_t)(mesh.F.size() / 3);
        fwrite("LPM1", 1, 4, stdout);
        fwrite(&nv, 4, 1, stdout);
        fwrite(&nf, 4, 1, stdout);
        fwrite(&seconds, 8, 1, stdout);
        fwrite(&cells, 4, 1, stdout);
        fwrite(&statics, 4, 1, stdout);
        fwrite(static_v.data(), 4, static_v.size(), stdout);
        fwrite(vf.data(), 4, vf.size(), stdout);
        fwrite(mesh.F.data(), 4, mesh.F.size(), stdout);
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
    lpm::passive_omp_wait(argv);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {  // a scene that cannot be read, or the GPU out of memory
        fprintf(stderr, "lpm_server: %s\n", e.what());
        return 1;
    }
}
