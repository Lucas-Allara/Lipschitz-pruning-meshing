// Octree build with Lipschitz (or interval) pruning and dual contouring on the GPU (Vulkan compute,
// shaders/). build() returns the same leaves and pruned trees as build_octree, so the
// CPU dual contouring of include/lpm/dc.h runs on it unchanged; only its last level keeps cells
// and trees. mesh() keeps the octree on the GPU and runs the dual contouring there too.
// dynamic_setup() and dynamic_frame() are the dynamic update of include/lpm/dynamic.h on the GPU.
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lpm/dc.h"
#include "lpm/dynamic.h"
#include "lpm/octree.h"

namespace lpm {

struct GpuStats {
    std::string device;
    double t_setup = 0.0;     // pipelines for this scene and upload of the scene and the root
    double t_levels = 0.0;    // wall time of the steps of the octree, including the read of the counters
    double t_gpu = 0.0;       // sum of the GPU timestamps of the octree dispatches
    double t_download = 0.0;  // build(): copy of the last level to the CPU
    double t_total = 0.0;     // everything but the creation of the Vulkan device
    std::vector<double> level_gpu;  // seconds per step, from the timestamps
};

// Coarse leaves of the GPU mesh, as the top-down adaptive dual contouring (AdaptiveParams with
// top_down, single_primitive_only) of include/lpm/adaptive.h.
struct CoarseParams {
    double tol = 0.02;  // residual of the quadric, relative to the cell size
    int min_depth = 3;  // first level that may hold coarse leaves (and never above depth - 5)
};

struct GpuMeshStats {
    double t_octree = 0.0;    // octree on the GPU
    double t_index = 0.0;     // first submission: grids, corner values, crossing edges, components
    double t_mesh = 0.0;      // second submission: crossings, vertices, faces, copy to the host
    double t_download = 0.0;  // unpacking of the mesh on the CPU
    double t_total = 0.0;
    // triangles is counted while the mesh is unpacked on the host: 0 when it stays on the GPU.
    uint64_t leaves = 0, crossings = 0, vertices = 0, triangles = 0, ambiguous = 0, prim_evals = 0, op_evals = 0;
    uint64_t coarse_leaves = 0;              // leaves of the finest level are `leaves`
    std::vector<uint64_t> coarse_per_level;  // with coarse leaves: per level
    std::vector<std::pair<std::string, double>> stage_gpu;  // GPU seconds per stage, from timestamps
};

struct GpuFrameStats {
    double t_descent = 0.0;   // root on the CPU and the levels of the dynamic descent (one submission with t_region)
    double t_region = 0.0;    // GPU time of the swallowed cells, dynamic leaves and ring
    double t_dc = 0.0;        // dual contouring of the region and assembly, with the copy to the host
    double t_download = 0.0;  // the mesh on the CPU
    double total = 0.0;
    double t_root = 0.0;      // root pruned on the CPU
    std::vector<double> level_gpu;  // descent: GPU seconds per level
    // kept_quads and new_quads: static faces kept and new faces (triangles with marching cubes).
    uint64_t dynamic_leaves = 0, culled = 0, region = 0, kept_quads = 0, new_quads = 0, new_vertices = 0;
};

// The Vulkan objects of a GpuPruner, for a renderer that draws from its buffers (apps/lpm_view.cpp).
struct GpuHandles {
    void* instance = nullptr;         // VkInstance
    void* physical_device = nullptr;  // VkPhysicalDevice
    void* device = nullptr;           // VkDevice
    void* queue = nullptr;            // VkQueue: compute, and graphics when the pruner was created for it
    uint32_t queue_family = 0;
};

// The mesh of a dynamic frame where the GPU keeps it, as device addresses: 3 doubles per vertex,
// the static vertices (nv_static) in one buffer and the new ones in another; 3 indices per
// triangle, the new vertices numbered from nv_static on. With coarse leaves, n_empty of the
// n_triangles have an index 0xffffffff (the other triangle of a face that is one), for the vertex
// shader to drop. Valid until the next call that uses the pruner.
struct GpuMeshBuffers {
    uint64_t static_v = 0, new_v = 0, faces = 0;
    uint32_t nv_static = 0, nv_new = 0, n_triangles = 0;
    uint64_t static_t = 0, new_t = 0;  // tree size of the cell of each vertex (32 bits), as static_v and new_v
    uint32_t n_empty = 0;
};

class GpuPruner {
public:
    // Creates the Vulkan instance and device; throws if no suitable GPU is found. With graphics,
    // the instance also enables instance_extensions (those of the window system), and the device
    // is a Vulkan 1.3 one whose queue draws and presents too (swapchain, dynamic rendering).
    explicit GpuPruner(const std::vector<std::string>& instance_extensions = {}, bool graphics = false);
    ~GpuPruner();
    GpuPruner(const GpuPruner&) = delete;
    GpuPruner& operator=(const GpuPruner&) = delete;
    // ORACLE_LIPSCHITZ, ORACLE_INTERVAL (the leaves and trees of build_octree with that oracle) or
    // ORACLE_FULL (no pruning), inherit = true and keep_empty_leaves = false. With ORACLE_LIPSCHITZ,
    // cull_factor may exceed 1 (the octree of dense sampling): the cells within cull_factor * R of
    // the surface keep their trees, as in build_octree, and the far leaves, which hold no tree, are
    // dropped.
    Octree build(const Scene& scene, const OctreeParams& params, Counters& counters, GpuStats& st);
    // Octree and dual contouring on the GPU, depth <= 9: the algorithm of dual_contour on
    // build_octree, in the same vertex and face order (with DCParams::marching_cubes, marching cubes
    // on the same leaves instead, which the CPU does not have). With ORACLE_FULL every evaluation uses the
    // full tree. `levels` receives the statistics of the octree levels. With coarse, cells whose
    // pruned tree is a single primitive may become coarse leaves while the octree is built, as
    // adaptive_mesh with top_down: the same vertices (the coarse ones after those of the finest
    // leaves) and the same faces, in another order. With to_host false the mesh stays on the GPU and
    // the Mesh returned is empty, which times the GPU work alone.
    Mesh mesh(const Scene& scene, const OctreeParams& params, const DCParams& dcp, Counters& c_oct, Counters& c_mesh,
              GpuStats& st, GpuMeshStats& ms, Octree* levels = nullptr, const CoarseParams* coarse = nullptr,
              bool to_host = true);
    // Dynamic update: the static octree (with its empty leaves) and the static dual contouring of
    // dyn are built once and stay on the GPU; each frame meshes the scene with the sphere at
    // `center`, the mesh of build_dynamic and patch_mesh: the static vertices, then the new ones;
    // the static faces that stay, then the new ones (triangles with DCParams::marching_cubes).
    // With coarse, the static mesh has the coarse leaves of mesh() (dual contouring only), and a frame
    // meshes each coarse leaf where a dynamic or swallowed cell lies by its finest leaves.
    void dynamic_setup(const DynamicScene& dyn, int depth, double margin, const DCParams& dcp, Counters& c,
                       const CoarseParams* coarse = nullptr);
    // With new_only, V and tree_n hold only the new vertices (the static ones are those of
    // static_mesh(), in front of them); F is always complete.
    Mesh dynamic_frame(const double center[3], Counters& c, GpuFrameStats& fs, bool new_only = false);
    // The same into `out`, whose memory is reused from frame to frame (fresh vectors of a large
    // mesh cost their page faults on every frame).
    void dynamic_frame(const double center[3], Counters& c, GpuFrameStats& fs, Mesh& out, bool new_only = false);
    // The same without copying the mesh to the host: it stays in the buffers returned.
    GpuMeshBuffers dynamic_frame_gpu(const double center[3], Counters& c, GpuFrameStats& fs);
    const Mesh& static_mesh() const;
    GpuHandles handles() const;
    // Frees the temporaries of the last octree build (they are reallocated when needed), for
    // example after a build that ran out of device memory.
    void release_build_buffers();
    // Frees the dynamic update (dynamic_setup must run again before the next frame) and the buffers
    // of the dual contouring, for example after a setup that ran out of device memory.
    void release_dynamic();
    std::string device_name() const;
    double init_seconds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lpm
