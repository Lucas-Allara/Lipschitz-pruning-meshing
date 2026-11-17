// Vulkan compute implementation of include/lpm/gpu.h.
//
// Octree: one dispatch of shaders/prune.comp (and prune_wide.comp for parents with large trees)
// per level, ping-pong buffers for the cells and trees of consecutive levels, and the counters of
// the level read back to size the next one. The interval pruning runs the same two shaders
// compiled with -DINTERVAL. With keep_empty_leaves the cells proven empty go to a
// separate list with their trees, and the static octree of a dynamic scene keeps every level.
//
// Coarse leaves (mesh() with CoarseParams): before a level of the octree is subdivided, shaders/
// probe.comp probes its cells with a single primitive (topology, a list of crossing sample edges,
// their crossings, the quadric of each cell) and flags those that become coarse leaves, which
// prune.comp does not subdivide; dc.comp then numbers their vertices after those of the finest
// leaves and adds the faces next to them.
//
// Dual contouring: shaders/dc.comp, one dispatch per stage, in two submissions. The first builds
// the grids, the corner values, the crossing edges and the components and reads back the number
// of crossing edges and vertices; the second computes the crossings, the vertices and the faces
// and copies the mesh to the host. The octree never leaves the GPU.
//
// Dynamic update (include/lpm/dynamic.h): the static scene keeps its octree (all levels, with a
// dense grid per level, each buffer cut to its used size) and its dual contouring on the GPU,
// and the value of every static tree at the center of its cell. A frame descends with
// shaders/dyn_prune.comp from the root, pruned on the CPU, where those values reduce most cells
// to one primitive and one operator, and patches the static mesh with the patch stages of dc.comp.
#include "lpm/gpu.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace lpm {

namespace {

#define SPIRV(name, file) const uint32_t name[] =
SPIRV(kPruneSpv, prune)
#include "prune.comp.inc"
    ;
SPIRV(kPruneWideSpv, prune_wide)
#include "prune_wide.comp.inc"
    ;
SPIRV(kPruneIvSpv, prune_interval)
#include "prune_interval.comp.inc"
    ;
SPIRV(kPruneWideIvSpv, prune_wide_interval)
#include "prune_wide_interval.comp.inc"
    ;
SPIRV(kScanSpv, scan)
#include "scan.comp.inc"
    ;
SPIRV(kScanAddSpv, scan_add)
#include "scan_add.comp.inc"
    ;
SPIRV(kGridSpv, grid)
#include "grid.comp.inc"
    ;
SPIRV(kDynSpv, dyn_prune)
#include "dyn_prune.comp.inc"
    ;
SPIRV(kDynWideSpv, dyn_prune_wide)
#include "dyn_prune_wide.comp.inc"
    ;
SPIRV(kDynValueSpv, dyn_prune_value)
#include "dyn_prune_value.comp.inc"
    ;
SPIRV(kDynArgsSpv, dyn_args)
#include "dyn_args.comp.inc"
    ;
SPIRV(kProbeSpv, probe)
#include "probe.comp.inc"
    ;
SPIRV(kDc1, dc_1)
#include "dc_1.comp.inc"
    ;
SPIRV(kDc2, dc_2)
#include "dc_2.comp.inc"
    ;
SPIRV(kDc3, dc_3)
#include "dc_3.comp.inc"
    ;
SPIRV(kDc4, dc_4)
#include "dc_4.comp.inc"
    ;
SPIRV(kDc5, dc_5)
#include "dc_5.comp.inc"
    ;
SPIRV(kDc6, dc_6)
#include "dc_6.comp.inc"
    ;
SPIRV(kDc7, dc_7)
#include "dc_7.comp.inc"
    ;
SPIRV(kDc8, dc_8)
#include "dc_8.comp.inc"
    ;
SPIRV(kDc9, dc_9)
#include "dc_9.comp.inc"
    ;
SPIRV(kDc10, dc_10)
#include "dc_10.comp.inc"
    ;
SPIRV(kDc11, dc_11)
#include "dc_11.comp.inc"
    ;
SPIRV(kDc12, dc_12)
#include "dc_12.comp.inc"
    ;
SPIRV(kDc13, dc_13)
#include "dc_13.comp.inc"
    ;
SPIRV(kDc14, dc_14)
#include "dc_14.comp.inc"
    ;
SPIRV(kDc15, dc_15)
#include "dc_15.comp.inc"
    ;
SPIRV(kDc16, dc_16)
#include "dc_16.comp.inc"
    ;
SPIRV(kDc17, dc_17)
#include "dc_17.comp.inc"
    ;
SPIRV(kDc18, dc_18)
#include "dc_18.comp.inc"
    ;
SPIRV(kDc19, dc_19)
#include "dc_19.comp.inc"
    ;
SPIRV(kDc20, dc_20)
#include "dc_20.comp.inc"
    ;
SPIRV(kDc21, dc_21)
#include "dc_21.comp.inc"
    ;
#undef SPIRV
struct Spirv {
    const uint32_t* code;
    size_t bytes;
};
constexpr int kDcStageCount = 21;
const Spirv kDcStages[kDcStageCount] = {
    {kDc1, sizeof(kDc1)},   {kDc2, sizeof(kDc2)},   {kDc3, sizeof(kDc3)},   {kDc4, sizeof(kDc4)},   {kDc5, sizeof(kDc5)},
    {kDc6, sizeof(kDc6)},   {kDc7, sizeof(kDc7)},   {kDc8, sizeof(kDc8)},   {kDc9, sizeof(kDc9)},   {kDc10, sizeof(kDc10)},
    {kDc11, sizeof(kDc11)}, {kDc12, sizeof(kDc12)}, {kDc13, sizeof(kDc13)}, {kDc14, sizeof(kDc14)}, {kDc15, sizeof(kDc15)},
    {kDc16, sizeof(kDc16)}, {kDc17, sizeof(kDc17)}, {kDc18, sizeof(kDc18)}, {kDc19, sizeof(kDc19)},
    {kDc20, sizeof(kDc20)}, {kDc21, sizeof(kDc21)}};
enum DcStage { SCATTER = 0, COMPACT, CORNERS, EDGES, CROSS, COMP, QEF, FACES, TOUCH, DYNBITS, RING, REGION, KEEP, NEWFLAGS, ASSEMBLE,
               CLEAR, LIST, COARSE_VERT, COARSE_FACES, MC_FACES, DISSOLVE };

#define VK_CHECK(x)                                                                                      \
    do {                                                                                                 \
        VkResult r_ = (x);                                                                               \
        if (r_ != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r_) + " in " #x); \
    } while (0)

struct GpuCell {
    uint32_t xyz[4];   // x, y, z, flag (far, or static in a region)
    uint32_t tree[4];  // offset, size
};
struct PruneCounters {
    uint32_t cells, entries, max_tree, single_prim;
    uint64_t prim_evals, op_evals;
    uint32_t far_cells, pad;
};
struct PrunePC {
    double aabb_min[4];
    double size[4];
    double r_keep, r_prune;
    uint32_t n_parents, wide, full, keep_empty;
    double margin;  // interval variant: the child boxes are widened by margin * size / 2
};
static_assert(sizeof(PrunePC) == 104 && offsetof(PrunePC, margin) == 96, "push constant layout of prune_common.glsl");

// Params of shaders/dc.comp, in its std430 layout.
struct DcParams {
    double aabb_min[4];
    double size[4];
    double fd_h;
    uint32_t N, n_cells, max_iters, full_n;
    uint64_t nodes, prims, ops, leaf_cells, tree_pool, cell_grid, cell_bits, cell_prefix, leaves, corner_val, edge_bits,
        edge_prefix, edges, crossings, comp, ncomp, vbase, V, F, vtree, counters;
    uint32_t frame, nv_static, depth, n_static_edges;
    uint64_t static_pool, static_corner_val, static_edge_bits, static_edge_prefix, static_crossings,
        static_cell_bits, static_cell_prefix, static_vbase, static_leaf_cells, static_edges, static_F, dyn_cells, dyn_bits,
        touched_bits, culled, ncomp_dyn, vbase_dyn, keep_static, keep_prefix, new_flags, new_prefix, out_F, patch_args;
    uint64_t frame_counts, cell_hi, cell_hi_prefix, cell_word_list, cell_word_count, cell_word_prefix, edge_hi, edge_hi_prefix,
        edge_word_list, edge_word_count, edge_word_prefix;
    uint64_t coarse_bits, coarse_prefix, coarse_records, coarse_F;
    uint32_t coarse_on, coarse_min, nv_fine, n_coarse;
    uint32_t coarse_word_off[12];
    uint64_t corner_list, static_leaves;
    uint32_t mc, project;  // marching cubes on the leaves; 1: QEF vertices onto the surface
    uint64_t dissolved;    // frame with coarse leaves: per coarse leaf, 1 when dissolved
};
static_assert(offsetof(DcParams, nodes) == 88 && offsetof(DcParams, frame) == 256 && offsetof(DcParams, static_pool) == 272 &&
                  offsetof(DcParams, coarse_bits) == 544 && offsetof(DcParams, coarse_on) == 576 && offsetof(DcParams, corner_list) == 640 &&
                  offsetof(DcParams, mc) == 656 && offsetof(DcParams, dissolved) == 664 && sizeof(DcParams) == 672,
              "layout of Params in dc.comp");
// Params and counters of shaders/probe.comp.
struct ProbeParams {
    double aabb_min[4];
    double size[4];
    double fd_h, tol;
    uint32_t P, level, n_cells, max_iters;
    uint64_t cells, tree_pool, nodes, prims, ops, bits, records, counters;
    uint64_t passed, edges, cross, level_counts;
    uint32_t edge_cap, pad;
};
static_assert(offsetof(ProbeParams, cells) == 96 && offsetof(ProbeParams, edge_cap) == 192, "layout of Params in probe.comp");
struct ProbeCounters {
    uint64_t prim_evals, op_evals;
    uint32_t found[16];
};
constexpr uint32_t kProbeRecord = 48;  // bytes per coarse leaf record
struct DcCounters {
    uint64_t prim_evals, op_evals;
    uint32_t ambiguous, empty_triangles;
};
struct DcPC {
    uint64_t params;
    uint32_t count, mode;
};
struct ScanPC {
    uint64_t data, out, sums;
    uint32_t n, mode;
};
struct ScanAddPC {
    uint64_t out, block;
    uint32_t n, pad;
};
struct GridPC {
    uint64_t cells, far, grid;
    uint32_t n_cells, n_far, n, pad;
};
// Level of shaders/dyn_prune.comp, in its std430 layout.
struct DynLevel {
    double aabb_min[4];
    double size[4];
    double r_keep, r_prune;
    uint32_t level, sphere_node, root_node, wide, culled_cap, cell_cap, entry_cap, pad;
    uint64_t nodes, prims, ops, parents, parent_pool, out_cells, out_pool, scratch, values, st_grid, st_cells, st_far, st_pool,
        culled, counters, st_value, st_far_value, args, culled_count;
};
static_assert(offsetof(DynLevel, nodes) == 112 && sizeof(DynLevel) == 264, "layout of Level in dyn_prune.comp");
struct DynArgsPC {
    uint64_t counters, args, culled_count;
    uint32_t cell_cap, entry_cap, wide, culled_cap, mode;
};
struct DynCounters {
    uint32_t cells, entries, culled, max_tree;
    uint64_t prim_evals, op_evals;
};
struct DynPC {
    uint64_t level;
    uint32_t n_parents, far;
};

constexpr uint32_t kGroup = 64;
// Parents with larger trees are pruned by prune_wide.comp, one workgroup per child; 1024 was the
// best of 256 to 2048 and no wide kernel at all on the three official scenes.
constexpr uint32_t kWide = 1024;
// The same for the dynamic descent, whose few cells per level leave the GPU mostly idle: 256 was
// the best of 64 to 1024 on Trees, Molecule and Monument (Monument d7, 5.4 ms -> 3.4 ms).
constexpr uint32_t kDynWide = 256;
constexpr uint32_t kBindings = 11;
constexpr uint32_t kMaxQueries = 256;
constexpr uint32_t kNone = 0xffffffffu;
constexpr uint32_t kDynLevelStride = 320;
// Shared memory of the largest workgroup, the WIDE variant of shaders/dyn_prune.comp: four arrays
// of 1024 uints, two of 1024 doubles and a few scalars. Vulkan guarantees 16 KB only.
constexpr uint32_t kSharedBytes = 4 * 4 * 1024 + 2 * 8 * 1024 + 256;

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress addr = 0;
    void* map = nullptr;
};

// Scene on the GPU: nodes, primitives, operators and the full tree.
struct SceneBufs {
    Buffer nodes, prims, ops, full_pool;
    uint32_t full_n = 0;
    int stack = 1;
};

// Buffers of one dual contouring (the static one of a dynamic scene persists between frames).
struct DcBufs {
    Buffer cell_grid, cell_bits, cell_prefix, leaves, corner_val, edge_bits, edge_prefix, edges, crossings, comp, ncomp, vbase, V,
        F, vtree;
    Buffer leaf_cells, dyn_bits, touched_bits, ncomp_dyn, vbase_dyn, keep_static, keep_prefix, new_flags, new_prefix, out_F;
    Buffer cell_hi, cell_hi_prefix, cell_word_list, cell_word_count, cell_word_prefix, edge_hi, edge_hi_prefix, edge_word_list,
        edge_word_count, edge_word_prefix, corner_list;
    uint32_t n_leaves = 0, n_edges = 0, n_vertices = 0, n_fine_vertices = 0;  // after a run (n_vertices: also the coarse ones)
    std::vector<Buffer*> all() {
        return {&cell_grid, &cell_bits,  &cell_prefix, &leaves,       &corner_val, &edge_bits,   &edge_prefix, &edges,
                &crossings, &comp,       &ncomp,       &vbase,        &V,          &F,           &vtree,       &leaf_cells,
                &dyn_bits,  &touched_bits, &ncomp_dyn, &vbase_dyn,    &keep_static, &keep_prefix, &new_flags,  &new_prefix,
                &out_F,     &cell_hi,    &cell_hi_prefix, &cell_word_list, &cell_word_count, &cell_word_prefix, &edge_hi,
                &edge_hi_prefix, &edge_word_list, &edge_word_count, &edge_word_prefix, &corner_list};
    }
};

// The levels of a static octree, kept on the GPU.
struct StaticLevels {
    std::vector<Buffer> cells, far, pool, grid;
    std::vector<Buffer> value, far_value;  // dynamic update: value of each tree at its center, NaN if not a fixed point
    std::vector<uint64_t> n_cells, n_far, n_entries;
    std::vector<double> size;  // 3 per level
    std::vector<TreeEntry> root_pool;  // tree of the root, on the CPU too
};

// The last level of an octree built on the GPU.
struct GpuLevels {
    Octree oct;     // levels with their statistics, without cells
    Buffer* cells = nullptr;  // the last level
    Buffer* pool = nullptr;
    uint64_t n_cells = 0, n_entries = 0;
    bool full = false;
    bool retry = false;  // the probe of a level needed more room for its edges: build again
};

}  // namespace

struct GpuPruner::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkPhysicalDeviceMemoryProperties memprops{};
    double timestamp_period = 1.0;  // nanoseconds per tick
    uint32_t max_groups = 65535;    // workgroups per dispatch (maxComputeWorkGroupCount[0])
    std::string name;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    double init_s = 0.0;

    // Octree: descriptor set with 11 storage buffers.
    VkShaderModule module = VK_NULL_HANDLE, module_wide = VK_NULL_HANDLE;
    VkShaderModule module_iv = VK_NULL_HANDLE, module_wide_iv = VK_NULL_HANDLE;  // interval pruning
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE, pipeline_wide = VK_NULL_HANDLE;
    VkPipeline pipeline_iv = VK_NULL_HANDLE, pipeline_wide_iv = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    // Everything else: buffer device addresses in push constants.
    VkShaderModule dc_modules[kDcStageCount] = {}, scan_module = VK_NULL_HANDLE, scan_add_module = VK_NULL_HANDLE,
                   grid_module = VK_NULL_HANDLE, dyn_module = VK_NULL_HANDLE, dyn_wide_module = VK_NULL_HANDLE,
                   dyn_value_module = VK_NULL_HANDLE, dyn_args_module = VK_NULL_HANDLE, probe_module = VK_NULL_HANDLE;
    VkPipelineLayout dc_layout = VK_NULL_HANDLE;
    VkPipeline dc[kDcStageCount] = {}, scan_pipe = VK_NULL_HANDLE, scan_add_pipe = VK_NULL_HANDLE, grid_pipe = VK_NULL_HANDLE,
               dyn_pipe = VK_NULL_HANDLE, dyn_wide_pipe = VK_NULL_HANDLE, dyn_value_pipe = VK_NULL_HANDLE,
               dyn_args_pipe = VK_NULL_HANDLE, probe_pipe = VK_NULL_HANDLE;
    int pipeline_stack = -1;

    // Coarse leaves of the last build (probe.comp): one bitset per level, concatenated from level 0
    // (coarse_word_off), the prefix sums of its words, and the records of every level, concatenated.
    struct CoarseRun {
        double tol = 0.02, fd_rel = 1e-6;
        int min_depth = 3, max_iters = 30;
    };
    Buffer coarse_bits, coarse_prefix, coarse_records, coarse_F, probe_params, probe_counters, probe_host;
    // The passes of probe.comp: the cells that passed the topology, their crossing sample edges
    // (room for probe_edge_cap, grown when a level needs more), and the counts of a level, which
    // hold the sizes of the indirect dispatches.
    Buffer probe_passed, probe_edges, probe_cross, probe_level_counts, probe_level_host;
    uint32_t probe_edge_cap = 1u << 20;
    std::vector<Buffer> coarse_level_records;
    bool coarse_on = false;
    uint32_t coarse_min = 0, coarse_words = 0, coarse_word_off[12] = {}, n_coarse = 0;
    std::vector<uint64_t> coarse_found;  // per level

    SceneBufs scene_main, scene_dyn;
    DcBufs dc_main, dc_static;
    // staging: host-visible for uploads; readback: host-visible and cached, so that the CPU reads it fast.
    Buffer cells[2], pool[2], far, scratch, values, counters, counters_host, staging, readback, dc_params, dc_counters, totals;
    std::vector<Buffer> scan_sums, scan_out;  // block totals of each recursion level of a prefix sum

    // Dynamic scene.
    bool dyn_ready = false;
    DynamicScene dyn;
    int dyn_depth = 0;
    double dyn_margin = 1e-3;
    DCParams dyn_dcp;
    StaticLevels st_levels;
    Mesh st_mesh;  // the static mesh on the CPU
    Buffer dyn_cells[2], dyn_pool[2], dyn_scratch, dyn_values, dyn_culled, dyn_counters, dyn_level_params, dummy;
    // The coarse leaves of a dynamic scene with coarse leaves (dynamic_setup), copied from those of
    // their own build, and the dissolved leaves of a frame.
    bool dyn_coarse = false;
    uint32_t dyn_n_coarse = 0, dyn_coarse_min = 0, dyn_coarse_word_off[12] = {};
    Buffer st_coarse_bits, st_coarse_prefix, st_coarse_records, dyn_dissolved;
    // The descent runs in one submission into buffers of these capacities (cells and tree entries
    // per level, swallowed cells per frame), grown when a frame does not fit.
    Buffer dyn_args, dyn_host;
    uint32_t dyn_cell_cap = 1u << 16, dyn_entry_cap = 1u << 20, dyn_culled_cap = 1u << 16;
    // The grids and bitsets of dc_main hold only what the last frame set (its region cells,
    // crossing edges and swallowed cells, so many of each), which the next frame clears.
    bool frame_clean = false;
    uint32_t prev_L = 0, prev_E = 0, prev_culled = 0;

    Impl(const std::vector<std::string>& instance_extensions, bool graphics) {
        Timer t;
        const uint32_t api = graphics ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "lpm";
        app.apiVersion = api;
        std::vector<const char*> iext;
        for (const std::string& e : instance_extensions) iext.push_back(e.c_str());
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = (uint32_t)iext.size();
        ici.ppEnabledExtensionNames = iext.data();
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());
        int best = -1;
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devs[i], &p);
            VkPhysicalDeviceFeatures f;
            vkGetPhysicalDeviceFeatures(devs[i], &f);
            if (!f.shaderFloat64 || !f.shaderInt64 || p.apiVersion < api) continue;
            if (best < 0 || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) best = (int)i;
        }
        if (best < 0) throw std::runtime_error(graphics ? "no Vulkan 1.3 device with 64-bit floats" : "no Vulkan 1.2 device with 64-bit floats");
        phys = devs[best];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys, &props);
        name = props.deviceName;
        timestamp_period = props.limits.timestampPeriod;
        // The dense stages dispatch one workgroup per 64 corners or edges of the grid (millions at
        // depth 9), far above the 65535 that Vulkan guarantees: groups() checks every dispatch.
        max_groups = props.limits.maxComputeWorkGroupCount[0];
        if (props.limits.maxComputeSharedMemorySize < kSharedBytes)
            throw std::runtime_error(std::string(props.deviceName) + " has " + std::to_string(props.limits.maxComputeSharedMemorySize) +
                                     " bytes of shared memory per workgroup; the dynamic descent needs " + std::to_string(kSharedBytes));
        vkGetPhysicalDeviceMemoryProperties(phys, &memprops);

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
        qfam = UINT32_MAX;
        const VkQueueFlags want = VK_QUEUE_COMPUTE_BIT | (graphics ? VK_QUEUE_GRAPHICS_BIT : 0);
        for (uint32_t i = 0; i < nq; i++)
            if ((qf[i].queueFlags & want) == want && qf[i].timestampValidBits > 0) { qfam = i; break; }
        if (qfam == UINT32_MAX) throw std::runtime_error(graphics ? "no graphics and compute queue with timestamps" : "no compute queue with timestamps");

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.dynamicRendering = VK_TRUE;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        if (graphics) f12.pNext = &f13;
        f12.shaderBufferInt64Atomics = VK_TRUE;
        f12.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f12;
        f2.features.shaderFloat64 = VK_TRUE;
        f2.features.shaderInt64 = VK_TRUE;
        VkPhysicalDeviceFeatures have;
        vkGetPhysicalDeviceFeatures(phys, &have);
        f2.features.largePoints = graphics && have.largePoints;  // the vertices of lpm_view, 3 pixels wide
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        const char* swapchain_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        if (graphics) {
            dci.enabledExtensionCount = 1;
            dci.ppEnabledExtensionNames = &swapchain_ext;
        }
        VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
        vkGetDeviceQueue(device, qfam, 0, &queue);

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = qfam;
        VK_CHECK(vkCreateCommandPool(device, &cpi, nullptr, &cmd_pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cmd_pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = kMaxQueries;
        VK_CHECK(vkCreateQueryPool(device, &qpi, nullptr, &queries));

        module = shader(kPruneSpv, sizeof(kPruneSpv));
        module_wide = shader(kPruneWideSpv, sizeof(kPruneWideSpv));
        module_iv = shader(kPruneIvSpv, sizeof(kPruneIvSpv));
        module_wide_iv = shader(kPruneWideIvSpv, sizeof(kPruneWideIvSpv));
        scan_module = shader(kScanSpv, sizeof(kScanSpv));
        scan_add_module = shader(kScanAddSpv, sizeof(kScanAddSpv));
        grid_module = shader(kGridSpv, sizeof(kGridSpv));
        dyn_module = shader(kDynSpv, sizeof(kDynSpv));
        dyn_wide_module = shader(kDynWideSpv, sizeof(kDynWideSpv));
        dyn_value_module = shader(kDynValueSpv, sizeof(kDynValueSpv));
        dyn_args_module = shader(kDynArgsSpv, sizeof(kDynArgsSpv));
        probe_module = shader(kProbeSpv, sizeof(kProbeSpv));
        for (int s = 0; s < kDcStageCount; s++) dc_modules[s] = shader(kDcStages[s].code, kDcStages[s].bytes);

        VkDescriptorSetLayoutBinding b[kBindings];
        for (uint32_t i = 0; i < kBindings; i++)
            b[i] = VkDescriptorSetLayoutBinding{i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = kBindings;
        dli.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dli, nullptr, &dsl));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PrunePC)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &layout));
        VkPushConstantRange dpcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 48};
        VkPipelineLayoutCreateInfo dpli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        dpli.pushConstantRangeCount = 1;
        dpli.pPushConstantRanges = &dpcr;
        VK_CHECK(vkCreatePipelineLayout(device, &dpli, nullptr, &dc_layout));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(device, &dpi, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = dpool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &dset));

        // The atomics of the shaders stay in device memory; the counters are copied to the host.
        counters = make_buffer(sizeof(PruneCounters), false);
        counters_host = make_buffer(64, true, true);
        dc_params = make_buffer(sizeof(DcParams), false);
        dc_counters = make_buffer(sizeof(DcCounters), false);
        dyn_counters = make_buffer(sizeof(DynCounters), false);
        totals = make_buffer(64, true, true);
        dummy = make_buffer(16, false);
        probe_params = make_buffer(sizeof(ProbeParams), false);
        probe_counters = make_buffer(sizeof(ProbeCounters), false);
        probe_host = make_buffer(sizeof(ProbeCounters), true, true);
        probe_level_counts = make_buffer(32, false);
        probe_level_host = make_buffer(32, true, true);
        init_s = t.seconds();
    }

    ~Impl() {
        if (device) {
            vkDeviceWaitIdle(device);
            std::vector<Buffer*> bufs = {&st_coarse_bits, &st_coarse_prefix, &st_coarse_records, &dyn_dissolved, &cells[0], &cells[1], &pool[0], &pool[1], &far, &scratch, &values, &counters, &counters_host,
                                         &staging, &readback, &dc_params, &dc_counters, &totals, &dyn_cells[0], &dyn_cells[1],
                                         &dyn_pool[0], &dyn_pool[1], &dyn_scratch, &dyn_values, &dyn_culled, &dyn_counters,
                                         &dyn_level_params, &dummy, &dyn_args, &dyn_host, &coarse_bits, &coarse_prefix,
                                         &coarse_records, &coarse_F, &probe_params, &probe_counters, &probe_host,
                                         &probe_passed, &probe_edges, &probe_cross, &probe_level_counts, &probe_level_host};
            for (Buffer& b : coarse_level_records) bufs.push_back(&b);
            for (SceneBufs* s : {&scene_main, &scene_dyn}) for (Buffer* b : {&s->nodes, &s->prims, &s->ops, &s->full_pool}) bufs.push_back(b);
            for (DcBufs* d : {&dc_main, &dc_static}) for (Buffer* b : d->all()) bufs.push_back(b);
            for (std::vector<Buffer>* v : {&scan_sums, &scan_out, &st_levels.cells, &st_levels.far, &st_levels.pool, &st_levels.grid,
                                                &st_levels.value, &st_levels.far_value})
                for (Buffer& b : *v) bufs.push_back(&b);
            for (Buffer* b : bufs) release(*b);
            for (VkPipeline p : {pipeline, pipeline_wide, pipeline_iv, pipeline_wide_iv, scan_pipe, scan_add_pipe, grid_pipe, dyn_pipe,
                                 dyn_wide_pipe, dyn_value_pipe, dyn_args_pipe, probe_pipe})
                if (p) vkDestroyPipeline(device, p, nullptr);
            for (VkPipeline p : dc) if (p) vkDestroyPipeline(device, p, nullptr);
            vkDestroyDescriptorPool(device, dpool, nullptr);
            vkDestroyPipelineLayout(device, layout, nullptr);
            vkDestroyPipelineLayout(device, dc_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, dsl, nullptr);
            for (VkShaderModule m : {module, module_wide, module_iv, module_wide_iv, scan_module, scan_add_module, grid_module, dyn_module,
                                       dyn_wide_module, dyn_value_module, dyn_args_module, probe_module})
                vkDestroyShaderModule(device, m, nullptr);
            for (VkShaderModule m : dc_modules) vkDestroyShaderModule(device, m, nullptr);
            vkDestroyQueryPool(device, queries, nullptr);
            vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, cmd_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    // The number of workgroups of a dispatch, which the device must allow.
    uint32_t groups(uint64_t n) const {
        if (n > max_groups)
            throw std::runtime_error("a dispatch of " + std::to_string(n) + " workgroups exceeds the " + std::to_string(max_groups) +
                                     " of this GPU: use a smaller depth");
        return (uint32_t)n;
    }

    VkShaderModule shader(const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = bytes;
        smi.pCode = code;
        VkShaderModule m;
        VK_CHECK(vkCreateShaderModule(device, &smi, nullptr, &m));
        return m;
    }

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("no suitable memory type");
    }

    // Device-local storage buffer, or a host-visible one (cached for readbacks); every buffer has a
    // device address.
    Buffer make_buffer(VkDeviceSize size, bool host, bool cached = false) {
        Buffer b;
        b.size = std::max<VkDeviceSize>(size, 16);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = b.size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &b.buf));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buf, &req);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &flags;
        mai.allocationSize = req.size;
        VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (host) want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                         (cached ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, want);
        if (vkAllocateMemory(device, &mai, nullptr, &b.mem) != VK_SUCCESS) {
            vkDestroyBuffer(device, b.buf, nullptr);
            throw std::runtime_error("out of GPU memory: " + std::to_string(size >> 20) + " MB more");
        }
        VK_CHECK(vkBindBufferMemory(device, b.buf, b.mem, 0));
        if (host) VK_CHECK(vkMapMemory(device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b.buf;
        b.addr = vkGetBufferDeviceAddress(device, &ai);
        return b;
    }

    void release(Buffer& b) {
        if (b.map) vkUnmapMemory(device, b.mem);
        if (b.buf) vkDestroyBuffer(device, b.buf, nullptr);
        if (b.mem) vkFreeMemory(device, b.mem, nullptr);
        b = Buffer{};
    }

    // Device-local buffer of at least `size` bytes, reallocated with some slack when too small.
    // Never called while a recorded command buffer uses the buffer.
    void ensure(Buffer& b, VkDeviceSize size) {
        if (b.buf && b.size >= size) return;
        release(b);
        b = make_buffer(size + size / 4, false);
    }
    // For the dense grids, whose size depends only on the depth: exactly that size (at depth 9 the
    // slack of ensure would be 1.5 GB over the grids of the static and the frame contouring).
    void ensure_exact(Buffer& b, VkDeviceSize size) {
        if (b.buf && b.size >= size) return;
        release(b);
        b = make_buffer(size, false);
    }
    // For the buffers of a dynamic frame, whose sizes change from frame to frame: more room, so
    // that an allocation (up to a millisecond) seldom falls in a frame.
    void ensure_frame(Buffer& b, VkDeviceSize size) {
        if (b.buf && b.size >= size) return;
        release(b);
        b = make_buffer(size + size / 2 + 4096, false);
    }
    static void indirect_barrier(VkCommandBuffer c) {
        barrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    }
    void dc_dispatch_indirect(VkCommandBuffer cb, DcStage s, VkDeviceSize offset, uint32_t mode = 0) {
        DcPC pc{dc_params.addr, 0, mode};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dc[s]);
        vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatchIndirect(cb, dyn_args.buf, offset);
    }
    // Moves the first `used` bytes of b into a buffer of that size, when b is much larger: the
    // levels kept for the dynamic update were allocated for the worst case of 8 children per cell.
    void shrink(Buffer& b, VkDeviceSize used) {
        used = std::max<VkDeviceSize>(used, 16);
        if (!b.buf || b.size <= used + used / 4 + (1u << 20)) return;
        Buffer n = make_buffer(used, false);
        submit([&](VkCommandBuffer cb) {
            prior_writes(cb);
            VkBufferCopy r{0, 0, used};
            vkCmdCopyBuffer(cb, b.buf, n.buf, 1, &r);
        });
        release(b);
        b = n;
    }

    template <class F>
    void submit(F record) {
        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        prior_writes(cmd);
        record(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkResetFences(device, 1, &fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    }

    static void barrier(VkCommandBuffer c, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst, VkAccessFlags da) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = sa;
        mb.dstAccessMask = da;
        vkCmdPipelineBarrier(c, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    static void compute_barrier(VkCommandBuffer c) {
        barrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    static void transfer_to_compute(VkCommandBuffer c) {
        barrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    static void compute_to_host(VkCommandBuffer c) {
        barrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT);
    }
    // Makes the writes of the previous submissions visible to this one.
    static void prior_writes(VkCommandBuffer c) {
        barrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    static void transfer_to_host(VkCommandBuffer c) {
        barrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    }

    // Copies host data into a device buffer through the staging buffer.
    void upload(Buffer& dst, const void* data, size_t bytes) {
        ensure(dst, bytes);
        if (bytes == 0) return;
        if (!staging.buf || staging.size < bytes) { release(staging); staging = make_buffer(bytes, true); }
        memcpy(staging.map, data, bytes);
        submit([&](VkCommandBuffer c) {
            VkBufferCopy r{0, 0, bytes};
            vkCmdCopyBuffer(c, staging.buf, dst.buf, 1, &r);
        });
    }

    void ensure_readback(size_t bytes) {
        if (readback.buf && readback.size >= bytes) return;
        release(readback);
        readback = make_buffer(bytes + bytes / 4, true, true);
    }

    void download(Buffer& src, void* data, size_t bytes) {
        if (bytes == 0) return;
        ensure_readback(bytes);
        submit([&](VkCommandBuffer c) {
            VkBufferCopy r{0, 0, bytes};
            vkCmdCopyBuffer(c, src.buf, readback.buf, 1, &r);
            transfer_to_host(c);
        });
        memcpy(data, readback.map, bytes);
    }

    VkPipeline compute_pipeline(VkShaderModule m, VkPipelineLayout l, int stack) {
        VkSpecializationMapEntry me{0, 0, sizeof(int32_t)};
        const int32_t value = stack;
        VkSpecializationInfo spec{1, &me, sizeof(value), &value};
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = m;
        cpi.stage.pName = "main";
        cpi.stage.pSpecializationInfo = &spec;
        cpi.layout = l;
        VkPipeline p;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &p));
        return p;
    }

    // All pipelines, specialized for a stack depth that covers every scene in use.
    void make_pipelines(int stack) {
        if (pipeline && pipeline_stack >= stack) return;
        vkDeviceWaitIdle(device);
        for (VkPipeline* p : {&pipeline, &pipeline_wide, &pipeline_iv, &pipeline_wide_iv, &scan_pipe, &scan_add_pipe, &grid_pipe, &dyn_pipe,
                              &dyn_wide_pipe, &dyn_value_pipe, &dyn_args_pipe, &probe_pipe})
            if (*p) vkDestroyPipeline(device, *p, nullptr);
        for (VkPipeline& p : dc) if (p) vkDestroyPipeline(device, p, nullptr);
        pipeline = compute_pipeline(module, layout, stack);
        pipeline_wide = compute_pipeline(module_wide, layout, stack);
        pipeline_iv = compute_pipeline(module_iv, layout, stack);
        pipeline_wide_iv = compute_pipeline(module_wide_iv, layout, stack);
        scan_pipe = compute_pipeline(scan_module, dc_layout, stack);
        scan_add_pipe = compute_pipeline(scan_add_module, dc_layout, stack);
        grid_pipe = compute_pipeline(grid_module, dc_layout, stack);
        dyn_pipe = compute_pipeline(dyn_module, dc_layout, stack);
        dyn_wide_pipe = compute_pipeline(dyn_wide_module, dc_layout, stack);
        dyn_value_pipe = compute_pipeline(dyn_value_module, dc_layout, stack);
        dyn_args_pipe = compute_pipeline(dyn_args_module, dc_layout, stack);
        probe_pipe = compute_pipeline(probe_module, dc_layout, stack);
        for (int s = 0; s < kDcStageCount; s++) dc[s] = compute_pipeline(dc_modules[s], dc_layout, stack);
        pipeline_stack = stack;
    }

    void bind(SceneBufs& sb, Buffer& in_cells, Buffer& in_pool, Buffer& out_cells, Buffer& out_pool, Buffer& out_far) {
        Buffer* bufs[kBindings] = {&sb.nodes, &sb.prims, &sb.ops, &in_cells, &in_pool, &out_cells, &out_pool, &scratch, &counters, &values, &out_far};
        VkDescriptorBufferInfo info[kBindings];
        VkWriteDescriptorSet w[kBindings];
        for (uint32_t i = 0; i < kBindings; i++) {
            info[i] = VkDescriptorBufferInfo{bufs[i]->buf, 0, VK_WHOLE_SIZE};
            w[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = dset;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &info[i];
        }
        vkUpdateDescriptorSets(device, kBindings, w, 0, nullptr);
    }

    // ------------------------------------------------------------------ prefix sums
    // Sizes the temporary buffers of the prefix sums of up to n values; called before recording.
    void prepare_scan(uint64_t n) {
        for (size_t lvl = 0;; lvl++) {
            const uint64_t blocks = (n + 1 + 1023) / 1024;
            if (scan_sums.size() <= lvl) { scan_sums.emplace_back(); scan_out.emplace_back(); }
            ensure(scan_sums[lvl], blocks * 4);
            ensure(scan_out[lvl], (blocks + 1) * 4);
            if (blocks <= 1) break;
            n = blocks;
        }
    }

    // out[0..n] = exclusive prefix sums of data[0..n) (bit counts of the words with mode 1).
    void scan(VkCommandBuffer cb, VkDeviceAddress data, VkDeviceAddress out, uint32_t n, uint32_t mode, size_t lvl = 0) {
        const uint32_t blocks = (n + 1 + 1023) / 1024;
        ScanPC pc{data, out, scan_sums[lvl].addr, n, mode};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scan_pipe);
        vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, groups(blocks), 1, 1);
        compute_barrier(cb);
        if (blocks > 1) {
            scan(cb, scan_sums[lvl].addr, scan_out[lvl].addr, blocks, 0, lvl + 1);
            ScanAddPC ap{out, scan_out[lvl].addr, n, 0};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scan_add_pipe);
            vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ap), &ap);
            vkCmdDispatch(cb, groups(((uint64_t)n + 1 + 255) / 256), 1, 1);
            compute_barrier(cb);
        }
    }

    // One invocation per element (or, with groups, one workgroup per element).
    void dc_dispatch(VkCommandBuffer cb, DcStage s, uint32_t count, uint32_t mode = 0, bool groups = false) {
        if (count == 0) return;
        DcPC pc{dc_params.addr, count, mode};
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dc[s]);
        vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, this->groups(groups ? count : ((uint64_t)count + kGroup - 1) / kGroup), 1, 1);
    }

    // ------------------------------------------------------------------ scene
    // Uploads the nodes, primitives and operators of `data` and the full tree of `tree`, whose
    // nodes are a prefix of those of `data` (the static scene of a dynamic one).
    void upload_scene(const Scene& tree, const Scene& data, SceneBufs& sb) {
        std::vector<uint32_t> nd(2 * std::max<size_t>(data.nodes.size(), 1));
        int sp = 0, max_stack = 1;
        for (size_t i = 0; i < data.nodes.size(); i++) {
            nd[2 * i] = (uint32_t)data.nodes[i].type;
            nd[2 * i + 1] = (uint32_t)data.nodes[i].idx;
            sp += data.nodes[i].type == NODE_PRIM ? 1 : -1;
            max_stack = std::max(max_stack, sp);
        }
        std::vector<double> pr(24 * std::max<size_t>(data.prims.size(), 1), 0.0);
        for (size_t i = 0; i < data.prims.size(); i++) prim_values(data.prims[i], &pr[24 * i]);
        // 0.25 / k is the division of blend_kernel, done once with the same result.
        std::vector<double> op(4 * std::max<size_t>(data.ops.size(), 1), 0.0);
        for (size_t i = 0; i < data.ops.size(); i++) {
            op[4 * i] = data.ops[i].k;
            op[4 * i + 1] = data.ops[i].s;
            op[4 * i + 2] = data.ops[i].k != 0.0 ? 0.25 / data.ops[i].k : 0.0;
        }
        sb.stack = max_stack + 2;  // + 2: the sphere and the union root of a dynamic tree
        make_pipelines(sb.stack);
        upload(sb.nodes, nd.data(), nd.size() * 4);
        upload(sb.prims, pr.data(), pr.size() * 8);
        upload(sb.ops, op.data(), op.size() * 8);
        // The tree entries keep the layout of TreeEntry on the GPU, so they are copied as they are.
        static_assert(sizeof(TreeEntry) == 12 && offsetof(TreeEntry, parent) == 4 && offsetof(TreeEntry, sign) == 8, "TreeEntry layout");
        const std::vector<TreeEntry> ft = full_tree(tree);
        upload(sb.full_pool, ft.data(), ft.size() * sizeof(TreeEntry));
        sb.full_n = (uint32_t)ft.size();
    }

    static void prim_values(const Primitive& P, double* v) {
        v[0] = P.type;
        for (int k = 0; k < 12; k++) v[1 + k] = P.m[k];
        for (int k = 0; k < 4; k++) v[13 + k] = P.data[k];
        for (int k = 0; k < 4; k++) v[17 + k] = P.corner_rounding[k];
        for (int k = 0; k < 2; k++) v[21 + k] = P.extrude_rounding[k];
        v[23] = 0.0;
    }

    // With coarse, the cells of the levels max(min_depth, depth - 5) to depth - 1 whose tree is a
    // single primitive are probed (probe.comp), and those that become coarse leaves get no children.
    GpuLevels build_levels(const Scene& scene, const OctreeParams& params, Counters& evals, GpuStats& st, SceneBufs& sb,
                           StaticLevels* keep, const CoarseRun* coarse = nullptr);
    // coarse_static: the static mesh of a dynamic scene with coarse leaves (dyn_coarse), whose leaves
    // are all of the finest level: the coarse ones replace them (FACES mode 1), their vertices after
    // the finest ones and their faces after those of the finest edges, in F.
    void run_dc(const Scene& scene, SceneBufs& sb, Buffer& leaf_cells, uint32_t n_cells, Buffer& tree_pool, bool full,
                const double size[3], int depth, const DCParams& dcp, DcBufs& b, GpuMeshStats& ms, Mesh* out,
                bool coarse_static = false);
    void dyn_setup(const DynamicScene& d, int depth, double margin, const DCParams& dcp, Counters& c, const CoarseParams* coarse);
    // The buffers of the dynamic update and of the dual contouring.
    void release_dynamic() {
        vkDeviceWaitIdle(device);
        for (DcBufs* x : {&dc_static, &dc_main}) for (Buffer* y : x->all()) release(*y);
        for (Buffer* y : {&dyn_cells[0], &dyn_cells[1], &dyn_pool[0], &dyn_pool[1], &dyn_scratch, &dyn_values, &dyn_culled, &st_coarse_bits,
                          &st_coarse_prefix, &st_coarse_records, &dyn_dissolved})
            release(*y);
        for (std::vector<Buffer>* v : {&st_levels.cells, &st_levels.far, &st_levels.pool, &st_levels.grid, &st_levels.value, &st_levels.far_value})
            for (Buffer& y : *v) release(y);
        release_build();
        release_coarse_build();
        dyn_ready = false;
        dyn_coarse = false;
        frame_clean = false;
    }
    // The buffers of a build with coarse leaves: the probe and its records, and the coarse leaves
    // themselves (a dynamic scene keeps a copy in st_coarse_*). The next build ensures them again.
    void release_coarse_build() {
        for (Buffer* x : {&probe_passed, &probe_edges, &probe_cross, &coarse_bits, &coarse_prefix, &coarse_records}) release(*x);
        for (Buffer& x : coarse_level_records) release(x);
    }
    // mode: 0 the whole mesh to out, 1 only the new vertices (and all faces), 2 nothing (last_mesh).
    void dyn_frame(const double center[3], Counters& c, GpuFrameStats& fs, int mode, Mesh& out);
    GpuMeshBuffers last_mesh;
    DynLevel level_params(int l) const;
    // The temporaries of the octree build, sized for its largest level.
    void release_build() { for (Buffer* x : {&scratch, &values, &far, &cells[0], &cells[1], &pool[0], &pool[1]}) release(*x); }
};

GpuLevels GpuPruner::Impl::build_levels(const Scene& scene, const OctreeParams& params, Counters& evals, GpuStats& st,
                                        SceneBufs& sb, StaticLevels* keep, const CoarseRun* coarse) {
    const bool full = params.oracle == ORACLE_FULL, interval = params.oracle == ORACLE_INTERVAL;
    // cull_factor > 1, the octree of dense sampling: the cells within cull_factor * R are kept with
    // their trees, as in build_octree, and the far leaves (constants, without trees) are dropped.
    const bool dense = params.cull_factor != 1.0;
    if ((!full && !interval && params.oracle != ORACLE_LIPSCHITZ) || !params.inherit ||
        (dense && (params.oracle != ORACLE_LIPSCHITZ || params.cull_factor < 1.0 || keep || coarse || params.keep_empty_leaves)) ||
        (params.keep_empty_leaves && (full || interval || !keep)))
        throw std::runtime_error("the GPU build supports the inherited Lipschitz or interval pruning (or none), with far leaves "
                                 "only in the dense octree of Lipschitz pruning");
    Timer t_setup;
    st = GpuStats{};
    st.device = name;
    GpuLevels g;
    g.full = full;
    g.oct.scene = &scene;
    g.oct.params = params;
    g.oct.full = full_tree(scene);

    // Level 0, the bounding box, pruned on the CPU as in build_octree.
    double size[3], R = 0.0, c[3];
    for (int a = 0; a < 3; a++) {
        size[a] = scene.aabb_max[a] - scene.aabb_min[a];
        c[a] = scene.aabb_min[a] + 0.5 * size[a];
        R += 0.25 * size[a] * size[a];
    }
    R = std::sqrt(R);
    std::vector<TreeEntry> root_pool;
    uint32_t root_n = sb.full_n;
    if (interval) {
        Scratch sc;
        Interval box[3];
        for (int a = 0; a < 3; a++) {
            const double m = params.margin * 0.5 * size[a];
            box[a] = Interval(scene.aabb_min[a] - m, scene.aabb_max[a] + m);
        }
        prune_interval(scene, TreeView{g.oct.full.data(), (uint32_t)g.oct.full.size()}, box, root_pool, root_n, sc, evals);
    } else if (!full) {
        Scratch sc;
        prune_lipschitz(scene, TreeView{g.oct.full.data(), (uint32_t)g.oct.full.size()}, c, R * (1.0 + params.margin), root_pool,
                        root_n, sc, evals);
    } else {
        Scratch sc;
        eval_tree(scene, TreeView{g.oct.full.data(), (uint32_t)g.oct.full.size()}, c, sc, evals);
    }
    {
        Level L0;
        for (int a = 0; a < 3; a++) L0.size[a] = size[a];
        L0.tested = 1;
        L0.sum_tree_n = L0.max_tree_n = root_n;
        L0.single_prim = root_n == 1;
        L0.num_cells = 1;
        g.oct.levels.push_back(std::move(L0));
    }
    GpuCell gc{{0, 0, 0, 0}, {0, root_n, 0, 0}};
    Buffer *in_cells = &cells[0], *in_pool = &pool[0];
    if (keep) {
        // A previous static octree (another scene) gives its memory back first.
        for (std::vector<Buffer>* v : {&keep->cells, &keep->far, &keep->pool, &keep->grid, &keep->value, &keep->far_value})
            for (Buffer& x : *v) release(x);
        *keep = StaticLevels{};
        keep->cells.resize(params.max_depth + 1);
        keep->far.resize(params.max_depth + 1);
        keep->pool.resize(params.max_depth + 1);
        keep->n_cells.assign(params.max_depth + 1, 0);
        keep->n_far.assign(params.max_depth + 1, 0);
        keep->n_entries.assign(params.max_depth + 1, 0);
        keep->size.assign(3 * (params.max_depth + 1), 0.0);
        for (int a = 0; a < 3; a++) keep->size[a] = size[a];
        keep->root_pool = root_pool;
        in_cells = &keep->cells[0];
        in_pool = &keep->pool[0];
        ensure(keep->far[0], 16);
        keep->n_cells[0] = 1;
        keep->n_entries[0] = root_pool.size();
    }
    upload(*in_cells, &gc, sizeof(gc));
    if (!full) upload(*in_pool, root_pool.data(), root_pool.size() * sizeof(TreeEntry));
    else ensure(*in_pool, 16);
    uint64_t n_cells = 1, n_entries = full ? 0 : root_pool.size(), parent_max = root_n;
    ensure(values, 16);
    ensure(scratch, 16);
    ensure(far, 16);
    for (Buffer& p : pool) ensure(p, 16);
    // Coarse leaves: a bitset per level, the probe counters, and the first level probed.
    coarse_on = coarse != nullptr;
    n_coarse = 0;
    coarse_found.assign(params.max_depth + 1, 0);
    int probe_min = params.max_depth;
    double probe_fd_h = 0.0;
    if (coarse) {
        if (keep || full || params.max_depth > 9) throw std::runtime_error("coarse leaves need a pruned build of depth <= 9");
        probe_min = std::max(coarse->min_depth, params.max_depth - 5);
        coarse_min = (uint32_t)probe_min;
        coarse_words = 0;
        for (int l = 0; l < params.max_depth; l++) {
            coarse_word_off[l] = coarse_words;
            coarse_words += (uint32_t)(((1ull << (3 * l)) + 31) / 32);
        }
        ensure_exact(coarse_bits, (uint64_t)coarse_words * 4);
        ensure(coarse_prefix, ((uint64_t)coarse_words + 1) * 4);
        if ((int)coarse_level_records.size() < params.max_depth) coarse_level_records.resize(params.max_depth);
        // The step of the finest level, as the finite differences of its leaves.
        probe_fd_h = coarse->fd_rel * std::min(size[0] * std::ldexp(1.0, -params.max_depth),
                                               std::min(size[1] * std::ldexp(1.0, -params.max_depth),
                                                        size[2] * std::ldexp(1.0, -params.max_depth)));
    }
    st.t_setup = t_setup.seconds();

    Timer t_levels;
    int level = 0, step = 0, ping = 1;
    while (level < params.max_depth) {
        Timer lt;
        // Every level subdivides 2 x 2 x 2. The 4 x 4 x 4 of the first levels of the paper was 15 to
        // 40 % slower in double precision on the three official scenes: the 64 children of a cell are
        // pruned from its larger tree, and the sequential walks over the trees, which dominate, do not
        // get shorter.
        const uint64_t per = 8;
        Level L;
        for (int a = 0; a < 3; a++) { size[a] *= 0.5; L.size[a] = size[a]; }
        const double r = 0.5 * std::sqrt(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]);
        L.tested = per * n_cells;
        const int next = level + 1;
        Buffer& out_cells = keep ? keep->cells[next] : cells[ping];
        Buffer& out_pool = keep ? keep->pool[next] : pool[ping];
        Buffer& out_far = keep ? keep->far[next] : far;
        PruneCounters cnt{};
        if (n_cells > 0) {
            ensure(out_cells, per * n_cells * sizeof(GpuCell));
            if (keep) ensure(out_far, per * n_cells * sizeof(GpuCell));
            const bool wide = !full && parent_max > kWide;
            if (full) {
                bind(sb, *in_cells, sb.full_pool, out_cells, sb.full_pool, far);
            } else {
                ensure(out_pool, per * n_entries * sizeof(TreeEntry));
                ensure(scratch, per * n_entries * 8);
                if (wide) ensure(values, per * n_entries * (interval ? 2 : 1) * sizeof(double));
                bind(sb, *in_cells, *in_pool, out_cells, out_pool, out_far);
            }
            // The probe of this level, before its cells are subdivided.
            const bool probe = coarse && level >= probe_min;
            ProbeParams pp{};
            if (probe) {
                ensure(coarse_level_records[level], n_cells * kProbeRecord);
                ensure(probe_passed, n_cells * 16);
                ensure(probe_edges, (uint64_t)probe_edge_cap * 8);
                ensure(probe_cross, (uint64_t)probe_edge_cap * 48);
                const double* ls = g.oct.levels[level].size;
                for (int a = 0; a < 3; a++) { pp.aabb_min[a] = scene.aabb_min[a]; pp.size[a] = ls[a]; }
                pp.fd_h = probe_fd_h;
                pp.tol = coarse->tol * std::min(ls[0], std::min(ls[1], ls[2]));
                pp.P = 1u << (params.max_depth - level);
                pp.level = (uint32_t)level;
                pp.n_cells = (uint32_t)n_cells;
                pp.max_iters = (uint32_t)coarse->max_iters;
                pp.cells = in_cells->addr;
                pp.tree_pool = in_pool->addr;
                pp.nodes = sb.nodes.addr; pp.prims = sb.prims.addr; pp.ops = sb.ops.addr;
                pp.bits = coarse_bits.addr + 4ull * coarse_word_off[level];
                pp.records = coarse_level_records[level].addr;
                pp.counters = probe_counters.addr;
                pp.passed = probe_passed.addr;
                pp.edges = probe_edges.addr;
                pp.cross = probe_cross.addr;
                pp.level_counts = probe_level_counts.addr;
                pp.edge_cap = probe_edge_cap;
            }
            PrunePC pc{};
            for (int a = 0; a < 3; a++) { pc.aabb_min[a] = scene.aabb_min[a]; pc.size[a] = size[a]; }
            pc.r_keep = r * std::max(1.0 + 1e-9, params.cull_factor);  // as the tests of build_octree
            pc.r_prune = r * (1.0 + params.margin);
            pc.n_parents = (uint32_t)n_cells;
            pc.wide = full ? UINT32_MAX : kWide;
            pc.full = full ? 1 : 0;
            pc.keep_empty = params.keep_empty_leaves ? 1 : 0;
            pc.margin = params.margin;
            const uint32_t q = 2 * (uint32_t)step;
            if (q + 1 >= kMaxQueries / 2) throw std::runtime_error("too many levels for the query pool");
            submit([&](VkCommandBuffer cb) {
                vkCmdResetQueryPool(cb, queries, q, 2);
                vkCmdFillBuffer(cb, counters.buf, 0, VK_WHOLE_SIZE, 0);
                if (coarse && step == 0) {
                    vkCmdFillBuffer(cb, coarse_bits.buf, 0, VK_WHOLE_SIZE, 0);
                    vkCmdFillBuffer(cb, probe_counters.buf, 0, VK_WHOLE_SIZE, 0);
                }
                if (probe) {
                    vkCmdUpdateBuffer(cb, probe_params.buf, 0, sizeof(pp), &pp);
                    vkCmdFillBuffer(cb, probe_level_counts.buf, 0, VK_WHOLE_SIZE, 0);
                }
                transfer_to_compute(cb);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, q);
                if (probe) {
                    // The four passes of probe.comp; the coarse leaves are flagged before prune.comp
                    // reads the cells.
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, probe_pipe);
                    auto pass = [&](uint32_t mode) {
                        DcPC ppc{probe_params.addr, (uint32_t)n_cells, mode};
                        vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ppc), &ppc);
                    };
                    pass(0);
                    vkCmdDispatch(cb, groups(n_cells), 1, 1);
                    compute_barrier(cb);
                    pass(1);
                    vkCmdDispatch(cb, 1, 1, 1);
                    indirect_barrier(cb);
                    pass(2);
                    vkCmdDispatchIndirect(cb, probe_level_counts.buf, 8);
                    compute_barrier(cb);
                    pass(3);
                    vkCmdDispatchIndirect(cb, probe_level_counts.buf, 20);
                    compute_barrier(cb);
                }
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, interval ? pipeline_iv : pipeline);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
                vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, groups((per * n_cells + kGroup - 1) / kGroup), 1, 1);
                // The two kernels write disjoint scratch ranges and share only the atomic counters.
                if (wide) {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, interval ? pipeline_wide_iv : pipeline_wide);
                    vkCmdDispatch(cb, groups(per * n_cells), 1, 1);
                }
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, q + 1);
                compute_to_host(cb);
                VkBufferCopy rg{0, 0, sizeof(PruneCounters)};
                vkCmdCopyBuffer(cb, counters.buf, counters_host.buf, 1, &rg);
                if (probe) {
                    VkBufferCopy rl{0, 0, 8};
                    vkCmdCopyBuffer(cb, probe_level_counts.buf, probe_level_host.buf, 1, &rl);
                }
                transfer_to_host(cb);
            });
            memcpy(&cnt, counters_host.map, sizeof(cnt));
            if (probe) {
                uint32_t lc[2];
                memcpy(lc, probe_level_host.map, sizeof(lc));
                if (lc[0] > probe_edge_cap) {
                    // Some cells of this level did not fit in the edge list: build again with room for them.
                    probe_edge_cap = lc[0] + lc[0] / 4;
                    g.retry = true;
                    return g;
                }
            }
            uint64_t ts[2];
            VK_CHECK(vkGetQueryPoolResults(device, queries, q, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const double gpu_s = (double)(ts[1] - ts[0]) * timestamp_period * 1e-9;
            st.level_gpu.push_back(gpu_s);
            st.t_gpu += gpu_s;
        } else if (keep) {
            ensure(out_cells, 16);
            ensure(out_pool, 16);
            ensure(out_far, 16);
        }
        step++;
        (interval ? evals.prim_ivals : evals.prim_evals) += cnt.prim_evals;
        (interval ? evals.op_ivals : evals.op_evals) += cnt.op_evals;
        n_cells = cnt.cells;
        n_entries = full ? 0 : cnt.entries;
        parent_max = cnt.max_tree;
        // The far leaves of a dense octree are dropped without being counted: every child tested is a
        // cell there, near or far, as in build_octree.
        L.far_cells = dense ? L.tested - n_cells : cnt.far_cells;
        L.num_cells = n_cells + L.far_cells;
        L.sum_tree_n = full ? n_cells * sb.full_n : cnt.entries;
        L.max_tree_n = cnt.max_tree;
        L.single_prim = cnt.single_prim;
        L.seconds = lt.seconds();
        g.oct.levels.push_back(std::move(L));
        if (keep) {
            shrink(out_cells, (VkDeviceSize)cnt.cells * sizeof(GpuCell));
            shrink(out_far, (VkDeviceSize)cnt.far_cells * sizeof(GpuCell));
            if (!full) shrink(out_pool, (VkDeviceSize)cnt.entries * sizeof(TreeEntry));
            keep->n_cells[next] = cnt.cells;
            keep->n_far[next] = cnt.far_cells;
            keep->n_entries[next] = cnt.entries;
            for (int a = 0; a < 3; a++) keep->size[3 * next + a] = size[a];
        }
        level = next;
        in_cells = &out_cells;
        in_pool = &out_pool;
        if (!keep) ping = 1 - ping;
    }
    if (coarse) {
        // The coarse leaves found, their records in one buffer (level by level), and the ranks of
        // their bitsets.
        submit([&](VkCommandBuffer cb) {
            VkBufferCopy r{0, 0, sizeof(ProbeCounters)};
            vkCmdCopyBuffer(cb, probe_counters.buf, probe_host.buf, 1, &r);
            transfer_to_host(cb);
        });
        ProbeCounters pcnt;
        memcpy(&pcnt, probe_host.map, sizeof(pcnt));
        evals.prim_evals += pcnt.prim_evals;
        evals.op_evals += pcnt.op_evals;
        for (int l = probe_min; l < params.max_depth; l++) { coarse_found[l] = pcnt.found[l]; n_coarse += pcnt.found[l]; }
        ensure(coarse_records, (uint64_t)n_coarse * kProbeRecord);
        prepare_scan(coarse_words);
        submit([&](VkCommandBuffer cb) {
            VkDeviceSize off = 0;
            for (int l = probe_min; l < params.max_depth; l++) {
                if (coarse_found[l] == 0) continue;
                VkBufferCopy r{0, off, coarse_found[l] * kProbeRecord};
                vkCmdCopyBuffer(cb, coarse_level_records[l].buf, coarse_records.buf, 1, &r);
                off += r.size;
            }
            transfer_to_compute(cb);
            scan(cb, coarse_bits.addr, coarse_prefix.addr, coarse_words, 1);
        });
    }
    st.t_levels = t_levels.seconds();
    g.cells = in_cells;
    g.pool = full ? &sb.full_pool : in_pool;
    g.n_cells = n_cells;
    g.n_entries = n_entries;
    return g;
}

// Dual contouring of the leaves `leaf_cells` (8 uints each) of depth `depth` into the buffers b;
// with `out` the mesh is copied to the host.
void GpuPruner::Impl::run_dc(const Scene& scene, SceneBufs& sb, Buffer& leaf_cells, uint32_t L, Buffer& tree_pool, bool full,
                             const double size[3], int depth, const DCParams& dcp, DcBufs& b, GpuMeshStats& ms, Mesh* out,
                             bool coarse_static) {
    Timer t_a;
    if (&b == &dc_main) frame_clean = false;
    const uint32_t N = 1u << depth, N1 = N + 1;
    const uint64_t cells3 = (uint64_t)N * N * N, corners = (uint64_t)N1 * N1 * N1, epos = 3 * corners;
    const uint32_t wc = (uint32_t)((cells3 + 31) / 32), we = (uint32_t)((epos + 31) / 32);
    ensure_exact(b.cell_grid, cells3 * 4);
    ensure_exact(b.cell_bits, (uint64_t)wc * 4);
    ensure_exact(b.cell_prefix, ((uint64_t)wc + 1) * 4);
    ensure(b.leaves, (uint64_t)L * 4);
    ensure_exact(b.corner_val, corners * 8);
    ensure_exact(b.edge_bits, (uint64_t)we * 4);
    ensure_exact(b.edge_prefix, ((uint64_t)we + 1) * 4);
    ensure(b.comp, (uint64_t)L * 8);
    ensure(b.ncomp, (uint64_t)L * 4);
    ensure(b.vbase, ((uint64_t)L + 1) * 4);
    prepare_scan(std::max<uint64_t>({wc, we, L}));

    DcParams P{};
    for (int a = 0; a < 3; a++) { P.aabb_min[a] = scene.aabb_min[a]; P.size[a] = size[a]; }
    P.fd_h = dcp.fd_rel * std::min(size[0], std::min(size[1], size[2]));
    P.N = N;
    P.n_cells = L;
    P.max_iters = (uint32_t)dcp.max_iters;
    P.full_n = full ? sb.full_n : 0;
    P.nodes = sb.nodes.addr; P.prims = sb.prims.addr; P.ops = sb.ops.addr;
    P.leaf_cells = leaf_cells.addr;
    P.tree_pool = full ? sb.full_pool.addr : tree_pool.addr;
    P.cell_grid = b.cell_grid.addr; P.cell_bits = b.cell_bits.addr; P.cell_prefix = b.cell_prefix.addr; P.leaves = b.leaves.addr;
    P.corner_val = b.corner_val.addr; P.edge_bits = b.edge_bits.addr; P.edge_prefix = b.edge_prefix.addr;
    P.comp = b.comp.addr; P.ncomp = b.ncomp.addr; P.vbase = b.vbase.addr;
    P.counters = dc_counters.addr;
    P.depth = (uint32_t)depth;
    // The coarse leaves of the last build, when it made them (only mesh() asks for them).
    const bool with_coarse = coarse_on && &b == &dc_main;
    if (with_coarse) {
        P.coarse_on = 1;
        P.coarse_min = coarse_min;
        P.n_coarse = n_coarse;
        P.coarse_bits = coarse_bits.addr;
        P.coarse_prefix = coarse_prefix.addr;
        P.coarse_records = coarse_records.addr;
        for (int l = 0; l < 12; l++) P.coarse_word_off[l] = coarse_word_off[l];
    }
    if (coarse_static) {
        P.coarse_on = 1;
        P.coarse_min = dyn_coarse_min;
        P.n_coarse = dyn_n_coarse;
        P.coarse_bits = st_coarse_bits.addr;
        P.coarse_prefix = st_coarse_prefix.addr;
        P.coarse_records = st_coarse_records.addr;
        for (int l = 0; l < 12; l++) P.coarse_word_off[l] = dyn_coarse_word_off[l];
    }
    P.project = with_coarse || !dcp.project_vertices ? 0 : 1;
    const uint32_t nc = with_coarse ? n_coarse : coarse_static ? dyn_n_coarse : 0;
    // Marching cubes on the leaves: the vertices are the crossings, and COMP counts the triangles.
    const bool mc = dcp.marching_cubes;
    if (mc && (with_coarse || coarse_static)) throw std::runtime_error("marching cubes has no coarse leaves");
    P.mc = mc ? 1 : 0;

    // Timestamps after each stage, in the second half of the query pool.
    const uint32_t q0 = kMaxQueries / 2;
    std::vector<const char*> names;
    auto stamp = [&](VkCommandBuffer cb, const char* n) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, q0 + (uint32_t)names.size());
        names.push_back(n);
    };
    auto read_stamps = [&](uint32_t first, uint32_t count) {
        std::vector<uint64_t> ts(count);
        VK_CHECK(vkGetQueryPoolResults(device, queries, q0 + first, count, count * 8, ts.data(), 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        for (uint32_t k = 1; k < count; k++)
            ms.stage_gpu.emplace_back(names[first + k], (double)(ts[k] - ts[k - 1]) * timestamp_period * 1e-9);
    };

    // First submission: grids, corners, crossing edges, components, and the counts.
    submit([&](VkCommandBuffer cb) {
        vkCmdResetQueryPool(cb, queries, q0, kMaxQueries / 2);
        stamp(cb, "start");
        vkCmdUpdateBuffer(cb, dc_params.buf, 0, sizeof(P), &P);
        vkCmdFillBuffer(cb, b.cell_grid.buf, 0, cells3 * 4, 0);
        vkCmdFillBuffer(cb, b.cell_bits.buf, 0, (uint64_t)wc * 4, 0);
        vkCmdFillBuffer(cb, b.edge_bits.buf, 0, (uint64_t)we * 4, 0);
        vkCmdFillBuffer(cb, dc_counters.buf, 0, VK_WHOLE_SIZE, 0);
        transfer_to_compute(cb);
        dc_dispatch(cb, SCATTER, L);
        compute_barrier(cb);
        scan(cb, b.cell_bits.addr, b.cell_prefix.addr, wc, 1);
        dc_dispatch(cb, COMPACT, wc, 0);
        stamp(cb, "leaf index");
        dc_dispatch(cb, CORNERS, (uint32_t)corners);
        compute_barrier(cb);
        stamp(cb, "corners");
        dc_dispatch(cb, EDGES, (uint32_t)epos);
        compute_barrier(cb);
        scan(cb, b.edge_bits.addr, b.edge_prefix.addr, we, 1);
        stamp(cb, "crossing edges");
        dc_dispatch(cb, COMP, L);
        compute_barrier(cb);
        scan(cb, b.ncomp.addr, b.vbase.addr, L, 0);
        stamp(cb, "components");
        compute_to_host(cb);
        VkBufferCopy r0{(VkDeviceSize)we * 4, 0, 4}, r1{(VkDeviceSize)L * 4, 4, 4};
        vkCmdCopyBuffer(cb, b.edge_prefix.buf, totals.buf, 1, &r0);
        vkCmdCopyBuffer(cb, b.vbase.buf, totals.buf, 1, &r1);
        transfer_to_host(cb);
    });
    uint32_t tot[2];
    memcpy(tot, totals.map, sizeof(tot));
    // With marching cubes the scan of COMP gives the triangles, and the vertices are the crossings.
    const uint32_t E = tot[0], n_tri = mc ? tot[1] : 0, nv_fine = mc ? E : tot[1], nv = nv_fine + nc;
    P.nv_fine = nv_fine;
    read_stamps(0, (uint32_t)names.size());
    const uint32_t first_b = (uint32_t)names.size();
    ms.t_index = t_a.seconds();

    // Second submission: crossings, vertices, faces, and the mesh to the host.
    Timer t_b;
    ensure(b.edges, (uint64_t)E * 4);
    ensure(b.crossings, mc ? 16 : (uint64_t)E * 6 * 8);
    // With coarse_static the faces of the coarse leaves follow those of the finest edges in F.
    const uint64_t slots = (uint64_t)E + (coarse_static ? 12ull * nc : 0);
    ensure(b.F, mc ? (uint64_t)n_tri * 3 * 4 : slots * 6 * 4);
    ensure(b.V, (uint64_t)nv * 3 * 8);
    ensure(b.vtree, (uint64_t)nv * 4);
    if (nc && !coarse_static) ensure(coarse_F, (uint64_t)nc * 12 * 6 * 4);
    P.edges = b.edges.addr; P.crossings = b.crossings.addr; P.F = b.F.addr; P.V = b.V.addr; P.vtree = b.vtree.addr;
    P.coarse_F = coarse_static ? b.F.addr + (uint64_t)E * 6 * 4 : nc ? coarse_F.addr : 0;
    const size_t bytes_v = (size_t)nv * 3 * 8, bytes_t = (size_t)nv * 4, bytes_f = mc ? (size_t)n_tri * 3 * 4 : (size_t)slots * 6 * 4,
                 bytes_cf = coarse_static ? 0 : (size_t)nc * 12 * 6 * 4;
    const size_t bytes_mesh = bytes_v + bytes_t + bytes_f + bytes_cf;
    if (out) ensure_readback(bytes_mesh + sizeof(DcCounters));
    else ensure_readback(sizeof(DcCounters));
    submit([&](VkCommandBuffer cb) {
        stamp(cb, "start");
        vkCmdUpdateBuffer(cb, dc_params.buf, 0, sizeof(P), &P);
        transfer_to_compute(cb);
        dc_dispatch(cb, COMPACT, we, 1);
        compute_barrier(cb);
        stamp(cb, "edge list");
        dc_dispatch(cb, CROSS, E);
        compute_barrier(cb);
        stamp(cb, "crossings");
        if (mc) dc_dispatch(cb, MC_FACES, L);
        else {
            dc_dispatch(cb, QEF, L);
            dc_dispatch(cb, FACES, E, coarse_static ? 1 : 0);
        }
        if (nc) {
            dc_dispatch(cb, COARSE_VERT, nc);
            dc_dispatch(cb, COARSE_FACES, 12 * nc);
        }
        compute_to_host(cb);
        stamp(cb, "vertices and faces");
        VkBufferCopy rv{0, 0, bytes_v}, rt{0, bytes_v, bytes_t}, rf{0, bytes_v + bytes_t, bytes_f},
            rcf{0, bytes_v + bytes_t + bytes_f, bytes_cf}, rc{0, out ? bytes_mesh : 0, sizeof(DcCounters)};
        if (out && bytes_v) vkCmdCopyBuffer(cb, b.V.buf, readback.buf, 1, &rv);
        if (out && bytes_t) vkCmdCopyBuffer(cb, b.vtree.buf, readback.buf, 1, &rt);
        if (out && bytes_f) vkCmdCopyBuffer(cb, b.F.buf, readback.buf, 1, &rf);
        if (out && bytes_cf) vkCmdCopyBuffer(cb, coarse_F.buf, readback.buf, 1, &rcf);
        vkCmdCopyBuffer(cb, dc_counters.buf, readback.buf, 1, &rc);
        transfer_to_host(cb);
        stamp(cb, "copy to host");
    });
    ms.t_mesh = t_b.seconds();
    read_stamps(first_b, (uint32_t)names.size() - first_b);
    b.n_leaves = L;
    b.n_edges = E;
    b.n_vertices = nv;
    b.n_fine_vertices = nv_fine;

    Timer t_down;
    const char* rb = (const char*)readback.map;
    DcCounters dcc;
    memcpy(&dcc, rb + (out ? bytes_mesh : 0), sizeof(dcc));
    if (out) {
        Mesh& mesh = *out;
        mesh = Mesh{};
        mesh.V.resize((size_t)nv * 3);
        memcpy(mesh.V.data(), rb, bytes_v);
        mesh.tree_n.resize(nv);
        memcpy(mesh.tree_n.data(), rb + bytes_v, bytes_t);
        // Two triangles per slot, one when the fourth index is kNone, none when the first is;
        // marching cubes writes its triangles packed.
        const uint32_t* f = (const uint32_t*)(rb + bytes_v + bytes_t);
        if (mc) mesh.F.assign(f, f + (size_t)n_tri * 3);
        else mesh.F.reserve((size_t)(E + 12 * nc) * 6);
        for (uint64_t u = 0; !mc && u < (uint64_t)E + 12ull * nc; u++) {
            if (f[6 * u] == kNone) continue;
            mesh.F.insert(mesh.F.end(), f + 6 * u, f + 6 * u + (f[6 * u + 3] == kNone ? 3 : 6));
        }
        ms.triangles = mesh.F.size() / 3;
    }
    ms.prim_evals = dcc.prim_evals;
    ms.op_evals = dcc.op_evals;
    ms.t_download = t_down.seconds();
    ms.leaves = L;
    ms.coarse_leaves = nc;
    ms.coarse_per_level = with_coarse ? coarse_found : std::vector<uint64_t>{};
    ms.crossings = E;
    ms.vertices = nv;
    ms.ambiguous = dcc.ambiguous;
}

GpuPruner::GpuPruner(const std::vector<std::string>& instance_extensions, bool graphics)
    : impl_(new Impl(instance_extensions, graphics)) {}
GpuHandles GpuPruner::handles() const {
    const Impl& g = *impl_;
    return GpuHandles{(void*)g.instance, (void*)g.phys, (void*)g.device, (void*)g.queue, g.qfam};
}
GpuPruner::~GpuPruner() = default;
std::string GpuPruner::device_name() const { return impl_->name; }
double GpuPruner::init_seconds() const { return impl_->init_s; }

Octree GpuPruner::build(const Scene& scene, const OctreeParams& params, Counters& counters, GpuStats& st) {
    Impl& g = *impl_;
    Timer t_total;
    g.upload_scene(scene, scene, g.scene_main);
    GpuLevels lv = g.build_levels(scene, params, counters, st, g.scene_main, nullptr);
    // The last level to the CPU, in the format of build_octree.
    Timer t_down;
    std::vector<GpuCell> cells(lv.n_cells);
    g.download(*lv.cells, cells.data(), cells.size() * sizeof(GpuCell));
    Level& last = lv.oct.levels.back();
    last.cells.resize(lv.n_cells);
    for (uint64_t i = 0; i < lv.n_cells; i++) {
        Cell& cc = last.cells[i];
        cc.x = cells[i].xyz[0]; cc.y = cells[i].xyz[1]; cc.z = cells[i].xyz[2];
        cc.tree_off = cells[i].tree[0];
        cc.tree_n = cells[i].tree[1];
    }
    if (!lv.full) {
        last.pool.resize(lv.n_entries);
        g.download(*lv.pool, last.pool.data(), lv.n_entries * sizeof(TreeEntry));
    }
    st.t_download = t_down.seconds();
    st.t_total = t_total.seconds();
    lv.oct.build_seconds = st.t_total;
    return std::move(lv.oct);
}

Mesh GpuPruner::mesh(const Scene& scene, const OctreeParams& params, const DCParams& dcp, Counters& c_oct, Counters& c_mesh,
                     GpuStats& st, GpuMeshStats& ms, Octree* levels, const CoarseParams* coarse, bool to_host) {
    Impl& g = *impl_;
    Timer t_total;
    ms = GpuMeshStats{};
    if (params.max_depth > 9) throw std::runtime_error("the GPU dual contouring needs depth <= 9 (dense grids)");
    if (params.cull_factor != 1.0) throw std::runtime_error("the GPU dual contouring needs cull_factor = 1");
    if (dcp.analytic_leaves) throw std::runtime_error("the GPU dual contouring has no analytic leaves");
    Timer t_oct;
    g.upload_scene(scene, scene, g.scene_main);
    Impl::CoarseRun cr;
    if (coarse) {
        cr.tol = coarse->tol;
        cr.min_depth = coarse->min_depth;
        cr.fd_rel = dcp.fd_rel;
        cr.max_iters = dcp.max_iters;
    }
    GpuLevels lv = g.build_levels(scene, params, c_oct, st, g.scene_main, nullptr, coarse ? &cr : nullptr);
    while (lv.retry) {
        c_oct = Counters{};
        lv = g.build_levels(scene, params, c_oct, st, g.scene_main, nullptr, &cr);
    }
    ms.t_octree = t_oct.seconds();
    Mesh mesh;
    g.run_dc(scene, g.scene_main, *lv.cells, (uint32_t)lv.n_cells, *lv.pool, lv.full, lv.oct.levels.back().size, params.max_depth,
             dcp, g.dc_main, ms, to_host ? &mesh : nullptr);
    c_mesh.prim_evals += ms.prim_evals;
    c_mesh.op_evals += ms.op_evals;
    ms.t_total = t_total.seconds();
    if (levels) *levels = std::move(lv.oct);
    return mesh;
}

// ---------------------------------------------------------------------- dynamic scene
void GpuPruner::Impl::dyn_setup(const DynamicScene& d, int depth, double margin, const DCParams& dcp, Counters& c,
                                const CoarseParams* coarse) {
    if (depth > 9) throw std::runtime_error("the GPU dynamic update needs depth <= 9 (dense grids)");
    if (dcp.analytic_leaves) throw std::runtime_error("the GPU dual contouring has no analytic leaves");
    // The buffers of a previous dynamic scene go back first: at depth 9 its grids and octree take
    // several GB, which the build of this one needs.
    release_dynamic();
    dyn = d;
    dyn_depth = depth;
    dyn_margin = margin;
    dyn_dcp = dcp;
    // The static scene T: the nodes before the sphere, its root without the union parent.
    Scene base = dyn.scene;
    base.nodes.resize(dyn.sphere_node);
    base.nodes.back().parent = -1;
    upload_scene(base, dyn.scene, scene_dyn);
    OctreeParams p;
    p.oracle = ORACLE_LIPSCHITZ;
    p.max_depth = depth;
    p.margin = margin;
    GpuStats st;
    if (coarse) {
        // The coarse leaves of the static scene, from a build of their own (that of the dynamic
        // update keeps every finest leaf), copied before the next build reuses their buffers.
        if (dcp.marching_cubes) throw std::runtime_error("coarse leaves are meshed by dual contouring only");
        CoarseRun cr;
        cr.tol = coarse->tol;
        cr.min_depth = coarse->min_depth;
        cr.fd_rel = dcp.fd_rel;
        cr.max_iters = dcp.max_iters;
        Counters cc;
        GpuLevels lv = build_levels(base, p, cc, st, scene_dyn, nullptr, &cr);
        while (lv.retry) {
            cc = Counters{};
            lv = build_levels(base, p, cc, st, scene_dyn, nullptr, &cr);
        }
        c.prim_evals += cc.prim_evals;
        c.op_evals += cc.op_evals;
        dyn_n_coarse = n_coarse;
        dyn_coarse_min = coarse_min;
        for (int l = 0; l < 12; l++) dyn_coarse_word_off[l] = coarse_word_off[l];
        ensure(st_coarse_bits, (uint64_t)coarse_words * 4);
        ensure(st_coarse_prefix, ((uint64_t)coarse_words + 1) * 4);
        ensure(st_coarse_records, std::max<uint64_t>(1, n_coarse) * kProbeRecord);
        submit([&](VkCommandBuffer cb) {
            VkBufferCopy rb{0, 0, (VkDeviceSize)coarse_words * 4}, rp{0, 0, ((VkDeviceSize)coarse_words + 1) * 4},
                rr{0, 0, (VkDeviceSize)n_coarse * kProbeRecord};
            if (rb.size) vkCmdCopyBuffer(cb, coarse_bits.buf, st_coarse_bits.buf, 1, &rb);
            vkCmdCopyBuffer(cb, coarse_prefix.buf, st_coarse_prefix.buf, 1, &rp);
            if (rr.size) vkCmdCopyBuffer(cb, coarse_records.buf, st_coarse_records.buf, 1, &rr);
        });
        release_build();
        release_coarse_build();
        dyn_coarse = true;
    }
    p.keep_empty_leaves = true;
    build_levels(base, p, c, st, scene_dyn, &st_levels);
    // A dense grid per level.
    st_levels.grid.resize(depth + 1);
    for (int l = 0; l <= depth; l++) {
        const uint64_t n = 1ull << l;
        ensure_exact(st_levels.grid[l], n * n * n * 4);
    }
    submit([&](VkCommandBuffer cb) {
        for (int l = 0; l <= depth; l++) {
            const uint64_t n = 1ull << l;
            vkCmdFillBuffer(cb, st_levels.grid[l].buf, 0, n * n * n * 4, 0);
        }
        transfer_to_compute(cb);
        for (int l = 0; l <= depth; l++) {
            const uint32_t total = (uint32_t)(st_levels.n_cells[l] + st_levels.n_far[l]);
            if (total == 0) continue;
            GridPC pc{st_levels.cells[l].addr, st_levels.far[l].addr, st_levels.grid[l].addr, (uint32_t)st_levels.n_cells[l],
                      (uint32_t)st_levels.n_far[l], 1u << l, 0};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, grid_pipe);
            vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, groups(((uint64_t)total + kGroup - 1) / kGroup), 1, 1);
        }
    });
    // The value of every static tree at the center of its cell, as the descent computes the center.
    st_levels.value.resize(depth + 1);
    st_levels.far_value.resize(depth + 1);
    ensure(dyn_level_params, (uint64_t)(depth + 1) * kDynLevelStride);
    submit([&](VkCommandBuffer cb) {
        for (int l = 1; l <= depth; l++) {
            ensure(st_levels.value[l], std::max<uint64_t>(1, st_levels.n_cells[l]) * 8);
            ensure(st_levels.far_value[l], std::max<uint64_t>(1, st_levels.n_far[l]) * 8);
            const DynLevel lp = level_params(l);
            const VkDeviceSize loff = (VkDeviceSize)l * kDynLevelStride;
            vkCmdUpdateBuffer(cb, dyn_level_params.buf, loff, sizeof(lp), &lp);
        }
        transfer_to_compute(cb);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dyn_value_pipe);
        for (int l = 1; l <= depth; l++)
            for (uint32_t far = 0; far < 2; far++) {
                const uint32_t n = (uint32_t)(far ? st_levels.n_far[l] : st_levels.n_cells[l]);
                if (n == 0) continue;
                DynPC pc{dyn_level_params.addr + (VkDeviceSize)l * kDynLevelStride, n, far};
                vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, groups(((uint64_t)n + kGroup - 1) / kGroup), 1, 1);
            }
    });
    // The static dual contouring, kept in dc_static, and its mesh on the CPU.
    GpuMeshStats ms;
    run_dc(base, scene_dyn, st_levels.cells[depth], (uint32_t)st_levels.n_cells[depth], st_levels.pool[depth], false,
           &st_levels.size[3 * depth], depth, dcp, dc_static, ms, &st_mesh, dyn_coarse);
    c.prim_evals += ms.prim_evals;
    c.op_evals += ms.op_evals;
    release_build();  // the frames do not use them
    release(dc_static.cell_grid);  // nor this one (512 MB at depth 9): they find the static leaves by rank
    frame_clean = false;
    dyn_ready = true;
}

// The parameters of level l of the dynamic descent that do not change between frames.
DynLevel GpuPruner::Impl::level_params(int l) const {
    const double* size = &st_levels.size[3 * l];
    DynLevel lp{};
    for (int a = 0; a < 3; a++) { lp.aabb_min[a] = dyn.scene.aabb_min[a]; lp.size[a] = size[a]; }
    const double R = 0.5 * std::sqrt(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]);
    lp.r_keep = R * (1.0 + 1e-9);
    lp.r_prune = R * (1.0 + dyn_margin);
    lp.level = (uint32_t)l;
    lp.sphere_node = dyn.sphere_node;
    lp.root_node = dyn.root_node;
    lp.wide = kDynWide;
    lp.nodes = scene_dyn.nodes.addr; lp.prims = scene_dyn.prims.addr; lp.ops = scene_dyn.ops.addr;
    lp.st_grid = st_levels.grid[l].addr; lp.st_cells = st_levels.cells[l].addr; lp.st_far = st_levels.far[l].addr;
    lp.st_pool = st_levels.pool[l].addr;
    lp.counters = dyn_counters.addr;
    lp.st_value = st_levels.value[l].addr; lp.st_far_value = st_levels.far_value[l].addr;
    return lp;
}

void GpuPruner::Impl::dyn_frame(const double center[3], Counters& c, GpuFrameStats& fs, int mode, Mesh& out) {
    if (!dyn_ready) throw std::runtime_error("dynamic_setup first");
    Timer t_total;
    fs = GpuFrameStats{};
    // The lists of the previous frame tell what to clear only if this frame ends: one cut short by
    // an exception leaves marks they do not describe, and the next frame clears everything.
    bool was_clean = frame_clean;
    frame_clean = false;
    const int D = dyn_depth;
    set_sphere_center(dyn, center);
    const Scene& scene = dyn.scene;

    // The sphere on the GPU.
    double sv[24];
    prim_values(scene.prims[dyn.sphere_prim], sv);

    // Root, on the CPU as in build_dynamic: the static root tree joined with the sphere.
    Timer t_desc;
    std::vector<TreeEntry> in, root_tree;
    {
        TreeView rt{st_levels.root_pool.data(), (uint32_t)st_levels.root_pool.size()};
        in.resize(rt.n + 2);
        for (uint32_t i = 0; i < rt.n; i++) { in[i] = rt.e[i]; if (in[i].parent < 0) in[i].parent = (int32_t)rt.n + 1; }
        in[rt.n] = TreeEntry{dyn.sphere_node, (int32_t)rt.n + 1, 1};
        in[rt.n + 1] = TreeEntry{dyn.root_node, -1, 1};
    }
    double rc[3], R0 = 0.0;
    for (int a = 0; a < 3; a++) { rc[a] = 0.5 * (scene.aabb_min[a] + scene.aabb_max[a]); R0 += 0.25 * st_levels.size[a] * st_levels.size[a]; }
    R0 = std::sqrt(R0);
    uint32_t root_n = 0;
    Scratch sc;
    const double rv = prune_lipschitz(scene, TreeView{in.data(), (uint32_t)in.size()}, rc, R0 * (1.0 + dyn_margin), root_tree, root_n, sc, c);
    fs.t_root = t_desc.seconds();
    bool root_culled = false, root_dynamic = false;
    if (std::fabs(rv) > R0 * (1.0 + 1e-9)) root_culled = true;
    else for (const TreeEntry& e : root_tree) root_dynamic = root_dynamic || e.node == dyn.sphere_node;

    // Descent and the region bits in one submission. Every level is dispatched indirectly, with
    // the arguments that shaders/dyn_args.comp computes on the GPU from the counters of the level
    // before, into buffers of fixed capacities; a frame that does not fit grows them and runs again.
    DcBufs& b = dc_main;
    DcBufs& s = dc_static;
    const uint32_t N = 1u << D, N1 = N + 1;
    const uint64_t cells3 = (uint64_t)N * N * N, corners = (uint64_t)N1 * N1 * N1, epos = 3 * corners;
    const uint32_t wc = (uint32_t)((cells3 + 31) / 32), we = (uint32_t)((epos + 31) / 32);
    const uint32_t Es = s.n_edges;
    // Marching cubes keeps the static triangles leaf by leaf (Ks static leaves) instead of quad by
    // quad (Es static crossing edges), and its static vertices are the static crossings.
    const bool mc = dyn_dcp.marching_cubes;
    // With coarse leaves the static faces of the coarse leaves (12 slots each) follow those of the
    // finest edges.
    const uint32_t ncs = dyn_coarse ? dyn_n_coarse : 0;
    const uint32_t Ks = mc ? s.n_leaves : Es + 12 * ncs;
    const uint32_t hc = (wc + 31) / 32, he = (we + 31) / 32;  // words of the second levels
    {
        // A grid or bitset allocated anew holds no frame: it is filled in full once.
        // (A frame finds its cells by their rank in cell_bits, without the dense cell_grid.)
        Buffer* grids[] = {&b.cell_bits, &b.dyn_bits, &b.touched_bits, &b.edge_bits, &b.cell_hi, &b.edge_hi, &dyn_culled};
        VkBuffer before[7];
        for (int k = 0; k < 7; k++) before[k] = grids[k]->buf;
        for (Buffer* x : {&b.cell_bits, &b.dyn_bits, &b.touched_bits}) ensure_exact(*x, (uint64_t)wc * 4);
        ensure_exact(b.edge_bits, (uint64_t)we * 4);
        ensure_exact(b.cell_hi, (uint64_t)hc * 4);
        ensure_exact(b.edge_hi, (uint64_t)he * 4);
        ensure_frame(dyn_culled, (uint64_t)dyn_culled_cap * 16);
        for (int k = 0; k < 7; k++) if (grids[k]->buf != before[k]) was_clean = false;
    }
    ensure_exact(b.cell_prefix, ((uint64_t)wc + 1) * 4);
    ensure_exact(b.corner_val, corners * 8);
    ensure_exact(b.edge_prefix, ((uint64_t)we + 1) * 4);
    ensure_exact(b.cell_hi_prefix, ((uint64_t)hc + 1) * 4);
    ensure_exact(b.edge_hi_prefix, ((uint64_t)he + 1) * 4);
    for (Buffer* x : {&b.leaf_cells, &b.edges}) ensure_frame(*x, 16);
    ensure(b.keep_static, (uint64_t)Ks * 4);
    ensure(b.keep_prefix, ((uint64_t)Ks + 1) * 4);
    prepare_scan(std::max<uint64_t>({hc, he, Ks}));
    ensure(dyn_level_params, (uint64_t)(D + 1) * kDynLevelStride);
    ensure(dyn_counters, (uint64_t)(D + 1) * sizeof(DynCounters));
    // Arguments: 32 bytes per level 1..D, then those of the patch, then the count of swallowed
    // cells, then frame_counts of dc.comp.
    const VkDeviceSize args_patch = (VkDeviceSize)(D + 1) * 32, culled_count_off = args_patch + 32, frame_counts_off = culled_count_off + 16;
    ensure(dyn_args, frame_counts_off + 32);
    if (!dyn_host.buf || dyn_host.size < (D + 1) * sizeof(DynCounters) + 32) {
        release(dyn_host);
        dyn_host = make_buffer((D + 1) * sizeof(DynCounters) + 32, true, true);
    }
    const uint32_t root_size = (uint32_t)root_tree.size();
    dyn_entry_cap = std::max(dyn_entry_cap, root_size);
    if (root_dynamic) {
        const size_t bytes = sizeof(GpuCell) + root_size * sizeof(TreeEntry);
        if (!staging.buf || staging.size < bytes) { release(staging); staging = make_buffer(bytes, true); }
        const GpuCell gc{{0, 0, 0, 0}, {0, root_size, 0, 0}};
        memcpy(staging.map, &gc, sizeof(gc));
        memcpy((char*)staging.map + sizeof(gc), root_tree.data(), root_size * sizeof(TreeEntry));
    }
    DynCounters root_cnt{};
    root_cnt.cells = root_dynamic ? 1 : 0;
    root_cnt.entries = root_dynamic ? root_size : 0;
    root_cnt.max_tree = root_dynamic ? root_size : 0;
    const uint32_t root_culled_count = root_culled ? 1 : 0, root_cell[4] = {0, 0, 0, 0};

    DcParams P{};
    const double* size = &st_levels.size[3 * D];
    for (int a = 0; a < 3; a++) { P.aabb_min[a] = scene.aabb_min[a]; P.size[a] = size[a]; }
    P.fd_h = dyn_dcp.fd_rel * std::min(size[0], std::min(size[1], size[2]));
    P.N = N;
    P.max_iters = (uint32_t)dyn_dcp.max_iters;
    P.nodes = scene_dyn.nodes.addr; P.prims = scene_dyn.prims.addr; P.ops = scene_dyn.ops.addr;
    P.cell_grid = b.cell_grid.addr; P.cell_bits = b.cell_bits.addr; P.cell_prefix = b.cell_prefix.addr;
    P.corner_val = b.corner_val.addr; P.edge_bits = b.edge_bits.addr; P.edge_prefix = b.edge_prefix.addr;
    P.counters = dc_counters.addr;
    P.frame = 1;
    P.nv_static = s.n_vertices;
    P.depth = (uint32_t)D;
    P.n_static_edges = Ks;
    P.mc = mc ? 1 : 0;
    P.project = dyn_dcp.project_vertices ? 1 : 0;
    if (dyn_coarse) {
        ensure(dyn_dissolved, std::max<uint64_t>(1, ncs) * 4);
        P.coarse_on = 1;
        P.coarse_min = dyn_coarse_min;
        P.n_coarse = ncs;
        P.nv_fine = s.n_fine_vertices;  // the coarse vertices follow the finest ones in the static mesh
        P.coarse_bits = st_coarse_bits.addr;
        P.coarse_prefix = st_coarse_prefix.addr;
        P.coarse_records = st_coarse_records.addr;
        for (int l = 0; l < 12; l++) P.coarse_word_off[l] = dyn_coarse_word_off[l];
        P.dissolved = dyn_dissolved.addr;
    }
    P.static_pool = st_levels.pool[D].addr;
    P.static_corner_val = s.corner_val.addr; P.static_edge_bits = s.edge_bits.addr; P.static_edge_prefix = s.edge_prefix.addr;
    P.static_crossings = mc ? s.V.addr : s.crossings.addr; P.static_cell_bits = s.cell_bits.addr; P.static_leaves = s.leaves.addr;
    P.static_cell_prefix = s.cell_prefix.addr; P.static_vbase = s.vbase.addr; P.static_leaf_cells = st_levels.cells[D].addr;
    P.static_edges = s.edges.addr; P.static_F = s.F.addr;
    P.dyn_bits = b.dyn_bits.addr; P.touched_bits = b.touched_bits.addr;
    P.keep_static = b.keep_static.addr; P.keep_prefix = b.keep_prefix.addr;
    P.frame_counts = dyn_args.addr + frame_counts_off;
    P.cell_hi = b.cell_hi.addr; P.cell_hi_prefix = b.cell_hi_prefix.addr;
    P.edge_hi = b.edge_hi.addr; P.edge_hi_prefix = b.edge_hi_prefix.addr;
    // The lists of the previous frame, which STAGE_CLEAR reads.
    P.leaf_cells = b.leaf_cells.addr;
    P.edges = b.edges.addr;

    uint32_t L = 0, n_culled = 0, nw = 0;
    std::vector<DynCounters> lc(D + 1);
    std::vector<uint64_t> ts(2 * (size_t)D + 2);
    for (int attempt = 0;; attempt++) {
        const bool clear_lists = was_clean && attempt == 0;
        ensure_frame(dyn_cells[0], (uint64_t)dyn_cell_cap * sizeof(GpuCell));
        ensure_frame(dyn_cells[1], (uint64_t)dyn_cell_cap * sizeof(GpuCell));
        ensure_frame(dyn_pool[0], (uint64_t)dyn_entry_cap * sizeof(TreeEntry));
        ensure_frame(dyn_pool[1], (uint64_t)dyn_entry_cap * sizeof(TreeEntry));
        ensure_frame(dyn_scratch, 8ull * dyn_entry_cap * 8);
        ensure_frame(dyn_values, 8ull * dyn_entry_cap * sizeof(double));
        if (attempt > 0) ensure_frame(dyn_culled, (uint64_t)dyn_culled_cap * 16);
        std::vector<DynLevel> lps(D + 1);
        for (int d = 1; d <= D; d++) {
            DynLevel& lp = lps[d];
            lp = level_params(d);
            lp.parents = dyn_cells[(d - 1) & 1].addr; lp.parent_pool = dyn_pool[(d - 1) & 1].addr;
            lp.out_cells = dyn_cells[d & 1].addr; lp.out_pool = dyn_pool[d & 1].addr;
            lp.scratch = dyn_scratch.addr; lp.values = dyn_values.addr;
            lp.culled = dyn_culled.addr;
            lp.culled_cap = dyn_culled_cap;
            lp.cell_cap = dyn_cell_cap;
            lp.entry_cap = dyn_entry_cap;
            lp.counters = dyn_counters.addr + (VkDeviceAddress)d * sizeof(DynCounters);
            lp.args = dyn_args.addr + (VkDeviceAddress)d * 32;
            lp.culled_count = dyn_args.addr + culled_count_off;
        }
        P.dyn_cells = dyn_cells[D & 1].addr;
        P.tree_pool = dyn_pool[D & 1].addr;
        P.culled = dyn_culled.addr;
        P.patch_args = dyn_args.addr + args_patch;
        auto args_dispatch = [&](VkCommandBuffer cb, int prev, VkDeviceSize args_off, uint32_t mode) {
            DynArgsPC ap{dyn_counters.addr + (VkDeviceAddress)prev * sizeof(DynCounters), dyn_args.addr + args_off,
                         dyn_args.addr + culled_count_off, dyn_cell_cap, dyn_entry_cap, kDynWide, dyn_culled_cap, mode};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dyn_args_pipe);
            vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ap), &ap);
            vkCmdDispatch(cb, 1, 1, 1);
        };
        submit([&](VkCommandBuffer cb) {
            vkCmdResetQueryPool(cb, queries, 0, 2 * (uint32_t)D + 2);
            if (root_dynamic) {
                VkBufferCopy r0{0, 0, sizeof(GpuCell)}, r1{sizeof(GpuCell), 0, root_n * sizeof(TreeEntry)};
                vkCmdCopyBuffer(cb, staging.buf, dyn_cells[0].buf, 1, &r0);
                vkCmdCopyBuffer(cb, staging.buf, dyn_pool[0].buf, 1, &r1);
            }
            vkCmdUpdateBuffer(cb, scene_dyn.prims.buf, 24 * 8 * (VkDeviceSize)dyn.sphere_prim, sizeof(sv), sv);
            for (int d = 1; d <= D; d++)
                vkCmdUpdateBuffer(cb, dyn_level_params.buf, (VkDeviceSize)d * kDynLevelStride, sizeof(DynLevel), &lps[d]);
            vkCmdFillBuffer(cb, dyn_counters.buf, 0, VK_WHOLE_SIZE, 0);
            vkCmdUpdateBuffer(cb, dyn_counters.buf, 0, sizeof(root_cnt), &root_cnt);
            vkCmdUpdateBuffer(cb, dyn_args.buf, culled_count_off, 4, &root_culled_count);
            if (root_culled) vkCmdUpdateBuffer(cb, dyn_culled.buf, 0, sizeof(root_cell), root_cell);
            vkCmdUpdateBuffer(cb, dc_params.buf, 0, sizeof(P), &P);
            if (!clear_lists) {
                for (Buffer* x : {&b.cell_bits, &b.dyn_bits, &b.touched_bits}) vkCmdFillBuffer(cb, x->buf, 0, (uint64_t)wc * 4, 0);
                vkCmdFillBuffer(cb, b.edge_bits.buf, 0, (uint64_t)we * 4, 0);
                vkCmdFillBuffer(cb, b.cell_hi.buf, 0, (uint64_t)hc * 4, 0);
                vkCmdFillBuffer(cb, b.edge_hi.buf, 0, (uint64_t)he * 4, 0);
            }
            vkCmdFillBuffer(cb, dc_counters.buf, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cb, dyn_args.buf, frame_counts_off, 32, 0);
            if (dyn_coarse) vkCmdFillBuffer(cb, dyn_dissolved.buf, 0, VK_WHOLE_SIZE, 0);
            transfer_to_compute(cb);
            if (clear_lists) {
                // Only what the previous frame set; before the descent records this frame's swallowed cells.
                dc_dispatch(cb, CLEAR, prev_L, 0);
                dc_dispatch(cb, CLEAR, prev_E, 1);
                dc_dispatch(cb, CLEAR, prev_culled, 2, true);
                compute_barrier(cb);
            }
            for (int d = 1; d <= D; d++) {
                const VkDeviceSize off = (VkDeviceSize)d * 32;
                args_dispatch(cb, d - 1, off, 0);
                indirect_barrier(cb);
                const DynPC pc{dyn_level_params.addr + (VkDeviceSize)d * kDynLevelStride, 0, 0};
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 2 * (uint32_t)(d - 1));
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dyn_pipe);
                vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatchIndirect(cb, dyn_args.buf, off);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dyn_wide_pipe);
                vkCmdPushConstants(cb, dc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatchIndirect(cb, dyn_args.buf, off + 12);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 2 * (uint32_t)(d - 1) + 1);
                compute_barrier(cb);
            }
            // The swallowed static leaves, the dynamic leaves and the ring, and their prefix sum.
            args_dispatch(cb, D, args_patch, 1);
            indirect_barrier(cb);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 2 * (uint32_t)D);
            dc_dispatch_indirect(cb, TOUCH, args_patch);
            dc_dispatch_indirect(cb, DYNBITS, args_patch + 12);
            compute_barrier(cb);
            dc_dispatch_indirect(cb, RING, args_patch + 12);
            if (dyn_coarse) dc_dispatch(cb, DISSOLVE, ncs, 0, true);  // the dissolved coarse leaves, by their finest leaves
            compute_barrier(cb);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 2 * (uint32_t)D + 1);
            compute_to_host(cb);
            // The counters of the levels, then the swallowed cells and frame_counts.
            VkBufferCopy rc{0, 0, (D + 1) * sizeof(DynCounters)}, rn{culled_count_off, (D + 1) * sizeof(DynCounters), 32};
            vkCmdCopyBuffer(cb, dyn_counters.buf, dyn_host.buf, 1, &rc);
            vkCmdCopyBuffer(cb, dyn_args.buf, dyn_host.buf, 1, &rn);
            transfer_to_host(cb);
        });
        const char* hp = (const char*)dyn_host.map;
        memcpy(lc.data(), hp, (D + 1) * sizeof(DynCounters));
        memcpy(&n_culled, hp + (D + 1) * sizeof(DynCounters), 4);
        memcpy(&L, hp + (D + 1) * sizeof(DynCounters) + 16, 4);
        memcpy(&nw, hp + (D + 1) * sizeof(DynCounters) + 20, 4);
        // Grown to twice what this attempt saw; the levels after one that did not fit saw nothing,
        // so a frame may take a few attempts, once.
        uint32_t need_cells = 0, need_entries = 0;
        for (int d = 1; d <= D; d++) { need_cells = std::max(need_cells, lc[d].cells); need_entries = std::max(need_entries, lc[d].entries); }
        const bool grow_cells = need_cells > dyn_cell_cap, grow_entries = need_entries > dyn_entry_cap, grow_culled = n_culled > dyn_culled_cap;
        if (!grow_cells && !grow_entries && !grow_culled) break;
        if (grow_cells) dyn_cell_cap = 2 * need_cells;
        if (grow_entries) dyn_entry_cap = 2 * need_entries;
        if (grow_culled) dyn_culled_cap = 2 * n_culled;
    }
    VK_CHECK(vkGetQueryPoolResults(device, queries, 0, 2 * (uint32_t)D + 2, ts.size() * 8, ts.data(), 8,
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    for (int d = 1; d <= D; d++) {
        fs.level_gpu.push_back((double)(ts[2 * d - 1] - ts[2 * d - 2]) * timestamp_period * 1e-9);
        c.prim_evals += lc[d].prim_evals;
        c.op_evals += lc[d].op_evals;
    }
    const uint32_t n_dyn = lc[D].cells;
    fs.dynamic_leaves = n_dyn;
    fs.culled = n_culled;
    fs.region = L;
    fs.t_region = (double)(ts[2 * D + 1] - ts[2 * D]) * timestamp_period * 1e-9;
    fs.t_descent = t_desc.seconds() - fs.t_region;

    // The region cells, now that their number is known.
    const uint32_t region_cap = std::max(1u, L);
    ensure_frame(b.leaf_cells, (uint64_t)region_cap * sizeof(GpuCell));
    ensure_frame(b.leaves, (uint64_t)region_cap * 4);
    ensure_frame(b.comp, (uint64_t)region_cap * 8);
    ensure_frame(b.ncomp, (uint64_t)region_cap * 4);
    ensure_frame(b.ncomp_dyn, (uint64_t)region_cap * 4);
    ensure_frame(b.vbase, ((uint64_t)region_cap + 1) * 4);
    ensure_frame(b.vbase_dyn, ((uint64_t)region_cap + 1) * 4);
    ensure_frame(b.cell_word_list, (uint64_t)std::max(1u, nw) * 4);
    ensure_frame(b.cell_word_count, (uint64_t)std::max(1u, nw) * 4);
    ensure_frame(b.cell_word_prefix, ((uint64_t)nw + 1) * 4);
    ensure_frame(b.corner_list, (uint64_t)region_cap * 8 * 4);
    prepare_scan(std::max<uint64_t>({hc, he, nw, region_cap, Ks}));
    P.cell_word_list = b.cell_word_list.addr; P.cell_word_count = b.cell_word_count.addr; P.cell_word_prefix = b.cell_word_prefix.addr;
    P.leaf_cells = b.leaf_cells.addr; P.leaves = b.leaves.addr;
    P.comp = b.comp.addr; P.ncomp = b.ncomp.addr; P.vbase = b.vbase.addr;
    P.ncomp_dyn = b.ncomp_dyn.addr; P.vbase_dyn = b.vbase_dyn.addr;
    P.n_cells = L;
    P.corner_list = b.corner_list.addr;

    // Second submission: corners, crossing edges, components, kept static quads, and the counts.
    Timer t_dc;
    submit([&](VkCommandBuffer cb) {
        vkCmdUpdateBuffer(cb, dc_params.buf, 0, sizeof(P), &P);
        transfer_to_compute(cb);
        // The words of the region bitset in use, in order, and their prefix sums.
        scan(cb, b.cell_hi.addr, b.cell_hi_prefix.addr, hc, 1);
        dc_dispatch(cb, LIST, hc, 0);
        compute_barrier(cb);
        scan(cb, b.cell_word_count.addr, b.cell_word_prefix.addr, nw, 0);
        dc_dispatch(cb, REGION, nw);
        compute_barrier(cb);
        dc_dispatch_indirect(cb, DYNBITS, args_patch + 12, 1);  // the dynamic leaves at their ranks
        compute_barrier(cb);
        dc_dispatch(cb, COMPACT, nw, 0);
        // The corners each leaf owns, listed, then evaluated in full subgroups.
        dc_dispatch(cb, CORNERS, 8 * L, 1);
        dc_dispatch(cb, KEEP, Ks);
        compute_barrier(cb);
        dc_dispatch(cb, CORNERS, 8 * L, 2);
        compute_barrier(cb);
        dc_dispatch(cb, EDGES, 12 * L, 1);
        compute_barrier(cb);
        scan(cb, b.keep_static.addr, b.keep_prefix.addr, Ks, 0);
        dc_dispatch(cb, COMP, L);
        compute_barrier(cb);
        scan(cb, b.ncomp_dyn.addr, b.vbase_dyn.addr, L, 0);
        compute_to_host(cb);
        VkBufferCopy r0{frame_counts_off + 8, 0, 8}, r1{(VkDeviceSize)L * 4, 8, 4}, r2{(VkDeviceSize)Ks * 4, 12, 4};
        vkCmdCopyBuffer(cb, dyn_args.buf, totals.buf, 1, &r0);
        vkCmdCopyBuffer(cb, b.vbase_dyn.buf, totals.buf, 1, &r1);
        vkCmdCopyBuffer(cb, b.keep_prefix.buf, totals.buf, 1, &r2);
        transfer_to_host(cb);
    });
    uint32_t tot[4];
    memcpy(tot, totals.map, sizeof(tot));
    // With marching cubes tot[2] and tot[3] count new and kept triangles, and the new vertices are the
    // crossings of the region.
    const uint32_t E = tot[0], nwe = tot[1], nv_new = mc ? E : tot[2], kept = tot[3], mc_new = mc ? tot[2] : 0;

    // Third submission: crossings, new vertices, faces, assembly, and the copy to the host.
    ensure_frame(b.edges, (uint64_t)E * 4);
    ensure_frame(b.crossings, mc ? 16 : (uint64_t)E * 6 * 8);
    ensure_frame(b.F, mc ? 16 : (uint64_t)E * 6 * 4);
    ensure_frame(b.V, (uint64_t)nv_new * 3 * 8);
    ensure_frame(b.vtree, (uint64_t)nv_new * 4);
    if (!mc) {  // the new faces of dual contouring, counted and placed after the kept ones
        ensure_frame(b.new_flags, (uint64_t)E * 4);
        ensure_frame(b.new_prefix, ((uint64_t)E + 1) * 4);
    }
    ensure_frame(b.out_F, mc ? ((uint64_t)kept + mc_new) * 3 * 4 : ((uint64_t)kept + E) * 6 * 4);
    ensure_frame(b.edge_word_list, (uint64_t)std::max(1u, nwe) * 4);
    ensure_frame(b.edge_word_count, (uint64_t)std::max(1u, nwe) * 4);
    ensure_frame(b.edge_word_prefix, ((uint64_t)nwe + 1) * 4);
    prepare_scan(std::max<uint64_t>({hc, he, nw, nwe, region_cap, Ks, E}));
    P.edge_word_list = b.edge_word_list.addr; P.edge_word_count = b.edge_word_count.addr; P.edge_word_prefix = b.edge_word_prefix.addr;
    P.edges = b.edges.addr; P.crossings = b.crossings.addr; P.F = b.F.addr; P.V = b.V.addr; P.vtree = b.vtree.addr;
    P.new_flags = b.new_flags.addr; P.new_prefix = b.new_prefix.addr; P.out_F = b.out_F.addr;
    const bool download = mode != 2;
    const size_t bytes_v = download ? (size_t)nv_new * 3 * 8 : 0, bytes_t = download ? (size_t)nv_new * 4 : 0,
                 bytes_f = !download ? 0 : mc ? ((size_t)kept + mc_new) * 3 * 4 : ((size_t)kept + E) * 6 * 4;
    ensure_readback(bytes_v + bytes_t + bytes_f + 8 + sizeof(DcCounters));
    submit([&](VkCommandBuffer cb) {
        vkCmdUpdateBuffer(cb, dc_params.buf, 0, sizeof(P), &P);
        transfer_to_compute(cb);
        // The words of the edge bitset in use, in order, and the sorted crossing edges.
        scan(cb, b.edge_hi.addr, b.edge_hi_prefix.addr, he, 1);
        dc_dispatch(cb, LIST, he, 1);
        compute_barrier(cb);
        scan(cb, b.edge_word_count.addr, b.edge_word_prefix.addr, nwe, 0);
        dc_dispatch(cb, COMPACT, nwe, 1);
        compute_barrier(cb);
        dc_dispatch(cb, CROSS, E);
        compute_barrier(cb);
        if (mc) {
            // The triangles of the new cells after the kept static ones, which ASSEMBLE copies.
            dc_dispatch(cb, MC_FACES, L);
            dc_dispatch(cb, ASSEMBLE, Ks, 0);
        } else {
            dc_dispatch(cb, QEF, L);
            dc_dispatch(cb, FACES, E);
            compute_barrier(cb);
            dc_dispatch(cb, NEWFLAGS, E);
            compute_barrier(cb);
            scan(cb, b.new_flags.addr, b.new_prefix.addr, E, 0);
            dc_dispatch(cb, ASSEMBLE, Ks, 0);  // the static faces of the finest edges, then those of the coarse leaves
            dc_dispatch(cb, ASSEMBLE, E, 1);
        }
        compute_to_host(cb);
        VkBufferCopy rv{0, 0, bytes_v}, rt{0, bytes_v, bytes_t}, rf{0, bytes_v + bytes_t, bytes_f},
            rn{(VkDeviceSize)E * 4, bytes_v + bytes_t + bytes_f, 4}, rc{0, bytes_v + bytes_t + bytes_f + 8, sizeof(DcCounters)};
        if (bytes_v) vkCmdCopyBuffer(cb, b.V.buf, readback.buf, 1, &rv);
        if (bytes_t) vkCmdCopyBuffer(cb, b.vtree.buf, readback.buf, 1, &rt);
        if (bytes_f) vkCmdCopyBuffer(cb, b.out_F.buf, readback.buf, 1, &rf);
        if (!mc) vkCmdCopyBuffer(cb, b.new_prefix.buf, readback.buf, 1, &rn);
        vkCmdCopyBuffer(cb, dc_counters.buf, readback.buf, 1, &rc);
        transfer_to_host(cb);
    });
    fs.t_dc = t_dc.seconds();

    Timer t_down;
    const char* rbp = (const char*)readback.map;
    uint32_t n_new = mc_new;
    if (!mc) memcpy(&n_new, rbp + bytes_v + bytes_t + bytes_f, 4);
    DcCounters dcc;
    memcpy(&dcc, rbp + bytes_v + bytes_t + bytes_f + 8, sizeof dcc);
    c.prim_evals += dcc.prim_evals;
    c.op_evals += dcc.op_evals;
    fs.kept_quads = kept;
    fs.new_quads = n_new;
    fs.new_vertices = nv_new;
    // What this frame set, for the clearing at the start of the next one.
    prev_L = L;
    prev_E = E;
    prev_culled = std::min(n_culled, dyn_culled_cap);
    frame_clean = true;
    if (!download) {
        last_mesh = GpuMeshBuffers{s.V.addr, b.V.addr, b.out_F.addr, s.n_vertices, nv_new, (mc ? 1 : 2) * (kept + n_new),
                                   s.vtree.addr, b.vtree.addr, mc ? 0u : dcc.empty_triangles};
        fs.t_download = t_down.seconds();
        fs.total = t_total.seconds();
        return;
    }
    const bool new_only = mode == 1;
    const size_t sv_n = new_only ? 0 : st_mesh.V.size(), st_n = new_only ? 0 : st_mesh.tree_n.size();
    out.vertex_cell.clear();
    out.V.resize(sv_n + (size_t)nv_new * 3);
    memcpy(out.V.data(), st_mesh.V.data(), sv_n * sizeof(double));
    memcpy(out.V.data() + sv_n, rbp, bytes_v);
    out.tree_n.resize(st_n + nv_new);
    memcpy(out.tree_n.data(), st_mesh.tree_n.data(), st_n * 4);
    memcpy(out.tree_n.data() + st_n, rbp + bytes_v, bytes_t);
    out.F.resize(((size_t)kept + n_new) * (mc ? 3 : 6));
    memcpy(out.F.data(), rbp + bytes_v + bytes_t, out.F.size() * 4);
    if (dyn_coarse) {
        // Next to a coarse leaf a face may be one triangle: the second of its slot is kNone.
        size_t w = 0;
        for (size_t t = 0; t < out.F.size(); t += 3)
            if (out.F[t] != kNone && out.F[t + 1] != kNone && out.F[t + 2] != kNone)
                for (int k = 0; k < 3; k++) out.F[w++] = out.F[t + k];
        out.F.resize(w);
    }
    fs.t_download = t_down.seconds();
    fs.total = t_total.seconds();
}

void GpuPruner::release_dynamic() { impl_->release_dynamic(); }

void GpuPruner::dynamic_setup(const DynamicScene& dyn, int depth, double margin, const DCParams& dcp, Counters& c,
                              const CoarseParams* coarse) {
    impl_->dyn_setup(dyn, depth, margin, dcp, c, coarse);
}
Mesh GpuPruner::dynamic_frame(const double center[3], Counters& c, GpuFrameStats& fs, bool new_only) {
    Mesh out;
    impl_->dyn_frame(center, c, fs, new_only ? 1 : 0, out);
    return out;
}
void GpuPruner::dynamic_frame(const double center[3], Counters& c, GpuFrameStats& fs, Mesh& out, bool new_only) {
    impl_->dyn_frame(center, c, fs, new_only ? 1 : 0, out);
}
GpuMeshBuffers GpuPruner::dynamic_frame_gpu(const double center[3], Counters& c, GpuFrameStats& fs) {
    Mesh unused;
    impl_->dyn_frame(center, c, fs, 2, unused);
    return impl_->last_mesh;
}
const Mesh& GpuPruner::static_mesh() const { return impl_->st_mesh; }
void GpuPruner::release_build_buffers() { impl_->release_build(); }

}  // namespace lpm
