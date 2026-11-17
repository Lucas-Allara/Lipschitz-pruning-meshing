// Declarations shared by prune.comp and prune_wide.comp: buffers, parameters, and the signed
// distance of the primitives and operators (as include/lpm/sdf.h), in double precision.
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

const uint NODE_PRIM = 1u;
const uint INACTIVE = 0u, SKIPPED = 1u, ACTIVE = 2u;
// Flags of the scratch entries.
const uint STATE_MASK = 3u, INACTIVE_ANC = 4u, ACTIVE_GLOBAL = 8u, NEGATIVE = 16u;
const uint NO_PARENT = 0xffffffffu;
// Flag (xyz.w) of a cell that probe.comp made a coarse leaf: it gets no children.
const uint COARSE_LEAF = 2u;

struct Cell { uvec4 xyz; uvec4 tree; };  // xyz: cell coordinates; tree: offset and size of its tree

layout(std430, binding = 0) readonly buffer Nodes { uvec2 nodes[]; };     // node type, index of its primitive or operator
layout(std430, binding = 1) readonly buffer Prims { double prims[]; };    // 24 values per primitive
layout(std430, binding = 2) readonly buffer Ops { dvec4 ops[]; };         // blending radius k, sign s, 0.25 / k
layout(std430, binding = 3) readonly buffer InCells { Cell in_cells[]; };
// Tree entries in the layout of TreeEntry (include/lpm/tree.h): node, parent in the same tree, sign
// in the low byte (1 or -1 as an int8).
layout(std430, binding = 4) readonly buffer InPool { uint in_pool[]; };
layout(std430, binding = 5) writeonly buffer OutCells { Cell out_cells[]; };
layout(std430, binding = 6) writeonly buffer OutPool { uint out_pool[]; };
layout(std430, binding = 7) buffer Scratch { uvec2 scratch[]; };          // flags, new parent then index in the output
layout(std430, binding = 8) buffer Counters {
    uint cells, entries, max_tree, single_prim;
    uint64_t prim_evals, op_evals;
    uint far_cells, pad;
} counters;
// With keep_empty, the cells proven empty, with their pruned trees (in out_pool); they are leaves.
layout(std430, binding = 10) writeonly buffer OutFar { Cell out_far[]; };

layout(push_constant) uniform Params {
    dvec4 aabb_min;
    dvec4 size;       // size of a child cell
    double r_keep;    // the child is kept when |f(center)| <= r_keep
    double r_prune;   // radius of the ball the tree is pruned for
    uint n_parents;
    uint wide;        // parents with larger trees are pruned by prune_wide.comp
    uint full;        // 1: no pruning; every cell keeps the full tree, entries [0, n) of in_pool
    uint keep_empty;  // 1: the cells proven empty go to out_far with their trees
    double margin;    // INTERVAL: each child box is widened by margin * size / 2 per axis
} pc;

// The cell of child k of a cell: bit 0 of k along x, bit 1 along y, bit 2 along z. Invocation (or
// workgroup) t takes child t & 7 of parent t >> 3.
uvec3 child_cell(uvec3 parent, uint child) {
    return 2u * parent + uvec3(child & 1u, (child >> 1) & 1u, child >> 2);
}
// Values (with INTERVAL, bounds) of the primitives of a tree, written by prune_wide.comp (same
// indexing as the scratch).
#ifdef INTERVAL
layout(std430, binding = 9) buffer Values { dvec2 values[]; };
#else
layout(std430, binding = 9) buffer Values { double values[]; };
#endif

#define PRIM_VALUE(i) prims[i]
#include "sdf.glsl"
#ifdef INTERVAL
#include "sdf_interval.glsl"
#endif

// Center of a child cell: the same arithmetic as build_octree.
dvec3 child_center(uvec3 cc) {
    const dvec3 lo = pc.aabb_min.xyz + dvec3(cc) * pc.size.xyz;
    return 0.5 * (lo + (lo + pc.size.xyz));
}

#ifdef INTERVAL
// Box of a child cell widened by the margin, as the interval oracle of build_octree.
void child_box(uvec3 cc, out dvec2 bx, out dvec2 by, out dvec2 bz) {
    const dvec3 lo = pc.aabb_min.xyz + dvec3(cc) * pc.size.xyz;
    const dvec3 hi = lo + pc.size.xyz;
    const dvec3 m = pc.margin * 0.5 * pc.size.xyz;
    bx = dvec2(lo.x - m.x, hi.x + m.x);
    by = dvec2(lo.y - m.y, hi.y + m.y);
    bz = dvec2(lo.z - m.z, hi.z + m.z);
}
#endif

