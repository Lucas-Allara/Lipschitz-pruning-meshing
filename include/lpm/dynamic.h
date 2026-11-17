// Dynamic update: the static scene T joined by a smooth union with a moving sphere S.
// The octree of T is built once, keeping the empty leaves with their pruned trees. Each
// frame descends from it: a cell whose root union is skipped with T winning keeps its
// whole static subtree, including the corner values, crossings and vertices of the static
// dual contouring; only the cells whose ball the sphere reaches are pruned and contoured.
#pragma once
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lpm/dc.h"

namespace lpm {

struct DynamicScene {
    Scene scene;  // nodes of T, then the sphere (n) and the union root (n + 1)
    uint32_t sphere_node = 0, root_node = 0, sphere_prim = 0;
};

inline DynamicScene make_dynamic_scene(const Scene& base, double k, double radius) {
    DynamicScene d;
    d.scene = base;
    Scene& s = d.scene;
    const uint32_t n = (uint32_t)s.nodes.size();
    Primitive P;
    P.type = PRIM_SPHERE;
    P.data[0] = radius;
    P.color = 0xaaaaff;
    d.sphere_prim = (uint32_t)s.prims.size();
    s.prims.push_back(P);
    Node sn;
    sn.type = NODE_PRIM; sn.idx = (int)d.sphere_prim; sn.parent = (int)n + 1; sn.sign = 1;
    s.nodes.push_back(sn);
    BinaryOp op;
    op.op = OP_UNION; op.k = k; op.s = 1.0;
    Node on;
    on.type = NODE_OP; on.idx = (int)s.ops.size(); on.left = (int)n - 1; on.right = (int)n; on.parent = -1; on.sign = 1;
    s.ops.push_back(op);
    s.nodes.push_back(on);
    s.nodes[n - 1].parent = (int)n + 1;
    d.sphere_node = n;
    d.root_node = n + 1;
    s.depth += 1;
    return d;
}

inline void set_sphere_center(DynamicScene& d, const double c[3]) {
    Primitive& P = d.scene.prims[d.sphere_prim];
    P.m[3] = -c[0]; P.m[7] = -c[1]; P.m[11] = -c[2];
}

// Path of the sphere of the official viewer, mapped into the bounding box.
inline void sphere_path(const Scene& s, double radius, double t, double c[3]) {
    double v[3] = {std::cos(t), std::cos(0.3 * t), std::sin(t)};
    const double sc = std::sin(t * 0.56 + 123.4);
    for (int a = 0; a < 3; a++) {
        v[a] *= sc;
        c[a] = s.aabb_min[a] + radius + (v[a] * 0.5 + 0.5) * (s.aabb_max[a] - s.aabb_min[a] - 2.0 * radius);
    }
}

// A closed orbit, so that an animation of one period loops: a circle around the vertical axis
// through the center of the bounding box, of radius r_frac times its smaller horizontal side,
// whose height goes from y_lo (at azimuth `phase`, in radians) to y_hi on the opposite side;
// y_lo and y_hi are fractions of the height of the box.
inline void orbit_path(const Scene& s, double t, double r_frac, double y_lo, double y_hi, double phase, double c[3]) {
    const double w = std::min(s.aabb_max[0] - s.aabb_min[0], s.aabb_max[2] - s.aabb_min[2]);
    const double h = s.aabb_max[1] - s.aabb_min[1];
    const double up = 0.5 * (1.0 - std::cos(t - phase));
    c[0] = 0.5 * (s.aabb_min[0] + s.aabb_max[0]) + r_frac * w * std::cos(t);
    c[1] = s.aabb_min[1] + (y_lo + (y_hi - y_lo) * up) * h;
    c[2] = 0.5 * (s.aabb_min[2] + s.aabb_max[2]) + r_frac * w * std::sin(t);
}

struct StaticOctree {
    Octree oct;
    std::vector<std::unordered_map<uint64_t, uint32_t>> maps;  // per level: cell key -> index
};

inline StaticOctree build_static(const Scene& base, int depth, double margin, Counters& counters) {
    StaticOctree st;
    OctreeParams p;
    p.oracle = ORACLE_LIPSCHITZ;
    p.max_depth = depth;
    p.margin = margin;
    p.keep_empty_leaves = true;
    st.oct = build_octree(base, p, counters);
    st.maps.resize(depth + 1);
    for (int l = 0; l <= depth; l++) {
        st.maps[l].reserve(st.oct.levels[l].cells.size() * 2);
        for (uint32_t i = 0; i < st.oct.levels[l].cells.size(); i++) {
            const Cell& c = st.oct.levels[l].cells[i];
            st.maps[l].emplace(cell_key(c.x, c.y, c.z), i);
        }
    }
    return st;
}

struct DynCell {
    uint32_t x, y, z;
    uint32_t tree_off = 0, tree_n = 0;
    double value = 0.0;
    uint8_t mode = 0;  // 0: the sphere is active, descend; 1: T wins on the whole ball, the static subtree stands
    uint8_t level = 0;
    int32_t static_idx = -1;
};

struct DynamicUpdate {
    std::vector<LeafRef> leaves;
    std::vector<DynCell> culled;  // cells discarded with |f'(c)| > R, at any level (used by the patch)
    std::vector<std::vector<TreeEntry>> pools;
    uint64_t tested = 0, dynamic_cells = 0, static_subtrees = 0, static_leaves = 0, dynamic_leaves = 0;
    double seconds = 0.0;
};

namespace detail {
// The static tree of a cell joined with the sphere under the union root.
inline void combine(TreeView ts, uint32_t sphere_node, uint32_t root_node, std::vector<TreeEntry>& out) {
    out.resize(ts.n + 2);
    for (uint32_t i = 0; i < ts.n; i++) {
        out[i] = ts.e[i];
        if (out[i].parent < 0) out[i].parent = (int32_t)ts.n + 1;
    }
    out[ts.n] = TreeEntry{sphere_node, (int32_t)ts.n + 1, 1};
    out[ts.n + 1] = TreeEntry{root_node, -1, 1};
}
inline bool has_node(const TreeEntry* e, uint32_t n, uint32_t node) {
    for (uint32_t i = 0; i < n; i++) if (e[i].node == node) return true;
    return false;
}
}  // namespace detail

inline DynamicUpdate build_dynamic(const DynamicScene& dyn, const StaticOctree& st, double margin, Counters& counters,
                                   bool collect_static_leaves = true) {
    Timer timer;
    const Scene& scene = dyn.scene;
    const Octree& so = st.oct;
    const int D = (int)so.levels.size() - 1;
    const int nthreads = omp_get_max_threads();
    DynamicUpdate du;
    du.pools.resize(D + 1);
    std::vector<std::vector<DynCell>> levels(D + 1);
    std::vector<DynCell> static_roots;  // cells of mode 1 whose static subtree provides the leaves

    // Level 0.
    {
        const Cell& sroot = so.levels[0].cells[0];
        std::vector<TreeEntry> in;
        detail::combine(so.tree_of(0, sroot), dyn.sphere_node, dyn.root_node, in);
        double c[3], R = 0.0;
        for (int a = 0; a < 3; a++) { c[a] = 0.5 * (scene.aabb_min[a] + scene.aabb_max[a]); R += 0.25 * so.levels[0].size[a] * so.levels[0].size[a]; }
        R = std::sqrt(R);
        Scratch sc;
        DynCell root;
        root.x = root.y = root.z = 0;
        root.value = prune_lipschitz(scene, TreeView{in.data(), (uint32_t)in.size()}, c, R * (1.0 + margin), du.pools[0], root.tree_n, sc, counters);
        root.static_idx = 0;
        root.level = 0;
        du.tested = 1;
        if (std::fabs(root.value) <= R * (1.0 + 1e-9)) {
            root.mode = detail::has_node(du.pools[0].data(), root.tree_n, dyn.sphere_node) ? 0 : 1;
            if (root.mode == 0) { levels[0].push_back(root); du.dynamic_cells++; }
            else if (!sroot.far) { static_roots.push_back(root); du.static_subtrees++; }
        } else {
            du.culled.push_back(root);
        }
    }

    for (int d = 1; d <= D; d++) {
        const std::vector<DynCell>& parents = levels[d - 1];
        const Level& SL = so.levels[d];
        const double* size = SL.size;
        const double R = 0.5 * std::sqrt(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]);
        std::vector<std::vector<DynCell>> cells_t(nthreads), statics_t(nthreads), culled_t(nthreads);
        std::vector<std::vector<TreeEntry>> pool_t(nthreads);
        std::vector<Counters> cnt_t(nthreads);
        std::vector<uint64_t> tested_t(nthreads, 0);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            Scratch sc;
            std::vector<TreeEntry> combined;
            std::vector<DynCell>& cells = cells_t[tid];
            std::vector<TreeEntry>& pool = pool_t[tid];
#pragma omp for schedule(dynamic, 16)
            for (size_t pi = 0; pi < parents.size(); pi++) {
                const DynCell& pc = parents[pi];
                const TreeView parent_tree{du.pools[d - 1].data() + pc.tree_off, pc.tree_n};
                for (int child = 0; child < 8; child++) {
                    DynCell cc;
                    cc.x = 2 * pc.x + (child & 1);
                    cc.y = 2 * pc.y + ((child >> 1) & 1);
                    cc.z = 2 * pc.z + ((child >> 2) & 1);
                    const uint32_t idx[3] = {cc.x, cc.y, cc.z};
                    double c[3];
                    for (int a = 0; a < 3; a++) c[a] = scene.aabb_min[a] + (idx[a] + 0.5) * size[a];
                    auto it = st.maps[d].find(cell_key(cc.x, cc.y, cc.z));
                    const Cell* scell = it == st.maps[d].end() ? nullptr : &SL.cells[it->second];
                    cc.static_idx = scell ? (int32_t)it->second : -1;
                    cc.level = (uint8_t)d;
                    // Input: the dynamic tree of the parent, or the static tree of this cell joined
                    // with the sphere, whichever is smaller; both are exact on the ball of the cell.
                    TreeView in = parent_tree;
                    if (scell && scell->tree_n + 2 < parent_tree.n) {
                        detail::combine(so.tree_of(d, *scell), dyn.sphere_node, dyn.root_node, combined);
                        in = TreeView{combined.data(), (uint32_t)combined.size()};
                    }
                    const size_t base = pool.size();
                    cc.value = prune_lipschitz(scene, in, c, R * (1.0 + margin), pool, cc.tree_n, sc, cnt_t[tid]);
                    tested_t[tid]++;
                    if (std::fabs(cc.value) > R * (1.0 + 1e-9)) { pool.resize(base); culled_t[tid].push_back(cc); continue; }
                    cc.tree_off = (uint32_t)base;
                    if (detail::has_node(pool.data() + base, cc.tree_n, dyn.sphere_node)) {
                        cc.mode = 0;
                        cells.push_back(cc);
                    } else {
                        // T wins on the whole ball: the field equals T here. The static subtree stands
                        // if it has surface; an empty static leaf, or no static cell at all (an
                        // ancestor was proven empty), means no surface.
                        pool.resize(base);
                        cc.mode = 1;
                        if (scell && !scell->far) statics_t[tid].push_back(cc);
                    }
                }
            }
        }
        std::vector<DynCell>& L = levels[d];
        for (int t = 0; t < nthreads; t++) {
            const uint32_t off = (uint32_t)du.pools[d].size();
            for (DynCell c : cells_t[t]) { c.tree_off += off; L.push_back(c); }
            du.pools[d].insert(du.pools[d].end(), pool_t[t].begin(), pool_t[t].end());
            static_roots.insert(static_roots.end(), statics_t[t].begin(), statics_t[t].end());
            du.culled.insert(du.culled.end(), culled_t[t].begin(), culled_t[t].end());
            du.static_subtrees += statics_t[t].size();
            du.tested += tested_t[t];
            counters.add(cnt_t[t]);
        }
        du.dynamic_cells += L.size();
        if (d < D) {
            std::sort(L.begin(), L.end(), [](const DynCell& a, const DynCell& b) { return cell_key(a.x, a.y, a.z) < cell_key(b.x, b.y, b.z); });
        }
    }

    // Leaves: the dynamic cells of the finest level, and the static leaves under every static root.
    for (const DynCell& c : levels[D]) {
        du.leaves.push_back(LeafRef{c.x, c.y, c.z, TreeView{du.pools[D].data() + c.tree_off, c.tree_n}, c.tree_n, false});
    }
    du.dynamic_leaves = du.leaves.size();
    std::vector<std::pair<int, uint32_t>> stack;
    if (!collect_static_leaves) static_roots.clear();
    for (size_t r = 0; r < static_roots.size(); r++) {
        const DynCell& root = static_roots[r];
        stack.clear();
        stack.push_back({(int)root.level, (uint32_t)root.static_idx});
        while (!stack.empty()) {
            auto [l, i] = stack.back();
            stack.pop_back();
            const Cell& c = so.levels[l].cells[i];
            if (c.far) continue;
            if (l == D) {
                du.leaves.push_back(LeafRef{c.x, c.y, c.z, so.tree_of(D, c), c.tree_n, true});
                continue;
            }
            for (int child = 0; child < 8; child++) {
                auto it = st.maps[l + 1].find(cell_key(2 * c.x + (child & 1), 2 * c.y + ((child >> 1) & 1), 2 * c.z + ((child >> 2) & 1)));
                if (it != st.maps[l + 1].end()) stack.push_back({l + 1, it->second});
            }
        }
    }
    du.static_leaves = du.leaves.size() - du.dynamic_leaves;
    du.seconds = timer.seconds();
    return du;
}


// ---------------------------------------------------------------- incremental patch
// The static mesh keeps, per quad, its crossing edge and four cells, so that a frame can
// drop the quads the sphere affects and append the ones of the recomputed region.
struct StaticMesh {
    Mesh mesh;
    DCCache cache;
    std::vector<QuadRecord> quads;
    std::vector<std::array<uint32_t, 6>> quad_box;  // per quad: min x, y, z and max x, y, z of its cells
    std::unordered_map<uint64_t, uint32_t> vertex_of_key;  // (cell, component) key -> static vertex id
};

inline void cell_coords(uint64_t key, uint32_t c[3]) {
    c[0] = (uint32_t)(key >> 42); c[1] = (uint32_t)((key >> 21) & 0x1fffff); c[2] = (uint32_t)(key & 0x1fffff);
}

inline StaticMesh build_static_mesh(const StaticOctree& st, const DCParams& dp, Counters& counters) {
    StaticMesh sm;
    const Octree& oct = st.oct;
    const int D = (int)oct.levels.size() - 1;
    const Level& L = oct.levels[D];
    std::vector<LeafRef> leaves;
    for (const Cell& c : L.cells) if (!c.far) leaves.push_back(LeafRef{c.x, c.y, c.z, oct.tree_of(D, c), c.tree_n, false});
    DCStats s0;
    sm.mesh = dual_contour_leaves(*oct.scene, D, L.size, std::move(leaves), dp, nullptr, &sm.cache, counters, s0, &sm.quads);
    sm.vertex_of_key.reserve(sm.mesh.vertex_cell.size() * 2);
    for (uint32_t i = 0; i < sm.mesh.vertex_cell.size(); i++) sm.vertex_of_key.emplace(sm.mesh.vertex_cell[i], i);
    sm.quad_box.resize(sm.quads.size());
    for (size_t q = 0; q < sm.quads.size(); q++) {
        std::array<uint32_t, 6>& b = sm.quad_box[q];
        b = {~0u, ~0u, ~0u, 0u, 0u, 0u};
        for (int k = 0; k < 4; k++) {
            uint32_t c[3];
            cell_coords(sm.quads[q].cells[k], c);
            for (int a = 0; a < 3; a++) { b[a] = std::min(b[a], c[a]); b[3 + a] = std::max(b[3 + a], c[a]); }
        }
    }
    return sm;
}

struct PatchStats {
    uint64_t dynamic_leaves = 0, touched = 0, ring = 0, kept_quads = 0, new_quads = 0, new_vertices = 0;
    double t_region = 0, t_dc = 0, t_assemble = 0, total = 0;
};

// The mesh of a frame: the static mesh minus the quads touching the cells the sphere reaches
// or swallows, plus the quads of the region re-contoured around the dynamic leaves. The
// static vertices keep their indices; the vertices of the dynamic cells are appended.
inline Mesh patch_mesh(const DynamicScene& dyn, const StaticOctree& st, const StaticMesh& sm, const DynamicUpdate& du,
                       const DCParams& dp, Counters& counters, PatchStats& ps, DCStats& ds) {
    Timer total;
    const Octree& so = st.oct;
    const int D = (int)so.levels.size() - 1;
    const Level& SL = so.levels[D];
    const int64_t N = 1ll << D;
    Timer tr;
    std::unordered_set<uint64_t> dynset, touched, ringset;
    dynset.reserve(du.leaves.size() * 2);
    for (const LeafRef& l : du.leaves) dynset.insert(cell_key(l.x, l.y, l.z));
    // Static leaves swallowed by the sphere: those under a culled cell of the dynamic descent.
    std::vector<std::pair<int, uint32_t>> stack;
    for (const DynCell& c : du.culled) {
        auto it = st.maps[c.level].find(cell_key(c.x, c.y, c.z));
        if (it == st.maps[c.level].end()) continue;
        stack.clear();
        stack.push_back({(int)c.level, it->second});
        while (!stack.empty()) {
            auto [l, i] = stack.back();
            stack.pop_back();
            const Cell& sc = so.levels[l].cells[i];
            if (sc.far) continue;
            if (l == D) { touched.insert(cell_key(sc.x, sc.y, sc.z)); continue; }
            for (int child = 0; child < 8; child++) {
                auto jt = st.maps[l + 1].find(cell_key(2 * sc.x + (child & 1), 2 * sc.y + ((child >> 1) & 1), 2 * sc.z + ((child >> 2) & 1)));
                if (jt != st.maps[l + 1].end()) stack.push_back({l + 1, jt->second});
            }
        }
    }
    // Region: the dynamic leaves and the ring of static leaves that share a corner with one.
    std::vector<LeafRef> region(du.leaves);
    uint32_t lo[3] = {~0u, ~0u, ~0u}, hi[3] = {0, 0, 0};
    auto grow = [&](uint32_t x, uint32_t y, uint32_t z) {
        const uint32_t c[3] = {x, y, z};
        for (int a = 0; a < 3; a++) { lo[a] = std::min(lo[a], c[a]); hi[a] = std::max(hi[a], c[a]); }
    };
    for (const LeafRef& l : du.leaves) {
        grow(l.x, l.y, l.z);
        for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
            if (!dx && !dy && !dz) continue;
            const int64_t x = (int64_t)l.x + dx, y = (int64_t)l.y + dy, z = (int64_t)l.z + dz;
            if (x < 0 || y < 0 || z < 0 || x >= N || y >= N || z >= N) continue;
            const uint64_t key = cell_key((uint32_t)x, (uint32_t)y, (uint32_t)z);
            if (dynset.count(key) || touched.count(key) || ringset.count(key)) continue;
            auto it = st.maps[D].find(key);
            if (it == st.maps[D].end() || SL.cells[it->second].far) continue;
            const Cell& sc = SL.cells[it->second];
            ringset.insert(key);
            region.push_back(LeafRef{sc.x, sc.y, sc.z, so.tree_of(D, sc), sc.tree_n, true});
        }
    }
    for (uint64_t key : touched) { uint32_t c[3]; cell_coords(key, c); grow(c[0], c[1], c[2]); }
    ps.dynamic_leaves = du.leaves.size();
    ps.touched = touched.size();
    ps.ring = ringset.size();
    ps.t_region = tr.seconds();

    Timer tdc;
    std::vector<QuadRecord> quads;
    Mesh local = dual_contour_leaves(dyn.scene, D, SL.size, std::move(region), dp, &sm.cache, nullptr, counters, ds, &quads);
    ps.t_dc = tdc.seconds();

    Timer ta;
    Mesh out;
    out.V = sm.mesh.V;
    std::vector<int64_t> map_local(local.vertex_cell.size(), -1);
    for (size_t i = 0; i < local.vertex_cell.size(); i++) {
        const uint64_t key = local.vertex_cell[i];
        if (dynset.count(cell_of_vertex_key(key))) {
            map_local[i] = (int64_t)(out.V.size() / 3);
            out.V.insert(out.V.end(), local.V.begin() + 3 * i, local.V.begin() + 3 * i + 3);
            ps.new_vertices++;
        } else {
            auto it = sm.vertex_of_key.find(key);
            if (it != sm.vertex_of_key.end()) map_local[i] = it->second;
        }
    }
    out.F.reserve(sm.mesh.F.size() + quads.size() * 6);
    const bool any_affected = !dynset.empty() || !touched.empty();
    for (size_t q = 0; q < sm.quads.size(); q++) {
        const QuadRecord& r = sm.quads[q];
        bool affected = false;
        if (any_affected) {
            const std::array<uint32_t, 6>& b = sm.quad_box[q];
            if (!(b[3] < lo[0] || b[0] > hi[0] || b[4] < lo[1] || b[1] > hi[1] || b[5] < lo[2] || b[2] > hi[2])) {
                for (int k = 0; k < 4 && !affected; k++) affected = dynset.count(r.cells[k]) || touched.count(r.cells[k]);
            }
        }
        if (affected) continue;
        out.F.insert(out.F.end(), {r.v[0], r.v[1], r.v[2], r.v[0], r.v[2], r.v[3]});
        ps.kept_quads++;
    }
    for (const QuadRecord& r : quads) {
        bool any_dyn = false;
        for (int k = 0; k < 4; k++) any_dyn = any_dyn || dynset.count(r.cells[k]);
        if (!any_dyn) continue;
        uint32_t v[4];
        bool ok = true;
        for (int k = 0; k < 4; k++) { const int64_t m = map_local[r.v[k]]; if (m < 0) { ok = false; break; } v[k] = (uint32_t)m; }
        if (!ok) continue;
        out.F.insert(out.F.end(), {v[0], v[1], v[2], v[0], v[2], v[3]});
        ps.new_quads++;
    }
    ps.t_assemble = ta.seconds();
    ps.total = total.seconds();
    return out;
}

}  // namespace lpm
