// Dual contouring of the finest octree level. Every cell that may contain the surface is
// at the maximum depth, so the leaves form a sparse uniform grid: corner values and edge
// crossings are shared through hash maps and evaluated once, with the pruned tree of any
// cell that contains them. A cache lets a dynamic update reuse the corners, crossings and
// vertices of the cells that a moving primitive cannot affect.
#pragma once
#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <cstdio>
#include <string>
#include <parallel/algorithm>
#include <unordered_map>
#include <vector>

#include "lpm/octree.h"

namespace lpm {

struct DCParams {
    int max_iters = 30;          // root finding iterations per crossing edge
    double fd_rel = 1e-6;        // finite difference step relative to the cell size
    bool analytic_leaves = false;  // cells whose tree is a single sphere: analytic crossing and normal
    // Dual contouring moves a vertex that ends farther than 0.05 cell from the surface onto it, with
    // up to three Newton steps inside its cell (not with coarse leaves). Near creases the steps
    // amplify rounding, so the CPU and the GPU compare their vertices without it (lpm_gpu).
    bool project_vertices = true;
    // GPU only (GpuPruner::mesh and dynamic_setup; the CPU dual contouring ignores it): marching cubes
    // on the leaves instead of dual contouring. The vertices are the crossing points and every cycle
    // of crossing edges of a leaf becomes a fan of triangles, with the same face pairings as dual
    // contouring, so the mesh is closed.
    bool marching_cubes = false;
};

struct DCStats {
    uint64_t cells = 0, corners = 0, crossings = 0, vertices = 0, triangles = 0, analytic_hits = 0;
    uint64_t reused_corners = 0, reused_crossings = 0, reused_vertices = 0, static_cells = 0, ambiguous_faces = 0;
    double t_index = 0, t_corners = 0, t_edges = 0, t_qef = 0, t_faces = 0, total = 0;
};

struct Mesh {
    std::vector<double> V;          // 3 per vertex
    std::vector<uint32_t> F;        // 3 per triangle
    std::vector<uint32_t> tree_n;   // per vertex: size of the pruned tree of its cell
    std::vector<uint64_t> vertex_cell;  // per vertex: its (cell, component) key, see vertex_key
};

// Key of a vertex: cell coordinates (20 bits each, depth up to 20) and component in the cell.
inline uint64_t vertex_key(uint32_t x, uint32_t y, uint32_t z, int comp) {
    return ((uint64_t)x << 44) | ((uint64_t)y << 24) | ((uint64_t)z << 4) | (uint64_t)comp;
}
inline uint64_t cell_of_vertex_key(uint64_t k) {
    return cell_key((uint32_t)(k >> 44), (uint32_t)((k >> 24) & 0xfffff), (uint32_t)((k >> 4) & 0xfffff));
}

// A quad of the dual contour: the crossing edge, its four cells and their vertices, oriented.
struct QuadRecord {
    uint64_t edge_key;
    uint64_t cells[4];
    uint32_t v[4];
};

// A leaf of the finest level with the tree that is exact on its closure. `is_static` marks a
// cell whose field is known not to have changed since the cache was filled.
struct LeafRef {
    uint32_t x, y, z;
    TreeView tree;
    uint32_t tree_n;
    bool is_static;
};

namespace detail {
struct Crossing {
    double p[3];
    double n[3];
};
}  // namespace detail

struct DCCache {
    std::unordered_map<uint64_t, double> corner;              // corner key -> value
    std::unordered_map<uint64_t, detail::Crossing> crossing;  // edge key -> crossing
    std::unordered_map<uint64_t, std::array<double, 3>> vertex;  // cell key -> vertex
};

namespace detail {
// Formats `n` records in parallel chunks with `line(i, text)` and writes them in order. The
// writers below use std::to_chars, which produces the characters of printf("%.17g") and "%u"
// (the C++ standard defines it so), several times faster.
template <class Line>
inline void write_lines(FILE* f, size_t n, size_t bytes_per_line, Line line) {
    const size_t chunk = 1 << 16, chunks = (n + chunk - 1) / chunk;
    std::vector<std::string> parts(chunks);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t c = 0; c < chunks; c++) {
        std::string& text = parts[c];
        const size_t b = c * chunk, e = std::min(n, b + chunk);
        text.reserve((e - b) * bytes_per_line);
        for (size_t i = b; i < e; i++) line(i, text);
    }
    for (const std::string& text : parts) fwrite(text.data(), 1, text.size(), f);
}
inline void append_double(std::string& text, double v) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general, 17);
    text.append(buf, r.ptr);
}
inline void append_uint(std::string& text, uint32_t v) {
    char buf[16];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v);
    text.append(buf, r.ptr);
}
// Closes f; true when every write to it succeeded (the writers below return it, and false when
// the file cannot be opened).
inline bool close_written(FILE* f) {
    const bool ok = !ferror(f);
    return fclose(f) == 0 && ok;
}
}  // namespace detail

// Text OBJ: "v %.17g %.17g %.17g" per vertex and "f %u %u %u" per triangle (1-based).
[[nodiscard]] inline bool write_obj(const Mesh& m, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    detail::write_lines(f, m.V.size() / 3, 72, [&](size_t i, std::string& t) {
        t += 'v';
        for (int k = 0; k < 3; k++) { t += ' '; detail::append_double(t, m.V[3 * i + k]); }
        t += '\n';
    });
    detail::write_lines(f, m.F.size() / 3, 24, [&](size_t i, std::string& t) {
        t += 'f';
        for (int k = 0; k < 3; k++) { t += ' '; detail::append_uint(t, m.F[3 * i + k] + 1); }
        t += '\n';
    });
    return detail::close_written(f);
}

// Binary little-endian PLY with double vertex coordinates, so that meshes still compare byte
// for byte: the vertex array as it is, then one (3, a, b, c) record per triangle.
[[nodiscard]] inline bool write_ply(const Mesh& m, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t nv = m.V.size() / 3, nf = m.F.size() / 3;
    fprintf(f, "ply\nformat binary_little_endian 1.0\nelement vertex %zu\nproperty double x\nproperty double y\nproperty double z\n"
               "element face %zu\nproperty list uchar uint vertex_indices\nend_header\n", nv, nf);
    fwrite(m.V.data(), sizeof(double), m.V.size(), f);  // x86 and ARM are little-endian
    std::vector<unsigned char> faces(13 * nf);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nf; i++) {
        faces[13 * i] = 3;
        memcpy(&faces[13 * i + 1], &m.F[3 * i], 12);
    }
    fwrite(faces.data(), 1, faces.size(), f);
    return detail::close_written(f);
}

// PLY for a .ply path, OBJ otherwise.
[[nodiscard]] inline bool write_mesh(const Mesh& m, const std::string& path) {
    const bool ply = path.size() >= 4 && path.compare(path.size() - 4, 4, ".ply") == 0;
    return ply ? write_ply(m, path) : write_obj(m, path);
}

[[nodiscard]] inline bool write_vertex_attribute(const std::vector<uint32_t>& a, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    detail::write_lines(f, a.size(), 4, [&](size_t i, std::string& t) {
        detail::append_uint(t, a[i]);
        t += '\n';
    });
    return detail::close_written(f);
}

namespace detail {

// Root of f along the segment p0 + t (p1 - p0), t in [0, 1], with f0 f1 < 0: bracketing
// secant steps (Illinois) with bisection fallback. It stops when f vanishes, when the iterate moves
// less than tol (the secant has converged on one side of the bracket), or when the bracket is
// narrower than tol; converged is false only when max_iters ran out first, where f at the result
// may not be near zero.
inline double edge_root(const Scene& s, TreeView t, const double p0[3], const double p1[3], double f0, double f1,
                        int max_iters, Scratch& sc, Counters& c, double tol, bool* converged = nullptr) {
    double a = 0.0, b = 1.0, fa = f0, fb = f1;
    int side = 0;
    double m = 0.5, prev = -1.0;
    if (converged) *converged = true;
    for (int it = 0; it < max_iters; it++) {
        m = (fa * b - fb * a) / (fa - fb);
        if (!(m > a && m < b)) m = 0.5 * (a + b);
        if (std::fabs(m - prev) < tol) return m;
        prev = m;
        double q[3];
        for (int k = 0; k < 3; k++) q[k] = p0[k] + m * (p1[k] - p0[k]);
        const double fm = eval_tree(s, t, q, sc, c);
        if (fm == 0.0) return m;
        if ((fm < 0.0) == (fa < 0.0)) {
            a = m; fa = fm;
            if (side == -1) fb *= 0.5;
            side = -1;
        } else {
            b = m; fb = fm;
            if (side == 1) fa *= 0.5;
            side = 1;
        }
        if (b - a < tol) return 0.5 * (a + b);
    }
    if (converged) *converged = false;
    return 0.5 * (a + b);
}

// Unit normal of f at the crossing x of an edge along axis, by forward differences of step h: f(x)
// is zero to within the tolerance of edge_root when it converged, else it is evaluated. A null
// gradient gives the axis, oriented by f0, the value at the start of the edge. Next to a tip these
// first-order normals can push the vertex of the quadric out of its cell; project_vertex brings it
// back to the surface.
inline void crossing_normal(const Scene& s, TreeView t, const double x[3], double h, bool converged, int axis, double f0,
                            double n[3], Scratch& sc, Counters& c) {
    const double fx = converged ? 0.0 : eval_tree(s, t, x, sc, c);
    double g[3], len = 0.0;
    for (int a = 0; a < 3; a++) {
        double q[3] = {x[0], x[1], x[2]};
        q[a] += h;
        g[a] = (eval_tree(s, t, q, sc, c) - fx) / h;
        len += g[a] * g[a];
    }
    len = std::sqrt(len);
    if (len > 1e-300) for (int a = 0; a < 3; a++) n[a] = g[a] / len;
    else { for (int a = 0; a < 3; a++) n[a] = 0.0; n[axis] = f0 < 0.0 ? 1.0 : -1.0; }
}

// Moves a vertex of the quadric onto the surface: up to three Newton steps along the gradient of
// the pruned tree of its cell (forward differences of step h), each clamped to the cell, while |f|
// exceeds tol. Where the cell holds detail finer than itself, the quadric can place the vertex more
// than a cell away from the surface, a spike in the mesh; a vertex already on the surface, on a
// sharp feature too, stays where it is.
inline void project_vertex(const Scene& s, TreeView t, double x[3], const double lo[3], const double hi[3], double h,
                           double tol, Scratch& sc, Counters& c) {
    for (int it = 0; it < 3; it++) {
        const double f = eval_tree(s, t, x, sc, c);
        if (std::fabs(f) <= tol) return;
        double g[3], g2 = 0.0;
        for (int a = 0; a < 3; a++) {
            double q[3] = {x[0], x[1], x[2]};
            q[a] += h;
            g[a] = (eval_tree(s, t, q, sc, c) - f) / h;
            g2 += g[a] * g[a];
        }
        if (g2 < 1e-300) return;
        for (int a = 0; a < 3; a++) x[a] = std::min(std::max(x[a] - f * g[a] / g2, lo[a]), hi[a]);
    }
}

inline bool single_sphere(const Scene& s, TreeView t, const Primitive** P, int* sign) {
    if (t.n != 1) return false;
    const Node& n = s.nodes[t.e[0].node];
    if (n.type != NODE_PRIM || s.prims[n.idx].type != PRIM_SPHERE) return false;
    *P = &s.prims[n.idx];
    *sign = t.e[0].sign;
    return true;
}

}  // namespace detail

// Dual contouring of a list of leaves of depth `depth` (cell size `size`). `read_cache`
// supplies the data of static cells; `write_cache` receives every corner, crossing and
// vertex of this run.
inline Mesh dual_contour_leaves(const Scene& scene, int depth, const double size[3], std::vector<LeafRef> leaves,
                                const DCParams& params, const DCCache* read_cache, DCCache* write_cache,
                                Counters& counters, DCStats& st, std::vector<QuadRecord>* quads_out = nullptr) {
    Timer total;
    const uint64_t N = 1ull << depth;
    const uint64_t N1 = N + 1;
    const int nthreads = omp_get_max_threads();
    Mesh mesh;

    // Every step of the index runs on all threads: sorts in parallel (the same order, as the keys
    // are unique or deduplicated after), and a cell is found among the few leaves of its row
    // (x, y), whose range in the sorted leaves row_start gives (the whole list when the depth
    // makes N^2 rows too many).
    Timer ti;
    __gnu_parallel::sort(leaves.begin(), leaves.end(), [](const LeafRef& a, const LeafRef& b) {
        return cell_key(a.x, a.y, a.z) < cell_key(b.x, b.y, b.z);
    });
    const size_t nl = leaves.size();
    std::vector<uint64_t> lkeys(nl);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nl; i++) lkeys[i] = cell_key(leaves[i].x, leaves[i].y, leaves[i].z);
    const bool rows = depth <= 12;
    std::vector<uint32_t> row_start(rows ? N * N + 1 : 0);
    if (rows) {
        // Leaf i starts the rows after the row of leaf i - 1 up to its own.
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nl; i++) {
            const uint64_t r = (uint64_t)leaves[i].x * N + leaves[i].y;
            const uint64_t first = i == 0 ? 0 : (uint64_t)leaves[i - 1].x * N + leaves[i - 1].y + 1;
            for (uint64_t q = first; q <= r; q++) row_start[q] = (uint32_t)i;
        }
        const uint64_t after = nl == 0 ? 0 : (uint64_t)leaves[nl - 1].x * N + leaves[nl - 1].y + 1;
        std::fill(row_start.begin() + after, row_start.end(), (uint32_t)nl);
    }
    st.cells = nl;
    for (const LeafRef& l : leaves) if (l.is_static) st.static_cells++;
    auto find_cell = [&](int64_t x, int64_t y, int64_t z) -> int64_t {
        if (x < 0 || y < 0 || z < 0 || x >= (int64_t)N || y >= (int64_t)N || z >= (int64_t)N) return -1;
        const uint64_t key = cell_key((uint32_t)x, (uint32_t)y, (uint32_t)z);
        auto lo = lkeys.begin(), hi = lkeys.end();
        if (rows) {
            const uint64_t r = (uint64_t)x * N + (uint64_t)y;
            lo = lkeys.begin() + row_start[r];
            hi = lkeys.begin() + row_start[r + 1];
        }
        auto it = std::lower_bound(lo, hi, key);
        return it == hi || *it != key ? -1 : (int64_t)(it - lkeys.begin());
    };
    auto corner_key = [&](uint64_t x, uint64_t y, uint64_t z) { return (x * N1 + y) * N1 + z; };

    // Unique corners.
    std::vector<uint64_t> ckeys(leaves.size() * 8);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < leaves.size(); i++) {
        const LeafRef& c = leaves[i];
        for (int k = 0; k < 8; k++)
            ckeys[8 * i + k] = corner_key(c.x + (k & 1), c.y + ((k >> 1) & 1), c.z + ((k >> 2) & 1));
    }
    __gnu_parallel::sort(ckeys.begin(), ckeys.end());
    ckeys.erase(std::unique(ckeys.begin(), ckeys.end()), ckeys.end());
    std::vector<uint32_t> cell_corner(leaves.size() * 8);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < leaves.size(); i++) {
        const LeafRef& c = leaves[i];
        for (int k = 0; k < 8; k++) {
            const uint64_t key = corner_key(c.x + (k & 1), c.y + ((k >> 1) & 1), c.z + ((k >> 2) & 1));
            cell_corner[8 * i + k] = (uint32_t)(std::lower_bound(ckeys.begin(), ckeys.end(), key) - ckeys.begin());
        }
    }
    st.corners = ckeys.size();
    st.t_index = ti.seconds();

    auto corner_point = [&](uint64_t key, double p[3]) {
        const uint64_t cz = key % N1, cy = (key / N1) % N1, cx = key / (N1 * N1);
        p[0] = scene.aabb_min[0] + cx * size[0];
        p[1] = scene.aabb_min[1] + cy * size[1];
        p[2] = scene.aabb_min[2] + cz * size[2];
    };
    // The adjacent cell with the smallest tree (every adjacent tree is exact at the corner),
    // and whether any adjacent cell is static.
    auto corner_owner = [&](uint64_t key, bool* any_static) -> int64_t {
        const int64_t cz = key % N1, cy = (key / N1) % N1, cx = key / (N1 * N1);
        int64_t best = -1;
        *any_static = false;
        for (int k = 0; k < 8; k++) {
            const int64_t id = find_cell(cx - (k & 1), cy - ((k >> 1) & 1), cz - ((k >> 2) & 1));
            if (id < 0) continue;
            if (leaves[id].is_static) *any_static = true;
            if (best < 0 || leaves[id].tree_n < leaves[best].tree_n) best = id;
        }
        return best;
    };

    // Corner values.
    Timer tc;
    std::vector<double> cval(ckeys.size());
    std::vector<Counters> cnt_t(nthreads);
    std::vector<uint64_t> reused_c(nthreads, 0), reused_e(nthreads, 0), reused_v(nthreads, 0);
#pragma omp parallel
    {
        Scratch sc;
        const int tid = omp_get_thread_num();
        Counters& cnt = cnt_t[tid];
#pragma omp for schedule(dynamic, 1024)
        for (size_t u = 0; u < ckeys.size(); u++) {
            bool any_static = false;
            const int64_t owner = corner_owner(ckeys[u], &any_static);
            if (read_cache && any_static) {
                auto it = read_cache->corner.find(ckeys[u]);
                if (it != read_cache->corner.end()) { cval[u] = it->second; reused_c[tid]++; continue; }
            }
            double p[3];
            corner_point(ckeys[u], p);
            cval[u] = eval_tree(scene, leaves[owner].tree, p, sc, cnt);
        }
    }
    st.t_corners = tc.seconds();

    // Sign-changing edges. Edge key = corner key of the lower corner * 3 + axis.
    Timer te;
    static const int edge_corners[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    std::vector<std::vector<uint64_t>> ekeys_t(nthreads);
#pragma omp parallel
    {
        std::vector<uint64_t>& ek = ekeys_t[omp_get_thread_num()];
#pragma omp for schedule(static)
        for (size_t i = 0; i < leaves.size(); i++) {
            for (int e = 0; e < 12; e++) {
                const uint32_t c0 = cell_corner[8 * i + edge_corners[e][0]];
                const uint32_t c1 = cell_corner[8 * i + edge_corners[e][1]];
                if ((cval[c0] < 0.0) != (cval[c1] < 0.0)) ek.push_back(ckeys[c0] * 3 + (uint64_t)(e / 4));
            }
        }
    }
    std::vector<uint64_t> ekeys;
    {
        std::vector<size_t> off(nthreads + 1, 0);
        for (int t = 0; t < nthreads; t++) off[t + 1] = off[t] + ekeys_t[t].size();
        ekeys.resize(off[nthreads]);
#pragma omp parallel for schedule(static, 1)
        for (int t = 0; t < nthreads; t++) std::copy(ekeys_t[t].begin(), ekeys_t[t].end(), ekeys.begin() + off[t]);
    }
    __gnu_parallel::sort(ekeys.begin(), ekeys.end());
    ekeys.erase(std::unique(ekeys.begin(), ekeys.end()), ekeys.end());
    st.crossings = ekeys.size();

    // The four cells around an edge, counterclockwise around the +axis direction.
    auto edge_cells = [&](uint64_t ekey, int64_t ids[4]) {
        const int axis = (int)(ekey % 3);
        const uint64_t key = ekey / 3;
        const int64_t cz = key % N1, cy = (key / N1) % N1, cx = key / (N1 * N1);
        static const int off[4][2] = {{-1, -1}, {0, -1}, {0, 0}, {-1, 0}};
        for (int q = 0; q < 4; q++) {
            int64_t x = cx, y = cy, z = cz;
            if (axis == 0) { y += off[q][0]; z += off[q][1]; }
            else if (axis == 1) { z += off[q][0]; x += off[q][1]; }
            else { x += off[q][0]; y += off[q][1]; }
            ids[q] = find_cell(x, y, z);
        }
    };

    std::vector<detail::Crossing> cross(ekeys.size());
    std::vector<uint64_t> analytic_t(nthreads, 0);
#pragma omp parallel
    {
        Scratch sc;
        const int tid = omp_get_thread_num();
        Counters& cnt = cnt_t[tid];
#pragma omp for schedule(dynamic, 256)
        for (size_t u = 0; u < ekeys.size(); u++) {
            const int axis = (int)(ekeys[u] % 3);
            const uint64_t key = ekeys[u] / 3;
            int64_t ids[4];
            edge_cells(ekeys[u], ids);
            int64_t owner = -1;
            bool any_static = false;
            for (int q = 0; q < 4; q++) {
                if (ids[q] < 0) continue;
                if (leaves[ids[q]].is_static) any_static = true;
                if (owner < 0 || leaves[ids[q]].tree_n < leaves[owner].tree_n) owner = ids[q];
            }
            detail::Crossing& cr = cross[u];
            if (read_cache && any_static) {
                auto it = read_cache->crossing.find(ekeys[u]);
                if (it != read_cache->crossing.end()) { cr = it->second; reused_e[tid]++; continue; }
            }
            const TreeView tree = leaves[owner].tree;
            double p0[3], p1[3];
            corner_point(key, p0);
            for (int k = 0; k < 3; k++) p1[k] = p0[k];
            p1[axis] += size[axis];
            const uint64_t cz = key % N1, cy = (key / N1) % N1, cx = key / (N1 * N1);
            uint64_t key1 = axis == 0 ? corner_key(cx + 1, cy, cz) : axis == 1 ? corner_key(cx, cy + 1, cz) : corner_key(cx, cy, cz + 1);
            const double f0 = cval[std::lower_bound(ckeys.begin(), ckeys.end(), key) - ckeys.begin()];
            const double f1 = cval[std::lower_bound(ckeys.begin(), ckeys.end(), key1) - ckeys.begin()];
            const Primitive* P = nullptr;
            int psign = 1;
            bool done = false;
            if (params.analytic_leaves && detail::single_sphere(scene, tree, &P, &psign)) {
                // |M p0 + t M e + tr| = r, with M rigid: t^2 + 2 b t + c = 0 along the unit edge direction.
                V3<double> q0 = to_primitive_space(*P, V3<double>{p0[0], p0[1], p0[2]});
                double e[3] = {0, 0, 0};
                e[axis] = 1.0;
                const double qe[3] = {P->m[0] * e[0] + P->m[1] * e[1] + P->m[2] * e[2],
                                      P->m[4] * e[0] + P->m[5] * e[1] + P->m[6] * e[2],
                                      P->m[8] * e[0] + P->m[9] * e[1] + P->m[10] * e[2]};
                const double b = q0.x * qe[0] + q0.y * qe[1] + q0.z * qe[2];
                const double c = q0.x * q0.x + q0.y * q0.y + q0.z * q0.z - P->data[0] * P->data[0];
                const double disc = b * b - c;
                if (disc >= 0.0) {
                    const double sq = std::sqrt(disc);
                    const double t1 = -b - sq, t2 = -b + sq;
                    double t = -1.0;
                    if (t1 >= 0.0 && t1 <= size[axis]) t = t1;
                    else if (t2 >= 0.0 && t2 <= size[axis]) t = t2;
                    if (t >= 0.0) {
                        for (int k = 0; k < 3; k++) cr.p[k] = p0[k];
                        cr.p[axis] += t;
                        const double qx = q0.x + t * qe[0], qy = q0.y + t * qe[1], qz = q0.z + t * qe[2];
                        const double len = std::sqrt(qx * qx + qy * qy + qz * qz);
                        const double g[3] = {qx / len, qy / len, qz / len};
                        cr.n[0] = (P->m[0] * g[0] + P->m[4] * g[1] + P->m[8] * g[2]) * psign;
                        cr.n[1] = (P->m[1] * g[0] + P->m[5] * g[1] + P->m[9] * g[2]) * psign;
                        cr.n[2] = (P->m[2] * g[0] + P->m[6] * g[1] + P->m[10] * g[2]) * psign;
                        analytic_t[tid]++;
                        done = true;
                    }
                }
            }
            if (!done) {
                bool converged;
                const double t = detail::edge_root(scene, tree, p0, p1, f0, f1, params.max_iters, sc, cnt, 1e-12, &converged);
                for (int k = 0; k < 3; k++) cr.p[k] = p0[k] + t * (p1[k] - p0[k]);
                const double h = params.fd_rel * std::min(size[0], std::min(size[1], size[2]));
                detail::crossing_normal(scene, tree, cr.p, h, converged, axis, f0, cr.n, sc, cnt);
            }
        }
    }
    for (int t = 0; t < nthreads; t++) st.analytic_hits += analytic_t[t];
    st.t_edges = te.seconds();

    // Vertices: one per connected component of crossing edges of the cell (manifold dual
    // contouring). Two crossing edges are joined when a face of the cell pairs them; on a face
    // with four crossing edges the sign of the field at the face center decides the pairing,
    // so the two cells sharing the face agree. The usual single vertex per cell results when
    // all the crossing edges form one component.
    Timer tq;
    static int edge_of[8][8];
    static bool edge_of_ready = false;
    if (!edge_of_ready) {
        for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++) edge_of[a][b] = -1;
        for (int e = 0; e < 12; e++) { edge_of[edge_corners[e][0]][edge_corners[e][1]] = e; edge_of[edge_corners[e][1]][edge_corners[e][0]] = e; }
        edge_of_ready = true;
    }
    std::vector<int8_t> comp(leaves.size() * 12, -1);   // component of each crossing edge of each cell
    std::vector<uint8_t> ncomp(leaves.size(), 0);
    std::vector<uint64_t> ambiguous_t(nthreads, 0);
#pragma omp parallel
    {
        Scratch sc;
        const int tid = omp_get_thread_num();
        Counters& cnt = cnt_t[tid];
#pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < leaves.size(); i++) {
            bool crossing[12];
            int any = 0;
            for (int e = 0; e < 12; e++) {
                const uint32_t c0 = cell_corner[8 * i + edge_corners[e][0]], c1 = cell_corner[8 * i + edge_corners[e][1]];
                crossing[e] = (cval[c0] < 0.0) != (cval[c1] < 0.0);
                any += crossing[e];
            }
            if (!any) continue;
            int parent[12];
            for (int e = 0; e < 12; e++) parent[e] = e;
            auto find = [&](int e) { while (parent[e] != e) { parent[e] = parent[parent[e]]; e = parent[e]; } return e; };
            auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) parent[a] = b; };
            for (int axis = 0; axis < 3; axis++) {
                const int bx = (axis + 1) % 3, cx = (axis + 2) % 3;
                for (int side = 0; side < 2; side++) {
                    // Corners of the face in cyclic order and the edges between them.
                    int corners[4];
                    const int ob[4] = {0, 1, 1, 0}, oc[4] = {0, 0, 1, 1};
                    for (int q = 0; q < 4; q++) corners[q] = (side << axis) | (ob[q] << bx) | (oc[q] << cx);
                    int fe[4], nc = 0;
                    for (int q = 0; q < 4; q++) { fe[q] = edge_of[corners[q]][corners[(q + 1) % 4]]; nc += crossing[fe[q]]; }
                    if (nc == 2) {
                        int first = -1;
                        for (int q = 0; q < 4; q++) if (crossing[fe[q]]) { if (first < 0) first = fe[q]; else unite(first, fe[q]); }
                    } else if (nc == 4) {
                        // Ambiguous face: corners 0 and 2 share a sign, 1 and 3 the other.
                        const bool s0 = cval[cell_corner[8 * i + corners[0]]] < 0.0;
                        double center[3];
                        const LeafRef& l = leaves[i];
                        const uint32_t idx[3] = {l.x, l.y, l.z};
                        for (int a = 0; a < 3; a++) center[a] = scene.aabb_min[a] + (idx[a] + 0.5) * size[a];
                        center[axis] = scene.aabb_min[axis] + (idx[axis] + side) * size[axis];
                        const bool sc_in = eval_tree(scene, leaves[i].tree, center, sc, cnt) < 0.0;
                        ambiguous_t[tid]++;
                        if (sc_in == s0) { unite(fe[0], fe[1]); unite(fe[2], fe[3]); }  // corners 1 and 3 are cut off
                        else { unite(fe[3], fe[0]); unite(fe[1], fe[2]); }              // corners 0 and 2 are cut off
                    }
                }
            }
            int root_id[12];
            for (int e = 0; e < 12; e++) root_id[e] = -1;
            int k = 0;
            for (int e = 0; e < 12; e++) {
                if (!crossing[e]) continue;
                const int r = find(e);
                if (root_id[r] < 0) root_id[r] = k++;
                comp[12 * i + e] = (int8_t)root_id[r];
            }
            ncomp[i] = (uint8_t)k;
        }
    }
    std::vector<uint32_t> vbase(leaves.size() + 1, 0);
    for (size_t i = 0; i < leaves.size(); i++) vbase[i + 1] = vbase[i] + ncomp[i];
    const uint32_t nv = vbase[leaves.size()];
    mesh.V.assign(3 * (size_t)nv, 0.0);
    mesh.tree_n.resize(nv);
    mesh.vertex_cell.resize(nv);
    const double h_min = std::min(size[0], std::min(size[1], size[2]));
#pragma omp parallel
    {
    Scratch sc;
    Counters& cnt = cnt_t[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 256)
    for (size_t i = 0; i < leaves.size(); i++) {
        if (!ncomp[i]) continue;
        const LeafRef& c = leaves[i];
        for (int q = 0; q < ncomp[i]; q++) { mesh.tree_n[vbase[i] + q] = c.tree_n; mesh.vertex_cell[vbase[i] + q] = vertex_key(c.x, c.y, c.z, q); }
        if (read_cache && c.is_static) {
            bool all = true;
            for (int q = 0; q < ncomp[i] && all; q++) {
                auto it = read_cache->vertex.find(vertex_key(c.x, c.y, c.z, q));
                if (it == read_cache->vertex.end()) { all = false; break; }
                for (int a = 0; a < 3; a++) mesh.V[3 * (vbase[i] + q) + a] = it->second[a];
            }
            if (all) { reused_v[omp_get_thread_num()] += ncomp[i]; continue; }
        }
        Eigen::Matrix3d ATA[6];
        Eigen::Vector3d ATb[6], mass[6];
        int count[6] = {0, 0, 0, 0, 0, 0};
        for (int q = 0; q < ncomp[i]; q++) { ATA[q].setZero(); ATb[q].setZero(); mass[q].setZero(); }
        for (int e = 0; e < 12; e++) {
            const int q = comp[12 * i + e];
            if (q < 0) continue;
            const uint32_t c0 = cell_corner[8 * i + edge_corners[e][0]];
            const uint64_t ekey = ckeys[c0] * 3 + (uint64_t)(e / 4);
            const size_t u = std::lower_bound(ekeys.begin(), ekeys.end(), ekey) - ekeys.begin();
            const detail::Crossing& cr = cross[u];
            Eigen::Vector3d n(cr.n[0], cr.n[1], cr.n[2]), p(cr.p[0], cr.p[1], cr.p[2]);
            ATA[q] += n * n.transpose();
            ATb[q] += n * n.dot(p);
            mass[q] += p;
            count[q]++;
        }
        double lo[3], hi[3];
        const uint32_t idx[3] = {c.x, c.y, c.z};
        for (int a = 0; a < 3; a++) { lo[a] = scene.aabb_min[a] + idx[a] * size[a]; hi[a] = lo[a] + size[a]; }
        for (int q = 0; q < ncomp[i]; q++) {
            mass[q] /= count[q];
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(ATA[q]);
            const Eigen::Vector3d ev = es.eigenvalues();
            const double lmax = ev.maxCoeff();
            Eigen::Matrix3d pinv = Eigen::Matrix3d::Zero();
            for (int k = 0; k < 3; k++) {
                if (ev[k] > 1e-2 * lmax && ev[k] > 1e-12) {
                    const Eigen::Vector3d v = es.eigenvectors().col(k);
                    pinv += v * v.transpose() / ev[k];
                }
            }
            Eigen::Vector3d x = mass[q] + pinv * (ATb[q] - ATA[q] * mass[q]);
            double v[3];
            for (int a = 0; a < 3; a++) v[a] = std::min(std::max(x[a], lo[a]), hi[a]);
            if (params.project_vertices) detail::project_vertex(scene, c.tree, v, lo, hi, params.fd_rel * h_min, 0.05 * h_min, sc, cnt);
            for (int a = 0; a < 3; a++) mesh.V[3 * (vbase[i] + q) + a] = v[a];
        }
    }
    }
    for (int t = 0; t < nthreads; t++) st.ambiguous_faces += ambiguous_t[t];
    st.t_qef = tq.seconds();

    // Faces: one quad per crossing edge, joining the component vertices of its four cells.
    Timer tf;
    auto local_edge = [&](const LeafRef& l, uint64_t corner, int axis) -> int {
        const uint64_t cz = corner % N1, cy = (corner / N1) % N1, cx = corner / (N1 * N1);
        const int bits[3] = {(int)(cx - l.x), (int)(cy - l.y), (int)(cz - l.z)};
        const int b = axis == 0 ? 1 : 0, c = axis == 2 ? 1 : 2;
        return 4 * axis + bits[b] + 2 * bits[c];
    };
    // In parallel: each thread takes a contiguous range of the edges and writes its quads after
    // those of the ranges before it, so that they keep the order of the edges.
    {
        const size_t E = ekeys.size();
        std::vector<size_t> made(nthreads + 1, 0);
        std::vector<std::vector<uint32_t>> f_t(nthreads);
        std::vector<std::vector<QuadRecord>> q_t(nthreads);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
            const size_t u0 = E * tid / nt, u1 = E * (tid + 1) / nt;
            std::vector<uint32_t>& f = f_t[tid];
            f.reserve(6 * (u1 - u0));
            for (size_t u = u0; u < u1; u++) {
                int64_t ids[4];
                edge_cells(ekeys[u], ids);
                const int axis = (int)(ekeys[u] % 3);
                const uint64_t key = ekeys[u] / 3;
                int64_t v[4];
                bool ok = true;
                for (int q = 0; q < 4; q++) {
                    if (ids[q] < 0) { ok = false; break; }
                    const int le = local_edge(leaves[ids[q]], key, axis);
                    const int cq = comp[12 * ids[q] + le];
                    if (cq < 0) { ok = false; break; }
                    v[q] = vbase[ids[q]] + cq;
                }
                if (!ok) continue;
                const double f0 = cval[std::lower_bound(ckeys.begin(), ckeys.end(), key) - ckeys.begin()];
                if (f0 >= 0.0) { std::swap(v[1], v[3]); std::swap(ids[1], ids[3]); }  // orient the normal from the inside to the outside
                f.insert(f.end(), {(uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2], (uint32_t)v[0], (uint32_t)v[2], (uint32_t)v[3]});
                if (quads_out) {
                    QuadRecord q;
                    q.edge_key = ekeys[u];
                    for (int k = 0; k < 4; k++) { const LeafRef& l = leaves[ids[k]]; q.cells[k] = cell_key(l.x, l.y, l.z); q.v[k] = (uint32_t)v[k]; }
                    q_t[tid].push_back(q);
                }
            }
            made[tid + 1] = f.size();
#pragma omp barrier
#pragma omp single
            {
                for (int t = 0; t < nt; t++) made[t + 1] += made[t];
                mesh.F.resize(made[nt]);
            }
            std::copy(f.begin(), f.end(), mesh.F.begin() + made[tid]);
        }
        if (quads_out)
            for (int t = 0; t < nthreads; t++) quads_out->insert(quads_out->end(), q_t[t].begin(), q_t[t].end());
    }
    st.t_faces = tf.seconds();

    if (write_cache) {
        write_cache->corner.clear(); write_cache->crossing.clear(); write_cache->vertex.clear();
        write_cache->corner.reserve(ckeys.size() * 2);
        for (size_t u = 0; u < ckeys.size(); u++) write_cache->corner.emplace(ckeys[u], cval[u]);
        write_cache->crossing.reserve(ekeys.size() * 2);
        for (size_t u = 0; u < ekeys.size(); u++) write_cache->crossing.emplace(ekeys[u], cross[u]);
        write_cache->vertex.reserve(nv * 2);
        for (uint32_t v = 0; v < nv; v++)
            write_cache->vertex.emplace(mesh.vertex_cell[v], std::array<double, 3>{mesh.V[3 * v], mesh.V[3 * v + 1], mesh.V[3 * v + 2]});
    }
    for (int t = 0; t < nthreads; t++) {
        counters.add(cnt_t[t]);
        st.reused_corners += reused_c[t]; st.reused_crossings += reused_e[t]; st.reused_vertices += reused_v[t];
    }
    st.vertices = mesh.V.size() / 3;
    st.triangles = mesh.F.size() / 3;
    st.total = total.seconds();
    return mesh;
}

inline Mesh dual_contour(const Octree& oct, const DCParams& params, Counters& counters, DCStats& st,
                         DCCache* write_cache = nullptr) {
    const int depth = (int)oct.levels.size() - 1;
    const Level& L = oct.levels[depth];
    std::vector<LeafRef> leaves;
    leaves.reserve(L.cells.size());
    for (const Cell& c : L.cells) {
        if (c.far) continue;
        leaves.push_back(LeafRef{c.x, c.y, c.z, oct.tree_of(depth, c), c.tree_n, false});
    }
    return dual_contour_leaves(*oct.scene, depth, L.size, std::move(leaves), params, nullptr, write_cache, counters, st);
}

}  // namespace lpm
