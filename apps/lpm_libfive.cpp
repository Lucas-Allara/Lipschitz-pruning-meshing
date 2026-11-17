// Meshes a scene with libfive (Keeter 2020): interval arithmetic octree with tape pruning
// and dual contouring. The scene is translated node by node into a libfive expression.
#include <libfive/render/brep/mesh.hpp>
#include <libfive/render/brep/region.hpp>
#include <libfive/render/brep/settings.hpp>
#include <libfive/tree/tree.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "lpm/scene.h"
#include "lpm/stats.h"

using libfive::Tree;
using namespace lpm;

// c > 0 ? a : b, through compare (-1, 0, 1); the tie takes b as in the shader.
static Tree select_pos(Tree c, Tree a, Tree b) { return b + (a - b) * max(compare(c, Tree(0.0)), Tree(0.0)); }
static Tree len2(Tree x, Tree y) { return sqrt(square(x) + square(y)); }
static Tree len3(Tree x, Tree y, Tree z) { return sqrt(square(x) + square(y) + square(z)); }

static Tree primitive_tree(const Primitive& P) {
    const Tree X = Tree::X(), Y = Tree::Y(), Z = Tree::Z();
    const Tree px = Tree(P.m[0]) * X + Tree(P.m[1]) * Y + Tree(P.m[2]) * Z + Tree(P.m[3]);
    const Tree py = Tree(P.m[4]) * X + Tree(P.m[5]) * Y + Tree(P.m[6]) * Z + Tree(P.m[7]);
    const Tree pz = Tree(P.m[8]) * X + Tree(P.m[9]) * Y + Tree(P.m[10]) * Z + Tree(P.m[11]);
    const Tree zero(0.0);
    switch (P.type) {
        case PRIM_SPHERE:
            return len3(px, py, pz) - Tree(P.data[0]);
        case PRIM_BOX: {
            const double hx = P.data[0] * 0.5, hy = P.data[1] * 0.5, hz = P.data[2] * 0.5;
            const double* r = P.corner_rounding;
            Tree rx = select_pos(px, Tree(r[0]), Tree(r[2]));
            Tree ry = select_pos(px, Tree(r[1]), Tree(r[3]));
            Tree rr = select_pos(pz, rx, ry);
            Tree qx = abs(px) - Tree(hx) + rr;
            Tree qz = abs(pz) - Tree(hz) + rr;
            Tree d2 = min(max(qx, qz), zero) + len2(max(qx, zero), max(qz, zero)) - rr;
            Tree er = select_pos(py, Tree(P.extrude_rounding[0]), Tree(P.extrude_rounding[1]));
            Tree ex = d2 + er;
            Tree ey = abs(py) - (Tree(hy) - er);
            return min(max(ex, ey), zero) + len2(max(ex, zero), max(ey, zero)) - er;
        }
        case PRIM_CYLINDER: {
            const double h = P.data[0] * 0.5, rad = P.data[1];
            Tree dx = len2(px, pz) - Tree(rad);
            Tree dy = abs(py) - Tree(h);
            return min(max(dx, dy), zero) + len2(max(dx, zero), max(dy, zero));
        }
        case PRIM_CONE: {
            const double radius = P.data[0], hh = P.data[1] * 0.5;
            const double ex = -radius, ey = 2.0 * hh;
            Tree p2x = len2(px, pz) - Tree(radius);
            Tree p2y = py + Tree(hh);
            Tree t = min(max((p2x * Tree(ex) + p2y * Tree(ey)) * Tree(1.0 / (ex * ex + ey * ey)), zero), Tree(1.0));
            Tree qx = p2x - Tree(ex) * t;
            Tree qy = p2y - Tree(ey) * t;
            Tree d = len2(qx, qy);
            return select_pos(max(qx, qy), d, -min(d, p2y));
        }
    }
    return Tree(1e20);
}

// libfive::Tree::optimized() descends into shared subtrees without memoization, so a blended
// operator, which references each operand in min(a, b) and in |a - b|, makes that pass
// exponential in the height of the tree (height 20: 4 s; height 25: more than two minutes).
// The pass is only an optimization: marking the tree as already optimized skips it, and
// the tape is then built from the DAG by Tree::walk(), which does count shared nodes once.
struct TreeAccess : libfive::Tree {
    explicit TreeAccess(const Tree& t) : Tree(t) {}
    Tree mark_optimized() const { return with_flags(1u); }  // TREE_FLAG_IS_OPTIMIZED = 1 << 0
};

static Tree op_tree(const BinaryOp& op, Tree a, Tree b) {
    const Tree s(op.s);
    Tree kern(0.0);
    if (op.k > 0.0) {
        Tree m = max(Tree(op.k) - abs(a - b), Tree(0.0));
        kern = square(m) * Tree(0.25 / op.k);
    }
    return s * (min(s * a, s * b) - kern);
}

int main(int argc, char** argv) {
    std::string scene_path, out_path, stats_path;
    int depth = 8, threads = 8;
    double max_err = 1e-8;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) exit(1); return argv[++i]; };
        if (a == "--scene") scene_path = next();
        else if (a == "--depth") depth = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--out") out_path = next();
        else if (a == "--stats") stats_path = next();
        else if (a == "--max-err") max_err = atof(next().c_str());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (scene_path.empty()) { fprintf(stderr, "usage: lpm_libfive --scene FILE [--depth D] [--threads T] [--out mesh.obj] [--stats stats.json]\n"); return 1; }

    Timer t_all;
    Scene scene = load_scene_json(scene_path);
    Timer t_tree;
    std::vector<Tree> stack;
    for (const Node& n : scene.nodes) {
        Tree t = n.type == NODE_PRIM ? primitive_tree(scene.prims[n.idx]) : Tree(0.0);
        if (n.type == NODE_OP) {
            Tree b = stack.back(); stack.pop_back();
            Tree a = stack.back(); stack.pop_back();
            t = op_tree(scene.ops[n.idx], a, b);
        }
        stack.push_back(n.sign < 0 ? -t : t);
    }
    Tree root = TreeAccess(stack.back()).mark_optimized();
    const double tree_s = t_tree.seconds();

    // The region is the scene bounding box; libfive subdivides until the smallest edge of
    // a cell is below min_feature, which gives 2^depth cells along the largest axis.
    double extent = 0.0;
    for (int a = 0; a < 3; a++) extent = std::max(extent, scene.aabb_max[a] - scene.aabb_min[a]);
    libfive::Region<3> region({scene.aabb_min[0], scene.aabb_min[1], scene.aabb_min[2]},
                              {scene.aabb_max[0], scene.aabb_max[1], scene.aabb_max[2]});
    libfive::BRepSettings settings;
    settings.min_feature = extent / (double)(1 << depth) * 1.0001;
    settings.workers = threads;
    settings.max_err = max_err;
    settings.alg = libfive::DUAL_CONTOURING;
    Timer t_render;
    auto mesh = libfive::Mesh::render(root, region, settings);
    const double render_s = t_render.seconds();
    if (!mesh) { fprintf(stderr, "libfive returned no mesh\n"); return 1; }

    if (!out_path.empty()) {
        FILE* f = fopen(out_path.c_str(), "w");
        for (const auto& v : mesh->verts) fprintf(f, "v %.9g %.9g %.9g\n", v.x(), v.y(), v.z());
        for (const auto& t : mesh->branes) fprintf(f, "f %u %u %u\n", t.x() + 1, t.y() + 1, t.z() + 1);
        fclose(f);
    }
    FILE* f = stats_path.empty() ? stdout : fopen(stats_path.c_str(), "w");
    fprintf(f, "{\"scene\": \"%s\", \"method\": \"libfive\", \"depth\": %d, \"threads\": %d, \"max_err\": %g, \"min_feature\": %.9g,\n",
            scene_path.c_str(), depth, threads, max_err, settings.min_feature);
    fprintf(f, " \"nodes\": %zu, \"prims\": %zu, \"time_tree\": %.6f, \"time_render\": %.6f, \"time_total\": %.6f,\n",
            scene.nodes.size(), scene.prims.size(), tree_s, render_s, t_all.seconds());
    fprintf(f, " \"vertices\": %zu, \"triangles\": %zu, \"peak_rss_mb\": %.1f}\n", mesh->verts.size() - 1, mesh->branes.size(), peak_rss_mb());
    if (f != stdout) fclose(f);
    return 0;
}
