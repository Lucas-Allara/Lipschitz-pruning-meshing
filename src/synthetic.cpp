#include "lpm/synthetic.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace lpm {
namespace {

struct Builder {
    Scene s;
    std::mt19937_64 rng;
    const SyntheticParams& p;
    Builder(const SyntheticParams& pp) : rng(pp.seed), p(pp) {}

    double uni(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }

    // Random rotation from a uniform unit quaternion, composed with the translation.
    void rigid(Primitive& P, const double c[3], bool rotate) {
        double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (rotate) {
            double q[4], n = 0.0;
            for (double& v : q) { v = std::normal_distribution<double>(0.0, 1.0)(rng); n += v * v; }
            n = std::sqrt(n);
            for (double& v : q) v /= n;
            const double w = q[0], x = q[1], y = q[2], z = q[3];
            R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w);     R[2] = 2 * (x * z + y * w);
            R[3] = 2 * (x * y + z * w);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
            R[6] = 2 * (x * z - y * w);     R[7] = 2 * (y * z + x * w);     R[8] = 1 - 2 * (x * x + y * y);
        }
        for (int r = 0; r < 3; r++) {
            for (int col = 0; col < 3; col++) P.m[4 * r + col] = R[3 * r + col];
            P.m[4 * r + 3] = -(R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]);
        }
    }

    std::vector<Primitive> pending;  // primitives generated up front (Morton order)
    size_t next_pending = 0;

    int add_leaf() {
        if (next_pending < pending.size()) {
            const Primitive& Q = pending[next_pending++];
            Node n;
            n.type = NODE_PRIM;
            n.idx = (int)s.prims.size();
            s.prims.push_back(Q);
            s.nodes.push_back(n);
            return (int)s.nodes.size() - 1;
        }
        Primitive P;
        double c[3];
        for (double& v : c) v = uni(-p.extent, p.extent);
        const double r = p.radius;
        int type = PRIM_SPHERE;
        if (p.shapes == "mixed") type = (int)std::uniform_int_distribution<int>(0, 3)(rng);
        P.type = type;
        switch (type) {
            case PRIM_SPHERE: P.data[0] = r * uni(0.5, 1.5); break;
            case PRIM_BOX:
                for (int a = 0; a < 3; a++) P.data[a] = 2.0 * r * uni(0.5, 1.5);
                for (int a = 0; a < 4; a++) P.corner_rounding[a] = (uni(0, 1) < 0.5) ? 0.0 : 0.25 * r * uni(0, 1);
                P.extrude_rounding[0] = (uni(0, 1) < 0.5) ? 0.0 : 0.2 * r * uni(0, 1);
                P.extrude_rounding[1] = (uni(0, 1) < 0.5) ? 0.0 : 0.2 * r * uni(0, 1);
                break;
            case PRIM_CYLINDER: P.data[0] = 2.0 * r * uni(0.5, 1.5); P.data[1] = r * uni(0.5, 1.5); break;
            case PRIM_CONE: P.data[0] = r * uni(0.5, 1.5); P.data[1] = 2.0 * r * uni(0.5, 1.5); break;
        }
        rigid(P, c, p.shapes == "mixed");
        P.color = (uint32_t)std::uniform_int_distribution<uint32_t>(0, 0xffffff)(rng);
        Node n;
        n.type = NODE_PRIM;
        n.idx = (int)s.prims.size();
        s.prims.push_back(P);
        s.nodes.push_back(n);
        return (int)s.nodes.size() - 1;
    }

    int add_op(int left, int right) {
        BinaryOp op;
        op.k = p.k;
        int kind = OP_UNION;
        if (p.ops == "mixed") {
            const double u = uni(0, 1);
            kind = u < 0.7 ? OP_UNION : (u < 0.9 ? OP_SUB : OP_INTER);
        }
        op.op = kind;
        op.s = kind == OP_UNION ? 1.0 : -1.0;
        Node n;
        n.type = NODE_OP;
        n.idx = (int)s.ops.size();
        n.left = left;
        n.right = right;
        s.ops.push_back(op);
        s.nodes.push_back(n);
        const int id = (int)s.nodes.size() - 1;
        s.nodes[left].parent = id;
        s.nodes[right].parent = id;
        s.nodes[right].sign = kind == OP_SUB ? -1 : 1;
        return id;
    }

    int balanced(int count) {  // emits the leaves and operators in post-order
        if (count == 1) return add_leaf();
        const int half = count / 2;
        const int l = balanced(half);
        const int r = balanced(count - half);
        return add_op(l, r);
    }
};

}  // namespace

namespace {
uint64_t part_by_2(uint64_t x) {
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}
}  // namespace

Scene make_synthetic(const SyntheticParams& p) {
    Builder b(p);
    if (p.morton) {
        // Generate every primitive first, then sort by the Morton code of its center so that
        // neighboring leaves of the tree are neighbors in space.
        std::vector<std::pair<uint64_t, Primitive>> items;
        for (int i = 0; i < p.n; i++) {
            b.add_leaf();
            const Primitive& P = b.s.prims.back();
            // center = -R^T t for a rigid world-to-primitive transform
            double c[3];
            for (int a = 0; a < 3; a++) c[a] = -(P.m[a] * P.m[3] + P.m[4 + a] * P.m[7] + P.m[8 + a] * P.m[11]);
            uint64_t q[3];
            for (int a = 0; a < 3; a++) q[a] = (uint64_t)std::min(std::max((c[a] + p.extent) / (2.0 * p.extent) * 2097151.0, 0.0), 2097151.0);
            items.push_back({part_by_2(q[0]) | (part_by_2(q[1]) << 1) | (part_by_2(q[2]) << 2), P});
        }
        b.s.prims.clear();
        b.s.nodes.clear();
        std::sort(items.begin(), items.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        for (auto& it : items) b.pending.push_back(it.second);
    }
    if (p.chain) {
        int prev = b.add_leaf();
        for (int i = 1; i < p.n; i++) {
            const int leaf = b.add_leaf();
            prev = b.add_op(prev, leaf);
        }
    } else {
        b.balanced(p.n);
    }
    Scene s = std::move(b.s);
    const double pad = 3.0 * p.radius + p.k + 0.05;
    for (int a = 0; a < 3; a++) { s.aabb_min[a] = -p.extent - pad; s.aabb_max[a] = p.extent + pad; }
    std::vector<int> h(s.nodes.size(), 1);
    for (size_t i = 0; i < s.nodes.size(); i++)
        if (s.nodes[i].type == NODE_OP) h[i] = 1 + std::max(h[s.nodes[i].left], h[s.nodes[i].right]);
    s.depth = h.empty() ? 0 : h.back();
    s.name = "synthetic";
    return s;
}

}  // namespace lpm
