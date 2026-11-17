#include "lpm/scene.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "json.hpp"

namespace lpm {
namespace {

using json = nlohmann::json;

struct Affine {  // 3x4 row major
    double m[12];
};

Affine identity() { return Affine{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}}; }

Affine compose(const Affine& a, const Affine& b) {  // a * b, applied as a(b(p))
    Affine c;
    for (int r = 0; r < 3; r++) {
        for (int col = 0; col < 4; col++) {
            double v = 0.0;
            for (int k = 0; k < 3; k++) v += a.m[4 * r + k] * b.m[4 * k + col];
            if (col == 3) v += a.m[4 * r + 3];
            c.m[4 * r + col] = v;
        }
    }
    return c;
}

Affine read_matrix(const json& j) {
    Affine a = identity();
    if (j.contains("matrix")) {
        const auto& m = j.at("matrix");
        for (int i = 0; i < 12; i++) a.m[i] = m.at(i).get<double>();
    }
    return a;
}

struct Pending {
    const json* j;
    Affine mat;      // product of the ancestors' matrices
    int sign;
    int parent;      // index in the temporary pre-order array
    int side;        // 0 left, 1 right
};

struct Temp {
    const json* j;
    Affine mat;
    int sign;
    int parent;
    int left = -1, right = -1;
};

}  // namespace

Scene load_scene_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open scene " + path);
    // Keys are read with at(): a scene without one of them is reported, not undefined behavior.
    json doc = json::parse(in);
    Scene scene;
    scene.name = path;
    for (int i = 0; i < 3; i++) {
        scene.aabb_min[i] = doc.at("aabb_min").at(i).get<double>();
        scene.aabb_max[i] = doc.at("aabb_max").at(i).get<double>();
    }

    // Pass 1: iterative pre-order expansion into a temporary array (the official
    // scenes nest more than a thousand levels deep).
    std::vector<Temp> temp;
    std::vector<Pending> stack;
    stack.push_back({&doc, identity(), 1, -1, 0});
    while (!stack.empty()) {
        Pending e = stack.back();
        stack.pop_back();
        const json& j = *e.j;
        int id = (int)temp.size();
        Affine node_mat = read_matrix(j);
        temp.push_back({&j, compose(e.mat, node_mat), e.sign, e.parent});
        if (e.parent >= 0) {
            if (e.side == 0) temp[e.parent].left = id; else temp[e.parent].right = id;
        }
        if (j.at("nodeType").get<std::string>() == "binaryOperator") {
            std::string mode = j.at("blendMode").get<std::string>();
            int sign_right = (mode == "sub") ? -1 : 1;
            // Push right first so that left is expanded first.
            stack.push_back({&j.at("rightChild"), temp.back().mat, sign_right, id, 1});
            stack.push_back({&j.at("leftChild"), temp.back().mat, 1, id, 0});
        }
    }

    // Pass 2: post-order numbering.
    std::vector<int> post_index(temp.size(), -1);
    std::vector<std::pair<int, int>> st;  // (temp id, state)
    st.push_back({0, 0});
    std::vector<int> order;
    while (!st.empty()) {
        auto [id, state] = st.back();
        st.pop_back();
        const Temp& t = temp[id];
        if (t.left < 0) { order.push_back(id); continue; }
        if (state == 0) {
            st.push_back({id, 1});
            st.push_back({t.right, 0});
            st.push_back({t.left, 0});
        } else {
            order.push_back(id);
        }
    }
    for (size_t i = 0; i < order.size(); i++) post_index[order[i]] = (int)i;

    scene.nodes.resize(order.size());
    for (size_t i = 0; i < order.size(); i++) {
        const Temp& t = temp[order[i]];
        const json& j = *t.j;
        Node& n = scene.nodes[i];
        n.sign = t.sign;
        n.parent = t.parent >= 0 ? post_index[t.parent] : -1;
        if (t.left >= 0) {
            n.type = NODE_OP;
            n.left = post_index[t.left];
            n.right = post_index[t.right];
            BinaryOp op;
            std::string mode = j.at("blendMode").get<std::string>();
            if (mode == "union") { op.op = OP_UNION; op.s = 1.0; }
            else if (mode == "sub") { op.op = OP_SUB; op.s = -1.0; }
            else if (mode == "inter") { op.op = OP_INTER; op.s = -1.0; }
            else throw std::runtime_error("unknown blend mode " + mode);
            op.k = j.value("blendRadius", 0.0);
            n.idx = (int)scene.ops.size();
            scene.ops.push_back(op);
        } else {
            n.type = NODE_PRIM;
            Primitive P;
            for (int q = 0; q < 12; q++) P.m[q] = t.mat.m[q];
            std::string pt = j.at("primitiveType").get<std::string>();
            if (pt == "sphere") {
                P.type = PRIM_SPHERE;
                P.data[0] = j.at("radius").get<double>();
            } else if (pt == "box") {
                P.type = PRIM_BOX;
                for (int q = 0; q < 3; q++) P.data[q] = j.at("sides").at(q).get<double>();
                // Replicates the 8-bit quantization of the official loader and shader:
                // encoded with scale 2 max(sx, sz), decoded with scale max(sx, sz).
                double scale = std::max(P.data[0], P.data[2]) * 2.0;
                for (int q = 0; q < 4; q++) {
                    double bevel = j.contains("bevel") ? j.at("bevel").at(q).get<double>() : 0.0;
                    uint32_t byte = scale > 0.0 ? (uint32_t)(float)(bevel * 255.0 * 2.0 / scale) : 0u;
                    byte &= 0xffu;
                    P.corner_rounding[q] = (double)byte * (scale * 0.5) / 255.0 / 2.0;
                }
            } else if (pt == "cylinder") {
                P.type = PRIM_CYLINDER;
                P.data[0] = j.at("height").get<double>();
                P.data[1] = j.at("radius").get<double>();
            } else if (pt == "cone") {
                P.type = PRIM_CONE;
                P.data[0] = j.at("radius").get<double>();
                P.data[1] = j.at("height").get<double>();
            } else {
                throw std::runtime_error("unknown primitive " + pt);
            }
            P.extrude_rounding[0] = j.value("round_x", 0.0);
            P.extrude_rounding[1] = j.value("round_y", 0.0);
            if (j.contains("color")) {
                uint32_t c = 0;
                for (int q = 0; q < 3; q++) {
                    const double channel = std::min(std::max(j.at("color").at(q).get<double>(), 0.0), 1.0);
                    uint32_t v = std::min((uint32_t)(channel * 255.99), 255u);
                    c |= v << (8 * q);
                }
                P.color = c;
            }
            n.idx = (int)scene.prims.size();
            scene.prims.push_back(P);
        }
    }
    // Height of the tree, computed bottom-up in post-order.
    std::vector<int> h(scene.nodes.size(), 1);
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        const Node& n = scene.nodes[i];
        if (n.type == NODE_OP) h[i] = 1 + std::max(h[n.left], h[n.right]);
    }
    scene.depth = h.empty() ? 0 : h.back();
    return scene;
}

}  // namespace lpm
