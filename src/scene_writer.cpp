#include <fstream>

#include "json.hpp"
#include "lpm/scene.h"
#include "lpm/synthetic.h"

namespace lpm {

bool write_scene_json(const Scene& s, const std::string& path) {
    using json = nlohmann::json;
    // Bottom-up assembly in post-order; the children objects are moved into the parent.
    std::vector<json> built(s.nodes.size());
    const json identity = json::array({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0});
    for (size_t i = 0; i < s.nodes.size(); i++) {
        const Node& n = s.nodes[i];
        json j;
        if (n.type == NODE_PRIM) {
            const Primitive& P = s.prims[n.idx];
            j["nodeType"] = "primitive";
            json m = json::array();
            for (int q = 0; q < 12; q++) m.push_back(P.m[q]);
            j["matrix"] = m;
            j["round_x"] = P.extrude_rounding[0];
            j["round_y"] = P.extrude_rounding[1];
            j["color"] = json::array({((P.color >> 0) & 0xff) / 255.0, ((P.color >> 8) & 0xff) / 255.0, ((P.color >> 16) & 0xff) / 255.0});
            switch (P.type) {
                case PRIM_SPHERE: j["primitiveType"] = "sphere"; j["radius"] = P.data[0]; break;
                case PRIM_BOX:
                    j["primitiveType"] = "box";
                    j["sides"] = json::array({P.data[0], P.data[1], P.data[2]});
                    j["bevel"] = json::array({2.0 * P.corner_rounding[0], 2.0 * P.corner_rounding[1], 2.0 * P.corner_rounding[2], 2.0 * P.corner_rounding[3]});
                    break;
                case PRIM_CYLINDER: j["primitiveType"] = "cylinder"; j["height"] = P.data[0]; j["radius"] = P.data[1]; break;
                case PRIM_CONE: j["primitiveType"] = "cone"; j["radius"] = P.data[0]; j["height"] = P.data[1]; break;
            }
        } else {
            const BinaryOp& op = s.ops[n.idx];
            j["nodeType"] = "binaryOperator";
            j["matrix"] = identity;
            j["blendMode"] = op.op == OP_UNION ? "union" : (op.op == OP_SUB ? "sub" : "inter");
            j["blendRadius"] = op.k;
            j["leftChild"] = std::move(built[n.left]);
            j["rightChild"] = std::move(built[n.right]);
        }
        built[i] = std::move(j);
    }
    json root = std::move(built.back());
    root["aabb_min"] = json::array({s.aabb_min[0], s.aabb_min[1], s.aabb_min[2]});
    root["aabb_max"] = json::array({s.aabb_max[0], s.aabb_max[1], s.aabb_max[2]});
    std::ofstream out(path);
    out << root.dump();
    out.close();
    return !out.fail();
}

}  // namespace lpm
