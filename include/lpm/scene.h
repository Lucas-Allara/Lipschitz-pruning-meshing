// Smooth CSG tree in the format of the official Lipschitz Pruning scenes.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lpm {

enum PrimType { PRIM_SPHERE = 0, PRIM_BOX = 1, PRIM_CYLINDER = 2, PRIM_CONE = 3 };
enum OpType { OP_UNION = 0, OP_SUB = 1, OP_INTER = 2 };
enum NodeType { NODE_OP = 0, NODE_PRIM = 1 };

struct Primitive {
    int type = PRIM_SPHERE;
    double m[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};  // world to primitive, 3x4 row major
    double data[4] = {0, 0, 0, 0};       // sphere: radius | box: sides | cylinder: height, radius | cone: radius, height
    double corner_rounding[4] = {0, 0, 0, 0};  // box, decoded as in the official shader
    double extrude_rounding[2] = {0, 0};       // box, rounding of the top and bottom faces
    uint32_t color = 0xffffff;
};

struct BinaryOp {
    int op = OP_UNION;
    double k = 0.0;   // blending radius
    double s = 1.0;   // +1 for min (union), -1 for max (intersection, difference)
};

// Nodes are stored in post-order: the children of a node precede it and the root is last.
struct Node {
    int type = NODE_PRIM;
    int idx = 0;       // index into prims or ops
    int parent = -1;
    int left = -1, right = -1;
    int sign = 1;      // the value of the node is multiplied by sign (-1 for the right child of a difference)
};

struct Scene {
    std::vector<Node> nodes;
    std::vector<Primitive> prims;
    std::vector<BinaryOp> ops;
    double aabb_min[3] = {-1, -1, -1};
    double aabb_max[3] = {1, 1, 1};
    std::string name;
    int depth = 0;  // height of the tree
};

Scene load_scene_json(const std::string& path);

}  // namespace lpm
