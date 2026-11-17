// The mesh, pulled from the buffers of the dynamic update (no vertex input). With flag 2 it draws
// the three edges of every triangle as lines, six indices per triangle.
#version 460
#extension GL_GOOGLE_include_directive : require
#include "view_common.glsl"

layout(location = 0) out vec3 world;
layout(location = 1) out float tree;  // log of the tree size

void main() {
    uint corner = gl_VertexIndex;
    if ((pc.flags & 2u) != 0u) {
        const uint ends[6] = uint[6](0u, 1u, 1u, 2u, 2u, 0u);
        corner = 3u * (gl_VertexIndex / 6u) + ends[gl_VertexIndex % 6u];
    }
    // A face next to a coarse leaf may be a single triangle: the other one of its slot has the index
    // 0xffffffff and is dropped (its three vertices go before the near plane, where it is clipped).
    const uint t3 = 3u * (corner / 3u);
    if (pc.faces.v[t3] == 0xffffffffu || pc.faces.v[t3 + 1u] == 0xffffffffu || pc.faces.v[t3 + 2u] == 0xffffffffu) {
        world = vec3(0.0);
        tree = 0.0;
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        gl_PointSize = 1.0;
        return;
    }
    const uint i = pc.faces.v[corner];
    dvec3 p;
    uint ts = 1u;
    if (i < pc.nv_static) {
        p = dvec3(pc.static_v.v[3u * i], pc.static_v.v[3u * i + 1u], pc.static_v.v[3u * i + 2u]);
        if (uint64_t(pc.static_t) != 0ul) ts = pc.static_t.v[i];
    } else {
        const uint k = i - pc.nv_static;
        p = dvec3(pc.new_v.v[3u * k], pc.new_v.v[3u * k + 1u], pc.new_v.v[3u * k + 2u]);
        if (uint64_t(pc.new_t) != 0ul) ts = pc.new_t.v[k];
    }
    world = vec3(p);
    tree = log(float(max(ts, 1u)));
    gl_Position = pc.mvp * vec4(world, 1.0);
    gl_PointSize = (pc.flags & 8u) != 0u ? 3.0 : 1.0;
}
