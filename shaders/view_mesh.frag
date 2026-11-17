// Flat shading with a light at the camera: the normal of the triangle from the derivatives of
// the position, turned towards the viewer. With flag 1 the color follows the size of the pruned
// tree of each vertex on the burgundy scale of the poster (1 to 20 nodes, logarithmic); the edges
// (flag 2) and the vertices (flag 4) are drawn in one translucent color.
#version 460
#extension GL_GOOGLE_include_directive : require
#include "view_common.glsl"

layout(location = 0) in vec3 world;
layout(location = 1) in float tree;
layout(location = 0) out vec4 color;

vec3 burgundy(float x) {
    const vec3 c[4] = vec3[4](vec3(0.965, 0.902, 0.925), vec3(0.788, 0.455, 0.561), vec3(0.510, 0.039, 0.196), vec3(0.227, 0.016, 0.086));
    const float s = 3.0 * clamp(x, 0.0, 1.0);
    const int k = min(int(s), 2);
    return mix(c[k], c[k + 1], s - float(k));
}

void main() {
    if ((pc.flags & 2u) != 0u) { color = vec4(0.17, 0.17, 0.18, 0.3); return; }  // blended over the triangles
    if ((pc.flags & 4u) != 0u) { color = vec4(0.51, 0.04, 0.20, 0.6); return; }
    vec3 n = normalize(cross(dFdx(world), dFdy(world)));
    const vec3 v = normalize(pc.eye.xyz - world);
    if (dot(n, v) < 0.0) n = -n;
    const float d = max(dot(n, v), 0.0);
    vec3 base = vec3(0.812, 0.839, 0.871);  // #cfd6de
    if ((pc.flags & 1u) != 0u) base = burgundy(tree / log(20.0));
    color = vec4(base * (0.28 + 0.72 * d) + 0.2 * pow(d, 60.0), 1.0);
}
