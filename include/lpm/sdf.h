// Signed distance of the primitives and smooth CSG operators, templated on the scalar
// type: double gives the point evaluation and Interval gives bounds over a box.
// The formulas replicate the official GLSL shader (eval.glsl / common.glsl).
#pragma once
#include "lpm/core.h"
#include "lpm/scene.h"

namespace lpm {

template <class T>
inline V3<T> to_primitive_space(const Primitive& P, const V3<T>& p) {
    V3<T> q;
    q.x = P.m[0] * p.x + P.m[1] * p.y + P.m[2] * p.z + P.m[3];
    q.y = P.m[4] * p.x + P.m[5] * p.y + P.m[6] * p.z + P.m[7];
    q.z = P.m[8] * p.x + P.m[9] * p.y + P.m[10] * p.z + P.m[11];
    return q;
}

// 2D rounded box with one radius per corner (sdRoundBox of the shader).
template <class T>
inline T sd_round_box_2d(T px, T pz, double bx, double bz, const double r[4]) {
    T rx = select_pos(px, T(r[0]), T(r[2]));
    T ry = select_pos(px, T(r[1]), T(r[3]));
    T rr = select_pos(pz, rx, ry);
    T qx = vabs(px) - bx + rr;
    T qz = vabs(pz) - bz + rr;
    return vmin(vmax(qx, qz), T(0.0)) + length2d(vmax(qx, T(0.0)), vmax(qz, T(0.0))) - rr;
}

template <class T>
inline T sd_extrude(T d, T z, T h, T r) {
    T qx = d + r;
    T qy = vabs(z) - h;
    return vmin(vmax(qx, qy), T(0.0)) + length2d(vmax(qx, T(0.0)), vmax(qy, T(0.0))) - r;
}

template <class T>
inline T sd_cone(const V3<T>& p, double radius, double half_height) {
    T p2x = length2d(p.x, p.z) - radius;
    T p2y = p.y + half_height;
    const double ex = -radius, ey = 2.0 * half_height;
    T t = vclamp01((p2x * ex + p2y * ey) / (ex * ex + ey * ey));
    T qx = p2x - ex * t;
    T qy = p2y - ey * t;
    T d = length2d(qx, qy);
    return select_pos(vmax(qx, qy), d, -vmin(d, p2y));
}

template <class T>
inline T eval_primitive(const Primitive& P, const V3<T>& pw) {
    V3<T> p = to_primitive_space(P, pw);
    switch (P.type) {
        case PRIM_SPHERE:
            return length3d(p) - P.data[0];
        case PRIM_BOX: {
            const double hx = P.data[0] * 0.5, hy = P.data[1] * 0.5, hz = P.data[2] * 0.5;
            T d2 = sd_round_box_2d(p.x, p.z, hx, hz, P.corner_rounding);
            T er = select_pos(p.y, T(P.extrude_rounding[0]), T(P.extrude_rounding[1]));
            return sd_extrude(d2, p.y, T(hy) - er, er);
        }
        case PRIM_CYLINDER: {
            const double h = P.data[0] * 0.5, r = P.data[1];
            T dx = length2d(p.x, p.z) - r;
            T dy = vabs(p.y) - h;
            return vmin(vmax(dx, dy), T(0.0)) + length2d(vmax(dx, T(0.0)), vmax(dy, T(0.0)));
        }
        case PRIM_CONE:
            return sd_cone(p, P.data[0], P.data[1] * 0.5);
    }
    return T(1e20);
}

// Quadratic blending kernel: phi(x, k) = max(k - x, 0)^2 / (4k).
template <class T>
inline T blend_kernel(T x, double k) {
    if (k == 0.0) return T(0.0);
    T m = vmax(T(k) - x, T(0.0));
    return vsqr(m) * (0.25 / k);
}

// OP(a, b) = s (min(s a, s b) - phi(|a - b|, k)); the sign of the right operand of a
// difference is already folded into the node sign, so SUB is a smooth max.
template <class T>
inline T eval_op(const BinaryOp& op, T a, T b) {
    const double s = op.s;
    return s * (vmin(s * a, s * b) - blend_kernel(vabs(a - b), op.k));
}

}  // namespace lpm
