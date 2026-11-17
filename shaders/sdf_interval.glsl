// Interval arithmetic of include/lpm/core.h and the interval evaluation of the primitives and
// operators of include/lpm/sdf.h, with the same operations in the same order, so that the GPU
// proves the same bounds as the CPU. An interval [lo, hi] is a dvec2 (lo, hi). The includer
// defines PRIM_VALUE(i) and includes sdf.glsl first.

dvec2 iv_add(dvec2 a, dvec2 b) { return dvec2(a.x + b.x, a.y + b.y); }
dvec2 iv_sub(dvec2 a, dvec2 b) { return dvec2(a.x - b.y, a.y - b.x); }
dvec2 iv_neg(dvec2 a) { return dvec2(-a.y, -a.x); }
dvec2 iv_addc(dvec2 a, double c) { return dvec2(a.x + c, a.y + c); }
dvec2 iv_subc(dvec2 a, double c) { return dvec2(a.x - c, a.y - c); }
dvec2 iv_mulc(double c, dvec2 a) { return c >= 0.0 ? dvec2(c * a.x, c * a.y) : dvec2(c * a.y, c * a.x); }
dvec2 iv_min(dvec2 a, dvec2 b) { return dvec2(min(a.x, b.x), min(a.y, b.y)); }
dvec2 iv_max(dvec2 a, dvec2 b) { return dvec2(max(a.x, b.x), max(a.y, b.y)); }
dvec2 iv_abs(dvec2 a) {
    if (a.x >= 0.0) return a;
    if (a.y <= 0.0) return iv_neg(a);
    return dvec2(0.0, max(-a.x, a.y));
}
dvec2 iv_sqr(dvec2 a) {
    if (a.x >= 0.0) return dvec2(a.x * a.x, a.y * a.y);
    if (a.y <= 0.0) return dvec2(a.y * a.y, a.x * a.x);
    return dvec2(0.0, max(a.x * a.x, a.y * a.y));
}
dvec2 iv_sqrt(dvec2 a) { return dvec2(sqrt(max(a.x, 0.0)), sqrt(max(a.y, 0.0))); }
double clamp01(double a) { return min(max(a, 0.0), 1.0); }
dvec2 iv_clamp01(dvec2 a) { return dvec2(clamp01(a.x), clamp01(a.y)); }
// When the sign of c is undetermined on the box, both branches are possible.
dvec2 iv_select_pos(dvec2 c, dvec2 a, dvec2 b) {
    if (c.x > 0.0) return a;
    if (c.y <= 0.0) return b;
    return dvec2(min(a.x, b.x), max(a.y, b.y));
}
dvec2 iv_length2(dvec2 x, dvec2 y) { return iv_sqrt(iv_add(iv_sqr(x), iv_sqr(y))); }

const dvec2 IV_ZERO = dvec2(0.0);

// The bound of a primitive over the box (bx, by, bz), as eval_primitive<Interval>.
dvec2 eval_primitive_iv(uint index, dvec2 bx, dvec2 by, dvec2 bz) {
    const uint b = 24u * index;
    const int type = int(PRIM_VALUE(b));
    const dvec2 px = iv_addc(iv_add(iv_add(iv_mulc(PRIM_VALUE(b + 1), bx), iv_mulc(PRIM_VALUE(b + 2), by)),
                                    iv_mulc(PRIM_VALUE(b + 3), bz)), PRIM_VALUE(b + 4));
    const dvec2 py = iv_addc(iv_add(iv_add(iv_mulc(PRIM_VALUE(b + 5), bx), iv_mulc(PRIM_VALUE(b + 6), by)),
                                    iv_mulc(PRIM_VALUE(b + 7), bz)), PRIM_VALUE(b + 8));
    const dvec2 pz = iv_addc(iv_add(iv_add(iv_mulc(PRIM_VALUE(b + 9), bx), iv_mulc(PRIM_VALUE(b + 10), by)),
                                    iv_mulc(PRIM_VALUE(b + 11), bz)), PRIM_VALUE(b + 12));
    if (type == 0) return iv_subc(iv_sqrt(iv_add(iv_add(iv_sqr(px), iv_sqr(py)), iv_sqr(pz))), PRIM_VALUE(b + 13));
    if (type == 1) {
        // sd_round_box_2d, then sd_extrude.
        const double hx = PRIM_VALUE(b + 13) * 0.5, hy = PRIM_VALUE(b + 14) * 0.5, hz = PRIM_VALUE(b + 15) * 0.5;
        const dvec2 rx = iv_select_pos(px, dvec2(PRIM_VALUE(b + 17)), dvec2(PRIM_VALUE(b + 19)));
        const dvec2 ry = iv_select_pos(px, dvec2(PRIM_VALUE(b + 18)), dvec2(PRIM_VALUE(b + 20)));
        const dvec2 rr = iv_select_pos(pz, rx, ry);
        const dvec2 qx = iv_add(iv_subc(iv_abs(px), hx), rr), qz = iv_add(iv_subc(iv_abs(pz), hz), rr);
        const dvec2 d2 = iv_sub(iv_add(iv_min(iv_max(qx, qz), IV_ZERO), iv_length2(iv_max(qx, IV_ZERO), iv_max(qz, IV_ZERO))), rr);
        const dvec2 er = iv_select_pos(py, dvec2(PRIM_VALUE(b + 21)), dvec2(PRIM_VALUE(b + 22)));
        const dvec2 h = iv_sub(dvec2(hy), er);
        const dvec2 ex = iv_add(d2, er), ey = iv_sub(iv_abs(py), h);
        return iv_sub(iv_add(iv_min(iv_max(ex, ey), IV_ZERO), iv_length2(iv_max(ex, IV_ZERO), iv_max(ey, IV_ZERO))), er);
    }
    if (type == 2) {
        const double h = PRIM_VALUE(b + 13) * 0.5, r = PRIM_VALUE(b + 14);
        const dvec2 dx = iv_subc(iv_length2(px, pz), r), dy = iv_subc(iv_abs(py), h);
        return iv_add(iv_min(iv_max(dx, dy), IV_ZERO), iv_length2(iv_max(dx, IV_ZERO), iv_max(dy, IV_ZERO)));
    }
    if (type == 3) {
        const double radius = PRIM_VALUE(b + 13), half_height = PRIM_VALUE(b + 14) * 0.5;
        const dvec2 p2x = iv_subc(iv_length2(px, pz), radius), p2y = iv_addc(py, half_height);
        const double ex = -radius, ey = 2.0 * half_height;
        // The division by a positive constant is a product by its inverse, as operator/ of core.h.
        const dvec2 t = iv_clamp01(iv_mulc(1.0 / (ex * ex + ey * ey), iv_add(iv_mulc(ex, p2x), iv_mulc(ey, p2y))));
        const dvec2 qx = iv_sub(p2x, iv_mulc(ex, t)), qy = iv_sub(p2y, iv_mulc(ey, t));
        const dvec2 d = iv_length2(qx, qy);
        return iv_select_pos(iv_max(qx, qy), d, iv_neg(iv_min(d, p2y)));
    }
    return dvec2(1e20lf);
}

// The bound of an operator from those of its operands, as eval_op<Interval>; op.z is 0.25 / k.
dvec2 eval_op_iv(dvec4 op, dvec2 a, dvec2 b) {
    const double k = op.x, s = op.y;
    dvec2 phi = IV_ZERO;
    if (k != 0.0) {
        const dvec2 m = iv_max(iv_sub(dvec2(k), iv_abs(iv_sub(a, b))), IV_ZERO);
        phi = iv_mulc(op.z, iv_sqr(m));
    }
    return iv_mulc(s, iv_sub(iv_min(iv_mulc(s, a), iv_mulc(s, b)), phi));
}

// The bound of a primitive widened by a relative epsilon, as inflate in include/lpm/tree.h.
dvec2 inflate(dvec2 v) {
    const double eps = 1e-12lf;
    return dvec2(v.x - eps * (1.0 + abs(v.x)), v.y + eps * (1.0 + abs(v.y)));
}
