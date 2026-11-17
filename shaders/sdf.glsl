// Signed distance of the primitives and smooth CSG operators, as include/lpm/sdf.h, in double
// precision. The includer defines PRIM_VALUE(i), the i-th value of the primitive array
// (24 values per primitive), before including this file.

double select_pos(double c, double a, double b) { return c > 0.0 ? a : b; }
double length2(double x, double y) { return sqrt(x * x + y * y); }

// Signed distance of a primitive, as eval_primitive in include/lpm/sdf.h. Layout of the 24
// values: type, world to primitive matrix (12), data (4), corner rounding (4), extrude rounding (2).
double eval_primitive(uint index, dvec3 w) {
    const uint b = 24u * index;
    const int type = int(PRIM_VALUE(b));
    const double px = PRIM_VALUE(b + 1) * w.x + PRIM_VALUE(b + 2) * w.y + PRIM_VALUE(b + 3) * w.z + PRIM_VALUE(b + 4);
    const double py = PRIM_VALUE(b + 5) * w.x + PRIM_VALUE(b + 6) * w.y + PRIM_VALUE(b + 7) * w.z + PRIM_VALUE(b + 8);
    const double pz = PRIM_VALUE(b + 9) * w.x + PRIM_VALUE(b + 10) * w.y + PRIM_VALUE(b + 11) * w.z + PRIM_VALUE(b + 12);
    if (type == 0) return sqrt(px * px + py * py + pz * pz) - PRIM_VALUE(b + 13);
    if (type == 1) {
        const double hx = PRIM_VALUE(b + 13) * 0.5, hy = PRIM_VALUE(b + 14) * 0.5, hz = PRIM_VALUE(b + 15) * 0.5;
        const double rx = select_pos(px, PRIM_VALUE(b + 17), PRIM_VALUE(b + 19));
        const double ry = select_pos(px, PRIM_VALUE(b + 18), PRIM_VALUE(b + 20));
        const double rr = select_pos(pz, rx, ry);
        const double qx = abs(px) - hx + rr, qz = abs(pz) - hz + rr;
        const double d2 = min(max(qx, qz), 0.0) + length2(max(qx, 0.0), max(qz, 0.0)) - rr;
        const double er = select_pos(py, PRIM_VALUE(b + 21), PRIM_VALUE(b + 22));
        const double ex = d2 + er, ey = abs(py) - (hy - er);
        return min(max(ex, ey), 0.0) + length2(max(ex, 0.0), max(ey, 0.0)) - er;
    }
    if (type == 2) {
        const double h = PRIM_VALUE(b + 13) * 0.5, r = PRIM_VALUE(b + 14);
        const double dx = length2(px, pz) - r, dy = abs(py) - h;
        return min(max(dx, dy), 0.0) + length2(max(dx, 0.0), max(dy, 0.0));
    }
    if (type == 3) {
        const double radius = PRIM_VALUE(b + 13), half_height = PRIM_VALUE(b + 14) * 0.5;
        const double p2x = length2(px, pz) - radius, p2y = py + half_height;
        const double ex = -radius, ey = 2.0 * half_height;
        const double t = clamp((p2x * ex + p2y * ey) / (ex * ex + ey * ey), 0.0, 1.0);
        const double qx = p2x - ex * t, qy = p2y - ey * t;
        const double d = length2(qx, qy);
        return select_pos(max(qx, qy), d, -min(d, p2y));
    }
    return 1e20lf;
}

// Smooth union, intersection or difference with the quadratic kernel, as eval_op. op.z is
// 0.25 / k, computed on the CPU: the same value as the division of the CPU code.
double eval_op(dvec4 op, double a, double b) {
    const double k = op.x, s = op.y;
    double phi = 0.0;
    if (k != 0.0) {
        const double m = max(k - abs(a - b), 0.0);
        phi = m * m * op.z;
    }
    return s * (min(s * a, s * b) - phi);
}
