// Root finding on an edge and the minimizer of a quadric, shared by dc.comp and probe.comp. The
// includer defines double eval_tree(uvec3 tree, dvec3 p) before including this file.

// Root of f along the edge p0 + t (p1 - p0): bracketing secant steps (Illinois) with bisection
// fallback, as detail::edge_root in include/lpm/dc.h (the same stops; converged is false only when
// max_iters ran out).
double edge_root(uvec3 tree, dvec3 p0, dvec3 p1, double f0, double f1, uint max_iters, out bool converged) {
    double a = 0.0, b = 1.0, fa = f0, fb = f1;
    int side = 0;
    double m = 0.5, prev = -1.0;
    converged = true;
    for (uint it = 0u; it < max_iters; it++) {
        m = (fa * b - fb * a) / (fa - fb);
        if (!(m > a && m < b)) m = 0.5 * (a + b);
        if (abs(m - prev) < 1e-12lf) return m;
        prev = m;
        const double fm = eval_tree(tree, p0 + m * (p1 - p0));
        if (fm == 0.0) return m;
        if ((fm < 0.0) == (fa < 0.0)) {
            a = m; fa = fm;
            if (side == -1) fb *= 0.5;
            side = -1;
        } else {
            b = m; fb = fm;
            if (side == 1) fa *= 0.5;
            side = 1;
        }
        if (b - a < 1e-12lf) return 0.5 * (a + b);
    }
    converged = false;
    return 0.5 * (a + b);
}

// Unit normal at the crossing x of an edge along axis by forward differences of step h, as
// detail::crossing_normal: f(x) is taken as zero when edge_root converged.
dvec3 crossing_normal(uvec3 tree, dvec3 x, double h, bool converged, uint axis, double f0) {
    const double fx = converged ? 0.0 : eval_tree(tree, x);
    dvec3 g;
    for (int a = 0; a < 3; a++) {
        dvec3 q = x;
        q[a] += h;
        g[a] = (eval_tree(tree, q) - fx) / h;
    }
    const double len = sqrt(g.x * g.x + g.y * g.y + g.z * g.z);
    dvec3 n = dvec3(0.0);
    if (len > 1e-300lf) n = g / len;
    else n[axis] = f0 < 0.0 ? 1.0 : -1.0;
    return n;
}

// A vertex of the quadric moved onto the surface, as detail::project_vertex: up to three Newton
// steps along the forward-difference gradient, clamped to the cell [lo, hi], while |f| > tol.
dvec3 project_vertex(uvec3 tree, dvec3 x, dvec3 lo, dvec3 hi, double h, double tol) {
    for (int it = 0; it < 3; it++) {
        const double f = eval_tree(tree, x);
        if (abs(f) <= tol) return x;
        dvec3 g;
        for (int a = 0; a < 3; a++) {
            dvec3 q = x;
            q[a] += h;
            g[a] = (eval_tree(tree, q) - f) / h;
        }
        const double g2 = g.x * g.x + g.y * g.y + g.z * g.z;
        if (g2 < 1e-300lf) return x;
        for (int a = 0; a < 3; a++) x[a] = min(max(x[a] - f * g[a] / g2, lo[a]), hi[a]);
    }
    return x;
}

// Eigen decomposition of a symmetric 3 x 3 matrix by cyclic Jacobi rotations: the eigenvalues
// end on the diagonal of a and the eigenvectors in the columns of v.
void jacobi(inout double a[3][3], out double v[3][3]) {
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) v[r][c] = r == c ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 50; sweep++) {
        const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        const double diag = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if (off == 0.0 || off <= 1e-34lf * diag) break;
        for (int pq = 0; pq < 3; pq++) {
            const int p = pq == 2 ? 1 : 0, q = pq == 0 ? 1 : 2;
            const double apq = a[p][q];
            if (apq == 0.0) continue;
            const double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            const double t = (theta >= 0.0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1.0));
            const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < 3; k++) { const double x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
            for (int k = 0; k < 3; k++) { const double x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
            for (int k = 0; k < 3; k++) { const double x = v[k][p], y = v[k][q]; v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
        }
    }
}

// Minimizer of the quadric (ata, atb) clamped to the box [lo, hi]: the pseudo-inverse keeps the
// eigenvalues above 1 % of the largest, around the mass point (the mean of the crossings).
dvec3 qef_solve(double ata[3][3], dvec3 atb, dvec3 mass, dvec3 lo, dvec3 hi) {
    double ev[3][3] = ata, vec[3][3];
    jacobi(ev, vec);
    const double lmax = max(ev[0][0], max(ev[1][1], ev[2][2]));
    double pinv[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int k = 0; k < 3; k++) {
        const double l = ev[k][k];
        if (l > 1e-2lf * lmax && l > 1e-12lf)
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) pinv[r][c] += vec[r][k] * vec[c][k] / l;
    }
    dvec3 res;
    for (int r = 0; r < 3; r++) res[r] = atb[r] - (ata[r][0] * mass.x + ata[r][1] * mass.y + ata[r][2] * mass.z);
    dvec3 x;
    for (int r = 0; r < 3; r++) x[r] = mass[r] + (pinv[r][0] * res.x + pinv[r][1] * res.y + pinv[r][2] * res.z);
    return clamp(x, lo, hi);
}
