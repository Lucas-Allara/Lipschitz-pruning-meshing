// Scalar helpers shared by the point and interval evaluations of the SDF primitives.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lpm {

// Closed interval [lo, hi]. The arithmetic is evaluated in double precision without
// directed rounding; the primitive evaluation inflates its result by a small epsilon.
struct Interval {
    double lo, hi;
    Interval() : lo(0.0), hi(0.0) {}
    Interval(double v) : lo(v), hi(v) {}
    Interval(double a, double b) : lo(a), hi(b) {}
    bool contains(double v) const { return lo <= v && v <= hi; }
    bool contains_zero() const { return lo <= 0.0 && 0.0 <= hi; }
    double width() const { return hi - lo; }
    static Interval hull(const Interval& a, const Interval& b) {
        return Interval(std::min(a.lo, b.lo), std::max(a.hi, b.hi));
    }
};

inline Interval operator+(Interval a, Interval b) { return Interval(a.lo + b.lo, a.hi + b.hi); }
inline Interval operator-(Interval a, Interval b) { return Interval(a.lo - b.hi, a.hi - b.lo); }
inline Interval operator-(Interval a) { return Interval(-a.hi, -a.lo); }
inline Interval operator+(Interval a, double c) { return Interval(a.lo + c, a.hi + c); }
inline Interval operator-(Interval a, double c) { return Interval(a.lo - c, a.hi - c); }
inline Interval operator+(double c, Interval a) { return a + c; }
inline Interval operator-(double c, Interval a) { return Interval(c - a.hi, c - a.lo); }
inline Interval operator*(double c, Interval a) {
    return c >= 0.0 ? Interval(c * a.lo, c * a.hi) : Interval(c * a.hi, c * a.lo);
}
inline Interval operator*(Interval a, double c) { return c * a; }
inline Interval operator/(Interval a, double c) { return (1.0 / c) * a; }  // c > 0 only
inline Interval operator*(Interval a, Interval b) {
    double p0 = a.lo * b.lo, p1 = a.lo * b.hi, p2 = a.hi * b.lo, p3 = a.hi * b.hi;
    return Interval(std::min(std::min(p0, p1), std::min(p2, p3)),
                    std::max(std::max(p0, p1), std::max(p2, p3)));
}

// Elementary functions, overloaded for double and Interval so that the primitive
// SDFs can be written once as templates.
inline double vmin(double a, double b) { return std::min(a, b); }
inline double vmax(double a, double b) { return std::max(a, b); }
inline double vabs(double a) { return std::fabs(a); }
inline double vsqr(double a) { return a * a; }
inline double vsqrt(double a) { return std::sqrt(a); }
inline double vclamp01(double a) { return std::min(std::max(a, 0.0), 1.0); }
// Branch on the sign of c: c > 0 gives a, otherwise b.
inline double select_pos(double c, double a, double b) { return c > 0.0 ? a : b; }

inline Interval vmin(Interval a, Interval b) { return Interval(std::min(a.lo, b.lo), std::min(a.hi, b.hi)); }
inline Interval vmax(Interval a, Interval b) { return Interval(std::max(a.lo, b.lo), std::max(a.hi, b.hi)); }
inline Interval vabs(Interval a) {
    if (a.lo >= 0.0) return a;
    if (a.hi <= 0.0) return -a;
    return Interval(0.0, std::max(-a.lo, a.hi));
}
inline Interval vsqr(Interval a) {
    if (a.lo >= 0.0) return Interval(a.lo * a.lo, a.hi * a.hi);
    if (a.hi <= 0.0) return Interval(a.hi * a.hi, a.lo * a.lo);
    return Interval(0.0, std::max(a.lo * a.lo, a.hi * a.hi));
}
inline Interval vsqrt(Interval a) { return Interval(std::sqrt(std::max(a.lo, 0.0)), std::sqrt(std::max(a.hi, 0.0))); }
inline Interval vclamp01(Interval a) { return Interval(vclamp01(a.lo), vclamp01(a.hi)); }
// When the sign of c is undetermined on the box, both branches are possible.
inline Interval select_pos(Interval c, Interval a, Interval b) {
    if (c.lo > 0.0) return a;
    if (c.hi <= 0.0) return b;
    return Interval::hull(a, b);
}

template <class T>
struct V3 {
    T x, y, z;
};

template <class T>
inline T length2d(T x, T y) { return vsqrt(vsqr(x) + vsqr(y)); }
template <class T>
inline T length3d(const V3<T>& v) { return vsqrt(vsqr(v.x) + vsqr(v.y) + vsqr(v.z)); }

}  // namespace lpm
