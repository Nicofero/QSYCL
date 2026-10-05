#pragma once

#include <cstddef>

namespace quantum {

// A gate-angle expression of the form   scale * theta[index] + offset.
//
//  * index >= 0  -> symbolic: depends on entry `index` of the parameter vector.
//  * index <  0  -> constant: evaluates to `offset` (scale is ignored).
//
// Circuit::new_param() hands out symbolic Params; plain doubles convert
// implicitly to constant Params, so existing code like `c.rx(0, 0.3)` keeps
// working. Simple affine arithmetic is supported so shared / scaled
// parameters are easy to write:  c.rz(1, 2.0 * theta + 0.5);
//
// Deliberately a small trivially-copyable POD: backends can copy arrays of
// these straight into device memory.
struct Param {
    int    index  = -1;
    double scale  = 1.0;
    double offset = 0.0;

    constexpr Param() = default;
    constexpr Param(double constant) : index(-1), scale(0.0), offset(constant) {}
    constexpr Param(int idx, double s, double o) : index(idx), scale(s), offset(o) {}

    constexpr bool is_symbolic() const { return index >= 0; }

    // Numeric value for a given parameter vector (theta may be null for constants).
    constexpr double resolve(const double* theta) const {
        return index >= 0 ? scale * theta[index] + offset : offset;
    }
};

constexpr Param operator*(double c, Param p) { p.scale *= c; p.offset *= c; return p; }
constexpr Param operator*(Param p, double c) { return c * p; }
constexpr Param operator+(Param p, double c) { p.offset += c; return p; }
constexpr Param operator+(double c, Param p) { return p + c; }
constexpr Param operator-(Param p, double c) { return p + (-c); }
constexpr Param operator-(double c, Param p) { return (-1.0 * p) + c; }
constexpr Param operator-(Param p)           { return -1.0 * p; }

} // namespace quantum
