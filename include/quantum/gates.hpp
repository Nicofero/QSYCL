#pragma once

#include "quantum/types.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace quantum::gates {

inline Matrix2x2 H() {
    double s = 1.0 / std::sqrt(2.0);
    return {Complex(s), Complex(s), Complex(s), Complex(-s)};
}
inline Matrix2x2 X() { return {Complex(0), Complex(1), Complex(1), Complex(0)}; }
inline Matrix2x2 Y() { return {Complex(0), Complex(0, -1), Complex(0, 1), Complex(0)}; }
inline Matrix2x2 Z() { return {Complex(1), Complex(0), Complex(0), Complex(-1)}; }
inline Matrix2x2 S() { return {Complex(1), Complex(0), Complex(0), Complex(0, 1)}; }
inline Matrix2x2 T() {
    return {Complex(1), Complex(0), Complex(0), Complex(std::cos(M_PI / 4), std::sin(M_PI / 4))};
}

inline Matrix2x2 RX(double theta) {
    double c = std::cos(theta / 2), s = std::sin(theta / 2);
    return {Complex(c), Complex(0, -s), Complex(0, -s), Complex(c)};
}
inline Matrix2x2 RY(double theta) {
    double c = std::cos(theta / 2), s = std::sin(theta / 2);
    return {Complex(c), Complex(-s), Complex(s), Complex(c)};
}
inline Matrix2x2 RZ(double theta) {
    double c = std::cos(theta / 2), s = std::sin(theta / 2);
    return {Complex(c, -s), Complex(0), Complex(0), Complex(c, s)};
}

} // namespace quantum::gates
