#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <string>

namespace quantum {

// A small, SYCL-device-friendly complex number.
// (std::complex has patchy device-side support across SYCL backends,
// so the runtime uses this plain struct everywhere instead.)
struct Complex {
    double re = 0.0;
    double im = 0.0;

    Complex() = default;
    Complex(double r, double i = 0.0) : re(r), im(i) {}

    Complex operator+(const Complex& o) const { return {re + o.re, im + o.im}; }
    Complex operator-(const Complex& o) const { return {re - o.re, im - o.im}; }
    Complex operator*(const Complex& o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
    Complex conj() const { return {re, -im}; }
    double norm() const { return re * re + im * im; }
};

// Every gate the Circuit API can emit. MEASURE is included so the runtime
// can treat measurement as just another scheduled operation.
enum class GateType {
    H, X, Y, Z, S, T,
    RX, RY, RZ,
    CNOT, CZ, SWAP,
    MEASURE
};

// A single scheduled operation. `qubits` holds 1 or 2 target indices
// depending on the gate; `params` holds the angle for parameterized gates.
struct GateOp {
    GateType type;
    std::vector<std::size_t> qubits;
    std::vector<double> params;
    std::string label; // optional, for debugging/printing circuits
};

// 2x2 unitary, row-major: [m00, m01, m10, m11]
using Matrix2x2 = std::array<Complex, 4>;

} // namespace quantum
