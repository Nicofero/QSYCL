#pragma once

#include "quantum/types.hpp"
#include "quantum/parameter.hpp"
#include <vector>
#include <cstddef>

namespace quantum {

// Circuit is the layer application code talks to. It knows nothing about
// SYCL, devices, or state vectors -- it just records a gate sequence.
// Methods return *this so calls can be chained: circuit.h(0).cnot(0,1);
class Circuit {
public:
    explicit Circuit(std::size_t num_qubits);

    // --- Single-qubit gates ---
    Circuit& h(std::size_t qubit);
    Circuit& x(std::size_t qubit);
    Circuit& y(std::size_t qubit);
    Circuit& z(std::size_t qubit);
    Circuit& s(std::size_t qubit);
    Circuit& sdg(std::size_t qubit);
    Circuit& t(std::size_t qubit);
    Circuit& tdg(std::size_t qubit);

    // --- Universal single-qubit gate ---
    Circuit& u(std::size_t qubit, Param theta, Param phi, Param lambda);

    // --- Parameterized single-qubit rotations (angle in radians) ---
    Circuit& rx(std::size_t qubit, Param theta);
    Circuit& ry(std::size_t qubit, Param theta);
    Circuit& rz(std::size_t qubit, Param theta);

    // --- Two-qubit gates ---
    Circuit& cnot(std::size_t control, std::size_t target);
    Circuit& cz(std::size_t control, std::size_t target);
    Circuit& swap(std::size_t qubit_a, std::size_t qubit_b);
    Circuit& crx(std::size_t control, std::size_t target, Param theta);
    Circuit& cry(std::size_t control, std::size_t target, Param theta);
    Circuit& crz(std::size_t control, std::size_t target, Param theta);

    // -- Universal controlled gate (control qubit, target qubit, 2x2 unitary) ---
    // Circuit& controlled(std::size_t control, std::size_t target, std::string label);

    // Mark a qubit for classical readout. If none are marked explicitly,
    // the runtime measures all qubits by default.
    Circuit& measure(std::size_t qubit);

    // --- Variational parameters ---
    // Allocate a fresh symbolic parameter (or n of them). Pass them wherever an
    // angle is expected; the same Param may be reused in several gates, and
    // affine expressions (2.0 * p + 0.1) are allowed. A circuit containing
    // symbolic angles must be bound before it can be executed:
    //     Circuit c(2);  auto th = c.new_params(2);
    //     c.ry(0, th[0]).cnot(0, 1).rz(1, th[1]);
    //     runtime.run(c.bind({0.1, 0.2}));
    Param new_param();
    std::vector<Param> new_params(std::size_t n);

    std::size_t num_params() const { return num_params_; }
    bool is_parametric() const;   // true if any gate has a symbolic angle

    // Returns a fully numeric copy with every symbolic angle resolved from
    // `theta` (size must equal num_params()). Optionally adds `shift` to the
    // (first) angle of gate `shift_op` after resolving -- the building block
    // of the parameter-shift rule. Semantics of shifted evaluation live HERE,
    // so host fallbacks and native backends agree on them.
    Circuit bind(const std::vector<double>& theta,
                 int shift_op = -1, double shift = 0.0) const;

    std::size_t num_qubits() const { return num_qubits_; }
    const std::vector<GateOp>& operations() const { return ops_; }
    const std::vector<std::size_t>& measured_qubits() const { return measured_; }

private:
    Circuit& add_gate(GateType type, std::vector<std::size_t> qubits,
                       std::vector<Param> args = {}, std::string label = "", bool dagger = false);

    std::size_t num_qubits_;
    std::size_t num_params_ = 0;
    std::vector<GateOp> ops_;
    std::vector<std::size_t> measured_;
};

} // namespace quantum
