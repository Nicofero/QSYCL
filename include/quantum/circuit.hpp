#pragma once

#include "quantum/types.hpp"
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
    Circuit& t(std::size_t qubit);

    // --- Parameterized single-qubit rotations (angle in radians) ---
    Circuit& rx(std::size_t qubit, double theta);
    Circuit& ry(std::size_t qubit, double theta);
    Circuit& rz(std::size_t qubit, double theta);

    // --- Two-qubit gates ---
    Circuit& cnot(std::size_t control, std::size_t target);
    Circuit& cz(std::size_t control, std::size_t target);
    Circuit& swap(std::size_t qubit_a, std::size_t qubit_b);

    // Mark a qubit for classical readout. If none are marked explicitly,
    // the runtime measures all qubits by default.
    Circuit& measure(std::size_t qubit);

    std::size_t num_qubits() const { return num_qubits_; }
    const std::vector<GateOp>& operations() const { return ops_; }
    const std::vector<std::size_t>& measured_qubits() const { return measured_; }

private:
    Circuit& add_gate(GateType type, std::vector<std::size_t> qubits,
                       std::vector<double> params = {}, std::string label = "");

    std::size_t num_qubits_;
    std::vector<GateOp> ops_;
    std::vector<std::size_t> measured_;
};

} // namespace quantum
