#include "quantum/circuit.hpp"
#include <stdexcept>
#include <string>

namespace quantum {

Circuit::Circuit(std::size_t num_qubits) : num_qubits_(num_qubits) {
    if (num_qubits == 0) {
        throw std::invalid_argument("Circuit must have at least 1 qubit");
    }
}

Circuit& Circuit::add_gate(GateType type, std::vector<std::size_t> qubits,
                            std::vector<Param> args, std::string label, bool dagger) {
    for (auto q : qubits) {
        if (q >= num_qubits_) {
            throw std::out_of_range("Qubit index out of range for this circuit");
        }
    }
    bool symbolic = false;
    std::vector<double> params;
    params.reserve(args.size());
    for (const auto& a : args) {
        if (a.is_symbolic()) {
            if (static_cast<std::size_t>(a.index) >= num_params_) {
                throw std::out_of_range("Param does not belong to this circuit (use Circuit::new_param())");
            }
            symbolic = true;
        }
        params.push_back(a.offset); // placeholder for symbolic entries (value at theta = 0)
    }
    GateOp op{type, std::move(qubits), std::move(params), std::move(label), dagger, {}};
    if (symbolic) op.symbolic = std::move(args);
    ops_.push_back(std::move(op));
    return *this;
}

Param Circuit::new_param() {
    return Param(static_cast<int>(num_params_++), 1.0, 0.0);
}

std::vector<Param> Circuit::new_params(std::size_t n) {
    std::vector<Param> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(new_param());
    return out;
}

bool Circuit::is_parametric() const {
    for (const auto& op : ops_) if (op.is_parametric()) return true;
    return false;
}

Circuit Circuit::bind(const std::vector<double>& theta, int shift_op, double shift) const {
    if (theta.size() != num_params_) {
        throw std::invalid_argument("Circuit::bind: expected " + std::to_string(num_params_) +
                                    " parameters, got " + std::to_string(theta.size()));
    }
    Circuit out(*this);
    out.num_params_ = 0; // the result has no free parameters left
    for (auto& op : out.ops_) {
        for (std::size_t i = 0; i < op.symbolic.size(); ++i) {
            op.params[i] = op.symbolic[i].resolve(theta.data());
        }
        op.symbolic.clear();
    }
    if (shift_op >= 0) {
        if (static_cast<std::size_t>(shift_op) >= out.ops_.size() || out.ops_[shift_op].params.empty()) {
            throw std::out_of_range("Circuit::bind: shift_op does not refer to a parameterized gate");
        }
        out.ops_[shift_op].params[0] += shift;
    }
    return out;
}

/**
 * @brief  Hadamard gate on qubit `qubit`.
 * 
 * Hadamard gate creates superposition: |0> -> (|0> + |1>)/sqrt(2), |1> -> (|0> - |1>)/sqrt(2).
 * 
 * @param  qubit:  qubit index
 * 
 * @return Reference to the circuit (for chaining)
 */
Circuit& Circuit::h(std::size_t qubit)  { return add_gate(GateType::H, {qubit}, {}, "H"); }
Circuit& Circuit::x(std::size_t qubit)  { return add_gate(GateType::X, {qubit}, {}, "X"); }
Circuit& Circuit::y(std::size_t qubit)  { return add_gate(GateType::Y, {qubit}, {}, "Y"); }
Circuit& Circuit::z(std::size_t qubit)  { return add_gate(GateType::Z, {qubit}, {}, "Z"); }
Circuit& Circuit::s(std::size_t qubit)  { return add_gate(GateType::S, {qubit}, {}, "S", false); }
Circuit& Circuit::sdg(std::size_t qubit) { return add_gate(GateType::S, {qubit}, {}, "SDG", true); }
Circuit& Circuit::t(std::size_t qubit)  { return add_gate(GateType::T, {qubit}, {}, "T", false); }
Circuit& Circuit::tdg(std::size_t qubit) { return add_gate(GateType::T, {qubit}, {}, "TDG", true); }

Circuit& Circuit::u(std::size_t qubit, Param theta, Param phi, Param lambda) {
    return add_gate(GateType::U, {qubit}, {theta, phi, lambda}, "U");
}

Circuit& Circuit::rx(std::size_t qubit, Param theta) {
    return add_gate(GateType::RX, {qubit}, {theta}, "RX");
}
Circuit& Circuit::ry(std::size_t qubit, Param theta) {
    return add_gate(GateType::RY, {qubit}, {theta}, "RY");
}
Circuit& Circuit::rz(std::size_t qubit, Param theta) {
    return add_gate(GateType::RZ, {qubit}, {theta}, "RZ");
}

Circuit& Circuit::cnot(std::size_t control, std::size_t target) {
    if (control == target) throw std::invalid_argument("CNOT control and target must differ");
    return add_gate(GateType::CNOT, {control, target}, {}, "CX");
}
Circuit& Circuit::cz(std::size_t control, std::size_t target) {
    if (control == target) throw std::invalid_argument("CZ control and target must differ");
    return add_gate(GateType::CZ, {control, target}, {}, "CZ");
}
Circuit& Circuit::swap(std::size_t qubit_a, std::size_t qubit_b) {
    if (qubit_a == qubit_b) throw std::invalid_argument("SWAP qubits must differ");
    return add_gate(GateType::SWAP, {qubit_a, qubit_b}, {}, "SWAP");
}

Circuit& Circuit::crx(std::size_t control, std::size_t target, Param theta) {
    if (control == target) throw std::invalid_argument("CRX control and target must differ");
    return add_gate(GateType::CRX, {control, target}, {theta}, "CRX");
}

Circuit& Circuit::cry(std::size_t control, std::size_t target, Param theta) {
    if (control == target) throw std::invalid_argument("CRY control and target must differ");
    return add_gate(GateType::CRY, {control, target}, {theta}, "CRY");
}

Circuit& Circuit::crz(std::size_t control, std::size_t target, Param theta) {
    if (control == target) throw std::invalid_argument("CRZ control and target must differ");
    return add_gate(GateType::CRZ, {control, target}, {theta}, "CRZ");
}

// TODO: implement control for arbitrary gates
// Circuit& Circuit::crtl(std::size_t control, std::size_t target, std::string label) {
//     if (control == target) throw std::invalid_argument("Controlled gate control and target must differ");
//     return add_gate(GateType::CONTROL, {control, target}, {}, label);
// }

Circuit& Circuit::measure(std::size_t qubit) {
    if (qubit >= num_qubits_) throw std::out_of_range("Qubit index out of range");
    measured_.push_back(qubit);
    return *this;
}

} // namespace quantum
