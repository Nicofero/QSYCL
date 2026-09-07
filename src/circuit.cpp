#include "quantum/circuit.hpp"
#include <stdexcept>

namespace quantum {

Circuit::Circuit(std::size_t num_qubits) : num_qubits_(num_qubits) {
    if (num_qubits == 0) {
        throw std::invalid_argument("Circuit must have at least 1 qubit");
    }
}

Circuit& Circuit::add_gate(GateType type, std::vector<std::size_t> qubits,
                            std::vector<double> params, std::string label, bool dagger) {
    for (auto q : qubits) {
        if (q >= num_qubits_) {
            throw std::out_of_range("Qubit index out of range for this circuit");
        }
    }
    ops_.push_back(GateOp{type, std::move(qubits), std::move(params), std::move(label), dagger});
    return *this;
}

Circuit& Circuit::h(std::size_t qubit)  { return add_gate(GateType::H, {qubit}, {}, "H"); }
Circuit& Circuit::x(std::size_t qubit)  { return add_gate(GateType::X, {qubit}, {}, "X"); }
Circuit& Circuit::y(std::size_t qubit)  { return add_gate(GateType::Y, {qubit}, {}, "Y"); }
Circuit& Circuit::z(std::size_t qubit)  { return add_gate(GateType::Z, {qubit}, {}, "Z"); }
Circuit& Circuit::s(std::size_t qubit)  { return add_gate(GateType::S, {qubit}, {}, "S", false); }
Circuit& Circuit::sdg(std::size_t qubit) { return add_gate(GateType::S, {qubit}, {}, "S†", true); }
Circuit& Circuit::t(std::size_t qubit)  { return add_gate(GateType::T, {qubit}, {}, "T", false); }
Circuit& Circuit::tdg(std::size_t qubit) { return add_gate(GateType::T, {qubit}, {}, "T†", true); }

Circuit& Circuit::u(std::size_t qubit, double theta, double phi, double lambda) {
    return add_gate(GateType::U, {qubit}, {theta, phi, lambda}, "U");
}

Circuit& Circuit::rx(std::size_t qubit, double theta) {
    return add_gate(GateType::RX, {qubit}, {theta}, "RX");
}
Circuit& Circuit::ry(std::size_t qubit, double theta) {
    return add_gate(GateType::RY, {qubit}, {theta}, "RY");
}
Circuit& Circuit::rz(std::size_t qubit, double theta) {
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

Circuit& Circuit::crx(std::size_t control, std::size_t target, double theta) {
    if (control == target) throw std::invalid_argument("CRX control and target must differ");
    return add_gate(GateType::CRX, {control, target}, {theta}, "CRX");
}

Circuit& Circuit::cry(std::size_t control, std::size_t target, double theta) {
    if (control == target) throw std::invalid_argument("CRY control and target must differ");
    return add_gate(GateType::CRY, {control, target}, {theta}, "CRY");
}

Circuit& Circuit::crz(std::size_t control, std::size_t target, double theta) {
    if (control == target) throw std::invalid_argument("CRZ control and target must differ");
    return add_gate(GateType::CRZ, {control, target}, {theta}, "CRZ");
}

// Circuit& Circuit::controlled(std::size_t control, std::size_t target, std::string label) {
//     if (control == target) throw std::invalid_argument("Controlled gate control and target must differ");
//     return add_gate(GateType::CONTROL, {control, target}, {}, label);
// }

Circuit& Circuit::measure(std::size_t qubit) {
    if (qubit >= num_qubits_) throw std::out_of_range("Qubit index out of range");
    measured_.push_back(qubit);
    return *this;
}

} // namespace quantum
