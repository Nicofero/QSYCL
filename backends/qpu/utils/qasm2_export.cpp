#include "qasm2_export.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace quantum {
namespace backends {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    return s;
}

bool one_of(const std::string& s, std::initializer_list<const char*> options) {
    for (const char* o : options) {
        if (s == o) return true;
    }
    return false;
}

} // namespace

std::string to_qasm2(std::size_t num_qubits,
                      const std::vector<GateOp>& gates,
                      const std::vector<std::size_t>& measured_qubits) {
    std::ostringstream out;
    out << "OPENQASM 2.0;\n";
    out << "include \"qelib1.inc\";\n";
    out << "qreg q[" << num_qubits << "];\n";
    out << "creg c[" << measured_qubits.size() << "];\n";

    for (const auto& op : gates) {
        std::string label = lower(op.label);
        // qelib1.inc calls the phase gate "u1"; "p" is the OpenQASM 3 /
        // newer-Qiskit spelling of the same gate, so alias it here
        // rather than emitting a name qelib1.inc doesn't define.
        std::string qasm_name = (label == "p") ? "u1" : label;

        if (one_of(label, {"id", "x", "y", "z", "h", "s", "sdg", "sx",
                            "sxdg", "t", "tdg"})) {
            if (op.qubits.empty()) {
                throw std::runtime_error(
                    "to_qasm2: gate '" + op.label + "' has no target qubit");
            }
            out << qasm_name << " q[" << op.qubits[0] << "];\n";
        } else if (one_of(label, {"cx", "cy", "cz", "swap", "ch"})) {
            if (op.qubits.size() < 2) {
                throw std::runtime_error(
                    "to_qasm2: gate '" + op.label + "' needs two qubits");
            }
            out << qasm_name << " q[" << op.qubits[0] << "],q[" << op.qubits[1] << "];\n";
        } else if (one_of(label, {"rx", "ry", "rz", "p", "u1"})) {
            if (op.qubits.empty() || op.params.empty()) {
                throw std::runtime_error(
                    "to_qasm2: gate '" + op.label + "' needs a qubit and a parameter");
            }
            out << qasm_name << "(" << op.params[0] << ") q[" << op.qubits[0] << "];\n";
        }   // THE CR* CASE REMAINS NOT DONE 
        else {
            throw std::runtime_error(
                "to_qasm2: gate '" + op.label +
                "' has no known OpenQASM 2.0 translation -- extend to_qasm2()");
        }
    }

    for (std::size_t clbit = 0; clbit < measured_qubits.size(); ++clbit) {
        out << "measure q[" << measured_qubits[clbit] << "] -> c[" << clbit << "];\n";
    }

    return out.str();
}

} // namespace backends
} // namespace quantum
