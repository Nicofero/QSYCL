#pragma once

#include "quantum/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace quantum {
namespace backends {

// Serializes a buffered gate list plus a measurement plan into OpenQASM
// 2.0 text -- QMIO's expected program format for the "openqasm"
// program_format (the only one QmioBackend implements; QIR text/bitcode
// are a documented gap, see qmio_backend.hpp).
//
// ASSUMED, matching the GateOp convention already established for the
// CUNQA backend: op.label names the gate (case-insensitive), op.qubits
// holds its operand qubit indices, op.params holds any rotation angle.
std::string to_qasm2(std::size_t num_qubits,
                      const std::vector<GateOp>& gates,
                      const std::vector<std::size_t>& measured_qubits);

} // namespace backends
} // namespace quantum
