#pragma once

#include "quantum/types.hpp"
#include <vector>
#include <memory>
#include <string>

namespace quantum {

// Backend is the boundary between the device-agnostic runtime and a
// specific SYCL device implementation (CPU, GPU, FPGA, custom...).
// Every backend owns its own state vector and knows how to apply gates
// to it on its target device. The runtime never touches SYCL directly --
// only backends do.
class Backend {
public:
    virtual ~Backend() = default;

    // Allocate/reset the state vector to |0...0> for `num_qubits` qubits.
    virtual void initialize(std::size_t num_qubits) = 0;

    // Apply a single gate operation to the current state.
    virtual void apply_gate(const GateOp& op) = 0;

    // Pull the full state vector back to host memory (2^n amplitudes).
    // Intended for small circuits / debugging, not production-scale sims.
    virtual std::vector<Complex> get_state() const = 0;

    // Per-basis-state probabilities |amplitude|^2, length 2^n.
    virtual std::vector<double> probabilities() const = 0;

    // Sample `shots` measurements in the computational basis, restricted
    // to `qubits` (empty = all qubits). Returns one bitstring-as-int per shot.
    virtual std::vector<unsigned long long> sample(
        const std::vector<std::size_t>& qubits, std::size_t shots) = 0;

    // Human-readable identifier, e.g. "CPU (SYCL: Intel(R) Xeon(R) ...)".
    virtual std::string device_name() const = 0;
};

} // namespace quantum
