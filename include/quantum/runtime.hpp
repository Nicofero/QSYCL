#pragma once

#include "quantum/circuit.hpp"
#include "quantum/backend.hpp"
#include "quantum/device_selector.hpp"
#include <memory>
#include <map>
#include <string>

namespace quantum {

// QuantumRuntime is the layer between the user-facing Circuit and a
// device Backend. It owns state-management concerns that are the same
// regardless of device: running gates in order, defaulting the readout
// set, and turning raw samples into a counts histogram.
class QuantumRuntime {
public:
    explicit QuantumRuntime(std::unique_ptr<Backend> backend);
    explicit QuantumRuntime(DeviceType device_type); // convenience: use BackendFactory

    // Resets backend state and applies every gate in the circuit, in order.
    void run(const Circuit& circuit);

    std::vector<Complex> state_vector() const;
    std::vector<double> probabilities() const;

    // Runs the circuit fresh, then samples `shots` measurements.
    std::vector<unsigned long long> sample(const Circuit& circuit, std::size_t shots = 1024);

    // Same as sample(), but returns a bitstring -> count histogram,
    // which is usually what you want for reporting results.
    std::map<std::string, std::size_t> sample_counts(const Circuit& circuit, std::size_t shots = 1024);

    std::string device_name() const;

private:
    std::unique_ptr<Backend> backend_;
    std::size_t num_qubits_ = 0;
};

} // namespace quantum
