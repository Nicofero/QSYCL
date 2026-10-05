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
    // Throws if the circuit still has free parameters: call Circuit::bind() first.
    void run(const Circuit& circuit);

    // Runs a fully bound circuit and returns <H> on the resulting state.
    double expectation(const Circuit& circuit, const Observable& obs);

    // Evaluates one parameterized ansatz for many parameter sets at once
    // (see EvalBatch). Backends may execute the whole batch in parallel.
    std::vector<double> expectation_batch(const Circuit& ansatz, const Observable& obs,
                                          const EvalBatch& batch);

    // Escape hatch for layers (like VQA) that talk to backend capabilities.
    Backend& backend() { return *backend_; }

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
