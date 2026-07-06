#include "quantum/runtime.hpp"
#include "quantum/backend_factory.hpp"
#include <numeric>
#include <bitset>

namespace quantum {

QuantumRuntime::QuantumRuntime(std::unique_ptr<Backend> backend)
    : backend_(std::move(backend)) {}

QuantumRuntime::QuantumRuntime(DeviceType device_type)
    : backend_(BackendFactory::create(device_type)) {}

void QuantumRuntime::run(const Circuit& circuit) {
    num_qubits_ = circuit.num_qubits();
    backend_->initialize(num_qubits_);
    for (const auto& op : circuit.operations()) {
        backend_->apply_gate(op);
    }
}

std::vector<Complex> QuantumRuntime::state_vector() const {
    return backend_->get_state();
}

std::vector<double> QuantumRuntime::probabilities() const {
    return backend_->probabilities();
}

std::vector<unsigned long long> QuantumRuntime::sample(const Circuit& circuit, std::size_t shots) {
    run(circuit);
    auto measured = circuit.measured_qubits();
    if (measured.empty()) {
        measured.resize(num_qubits_);
        std::iota(measured.begin(), measured.end(), 0);
    }
    return backend_->sample(measured, shots);
}

std::map<std::string, std::size_t> QuantumRuntime::sample_counts(const Circuit& circuit, std::size_t shots) {
    auto measured = circuit.measured_qubits();
    std::size_t num_readout = measured.empty() ? circuit.num_qubits() : measured.size();

    auto raw = sample(circuit, shots);
    std::map<std::string, std::size_t> counts;
    for (auto value : raw) {
        std::string bits;
        for (std::size_t k = 0; k < num_readout; ++k) {
            bits += ((value >> (num_readout - 1 - k)) & 1ULL) ? '1' : '0';
        }
        counts[bits]++;
    }
    return counts;
}

std::string QuantumRuntime::device_name() const {
    return backend_->device_name();
}

} // namespace quantum
