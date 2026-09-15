#include "qpu_backend.hpp"

#include <stdexcept>

namespace quantum {
namespace backends {

QPUBackend::QPUBackend(std::unique_ptr<Backend> fallback)
    : fallback_(std::move(fallback)) {}

QPUBackend::~QPUBackend() = default;

void QPUBackend::initialize(std::size_t num_qubits) {
    num_qubits_ = num_qubits;
    gate_buffer_.clear();
    if (!device_available_) {
        fallback_->initialize(num_qubits);
    }
}

void QPUBackend::apply_gate(const GateOp& op) {
    if (!device_available_) {
        fallback_->apply_gate(op);
        return;
    }
    gate_buffer_.push_back(op); // buffered, not executed -- see header
}

std::vector<Complex> QPUBackend::get_state() const {
    if (!device_available_) {
        return fallback_->get_state();
    }
    throw std::runtime_error(
        "get_state() is not supported on a real/remote QPU backend: no "
        "real QPU can report exact statevector amplitudes. Use sample() "
        "instead.");
}

std::vector<double> QPUBackend::probabilities() const {
    if (!device_available_) {
        return fallback_->probabilities();
    }
    throw std::runtime_error(
        "probabilities() is not supported on a real/remote QPU backend: "
        "no real QPU can report exact probabilities. Use sample() "
        "instead.");
}

std::vector<unsigned long long> QPUBackend::sample(
    const std::vector<std::size_t>& qubits, std::size_t shots) {
    if (!device_available_) {
        return fallback_->sample(qubits, shots);
    }

    std::vector<std::size_t> measured = qubits;
    if (measured.empty()) {
        measured.resize(num_qubits_);
        for (std::size_t i = 0; i < num_qubits_; ++i) measured[i] = i;
    }

    try {
        return submit_circuit(measured, shots).outcomes;
    } catch (const std::exception&) {
        // See the class-level "Retry ownership" note in the header: a
        // subclass is expected to have already done whatever retrying
        // is safe for its own protocol before letting an exception
        // reach here, so this never retries submit_circuit() itself --
        // it only decides whether to fall back or propagate.
        recheck_device();
        if (!device_available_) {
            fallback_->initialize(num_qubits_);
            for (const auto& op : gate_buffer_) {
                fallback_->apply_gate(op);
            }
            return fallback_->sample(qubits, shots);
        }
        throw; // device still considered available -- surface the real error
    }
}

} // namespace backends
} // namespace quantum