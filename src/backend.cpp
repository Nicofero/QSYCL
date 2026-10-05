#include "quantum/backend.hpp"
#include <stdexcept>

namespace quantum {

double Backend::expectation(const Observable& obs) {
    return expectation_value(get_state(), obs);
}

std::vector<double> Backend::expectation_batch(const Circuit& ansatz,
                                               const Observable& obs,
                                               const EvalBatch& batch) {
    if (obs.num_qubits() != ansatz.num_qubits()) {
        throw std::invalid_argument("expectation_batch: observable and ansatz qubit counts differ");
    }
    if (batch.count != 0 && batch.num_params != ansatz.num_params()) {
        throw std::invalid_argument("expectation_batch: batch parameter count does not match ansatz");
    }
    std::vector<double> out;
    out.reserve(batch.count);
    for (std::size_t i = 0; i < batch.count; ++i) {
        const int op = batch.shift_op.empty() ? -1 : batch.shift_op[i];
        const double delta = batch.shift.empty() ? 0.0 : batch.shift[i];
        Circuit bound = ansatz.bind(batch.theta_of(i), op, delta);
        initialize(bound.num_qubits());
        for (const auto& gate : bound.operations()) apply_gate(gate);
        out.push_back(expectation(obs));
    }
    return out;
}

} // namespace quantum
