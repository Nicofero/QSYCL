#pragma once

#include "quantum/backend.hpp"
#include <sycl/sycl.hpp>
#include <random>

namespace quantum::backends {

// Full state-vector simulator running on a SYCL CPU device.
// Simple and exact, but memory scales as O(2^num_qubits) -- fine for
// development and small circuits (roughly up to ~26-28 qubits on a
// machine with tens of GB of RAM).
class CUNQABackend : public Backend {
public:
    CUNQABackend();

    void initialize(std::size_t num_qubits) override;
    void apply_gate(const GateOp& op) override;
    std::vector<Complex> get_state() const override;
    std::vector<double> probabilities() const override;
    std::vector<unsigned long long> sample(
        const std::vector<std::size_t>& qubits, std::size_t shots) override;
    std::string device_name() const override;

private:
    void apply_single_qubit_matrix(std::size_t qubit, const Matrix2x2& m);
    void apply_controlled_matrix(std::size_t control, std::size_t target, const Matrix2x2& m);
    void apply_swap_gate(std::size_t qubit_a, std::size_t qubit_b);

    sycl::queue queue_;
    std::vector<Complex> state_;   // host mirror, kept in sync after each gate
    std::size_t num_qubits_ = 0;
    std::mt19937_64 rng_{std::random_device{}()};
};

} // namespace quantum::backends
