#pragma once

#include "quantum/backend.hpp"
#include <sycl/sycl.hpp>
#include <random>

namespace quantum::backends {

// Same simulation algorithm as CPUBackend, but the state vector lives
// entirely on the device via USM (Unified Shared Memory) instead of a
// sycl::buffer, so it stays resident on the GPU across gates instead of
// syncing back to host after every call.
class GPUBackend : public Backend {
public:
    GPUBackend();
    ~GPUBackend() override;

    // Owns a raw USM allocation -- disable copying so we never get a
    // double-free from two backends pointing at the same device memory.
    GPUBackend(const GPUBackend&) = delete;
    GPUBackend& operator=(const GPUBackend&) = delete;

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
    void free_state();

    mutable sycl::queue queue_; // mutable: get_state()/probabilities() are
                                 // const but still need to issue a memcpy
    Complex* state_dev_ = nullptr; // USM device pointer -- NEVER dereference on host
    std::size_t dim_ = 0;
    std::size_t num_qubits_ = 0;
    std::mt19937_64 rng_{std::random_device{}()};
};

} // namespace quantum::backends
