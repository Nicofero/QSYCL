#include "gpu_backend.hpp"
#include "quantum/gates.hpp"
#include "quantum/device_selector.hpp"
#include <stdexcept>
#include <numeric>

namespace quantum::backends {

GPUBackend::GPUBackend()
    : queue_(DeviceSelector::make_queue(DeviceType::GPU)) {}

GPUBackend::~GPUBackend() { free_state(); }

void GPUBackend::free_state() {
    if (state_dev_) {
        sycl::free(state_dev_, queue_); // must free with the SAME queue/context that allocated it
        state_dev_ = nullptr;
    }
}

void GPUBackend::initialize(std::size_t num_qubits) {
    if (num_qubits == 0 || num_qubits > 30) {
        throw std::invalid_argument("GPUBackend: num_qubits must be in [1, 30]");
    }
    free_state(); // guard against re-running a circuit without leaking the old allocation

    num_qubits_ = num_qubits;
    dim_ = std::size_t(1) << num_qubits;

    state_dev_ = sycl::malloc_device<Complex>(dim_, queue_);
    if (!state_dev_) {
        // malloc_device returns nullptr on failure rather than throwing --
        // check it explicitly, or any kernel that uses it will segfault.
        throw std::runtime_error("GPUBackend: sycl::malloc_device failed (out of device memory?)");
    }

    // Zero the state, then set |0...0> to amplitude 1. Both done ON DEVICE --
    // we never touch state_dev_ from host code.
    Complex* state = state_dev_; // capture the pointer VALUE, not `this`
    queue_.parallel_for(sycl::range<1>(dim_), [=](sycl::id<1> idx) {
        state[idx[0]] = Complex(0.0, 0.0);
    }).wait();
    queue_.submit([&](sycl::handler& h) {
        h.single_task([=]() { state[0] = Complex(1.0, 0.0); });
    }).wait();
}

void GPUBackend::apply_single_qubit_matrix(std::size_t qubit, const Matrix2x2& m) {
    if (!state_dev_) throw std::runtime_error("GPUBackend::apply_gate called before initialize()");

    std::size_t mask = std::size_t(1) << qubit;
    Complex* state = state_dev_; // local copies -- these are what the lambda captures by value
    Matrix2x2 mat = m;

    queue_.parallel_for(sycl::range<1>(dim_ / 2), [=](sycl::id<1> idx) {
        std::size_t i = idx[0];
        std::size_t low = i & (mask - 1);
        std::size_t high = i & ~(mask - 1);
        std::size_t i0 = (high << 1) | low;
        std::size_t i1 = i0 | mask;

        Complex a0 = state[i0];
        Complex a1 = state[i1];
        state[i0] = mat[0] * a0 + mat[1] * a1;
        state[i1] = mat[2] * a0 + mat[3] * a1;
    }).wait();
}

void GPUBackend::apply_controlled_matrix(std::size_t control, std::size_t target, const Matrix2x2& m) {
    if (!state_dev_) throw std::runtime_error("GPUBackend::apply_gate called before initialize()");

    std::size_t mask_t = std::size_t(1) << target;
    std::size_t mask_c = std::size_t(1) << control;
    Complex* state = state_dev_;
    Matrix2x2 mat = m;

    queue_.parallel_for(sycl::range<1>(dim_ / 2), [=](sycl::id<1> idx) {
        std::size_t i = idx[0];
        std::size_t low = i & (mask_t - 1);
        std::size_t high = i & ~(mask_t - 1);
        std::size_t i0 = (high << 1) | low;
        std::size_t i1 = i0 | mask_t;

        if (i0 & mask_c) {
            Complex a0 = state[i0];
            Complex a1 = state[i1];
            state[i0] = mat[0] * a0 + mat[1] * a1;
            state[i1] = mat[2] * a0 + mat[3] * a1;
        }
    }).wait();
}

void GPUBackend::apply_swap_gate(std::size_t qubit_a, std::size_t qubit_b) {
    if (!state_dev_) throw std::runtime_error("GPUBackend::apply_gate called before initialize()");

    std::size_t mask_a = std::size_t(1) << qubit_a;
    std::size_t mask_b = std::size_t(1) << qubit_b;
    Complex* state = state_dev_;

    queue_.parallel_for(sycl::range<1>(dim_), [=](sycl::id<1> idx) {
        std::size_t i = idx[0];
        if ((i & mask_a) == 0 && (i & mask_b) != 0) {
            std::size_t j = (i | mask_a) & ~mask_b;
            Complex tmp = state[i];
            state[i] = state[j];
            state[j] = tmp;
        }
    }).wait();
}

void GPUBackend::apply_gate(const GateOp& op) {
    using namespace quantum::gates;
    switch (op.type) {
        case GateType::H:  apply_single_qubit_matrix(op.qubits[0], H()); break;
        case GateType::X:  apply_single_qubit_matrix(op.qubits[0], X()); break;
        case GateType::Y:  apply_single_qubit_matrix(op.qubits[0], Y()); break;
        case GateType::Z:  apply_single_qubit_matrix(op.qubits[0], Z()); break;
        case GateType::S:  if (op.dagger) apply_single_qubit_matrix(op.qubits[0], Sdg());
                           else apply_single_qubit_matrix(op.qubits[0], S()); break;
        case GateType::T:  if (op.dagger) apply_single_qubit_matrix(op.qubits[0], Tdg());
                           else apply_single_qubit_matrix(op.qubits[0], T()); break;
        case GateType::U:  apply_single_qubit_matrix(op.qubits[0], U(op.params[0], op.params[1], op.params[2])); break;
        case GateType::RX: apply_single_qubit_matrix(op.qubits[0], RX(op.params[0])); break;
        case GateType::RY: apply_single_qubit_matrix(op.qubits[0], RY(op.params[0])); break;
        case GateType::RZ: apply_single_qubit_matrix(op.qubits[0], RZ(op.params[0])); break;
        case GateType::CNOT: apply_controlled_matrix(op.qubits[0], op.qubits[1], X()); break;
        case GateType::CZ:   apply_controlled_matrix(op.qubits[0], op.qubits[1], Z()); break;
        case GateType::CRX:  apply_controlled_matrix(op.qubits[0], op.qubits[1], RX(op.params[0])); break;
        case GateType::CRY:  apply_controlled_matrix(op.qubits[0], op.qubits[1], RY(op.params[0])); break;
        case GateType::CRZ:  apply_controlled_matrix(op.qubits[0], op.qubits[1], RZ(op.params[0])); break;
        case GateType::SWAP: apply_swap_gate(op.qubits[0], op.qubits[1]); break;
        case GateType::MEASURE: break;
    }
}

std::vector<Complex> GPUBackend::get_state() const {
    if (!state_dev_) throw std::runtime_error("GPUBackend::get_state called before initialize()");
    std::vector<Complex> host(dim_); // host-side buffer we're allowed to dereference
    queue_.memcpy(host.data(), state_dev_, dim_ * sizeof(Complex)).wait(); // device -> host copy
    return host;
}

std::vector<double> GPUBackend::probabilities() const {
    auto host_state = get_state(); // reuse the safe copy-back path above
    std::vector<double> probs(host_state.size());
    for (std::size_t i = 0; i < host_state.size(); ++i) probs[i] = host_state[i].norm();
    return probs;
}

std::vector<unsigned long long> GPUBackend::sample(
    const std::vector<std::size_t>& qubits, std::size_t shots) {
    auto probs = probabilities();
    std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());

    std::vector<unsigned long long> results;
    results.reserve(shots);

    std::vector<std::size_t> which = qubits.empty()
        ? [&] { std::vector<std::size_t> all(num_qubits_); std::iota(all.begin(), all.end(), 0); return all; }()
        : qubits;

    for (std::size_t s = 0; s < shots; ++s) {
        std::size_t full_index = dist(rng_);
        unsigned long long readout = 0;
        for (std::size_t k = 0; k < which.size(); ++k) {
            if (full_index & (std::size_t(1) << which[k])) readout |= (1ULL << k);
        }
        results.push_back(readout);
    }
    return results;
}

std::string GPUBackend::device_name() const {
    return "GPU (SYCL: " + queue_.get_device().get_info<sycl::info::device::name>() + ")";
}

} // namespace quantum::backends
