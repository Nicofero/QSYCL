#include "cpu_backend.hpp"
#include "quantum/gates.hpp"
#include "quantum/device_selector.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace quantum::backends {

CPUBackend::CPUBackend()
    : queue_(DeviceSelector::make_queue(DeviceType::CPU)) {}

void CPUBackend::initialize(std::size_t num_qubits) {
    if (num_qubits == 0 || num_qubits > 30) {
        throw std::invalid_argument("CPUBackend: num_qubits must be in [1, 30] for this reference implementation");
    }
    num_qubits_ = num_qubits;
    std::size_t dim = std::size_t(1) << num_qubits;
    state_.assign(dim, Complex(0.0, 0.0));
    state_[0] = Complex(1.0, 0.0);
}

void CPUBackend::apply_single_qubit_matrix(std::size_t qubit, const Matrix2x2& m) {
    std::size_t dim = state_.size();
    std::size_t mask = std::size_t(1) << qubit;

    sycl::buffer<Complex, 1> buf(state_.data(), sycl::range<1>(dim));
    queue_.submit([&](sycl::handler& h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);
        Matrix2x2 mat = m; // captured by value, kernel-safe
        h.parallel_for(sycl::range<1>(dim / 2), [=](sycl::id<1> idx) {
            std::size_t i = idx[0];
            std::size_t low = i & (mask - 1);
            std::size_t high = i & ~(mask - 1);
            std::size_t i0 = (high << 1) | low;
            std::size_t i1 = i0 | mask;

            Complex a0 = acc[i0];
            Complex a1 = acc[i1];
            acc[i0] = mat[0] * a0 + mat[1] * a1;
            acc[i1] = mat[2] * a0 + mat[3] * a1;
        });
    }).wait();
}

void CPUBackend::apply_controlled_matrix(std::size_t control, std::size_t target, const Matrix2x2& m) {
    std::size_t dim = state_.size();
    std::size_t mask_t = std::size_t(1) << target;
    std::size_t mask_c = std::size_t(1) << control;

    sycl::buffer<Complex, 1> buf(state_.data(), sycl::range<1>(dim));
    queue_.submit([&](sycl::handler& h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);
        Matrix2x2 mat = m;
        h.parallel_for(sycl::range<1>(dim / 2), [=](sycl::id<1> idx) {
            std::size_t i = idx[0];
            std::size_t low = i & (mask_t - 1);
            std::size_t high = i & ~(mask_t - 1);
            std::size_t i0 = (high << 1) | low;
            std::size_t i1 = i0 | mask_t;

            if (i0 & mask_c) { // control bit is identical on i0 and i1
                Complex a0 = acc[i0];
                Complex a1 = acc[i1];
                acc[i0] = mat[0] * a0 + mat[1] * a1;
                acc[i1] = mat[2] * a0 + mat[3] * a1;
            }
        });
    }).wait();
}

void CPUBackend::apply_swap_gate(std::size_t qubit_a, std::size_t qubit_b) {
    std::size_t dim = state_.size();
    std::size_t mask_a = std::size_t(1) << qubit_a;
    std::size_t mask_b = std::size_t(1) << qubit_b;

    sycl::buffer<Complex, 1> buf(state_.data(), sycl::range<1>(dim));
    queue_.submit([&](sycl::handler& h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(dim), [=](sycl::id<1> idx) {
            std::size_t i = idx[0];
            if ((i & mask_a) == 0 && (i & mask_b) != 0) {
                std::size_t j = (i | mask_a) & ~mask_b;
                Complex tmp = acc[i];
                acc[i] = acc[j];
                acc[j] = tmp;
            }
        });
    }).wait();
}

void CPUBackend::apply_gate(const GateOp& op) {
    using namespace quantum::gates;
    switch (op.type) {
        case GateType::H:  apply_single_qubit_matrix(op.qubits[0], H()); break;
        case GateType::X:  apply_single_qubit_matrix(op.qubits[0], X()); break;
        case GateType::Y:  apply_single_qubit_matrix(op.qubits[0], Y()); break;
        case GateType::Z:  apply_single_qubit_matrix(op.qubits[0], Z()); break;
        case GateType::S:  apply_single_qubit_matrix(op.qubits[0], S()); break;
        case GateType::T:  apply_single_qubit_matrix(op.qubits[0], T()); break;
        case GateType::RX: apply_single_qubit_matrix(op.qubits[0], RX(op.params[0])); break;
        case GateType::RY: apply_single_qubit_matrix(op.qubits[0], RY(op.params[0])); break;
        case GateType::RZ: apply_single_qubit_matrix(op.qubits[0], RZ(op.params[0])); break;
        case GateType::CNOT: apply_controlled_matrix(op.qubits[0], op.qubits[1], X()); break;
        case GateType::CZ:   apply_controlled_matrix(op.qubits[0], op.qubits[1], Z()); break;
        case GateType::SWAP: apply_swap_gate(op.qubits[0], op.qubits[1]); break;
        case GateType::MEASURE: /* handled by runtime via sample(); no-op on state */ break;
    }
}

std::vector<Complex> CPUBackend::get_state() const {
    return state_;
}

std::vector<double> CPUBackend::probabilities() const {
    std::vector<double> probs(state_.size());
    for (std::size_t i = 0; i < state_.size(); ++i) probs[i] = state_[i].norm();
    return probs;
}

std::vector<unsigned long long> CPUBackend::sample(
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

std::string CPUBackend::device_name() const {
    return "CPU (SYCL: " + queue_.get_device().get_info<sycl::info::device::name>() + ")";
}

} // namespace quantum::backends
