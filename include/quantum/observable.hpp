#pragma once

#include "quantum/types.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace quantum {

// One Pauli string in a form that is cheap to evaluate on ANY device.
//
// For P = (tensor product of I/X/Y/Z) acting on a basis state |j>:
//     P|j> = i^{ny} * (-1)^{popcount(j & z)} * |j XOR x>
// where x has a bit for every X or Y factor, z has a bit for every Z or Y
// factor, and ny is the number of Y factors. So
//     <psi|P|psi> = i^{ny} * sum_j conj(psi[j ^ x]) * (-1)^{popcount(j & z)} * psi[j]
// A device backend only needs these three integers per term (see
// Observable::masks()); no strings or host objects reach the kernel.
struct PauliMask {
    double        coeff;
    std::uint64_t x;
    std::uint64_t z;
    std::uint8_t  ny; // number of Y factors (only ny mod 4 matters)
};

// A real-weighted sum of Pauli strings:  H = sum_k c_k P_k   (Hermitian).
class Observable {
public:
    explicit Observable(std::size_t num_qubits);

    // Dense form, character k acts on qubit k. Example: add_term(0.5, "ZZI")
    // is 0.5 * Z_0 Z_1 on 3 qubits.
    Observable& add_term(double coeff, const std::string& paulis);

    // Sparse form: add_term(1.0, {{'Z', 0}, {'Z', 1}}).
    Observable& add_term(double coeff, const std::vector<std::pair<char, std::size_t>>& factors);

    std::size_t num_qubits() const { return num_qubits_; }
    const std::vector<PauliMask>& masks() const { return masks_; }
    std::size_t num_terms() const { return masks_.size(); }

    // True if every term is Z/I only. Then <H> = sum_j |psi_j|^2 h_j for a
    // precomputed diagonal h -- a big shortcut (QAOA, Ising, MaxCut...).
    bool is_diagonal() const;

    // Diagonal entries h_j for a diagonal observable (length 2^n). Throws otherwise.
    std::vector<double> diagonal() const;

private:
    std::size_t num_qubits_;
    std::vector<PauliMask> masks_;
};

// Host reference implementation of <psi|H|psi> from a full state vector.
// Used as the default by Backend::expectation(); native backends can (and
// should) compute this on-device instead.
double expectation_value(const std::vector<Complex>& state, const Observable& obs);

} // namespace quantum
