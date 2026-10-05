#include "quantum/observable.hpp"
#include <stdexcept>

namespace quantum {

namespace {
inline int parity(std::uint64_t v) {
    v ^= v >> 32; v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
    return static_cast<int>(v & 1ULL);
}
} // namespace

Observable::Observable(std::size_t num_qubits) : num_qubits_(num_qubits) {
    if (num_qubits == 0 || num_qubits > 64) {
        throw std::invalid_argument("Observable: num_qubits must be in [1, 64]");
    }
}

Observable& Observable::add_term(double coeff, const std::string& paulis) {
    if (paulis.size() != num_qubits_) {
        throw std::invalid_argument("Observable::add_term: Pauli string length must equal num_qubits");
    }
    std::vector<std::pair<char, std::size_t>> factors;
    for (std::size_t k = 0; k < paulis.size(); ++k) {
        if (paulis[k] != 'I') factors.emplace_back(paulis[k], k);
    }
    return add_term(coeff, factors);
}

Observable& Observable::add_term(double coeff, const std::vector<std::pair<char, std::size_t>>& factors) {
    PauliMask m{coeff, 0, 0, 0};
    std::uint64_t seen = 0;
    for (const auto& [op, q] : factors) {
        if (q >= num_qubits_) throw std::out_of_range("Observable::add_term: qubit index out of range");
        const std::uint64_t bit = std::uint64_t(1) << q;
        if (seen & bit) throw std::invalid_argument("Observable::add_term: qubit repeated in one term");
        seen |= bit;
        switch (op) {
            case 'I': break;
            case 'X': m.x |= bit; break;
            case 'Y': m.x |= bit; m.z |= bit; ++m.ny; break;
            case 'Z': m.z |= bit; break;
            default: throw std::invalid_argument("Observable::add_term: Pauli must be one of I, X, Y, Z");
        }
    }
    masks_.push_back(m);
    return *this;
}

bool Observable::is_diagonal() const {
    for (const auto& m : masks_) if (m.x != 0) return false;
    return true;
}

std::vector<double> Observable::diagonal() const {
    if (!is_diagonal()) throw std::logic_error("Observable::diagonal: observable has X/Y terms");
    if (num_qubits_ > 30) throw std::length_error("Observable::diagonal: too many qubits");
    std::vector<double> h(std::size_t(1) << num_qubits_, 0.0);
    for (std::size_t j = 0; j < h.size(); ++j) {
        double v = 0.0;
        for (const auto& m : masks_) v += m.coeff * (parity(j & m.z) ? -1.0 : 1.0);
        h[j] = v;
    }
    return h;
}

double expectation_value(const std::vector<Complex>& psi, const Observable& obs) {
    if (psi.size() != (std::size_t(1) << obs.num_qubits())) {
        throw std::invalid_argument("expectation_value: state size does not match observable qubits");
    }
    double total = 0.0;
    for (const auto& m : obs.masks()) {
        double re = 0.0, im = 0.0;
        for (std::size_t j = 0; j < psi.size(); ++j) {
            const double sign = parity(j & m.z) ? -1.0 : 1.0;
            const Complex t = psi[j ^ m.x].conj() * psi[j]; // conj(psi[j^x]) * psi[j]
            re += sign * t.re;
            im += sign * t.im;
        }
        // multiply by i^{ny}; the expectation of a Hermitian operator is real
        switch (m.ny & 3) {
            case 0: break;                              //  1
            case 1: { double r = -im; im = re; re = r; break; } //  i
            case 2: re = -re; im = -im; break;          // -1
            case 3: { double r = im; im = -re; re = r; break; } // -i
        }
        total += m.coeff * re;
    }
    return total;
}

} // namespace quantum
