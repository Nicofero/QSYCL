#include "quantum/algorithms/hamiltonians.hpp"
#include <stdexcept>
#include <string>

namespace quantum::hamiltonians {

Observable maxcut(std::size_t num_nodes, const std::vector<Edge>& edges, bool include_constant) {
    Observable H(num_nodes);
    double constant = 0.0;
    for (const auto& e : edges) {
        if (e.u >= num_nodes || e.v >= num_nodes)
            throw std::out_of_range("maxcut: edge endpoint out of range");
        if (e.u == e.v) throw std::invalid_argument("maxcut: self-loops are not allowed");
        H.add_term(0.5 * e.weight, {{'Z', e.u}, {'Z', e.v}});
        constant -= 0.5 * e.weight;
    }
    if (include_constant && constant != 0.0) H.add_term(constant, std::string(num_nodes, 'I'));
    return H;
}

double cut_value(const std::vector<Edge>& edges, unsigned long long bits) {
    double cut = 0.0;
    for (const auto& e : edges)
        if (((bits >> e.u) & 1ULL) != ((bits >> e.v) & 1ULL)) cut += e.weight;
    return cut;
}

std::pair<double, unsigned long long> maxcut_brute_force(std::size_t num_nodes, const std::vector<Edge>& edges) {
    if (num_nodes == 0 || num_nodes > 26) throw std::invalid_argument("maxcut_brute_force: num_nodes must be in [1, 26]");
    double best = -1.0;
    unsigned long long best_bits = 0;
    // Fix the top node on side 0: a cut and its complement have equal weight.
    const unsigned long long limit = 1ULL << (num_nodes - 1);
    for (unsigned long long x = 0; x < limit; ++x) {
        const double c = cut_value(edges, x);
        if (c > best) { best = c; best_bits = x; }
    }
    return {best, best_bits};
}

static void check_chain(std::size_t n, bool periodic) {
    if (n < 2) throw std::invalid_argument("chain Hamiltonian needs at least 2 sites");
    if (periodic && n < 3) throw std::invalid_argument("periodic chain needs at least 3 sites");
}

Observable ising_chain(std::size_t n, double J, double h, bool periodic) {
    check_chain(n, periodic);
    Observable H(n);
    const std::size_t bonds = periodic ? n : n - 1;
    for (std::size_t i = 0; i < bonds; ++i) H.add_term(-J, {{'Z', i}, {'Z', (i + 1) % n}});
    if (h != 0.0)
        for (std::size_t i = 0; i < n; ++i) {
            // Spelled out: a one-entry braced list is ambiguous between the
            // string and vector overloads of add_term().
            const std::vector<std::pair<char, std::size_t>> x_i = {{'X', i}};
            H.add_term(-h, x_i);
        }
    return H;
}

Observable heisenberg_chain(std::size_t n, double J, double delta, bool periodic) {
    check_chain(n, periodic);
    Observable H(n);
    const std::size_t bonds = periodic ? n : n - 1;
    for (std::size_t i = 0; i < bonds; ++i) {
        const std::size_t j = (i + 1) % n;
        H.add_term(J, {{'X', i}, {'X', j}});
        H.add_term(J, {{'Y', i}, {'Y', j}});
        if (delta != 0.0) H.add_term(J * delta, {{'Z', i}, {'Z', j}});
    }
    return H;
}

Observable from_pauli_terms(std::size_t num_qubits, const std::vector<std::pair<double, std::string>>& terms) {
    Observable H(num_qubits);
    for (const auto& [coeff, paulis] : terms) H.add_term(coeff, paulis);
    return H;
}

} // namespace quantum::hamiltonians
