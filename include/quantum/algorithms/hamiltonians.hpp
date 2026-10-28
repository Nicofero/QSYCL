#pragma once

#include "quantum/observable.hpp"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace quantum::hamiltonians {

// Builders for common problem Hamiltonians, returned as Observables ready for
// VQA / vqe() / qaoa(). Convention (matches Observable): qubit q is character q
// of a Pauli string, and bit q of a basis-state index.

// A weighted undirected graph edge.
struct Edge {
    std::size_t u, v;
    double weight = 1.0;
};

// MaxCut cost to MINIMIZE:  C = sum_{(u,v)} w_uv * (Z_u Z_v - 1) / 2.
// For a bitstring x (Z=+1 for bit 0, Z=-1 for bit 1) the energy is exactly
// -(weight of the edges cut by x), so min C = -MaxCut. Diagonal, hence usable
// directly as the QAOA cost. The constant -sum(w)/2 is KEPT, so reported
// energies are true negative cut values; pass include_constant=false to get the
// bare sum w/2 Z Z form (same minimizer, shifted energies).
Observable maxcut(std::size_t num_nodes, const std::vector<Edge>& edges, bool include_constant = true);

// Cut weight of the partition encoded by bit q of `bits` (bit q = side of node q).
double cut_value(const std::vector<Edge>& edges, unsigned long long bits);

// Brute-force MaxCut (2^(n-1) cuts; for tests and small instances, n <= 26).
// Returns {best cut weight, one optimal bitstring}.
std::pair<double, unsigned long long> maxcut_brute_force(std::size_t num_nodes, const std::vector<Edge>& edges);

// Transverse-field Ising chain:  H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
// (open chain, or periodic if `periodic`; periodic needs n >= 3 to avoid a
// double bond). Not diagonal when h != 0, so use it with vqe(), not qaoa().
Observable ising_chain(std::size_t n, double J, double h, bool periodic = false);

// Heisenberg XXZ chain:  H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1})
// Delta = 1 is the isotropic Heisenberg model; Delta = 0 is the XX model.
Observable heisenberg_chain(std::size_t n, double J = 1.0, double delta = 1.0, bool periodic = false);

// A Hamiltonian given directly as weighted Pauli strings, e.g.
//   from_pauli_terms(2, {{0.5, "ZZ"}, {-1.0, "XI"}}).
Observable from_pauli_terms(std::size_t num_qubits, const std::vector<std::pair<double, std::string>>& terms);

} // namespace quantum::hamiltonians
