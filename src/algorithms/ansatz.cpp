#include "quantum/algorithms/ansatz.hpp"
#include <cctype>
#include <random>
#include <stdexcept>

namespace quantum::ansatz {

std::vector<std::pair<std::size_t, std::size_t>> entanglement_pairs(std::size_t n, Entanglement e) {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    if (n < 2) return pairs;
    switch (e) {
        case Entanglement::Linear:
            for (std::size_t i = 0; i + 1 < n; ++i) pairs.emplace_back(i, i + 1);
            break;
        case Entanglement::Circular:
            for (std::size_t i = 0; i + 1 < n; ++i) pairs.emplace_back(i, i + 1);
            if (n > 2) pairs.emplace_back(n - 1, 0);   // for n == 2 this would just repeat (0,1)
            break;
        case Entanglement::Full:
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = i + 1; j < n; ++j) pairs.emplace_back(i, j);
            break;
        case Entanglement::Pairwise:
            for (std::size_t i = 0; i + 1 < n; i += 2) pairs.emplace_back(i, i + 1);
            for (std::size_t i = 1; i + 1 < n; i += 2) pairs.emplace_back(i, i + 1);
            break;
    }
    return pairs;
}

namespace {

void rotation_layer(Circuit& c, char axis, const std::vector<Param>& angles) {
    for (std::size_t q = 0; q < angles.size(); ++q) {
        switch (axis) {
            case 'X': c.rx(q, angles[q]); break;
            case 'Y': c.ry(q, angles[q]); break;
            case 'Z': c.rz(q, angles[q]); break;
        }
    }
}

std::string normalize_rotations(const std::string& rotations) {
    if (rotations.empty()) throw std::invalid_argument("ansatz: rotation string must not be empty");
    std::string out;
    for (char ch : rotations) {
        const char up = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (up != 'X' && up != 'Y' && up != 'Z')
            throw std::invalid_argument(std::string("ansatz: rotation must be X, Y or Z, got '") + ch + "'");
        out += up;
    }
    return out;
}

} // namespace

Circuit& append_two_local(Circuit& circuit, const std::string& rotations, std::size_t reps,
                          Entanglement entanglement, bool final_rotation_layer) {
    const std::string rots = normalize_rotations(rotations);
    const std::size_t n = circuit.num_qubits();
    const auto pairs = entanglement_pairs(n, entanglement);

    auto layer = [&] {
        for (char axis : rots) rotation_layer(circuit, axis, circuit.new_params(n));
    };
    for (std::size_t r = 0; r < reps; ++r) {
        layer();
        for (const auto& [c, t] : pairs) circuit.cnot(c, t);
    }
    if (final_rotation_layer || reps == 0) layer();   // reps == 0 would otherwise add nothing
    return circuit;
}

Circuit two_local(std::size_t num_qubits, const std::string& rotations, std::size_t reps,
                  Entanglement entanglement, bool final_rotation_layer) {
    Circuit c(num_qubits);
    append_two_local(c, rotations, reps, entanglement, final_rotation_layer);
    return c;
}

Circuit& append_real_amplitudes(Circuit& circuit, std::size_t reps, Entanglement entanglement) {
    return append_two_local(circuit, "Y", reps, entanglement, true);
}
Circuit real_amplitudes(std::size_t num_qubits, std::size_t reps, Entanglement entanglement) {
    Circuit c(num_qubits);
    append_real_amplitudes(c, reps, entanglement);
    return c;
}

Circuit& append_efficient_su2(Circuit& circuit, std::size_t reps, Entanglement entanglement) {
    return append_two_local(circuit, "YZ", reps, entanglement, true);
}
Circuit efficient_su2(std::size_t num_qubits, std::size_t reps, Entanglement entanglement) {
    Circuit c(num_qubits);
    append_efficient_su2(c, reps, entanglement);
    return c;
}

Circuit hardware_efficient(std::size_t num_qubits, std::size_t layers, Entanglement entanglement) {
    return efficient_su2(num_qubits, layers, entanglement);
}

Circuit& append_basis_state(Circuit& circuit, unsigned long long bits) {
    for (std::size_t q = 0; q < circuit.num_qubits() && q < 64; ++q)
        if ((bits >> q) & 1ULL) circuit.x(q);
    if (circuit.num_qubits() < 64 && (bits >> circuit.num_qubits()) != 0)
        throw std::out_of_range("append_basis_state: bitstring has bits set beyond the circuit's qubits");
    return circuit;
}

Circuit& append_qaoa_layers(Circuit& circuit, const Observable& cost, std::size_t p) {
    if (cost.num_qubits() != circuit.num_qubits())
        throw std::invalid_argument("append_qaoa_layers: cost Hamiltonian and circuit must have the same number of qubits");
    if (!cost.is_diagonal())
        throw std::invalid_argument("append_qaoa_layers: the cost Hamiltonian must be diagonal (only Z/I terms)");
    const std::size_t n = circuit.num_qubits();

    for (std::size_t layer = 0; layer < p; ++layer) {
        const Param gamma = circuit.new_param();
        const Param beta = circuit.new_param();

        // exp(-i gamma c Z_{a} Z_{b} ...) = CNOT fan-in onto the last qubit of the
        // term, RZ(2 gamma c) there (RZ(t) = exp(-i t Z / 2)), then undo the fan-in.
        for (const auto& term : cost.masks()) {
            if (term.z == 0) continue;                  // identity: global phase only
            std::vector<std::size_t> qs;
            for (std::size_t q = 0; q < n; ++q)
                if ((term.z >> q) & 1ULL) qs.push_back(q);
            const std::size_t target = qs.back();
            for (std::size_t i = 0; i + 1 < qs.size(); ++i) circuit.cnot(qs[i], target);
            circuit.rz(target, (2.0 * term.coeff) * gamma);
            for (std::size_t i = qs.size() - 1; i-- > 0;) circuit.cnot(qs[i], target);
        }
        // exp(-i beta X) = RX(2 beta) on every qubit.
        for (std::size_t q = 0; q < n; ++q) circuit.rx(q, 2.0 * beta);
    }
    return circuit;
}

Circuit qaoa(const Observable& cost, std::size_t p) {
    Circuit c(cost.num_qubits());
    for (std::size_t q = 0; q < c.num_qubits(); ++q) c.h(q);
    append_qaoa_layers(c, cost, p);
    return c;
}

std::vector<double> random_parameters(std::size_t count, unsigned long long seed, double lo, double hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    std::vector<double> v(count);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<double> qaoa_linear_ramp(std::size_t p, double delta) {
    std::vector<double> theta;
    theta.reserve(2 * p);
    for (std::size_t k = 0; k < p; ++k) {
        const double s = (static_cast<double>(k) + 0.5) / static_cast<double>(p);
        theta.push_back(s * delta);           // gamma_k grows
        theta.push_back(-(1.0 - s) * delta);  // beta_k shrinks (sign: see header; verified by tests)
    }
    return theta;
}

} // namespace quantum::ansatz
