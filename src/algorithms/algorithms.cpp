#include "quantum/algorithms/algorithms.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace quantum {

namespace {

int parity64(unsigned long long v) {
    v ^= v >> 32; v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
    return static_cast<int>(v & 1ULL);
}

// Runs `restarts` optimizations from the given starting points and keeps the best.
struct Sweep {
    VQAResult best;
    std::size_t best_restart = 0;
    std::vector<double> energies;
    std::size_t evaluations = 0;
};

Sweep run_restarts(QuantumRuntime& runtime, const Circuit& ansatz, const Observable& H,
                   const VQAOptions& base, std::size_t restarts,
                   const std::vector<std::vector<double>>& starts) {
    if (restarts == 0) throw std::invalid_argument("restarts must be at least 1");
    VQA vqa(runtime, ansatz, H);
    Sweep sweep;
    double best_energy = std::numeric_limits<double>::infinity();
    for (std::size_t r = 0; r < restarts; ++r) {
        VQAOptions opt = base;
        opt.seed = base.seed + r;   // SPSA's perturbations differ per restart too
        VQAResult res = vqa.minimize(starts.at(r), opt);
        sweep.evaluations += res.evaluations;
        sweep.energies.push_back(res.energy);
        // `!(a < b)` style so a NaN energy can never be selected as the best.
        if (res.energy < best_energy) {
            best_energy = res.energy;
            sweep.best = std::move(res);
            sweep.best_restart = r;
        }
    }
    if (!(best_energy < std::numeric_limits<double>::infinity()))
        throw std::runtime_error("all restarts produced a non-finite energy");
    return sweep;
}

} // namespace

double diagonal_cost(const Observable& cost, unsigned long long bits) {
    if (!cost.is_diagonal()) throw std::invalid_argument("diagonal_cost: observable has X/Y terms");
    double total = 0.0;
    for (const auto& t : cost.masks()) total += t.coeff * (parity64(bits & t.z) ? -1.0 : 1.0);
    return total;
}

VQEResult vqe(QuantumRuntime& runtime, const Circuit& ansatz, const Observable& hamiltonian,
              const VQEOptions& options) {
    const std::size_t P = ansatz.num_params();
    if (!options.initial_point.empty() && options.initial_point.size() != P)
        throw std::invalid_argument("vqe: initial_point has " + std::to_string(options.initial_point.size()) +
                                    " entries but the ansatz has " + std::to_string(P) + " parameters");

    std::vector<std::vector<double>> starts;
    for (std::size_t r = 0; r < std::max<std::size_t>(options.restarts, 1); ++r) {
        if (r == 0 && !options.initial_point.empty()) starts.push_back(options.initial_point);
        else starts.push_back(ansatz::random_parameters(P, options.seed + r, -options.init_range, options.init_range));
    }
    Sweep s = run_restarts(runtime, ansatz, hamiltonian, options.vqa, options.restarts, starts);

    VQEResult out;
    out.best = std::move(s.best);
    out.best_restart = s.best_restart;
    out.restart_energies = std::move(s.energies);
    out.total_evaluations = s.evaluations;
    return out;
}

QAOAResult qaoa(QuantumRuntime& runtime, const Observable& cost, const QAOAOptions& options) {
    if (options.p == 0) throw std::invalid_argument("qaoa: p must be at least 1");
    if (!cost.is_diagonal())
        throw std::invalid_argument("qaoa: the cost Hamiltonian must be diagonal (only Z/I terms); use vqe() for other Hamiltonians");

    const Circuit circuit = ansatz::qaoa(cost, options.p);
    const std::size_t P = circuit.num_params();   // 2p
    if (!options.initial_point.empty() && options.initial_point.size() != P)
        throw std::invalid_argument("qaoa: initial_point must have 2*p = " + std::to_string(P) + " entries");

    std::vector<std::vector<double>> starts;
    for (std::size_t r = 0; r < std::max<std::size_t>(options.restarts, 1); ++r) {
        if (r == 0 && !options.initial_point.empty()) starts.push_back(options.initial_point);
        else if (r == 0) starts.push_back(ansatz::qaoa_linear_ramp(options.p));
        else starts.push_back(ansatz::random_parameters(P, options.seed + r, -options.init_range, options.init_range));
    }
    Sweep s = run_restarts(runtime, circuit, cost, options.vqa, options.restarts, starts);

    QAOAResult out;
    out.best = std::move(s.best);
    out.best_restart = s.best_restart;
    out.restart_energies = std::move(s.energies);
    out.total_evaluations = s.evaluations;

    // Read the answer off the optimized state: sample it and keep the cheapest bitstring.
    const auto samples = runtime.sample(circuit.bind(out.best.theta), options.sample_shots);
    if (samples.empty()) throw std::runtime_error("qaoa: the backend returned no samples");
    double best_cost = std::numeric_limits<double>::infinity();
    for (auto x : samples) {
        ++out.counts[x];
    }
    for (const auto& [bits, n] : out.counts) {
        (void)n;
        const double c = diagonal_cost(cost, bits);
        if (c < best_cost) { best_cost = c; out.best_bitstring = bits; }
    }
    out.best_cost = best_cost;
    return out;
}

} // namespace quantum
