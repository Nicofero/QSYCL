#pragma once

#include "quantum/algorithms/ansatz.hpp"
#include "quantum/observable.hpp"
#include "quantum/runtime.hpp"
#include "quantum/vqa.hpp"
#include <cstddef>
#include <map>
#include <vector>

namespace quantum {

// Ready-made variational algorithms built on VQA. They add what a bare
// VQA::minimize() leaves to the caller: choosing starting points, restarting
// from several of them (variational landscapes are non-convex), and for QAOA
// turning the optimized state into an actual answer (a bitstring).
//
// Both run wherever the QuantumRuntime's backend runs: state-vector CPU/GPU
// (native batched kernels), or shot-based CUNQA/QPU. Optimizer settings,
// including shots and optimizer_device, are the usual VQAOptions.

// ---------------------------------------------------------------------------
// VQE: minimize <ansatz(theta)| H |ansatz(theta)>.
// ---------------------------------------------------------------------------
struct VQEOptions {
    VQAOptions vqa;                    // optimizer, gradient method, shots, devices, ...

    std::size_t restarts = 1;          // independent optimizations; the lowest energy wins
    unsigned long long seed = 1234;    // restart k draws its start from seed + k

    // Starting point of restart 0. Empty = random like the others. A user point
    // is never perturbed, so restarts == 1 with a point is exactly VQA::minimize.
    std::vector<double> initial_point;

    // Random starts are uniform in [-init_range, init_range]. Small values suit
    // deep ansatzes, where large random angles land on barren plateaus.
    double init_range = 3.14159265358979323846;
};

struct VQEResult {
    VQAResult best;                    // the winning restart (theta, energy, history, ...)
    std::size_t best_restart = 0;
    std::vector<double> restart_energies;   // final energy of every restart
    std::size_t total_evaluations = 0;      // circuit executions over ALL restarts
};

VQEResult vqe(QuantumRuntime& runtime, const Circuit& ansatz, const Observable& hamiltonian,
              const VQEOptions& options = {});

// ---------------------------------------------------------------------------
// QAOA for a diagonal (Z-only) cost Hamiltonian, e.g. hamiltonians::maxcut().
// Builds ansatz::qaoa(cost, p), optimizes it, then samples the optimized state
// and reports the best bitstring found.
// ---------------------------------------------------------------------------
struct QAOAOptions {
    std::size_t p = 1;                 // number of QAOA layers
    VQAOptions vqa;
    std::size_t restarts = 1;
    unsigned long long seed = 1234;

    // Parameter order is [gamma_0, beta_0, gamma_1, beta_1, ...] (see ansatz.hpp).
    // Empty: restart 0 starts from the linear-ramp heuristic, later restarts from
    // random angles in [-init_range, init_range].
    std::vector<double> initial_point;
    double init_range = 3.14159265358979323846;

    std::size_t sample_shots = 4096;   // shots used to read the answer off the final state
};

struct QAOAResult {
    VQAResult best;                    // optimized parameters and expected cost
    std::size_t best_restart = 0;
    std::vector<double> restart_energies;
    std::size_t total_evaluations = 0;

    // The answer. Bit q of `best_bitstring` is qubit q (for MaxCut: the side of node q).
    // It is the LOWEST-COST bitstring among the sampled ones, which for a cost
    // that is being minimized is what you actually want -- not the most frequent.
    unsigned long long best_bitstring = 0;
    double best_cost = 0.0;            // cost of best_bitstring (exact, from the Hamiltonian)
    std::map<unsigned long long, std::size_t> counts;  // sampled bitstring -> occurrences
};

QAOAResult qaoa(QuantumRuntime& runtime, const Observable& cost, const QAOAOptions& options = {});

// Cost of one basis state for a diagonal Observable: sum_t c_t * (-1)^popcount(bits & z_t).
// Throws if the observable has X/Y terms. Constants (identity terms) are included.
double diagonal_cost(const Observable& cost, unsigned long long bits);

} // namespace quantum
