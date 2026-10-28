// QAOA for MaxCut with the library's Hamiltonian builder and qaoa() driver.
#include "quantum/algorithms/algorithms.hpp"
#include "quantum/algorithms/hamiltonians.hpp"
#include <iostream>

int main() {
    using namespace quantum;

    // A small weighted graph (the weights are the edge weights of the cut).
    const std::size_t n = 6;
    const std::vector<hamiltonians::Edge> edges = {
        {0, 1, 1.0}, {1, 2, 2.0}, {2, 3, 1.0}, {3, 4, 1.5}, {4, 5, 1.0}, {5, 0, 2.0}, {0, 3, 0.5}};

    // Energy of a bitstring == -(cut value), so minimizing the energy maximizes the cut.
    const Observable cost = hamiltonians::maxcut(n, edges);

    QuantumRuntime runtime(DeviceType::CUNQA);
    std::cout << "Running on: " << runtime.device_name() << "\n";

    QAOAOptions opt;
    opt.p = 3;                       // QAOA depth: 2p parameters [gamma_0, beta_0, gamma_1, ...]
    opt.restarts = 1;                // restart 0 uses the linear-ramp start, the rest are random
    opt.vqa.optimizer = OptimizerKind::Adam;
    opt.vqa.learning_rate = 0.05;
    opt.vqa.max_iterations = 300;
    opt.sample_shots = 4096;         // final state is sampled to read out a bitstring


    const QAOAResult res = qaoa(runtime, cost, opt);
    const auto exact = hamiltonians::maxcut_brute_force(n, edges);

    std::cout << "<C> = " << res.best.energy << "\n"
              << "best sampled cut = " << hamiltonians::cut_value(edges, res.best_bitstring)
              << "  (optimum " << exact.first << ")\n"
              << "bitstring = ";
    for (std::size_t q = 0; q < n; ++q) std::cout << ((res.best_bitstring >> q) & 1ULL);   // character q = qubit q
    std::cout << "\n";
}
