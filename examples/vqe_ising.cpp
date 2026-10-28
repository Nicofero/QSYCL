// VQE for the transverse-field Ising chain with the ansatz library.
//   H = -J sum Z_i Z_{i+1} - h sum X_i      (10 qubits, open chain)
// Swap the DeviceType to run on the GPU or on CUNQA/QPU -- nothing else changes.
#include "quantum/algorithms/algorithms.hpp"
#include "quantum/algorithms/hamiltonians.hpp"
#include <iostream>

int main() {
    using namespace quantum;

    const std::size_t n = 10;
    Observable H = hamiltonians::ising_chain(n, /*J=*/1.0, /*h=*/0.8, /*periodic=*/false);
    Circuit ansatz = ansatz::real_amplitudes(n, /*reps=*/3, ansatz::Entanglement::Linear);

    QuantumRuntime runtime(DeviceType::CUNQA);
    std::cout << "Running on: " << runtime.device_name() << ", " << ansatz.num_params() << " parameters\n";

    VQEOptions opt;
    opt.restarts = 1;                 // random restarts; the best energy wins
    opt.init_range = 0.5;             // start near |0...0|: avoids barren-plateau-ish random starts
    opt.vqa.optimizer = OptimizerKind::Adam;
    opt.vqa.learning_rate = 0.05;
    opt.vqa.max_iterations = 300;
    opt.vqa.optimizer_device = DeviceType::CPU;   // may differ from the circuit device

    const VQEResult res = vqe(runtime, ansatz, H, opt);
    std::cout << "ground-state energy estimate: " << res.best.energy << "\n";
    for (std::size_t r = 0; r < res.restart_energies.size(); ++r)
        std::cout << "  restart " << r << ": " << res.restart_energies[r] << (r == res.best_restart ? "  <- best" : "") << "\n";
    std::cout << "total circuit evaluations: " << res.total_evaluations << "\n";
}
