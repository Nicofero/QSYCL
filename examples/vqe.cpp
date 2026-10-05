#include "quantum/vqa.hpp"
#include <cmath>
#include <iostream>

// Minimal VQE: find the ground state of H = Z0 Z1 + X0 X1 (exact energy: -2).
// The same code runs on any backend -- only the DeviceType changes.
int main() {
    using namespace quantum;

    // 1. Parameterized ansatz. Params are handles, not numbers.
    Circuit ansatz(2);
    auto th = ansatz.new_params(2);
    ansatz.ry(0, th[0]).cnot(0, 1).x(1).ry(1, th[1]);

    // 2. Observable as a sum of Pauli strings (character k acts on qubit k).
    Observable H(2);
    H.add_term(1.0, "ZZ").add_term(1.0, "XX");

    // 3. Pick a device and let VQA choose the best execution strategy the
    //    backend supports (native -> batched -> sequential).
    QuantumRuntime runtime(DeviceType::CUNQA);
    std::cout << "Running on: " << runtime.device_name() << "\n";

    VQA vqa(runtime, ansatz, H);
    VQAOptions opt;
    opt.optimizer = OptimizerKind::Adam;
    opt.gradient = GradientMethod::ParameterShift;
    opt.learning_rate = 0.1;
    opt.max_iterations = 200;
    // The optimizer's parameter-update kernels run on THIS device,
    // independent of the DeviceType circuits execute on above (only
    // honoured natively by CPUBackend/GPUBackend; e.g. set this to
    // DeviceType::GPU while `runtime` above uses DeviceType::CPU circuits,
    // or leave it CPU while circuits run on a GPU, as here).
    opt.optimizer_device = DeviceType::CPU;

    VQAResult res = vqa.minimize({0.4, 0.4}, opt);

    std::cout << "E = " << res.energy << "  (exact -2)\n"
              << "iterations = " << res.iterations
              << ", circuit evaluations = " << res.evaluations
              << ", native = " << std::boolalpha << res.used_native << "\n"
              << "theta = [" << res.theta[0] << ", " << res.theta[1] << "]\n";
    return 0;
}
