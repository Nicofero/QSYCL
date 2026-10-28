#include "quantum/vqa.hpp"
#include <cmath>
#include <iostream>
#include <chrono>

// Minimal VQE: find the ground state of H = Z0 Z1 + X0 X1 (exact energy: -2).
// The same code runs on any backend -- only the DeviceType changes.
int main() {
    using namespace quantum;

    // 1. Parameterized ansatz. Params are handles, not numbers.
    // Circuit ansatz(5);
    int nparams = 2;
    // auto th = ansatz.new_params(30);
    // ansatz.ry(0, th[0]).rz(0, th[1])
    // .ry(1, th[2]).rz(1, th[3])
    // .ry(2, th[4]).rz(2, th[5])
    // .ry(3, th[6]).rz(3, th[7])
    // .ry(4, th[8]).rz(4, th[9])

    // .cnot(0, 1).cnot(1, 2).cnot(2, 3).cnot(3, 4)

    // .ry(0, th[10]).rz(0, th[11])
    // .ry(1, th[12]).rz(1, th[13])
    // .ry(2, th[14]).rz(2, th[15])
    // .ry(3, th[16]).rz(3, th[17])
    // .ry(4, th[18]).rz(4, th[19])

    // .cnot(0, 1).cnot(1, 2).cnot(2, 3).cnot(3, 4)

    // .ry(0, th[20]).rz(0, th[21])
    // .ry(1, th[22]).rz(1, th[23])
    // .ry(2, th[24]).rz(2, th[25])
    // .ry(3, th[26]).rz(3, th[27])
    // .ry(4, th[28]).rz(4, th[29]);
    Circuit ansatz(2);
    auto th = ansatz.new_params(nparams);
    ansatz.ry(0, th[0]).cnot(0, 1).x(1).ry(1, th[1]);
    Observable H(2);
    H.add_term(1.0, "ZZ")
     .add_term(1.0, "XX");

    // 2. Observable as a sum of Pauli strings (character k acts on qubit k).
    // Observable H(5);
    // H.add_term(1.0, "ZZIII")
    //  .add_term(1.0, "IZZII")
    //  .add_term(1.0, "IIZZI")
    //  .add_term(1.0, "IIIZZ")
    //  .add_term(1.0, "XIIII")
    //  .add_term(1.0, "IXIII")
    //  .add_term(1.0, "IIXII")
    //  .add_term(1.0, "IIIXI")
    //  .add_term(1.0, "IIIIX");

    // 3. Pick a device and let VQA choose the best execution strategy the
    //    backend supports (native -> batched -> sequential).
    QuantumRuntime runtime(DeviceType::CUNQA);
    std::cout << "Running on: " << runtime.device_name() << "\n";

    VQA vqa(runtime, ansatz, H);
    VQAOptions opt;
    opt.optimizer = OptimizerKind::Adam;
    opt.gradient = GradientMethod::ParameterShift;
    opt.learning_rate = 0.1;
    opt.max_iterations = 500;
    // The optimizer's parameter-update kernels run on THIS device,
    // independent of the DeviceType circuits execute on above (only
    // honoured natively by CPUBackend/GPUBackend; e.g. set this to
    // DeviceType::GPU while `runtime` above uses DeviceType::CPU circuits,
    // or leave it CPU while circuits run on a GPU, as here).
    opt.optimizer_device = DeviceType::CPU;

    opt.prefer_native = true; // force host-driven loop

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> th0(nparams, 0.4);

    VQAResult res = vqa.minimize(th0, opt);

    auto end = std::chrono::high_resolution_clock::now();
     auto duration = std::chrono::duration_cast<
        std::chrono::milliseconds
    >(end - start);
    std::cout << "VQE completed in " << duration.count() << " milliseconds\n";

    std::cout << "E = " << res.energy << "  (exact -2)\n"
              << "iterations = " << res.iterations
              << ", circuit evaluations = " << res.evaluations
              << ", native = " << std::boolalpha << res.used_native << "\n"
              << "theta = [" << res.theta[0] << ", " << res.theta[1] << "]\n";
    return 0;
}
