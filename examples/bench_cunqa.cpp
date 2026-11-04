// Benchmark: SYCL-native VQA loop vs the host-driven loop, over problem sizes.
//
// For each qubit count it runs the SAME optimization (same ansatz, Hamiltonian,
// start point, fixed iteration count, early stopping disabled) twice on the same
// backend:
//     native = whole loop on the SYCL device (prefer_native = true)
//     host   = optimizer on the host, circuits batched (prefer_native = false)
// and prints time per iteration plus the speed-up. The crossover (speed-up > 1)
// tells you where native starts to pay off on YOUR hardware.
//
// Usage: ./bench_native [circuit_device] [optimizer_device] [min_qubits] [max_qubits] [iterations] [reps]
//        devices: cpu | gpu   (defaults: cpu cpu 4 16 20 2)
//        reps = ansatz depth (RealAmplitudes reps); more reps -> more parameters.
#include "quantum/algorithms/ansatz.hpp"
#include "quantum/algorithms/hamiltonians.hpp"
#include "quantum/vqa.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace quantum;
using Clock = std::chrono::steady_clock;

static DeviceType parse_device(const char* s) {
    if (!std::strcmp(s, "gpu")) return DeviceType::GPU;
    if (!std::strcmp(s, "cpu")) return DeviceType::CPU;
    if (!std::strcmp(s, "cunqa")) return DeviceType::CUNQA;
    throw std::invalid_argument(std::string("unknown device '") + s + "' (use cpu or gpu)");
}

struct Timed { double seconds; VQAResult res; };

static Timed run(QuantumRuntime& rt, const Circuit& ansatz, const Observable& H, VQAOptions opt,
                 const std::vector<double>& theta0, bool native) {
    opt.prefer_native = native;
    VQA vqa(rt, ansatz, H);
    const auto t0 = Clock::now();
    VQAResult res = vqa.minimize(theta0, opt);
    const auto t1 = Clock::now();
    return {std::chrono::duration<double>(t1 - t0).count(), std::move(res)};
}

int main(int argc, char** argv) {
    try {
        const DeviceType circuit_dev = parse_device(argc > 1 ? argv[1] : "cpu");
        const DeviceType optimizer_dev = parse_device(argc > 2 ? argv[2] : "cpu");
        const std::size_t min_q = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 4;
        const std::size_t max_q = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 16;
        const std::size_t iters = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 20;
        const std::size_t reps = argc > 6 ? std::strtoul(argv[6], nullptr, 10) : 2;

        QuantumRuntime rt(circuit_dev);
        std::printf("circuits: %s | optimizer device: %s | %zu iterations | RealAmplitudes reps=%zu\n",
                    rt.device_name().c_str(), optimizer_dev == DeviceType::GPU ? "GPU" : "CPU", iters, reps);
        std::printf("%6s %7s %9s | %12s %12s | %8s | %s\n", "qubits", "params", "evals/it", "native ms/it",
                    "host ms/it", "speedup", "native used / energy diff");

        for (std::size_t n = min_q; n <= max_q; n += (n < 8 ? 2 : 1)) {
            const Observable H = hamiltonians::ising_chain(n, 1.0, 0.8, false);
            const Circuit ansatz = ansatz::real_amplitudes(n, reps, ansatz::Entanglement::Linear);
            const std::vector<double> theta0 = ansatz::random_parameters(ansatz.num_params(), 7, -0.5, 0.5);

            VQAOptions opt;
            opt.optimizer = OptimizerKind::Adam;
            opt.gradient = GradientMethod::ParameterShift;
            opt.learning_rate = 0.05;
            opt.max_iterations = iters;
            opt.tolerance = -1.0;                 // never stop early: both runs do exactly `iters` iterations
            opt.optimizer_device = optimizer_dev;

            // Warm-up: the first SYCL run pays for kernel JIT/compilation and allocations.
            VQAOptions warm = opt; warm.max_iterations = 2;
            run(rt, ansatz, H, warm, theta0, true);
            run(rt, ansatz, H, warm, theta0, false);

            const Timed nat = run(rt, ansatz, H, opt, theta0, true);
            const Timed host = run(rt, ansatz, H, opt, theta0, false);

            const double nat_ms = 1e3 * nat.seconds / static_cast<double>(nat.res.iterations);
            const double host_ms = 1e3 * host.seconds / static_cast<double>(host.res.iterations);
            const double evals_per_it = static_cast<double>(host.res.evaluations) / static_cast<double>(host.res.iterations);
            std::printf("%6zu %7zu %9.1f | %12.3f %12.3f | %7.2fx | %s, dE=%.1e\n", n, ansatz.num_params(),
                        evals_per_it, nat_ms, host_ms, host_ms / nat_ms,
                        nat.res.used_native ? "native" : "NOT native (backend has no native path)",
                        std::fabs(nat.res.energy - host.res.energy));
        }
        std::puts("\nspeedup > 1: native is faster; < 1: the host loop wins (launch/transfer overhead dominates).");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}