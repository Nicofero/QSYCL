// Benchmark of the three VQA execution strategies, over problem sizes.
//
//   native      : whole loop on the SYCL device (prefer_native = true)
//   host+batch  : optimizer on the host, but every iteration's circuits are
//                 evaluated by the backend's batched kernels (prefer_native = false)
//   sequential  : the generic fallback any user-defined backend gets for free:
//                 one circuit at a time through initialize()/apply_gate() only
//
// IMPORTANT when reading the numbers: "native" and "host+batch" run the SAME
// batched circuit kernels. They differ only in the optimizer update (Adam on
// O(num_params) numbers) and a few tiny copies per iteration, which is nothing
// next to the circuit work, so native/host+batch ~ 1.0 is the EXPECTED result.
// The large gain comes from batching the circuits (host+batch vs sequential).
//
// Usage: ./bench_native [device] [optimizer_device] [min_qubits] [max_qubits] [iterations] [reps]
//        device / optimizer_device: cpu | gpu   (defaults: cpu cpu 4 14 20 2)
//        reps = ansatz depth (RealAmplitudes reps); more reps -> more parameters.
// The sequential run is slow, so it does fewer iterations (time is reported per iteration).
#include "quantum/algorithms/ansatz.hpp"
#include "quantum/algorithms/hamiltonians.hpp"
#include "quantum/backend_factory.hpp"
#include "quantum/vqa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

using namespace quantum;
using Clock = std::chrono::steady_clock;

// Forwards ONLY the original pure virtuals to a real backend. It does not override
// expectation_batch()/run_native_vqa(), so VQA gets the generic sequential path.
class SequentialBackend : public Backend {
public:
    explicit SequentialBackend(std::unique_ptr<Backend> inner) : inner_(std::move(inner)) {}
    void initialize(std::size_t n) override { inner_->initialize(n); }
    void apply_gate(const GateOp& op) override { inner_->apply_gate(op); }
    std::vector<Complex> get_state() const override { return inner_->get_state(); }
    std::vector<double> probabilities() const override { return inner_->probabilities(); }
    std::vector<unsigned long long> sample(const std::vector<std::size_t>& q, std::size_t shots) override {
        return inner_->sample(q, shots);
    }
    std::string device_name() const override { return inner_->device_name() + " (sequential)"; }
private:
    std::unique_ptr<Backend> inner_;
};

static DeviceType parse_device(const char* s) {
    if (!std::strcmp(s, "gpu")) return DeviceType::GPU;
    if (!std::strcmp(s, "cpu")) return DeviceType::CPU;
    if (!std::strcmp(s, "cunqa")) return DeviceType::CUNQA;
    throw std::invalid_argument(std::string("unknown device '") + s + "' (use cpu or gpu)");
}

// Returns milliseconds per iteration.
static double run(QuantumRuntime& rt, const Circuit& ansatz, const Observable& H, VQAOptions opt,
                  const std::vector<double>& theta0, bool native, VQAResult* out = nullptr) {
    opt.prefer_native = native;
    VQA vqa(rt, ansatz, H);
    const auto t0 = Clock::now();
    VQAResult res = vqa.minimize(theta0, opt);
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    const double ms_per_it = 1e3 * s / static_cast<double>(std::max<std::size_t>(res.iterations, 1));
    if (out) *out = std::move(res);
    return ms_per_it;
}

int main(int argc, char** argv) {
    try {
        const DeviceType dev = parse_device(argc > 1 ? argv[1] : "cpu");
        const DeviceType optimizer_dev = parse_device(argc > 2 ? argv[2] : "cpu");
        const std::size_t min_q = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 4;
        const std::size_t max_q = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 14;
        const std::size_t iters = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 20;
        const std::size_t reps = argc > 6 ? std::strtoul(argv[6], nullptr, 10) : 2;
        const std::size_t seq_iters = std::max<std::size_t>(2, iters / 5);

        QuantumRuntime rt(dev);
        QuantumRuntime rt_seq(std::make_unique<SequentialBackend>(BackendFactory::create(dev)));
        std::printf("circuits: %s | optimizer device: %s | %zu iterations (sequential: %zu) | RealAmplitudes reps=%zu\n",
                    rt.device_name().c_str(), optimizer_dev == DeviceType::GPU ? "GPU" : "CPU", iters, seq_iters, reps);
        std::printf("%6s %7s %9s | %10s %10s %10s | %13s %15s | %s\n", "qubits", "params", "evals/it",
                    "native", "host+batch", "sequential", "native/host", "batch/sequential", "check");
        std::printf("%6s %7s %9s | %10s %10s %10s |\n", "", "", "", "ms/it", "ms/it", "ms/it");

        for (std::size_t n = min_q; n <= max_q; n += (n < 8 ? 2 : 1)) {
            const Observable H = hamiltonians::ising_chain(n, 1.0, 0.8, false);
            const Circuit ansatz = ansatz::real_amplitudes(n, reps, ansatz::Entanglement::Linear);
            const std::vector<double> theta0 = ansatz::random_parameters(ansatz.num_params(), 7, -0.5, 0.5);

            VQAOptions opt;
            opt.optimizer = OptimizerKind::Adam;
            opt.gradient = GradientMethod::ParameterShift;
            opt.learning_rate = 0.05;
            opt.max_iterations = iters;
            opt.tolerance = -1.0;                 // never stop early: every run does exactly max_iterations
            opt.optimizer_device = optimizer_dev;

            // Warm-up: the first SYCL run pays for kernel JIT/compilation and allocations.
            VQAOptions warm = opt; warm.max_iterations = 2;
            run(rt, ansatz, H, warm, theta0, true);
            run(rt, ansatz, H, warm, theta0, false);

            VQAResult nat, host, seq;
            const double nat_ms = run(rt, ansatz, H, opt, theta0, true, &nat);
            const double host_ms = run(rt, ansatz, H, opt, theta0, false, &host);
            VQAOptions sopt = opt; sopt.max_iterations = seq_iters;
            const double seq_ms = run(rt_seq, ansatz, H, sopt, theta0, false, &seq);

            // Same start, same maths -> native and host must agree (and the sequential run must
            // have reproduced the host run's first seq_iters energies).
            const double dE = std::fabs(nat.energy - host.energy);
            double dseq = 0.0;
            for (std::size_t k = 0; k < seq.history.size() && k < host.history.size(); ++k)
                dseq = std::max(dseq, std::fabs(seq.history[k] - host.history[k]));
            const double evals_per_it = static_cast<double>(host.evaluations) / static_cast<double>(std::max<std::size_t>(host.iterations, 1));
            std::printf("%6zu %7zu %9.1f | %10.3f %10.3f %10.3f | %12.2fx %14.2fx | %s dE=%.0e seq=%.0e\n", n,
                        ansatz.num_params(), evals_per_it, nat_ms, host_ms, seq_ms, host_ms / nat_ms,
                        seq_ms / host_ms, nat.used_native ? "native" : "NOT-native", dE, dseq);
        }
        std::puts("\nnative/host     : > 1 means the on-device optimizer loop is faster than the host loop (expect ~1:\n"
                  "                  both use the same batched circuit kernels; only an O(params) update differs).\n"
                  "batch/sequential: > 1 means batching the circuits helps (this is where the real gain is).\n"
                  "check           : dE = |native - host| final energy, seq = max energy gap vs host over the sequential\n"
                  "                  run's iterations; both should be ~1e-10 or smaller.");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}