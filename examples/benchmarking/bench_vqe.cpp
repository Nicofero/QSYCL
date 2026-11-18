// General VQE benchmark: every execution method on every backend.
//
// For each (circuit device x optimizer device x [CUNQA vQPU count] x qubit
// count x method) it runs the SAME VQE optimization -- same Hamiltonian,
// ansatz, start point, fixed iteration count, early stopping disabled -- and
// reports the time per iteration. Results go to stdout as a table and,
// optionally, to a CSV file for plotting.
//
// Execution methods
//   native      VQA::minimize with prefer_native = true. The optimizer's
//               update kernels run on --optimizers' SYCL device; circuits run
//               wherever the backend runs them (batched SYCL kernels on
//               CPU/GPU, batched across all vQPUs on CUNQA, shots on QMIO).
//   host        prefer_native = false: optimizer on the host, but every
//               iteration's circuits still go through the backend's batched
//               expectation_batch().
//   sequential  The generic path any backend gets for free: one circuit at a
//               time through initialize()/apply_gate() and then either an exact
//               expectation (CPU/GPU) or a sample() call per measurement group
//               (CUNQA/QMIO). No batching at all.
//
// CUNQA vQPU scaling
//   For --devices cunqa, every value in --vqpus is run as a separate
//   configuration: the backend is rebuilt with CUNQA_MAX_VQPUS=<k>, so it only
//   uses the first k vQPUs of the registry ($STORE/.cunqa/qpus.json). 0 means
//   "every registered vQPU". Raise at least max(--vqpus) vQPUs beforehand.
//
// Usage (all flags optional):
//   ./bench_vqe --devices cpu,gpu,cunqa,qmio   circuit devices to benchmark
//               --optimizers cpu,gpu           optimizer devices (native method only uses it)
//               --methods native,host,sequential
//               --min-qubits 4 --max-qubits 12 --step 2
//               --iters 10                     iterations per run (native/host)
//               --seq-iters 2                  iterations per sequential run (it is slow)
//               --reps 2                       ansatz depth (RealAmplitudes reps)
//               --hamiltonian ising|heisenberg (heisenberg: 3 measurement groups instead of 2)
//               --shots 1024                   shots per circuit (shot-based backends only)
//               --vqpus 1,2,4,8                CUNQA vQPU counts to sweep (0 = all registered)
//               --warmup 1                     warm-up iterations before every timed run (0 = none)
//               --csv results.csv              also write every row as CSV (appends; header once)
#include "quantum/algorithms/ansatz.hpp"
#include "quantum/algorithms/hamiltonians.hpp"
#include "quantum/backend_factory.hpp"
#include "quantum/shot_expectation.hpp"
#include "quantum/vqa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace quantum;
using Clock = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------- CLI ----

struct Config {
    std::vector<std::string> devices = {"cpu"};
    std::vector<std::string> optimizers = {"cpu"};
    std::vector<std::string> methods = {"native", "host", "sequential"};
    std::size_t min_q = 4, max_q = 12, step = 2;
    std::size_t iters = 10, seq_iters = 2, reps = 2;
    std::string hamiltonian = "ising";
    std::size_t shots = 1024;
    std::vector<std::size_t> vqpus = {0};
    std::size_t warmup = 1;
    std::string csv;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
}

Config parse_args(int argc, char** argv) {
    Config c;
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "-h" || key == "--help") {
            std::puts("see the header comment of examples/benchmarking/bench_vqe.cpp for every flag");
            std::exit(0);
        }
        if (key.rfind("--", 0) != 0 || i + 1 >= argc)
            throw std::invalid_argument("expected '--flag value', got '" + key + "'");
        kv[key.substr(2)] = argv[++i];
    }
    auto num = [&](const char* k, std::size_t& dst) {
        if (kv.count(k)) dst = std::stoul(kv.at(k));
    };
    if (kv.count("devices")) c.devices = split(kv.at("devices"));
    if (kv.count("optimizers")) c.optimizers = split(kv.at("optimizers"));
    if (kv.count("methods")) c.methods = split(kv.at("methods"));
    if (kv.count("hamiltonian")) c.hamiltonian = kv.at("hamiltonian");
    if (kv.count("csv")) c.csv = kv.at("csv");
    if (kv.count("vqpus")) {
        c.vqpus.clear();
        for (auto& v : split(kv.at("vqpus"))) c.vqpus.push_back(std::stoul(v));
    }
    num("min-qubits", c.min_q); num("max-qubits", c.max_q); num("step", c.step);
    num("iters", c.iters); num("seq-iters", c.seq_iters); num("reps", c.reps);
    num("shots", c.shots); num("warmup", c.warmup);
    if (c.step == 0) c.step = 1;
    return c;
}

DeviceType parse_device(const std::string& s) {
    if (s == "cpu") return DeviceType::CPU;
    if (s == "gpu") return DeviceType::GPU;
    if (s == "cunqa") return DeviceType::CUNQA;
    if (s == "qmio") return DeviceType::QMIO;
    throw std::invalid_argument("unknown device '" + s + "' (use cpu, gpu, cunqa or qmio)");
}

bool is_shot_based(DeviceType d) { return d == DeviceType::CUNQA || d == DeviceType::QMIO; }

// ----------------------------------------------------- sequential path ----

// Forwards only Backend's original pure virtuals to a real backend, so VQA
// gets no batching and no native loop: one circuit at a time. For exact
// backends this is Backend's default expectation_batch (state read back per
// circuit); for shot-based ones it is one sample() per measurement group
// (get_state() is unavailable there, so the default would throw).
class SequentialBackend : public Backend {
public:
    SequentialBackend(std::unique_ptr<Backend> inner, bool shot_based)
        : inner_(std::move(inner)), shot_based_(shot_based) {}

    void initialize(std::size_t n) override { inner_->initialize(n); }
    void apply_gate(const GateOp& op) override { inner_->apply_gate(op); }
    std::vector<Complex> get_state() const override { return inner_->get_state(); }
    std::vector<double> probabilities() const override { return inner_->probabilities(); }
    std::vector<unsigned long long> sample(const std::vector<std::size_t>& q, std::size_t shots) override {
        return inner_->sample(q, shots);
    }
    std::string device_name() const override { return inner_->device_name() + " (sequential)"; }

    std::vector<double> expectation_batch(const Circuit& ansatz, const Observable& obs,
                                          const EvalBatch& batch) override {
        if (shot_based_)
            return native::shot_based_expectation_batch(*this, ansatz, obs, batch, batch.shots);
        return Backend::expectation_batch(ansatz, obs, batch);
    }

private:
    std::unique_ptr<Backend> inner_;
    bool shot_based_;
};

// ------------------------------------------------------------ helpers ----

// "CUNQA (3 vQPU(s): ...)" -> 3; a fallback/other name -> 0.
std::size_t vqpus_in_use(const std::string& device_name) {
    std::size_t n = 0;
    if (std::sscanf(device_name.c_str(), "CUNQA (%zu vQPU", &n) == 1) return n;
    return 0;
}

std::size_t measurement_groups(const Observable& H) { return native::group_qwc(H).size(); }

struct RunResult {
    double ms_per_iter = 0.0;
    double total_s = 0.0;
    VQAResult res;
};

RunResult run_vqe(QuantumRuntime& rt, const Circuit& ansatz, const Observable& H, VQAOptions opt,
                  const std::vector<double>& theta0, bool native) {
    opt.prefer_native = native;
    VQA vqa(rt, ansatz, H);
    const auto t0 = Clock::now();
    VQAResult res = vqa.minimize(theta0, opt);
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    RunResult r;
    r.total_s = s;
    r.ms_per_iter = 1e3 * s / static_cast<double>(std::max<std::size_t>(res.iterations, 1));
    r.res = std::move(res);
    return r;
}

class CsvSink {
public:
    explicit CsvSink(const std::string& path) {
        if (path.empty()) return;
        std::ifstream probe(path);
        const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
        out_.open(path, std::ios::app);
        if (!out_) throw std::runtime_error("cannot open CSV file '" + path + "'");
        if (fresh)
            out_ << "device,optimizer_device,method,vqpus,qubits,params,hamiltonian,groups,"
                    "circuits_per_iter,shots,iterations,evaluations,ms_per_iter,total_s,"
                    "final_energy,used_native,backend_name\n";
    }
    template <typename... Ts>
    void row(const Ts&... fields) {
        if (!out_.is_open()) return;
        bool first = true;
        ((out_ << (first ? "" : ",") << fields, first = false), ...);
        out_ << "\n";
        out_.flush(); // keep partial results if the SLURM job is killed
    }

private:
    std::ofstream out_;
};

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        CsvSink csv(cfg.csv);

        std::printf("VQE benchmark | hamiltonian=%s | RealAmplitudes reps=%zu | iters=%zu (sequential %zu) | "
                    "shots=%zu | warm-up=%zu\n\n",
                    cfg.hamiltonian.c_str(), cfg.reps, cfg.iters, cfg.seq_iters, cfg.shots, cfg.warmup);
        std::printf("%-6s %-4s %-10s %5s %6s %6s %8s | %12s %10s | %14s %s\n", "device", "opt", "method",
                    "vqpus", "qubits", "params", "circ/it", "ms/iter", "total s", "final energy", "notes");

        for (const auto& dev_name : cfg.devices) {
            const DeviceType dev = parse_device(dev_name);
            // vQPU-count sweep only means something for CUNQA.
            const std::vector<std::size_t> vqpu_sweep =
                dev == DeviceType::CUNQA ? cfg.vqpus : std::vector<std::size_t>{0};

            for (const auto& opt_name : cfg.optimizers) {
                const DeviceType opt_dev = parse_device(opt_name);

                for (std::size_t k : vqpu_sweep) {
                    if (dev == DeviceType::CUNQA) {
                        // Read by CUNQABackend::discover_qpus() at construction.
                        setenv("CUNQA_MAX_VQPUS", std::to_string(k).c_str(), 1);
                    }

                    std::unique_ptr<QuantumRuntime> rt, rt_seq;
                    try {
                        rt = std::make_unique<QuantumRuntime>(dev);
                        rt_seq = std::make_unique<QuantumRuntime>(
                            std::make_unique<SequentialBackend>(BackendFactory::create(dev), is_shot_based(dev)));
                    } catch (const std::exception& e) {
                        std::printf("%-6s skipped: %s\n", dev_name.c_str(), e.what());
                        break; // backend not built / not available: skip the whole device
                    }
                    const std::string backend_name = rt->device_name();
                    const std::size_t k_used = dev == DeviceType::CUNQA ? vqpus_in_use(backend_name) : 0;
                    if (dev == DeviceType::CUNQA && k_used == 0) {
                        std::printf("%-6s WARNING: no vQPU found, CUNQA is running on its CPU fallback (%s)\n",
                                    dev_name.c_str(), backend_name.c_str());
                    }
                    if (dev == DeviceType::CUNQA && k != 0 && k_used != 0 && k_used < k) {
                        std::printf("%-6s WARNING: asked for %zu vQPUs but only %zu are registered\n",
                                    dev_name.c_str(), k, k_used);
                    }

                    for (std::size_t n = cfg.min_q; n <= cfg.max_q; n += cfg.step) {
                        const Observable H = cfg.hamiltonian == "heisenberg"
                                                 ? hamiltonians::heisenberg_chain(n, 1.0, 1.0, false)
                                                 : hamiltonians::ising_chain(n, 1.0, 0.8, false);
                        const Circuit ansatz = ansatz::real_amplitudes(n, cfg.reps, ansatz::Entanglement::Linear);
                        const std::vector<double> theta0 =
                            ansatz::random_parameters(ansatz.num_params(), 7, -0.5, 0.5);
                        const std::size_t groups = measurement_groups(H);

                        VQAOptions opt;
                        opt.optimizer = OptimizerKind::Adam;
                        opt.gradient = GradientMethod::ParameterShift;
                        opt.learning_rate = 0.05;
                        opt.tolerance = -1.0;   // never stop early: every run does exactly max_iterations
                        opt.optimizer_device = opt_dev;
                        opt.shots = cfg.shots;

                        for (const auto& method : cfg.methods) {
                            const bool sequential = method == "sequential";
                            const bool native = method == "native";
                            if (!sequential && !native && method != "host")
                                throw std::invalid_argument("unknown method '" + method +
                                                            "' (use native, host or sequential)");
                            // The optimizer device only matters for the native loop: the host
                            // and sequential methods update theta on the host. Run them once,
                            // under the first optimizer, instead of repeating identical work.
                            if (!native && opt_name != cfg.optimizers.front()) continue;

                            QuantumRuntime& runtime = sequential ? *rt_seq : *rt;
                            VQAOptions o = opt;
                            o.max_iterations = sequential ? cfg.seq_iters : cfg.iters;

                            RunResult r;
                            try {
                                if (cfg.warmup > 0) {
                                    VQAOptions w = o;
                                    w.max_iterations = cfg.warmup;
                                    run_vqe(runtime, ansatz, H, w, theta0, native);
                                }
                                r = run_vqe(runtime, ansatz, H, o, theta0, native);
                            } catch (const std::exception& e) {
                                std::printf("%-6s %-4s %-10s %5zu %6zu  FAILED: %s\n", dev_name.c_str(),
                                            opt_name.c_str(), method.c_str(), k_used, n, e.what());
                                continue;
                            }

                            const double circ_per_it =
                                static_cast<double>(r.res.evaluations) /
                                static_cast<double>(std::max<std::size_t>(r.res.iterations, 1)) *
                                static_cast<double>(is_shot_based(dev) ? groups : 1);
                            const std::string notes =
                                native && !r.res.used_native ? "native requested but backend ran host loop" : "";
                            std::printf("%-6s %-4s %-10s %5zu %6zu %6zu %8.0f | %12.3f %10.3f | %14.6f %s\n",
                                        dev_name.c_str(), native ? opt_name.c_str() : "host", method.c_str(),
                                        k_used, n, ansatz.num_params(), circ_per_it, r.ms_per_iter, r.total_s,
                                        r.res.energy, notes.c_str());
                            std::fflush(stdout);
                            csv.row(dev_name, native ? opt_name : std::string("host"), method, k_used, n,
                                    ansatz.num_params(), cfg.hamiltonian, groups, circ_per_it,
                                    is_shot_based(dev) ? cfg.shots : 0, r.res.iterations, r.res.evaluations,
                                    r.ms_per_iter, r.total_s, r.res.energy, r.res.used_native ? 1 : 0,
                                    quoted(backend_name));
                        }
                    }
                }
            }
        }
        std::puts("\ncirc/it: circuits submitted per iteration (shot-based backends: one per evaluation and "
                  "measurement group).\nvqpus  : vQPUs CUNQA actually used (0 for every other device).");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
