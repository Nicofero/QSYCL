// Circuit benchmark without any VQE: how fast is ONE circuit on each backend, how
// far can each backend go, and how many independent circuits per second can it
// sustain? Two modes (--modes latency,throughput):
//
// LATENCY (default) -- one circuit at a time.
// Nothing is batched: every measurement is one circuit run
// through the plain QuantumRuntime path -- initialize(), apply_gate() for each
// gate, then sample() -- exactly what a user calling runtime.sample(circuit)
// gets. For each device the qubit count grows until one of the limits is hit:
//
//   time_limit    the median time of one circuit exceeded --time-limit seconds
//   memory_limit  an exact simulator would need more than --max-memory-gb for
//                 its state vector (checked BEFORE running, so the job is not
//                 OOM-killed)
//   failed        the backend threw (out of memory, vQPU too small, ...)
//
// Timings per repetition (ms):
//   exec    runtime.run(): gate application. CPU/GPU: the real simulation.
//           CUNQA/QMIO: only buffers the gates locally (~0).
//   sample  backend.sample(shots). CPU/GPU: drawing shots from the state.
//           CUNQA/QMIO: the whole remote round trip (serialize, send, execute,
//           receive, parse) -- i.e. the real cost of the circuit.
//   total   exec + sample.
// A single circuit only ever runs on one vQPU, so latency is measured once per
// device, with every registered vQPU available.
//
// THROUGHPUT -- B independent circuits submitted together.
// The same circuit family, but B copies (real_amplitudes: B different random
// parameter sets) handed to the backend in ONE Backend::expectation_batch()
// call with a Z...Z observable, so every copy is exactly one circuit and
// nothing VQE-specific (gradients, optimizer) is involved. This is the path
// that spreads circuits over vQPUs: on CUNQA every value of --vqpus is run as a
// separate configuration (CUNQA_MAX_VQPUS=<k>, 0 = all registered). On CPU/GPU
// it is the batched [B][2^n] state-vector kernel. Reported as circuits/second.
// For one (device, circuit, qubits) the batch sizes grow until --time-limit or
// the memory guard (B state vectors on CPU/GPU) is hit.
//
// Usage (all flags optional):
//   ./bench_circuit --devices cpu,gpu,cunqa,qmio
//                   --circuits real_amplitudes,ghz,qft
//                   --min-qubits 2 --max-qubits 34 --step 1
//                   --reps 2               RealAmplitudes depth (real_amplitudes only)
//                   --shots 1024
//                   --repeats 5            timed repetitions per point (median reported)
//                   --warmup 1             untimed repetitions before them
//                   --time-limit 120       seconds per circuit before the sweep stops
//                   --max-memory-gb 0      exact-simulator memory guard (0 = no guard)
//                   --csv circuit.csv      latency rows as CSV (appends; header once)
//                   --modes latency        latency, throughput, or both (comma-separated)
//                   --batch-sizes 1,8,32,128    throughput: circuits per submission
//                   --throughput-qubits 4,8,12  throughput: qubit counts to test
//                   --vqpus 1,2,4,8        throughput on CUNQA: vQPU counts (0 = all registered)
//                   --throughput-csv circuit_throughput.csv   throughput rows as CSV
#include "quantum/algorithms/ansatz.hpp"
#include "quantum/observable.hpp"
#include "quantum/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace quantum;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Config {
    std::vector<std::string> devices = {"cpu"};
    std::vector<std::string> circuits = {"real_amplitudes", "ghz", "qft"};
    std::size_t min_q = 2, max_q = 34, step = 1, reps = 2;
    std::size_t shots = 1024, repeats = 5, warmup = 1;
    double time_limit_s = 120.0;
    double max_memory_gb = 0.0;
    std::string csv;
    std::vector<std::string> modes = {"latency"};
    std::vector<std::size_t> batch_sizes = {1, 8, 32, 128};
    std::vector<std::size_t> tp_qubits = {4, 8, 12};
    std::vector<std::size_t> vqpus = {0};
    std::string tp_csv;

    bool has_mode(const std::string& m) const {
        return std::find(modes.begin(), modes.end(), m) != modes.end();
    }
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
            std::puts("see the header comment of examples/benchmarking/bench_circuit.cpp for every flag");
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
    if (kv.count("circuits")) c.circuits = split(kv.at("circuits"));
    if (kv.count("csv")) c.csv = kv.at("csv");
    if (kv.count("throughput-csv")) c.tp_csv = kv.at("throughput-csv");
    if (kv.count("modes")) c.modes = split(kv.at("modes"));
    auto nums = [&](const char* k, std::vector<std::size_t>& dst) {
        if (!kv.count(k)) return;
        dst.clear();
        for (auto& v : split(kv.at(k))) dst.push_back(std::stoul(v));
    };
    nums("batch-sizes", c.batch_sizes);
    nums("throughput-qubits", c.tp_qubits);
    nums("vqpus", c.vqpus);
    for (auto& m : c.modes)
        if (m != "latency" && m != "throughput")
            throw std::invalid_argument("unknown mode '" + m + "' (use latency, throughput)");
    if (kv.count("time-limit")) c.time_limit_s = std::stod(kv.at("time-limit"));
    if (kv.count("max-memory-gb")) c.max_memory_gb = std::stod(kv.at("max-memory-gb"));
    num("min-qubits", c.min_q); num("max-qubits", c.max_q); num("step", c.step);
    num("reps", c.reps); num("shots", c.shots); num("repeats", c.repeats); num("warmup", c.warmup);
    if (c.step == 0) c.step = 1;
    if (c.repeats == 0) c.repeats = 1;
    return c;
}

DeviceType parse_device(const std::string& s) {
    if (s == "cpu") return DeviceType::CPU;
    if (s == "gpu") return DeviceType::GPU;
    if (s == "cunqa") return DeviceType::CUNQA;
    if (s == "qmio") return DeviceType::QMIO;
    throw std::invalid_argument("unknown device '" + s + "' (use cpu, gpu, cunqa or qmio)");
}

bool is_exact_simulator(DeviceType d) { return d == DeviceType::CPU || d == DeviceType::GPU; }

// ------------------------------------------------------------ circuits ----
// Only gates every backend can run (CUNQA has no generic U gate, so the QFT's
// controlled phase is built from CRZ + RZ: equal to CP up to a global phase,
// which changes nothing about the cost).

// real_amplitudes stays parametric here (throughput binds a different
// parameter set per copy); every other family has no parameters.
Circuit make_template(const std::string& kind, std::size_t n, std::size_t reps) {
    if (kind == "real_amplitudes")
        return ansatz::real_amplitudes(n, reps, ansatz::Entanglement::Linear);
    Circuit c(n);
    if (kind == "ghz") {
        c.h(0);
        for (std::size_t q = 0; q + 1 < n; ++q) c.cnot(q, q + 1);
        return c;
    }
    if (kind == "qft") {
        // Start from a non-trivial basis state so the QFT isn't acting on |0..0>.
        for (std::size_t q = 0; q < n; q += 2) c.x(q);
        for (std::size_t i = 0; i < n; ++i) {
            c.h(i);
            for (std::size_t j = i + 1; j < n; ++j) {
                const double theta = kPi / static_cast<double>(1ULL << std::min<std::size_t>(j - i, 62));
                c.rz(i, theta / 2).crz(i, j, theta);   // controlled phase(theta), up to global phase
            }
        }
        for (std::size_t q = 0; q < n / 2; ++q) c.swap(q, n - 1 - q);
        return c;
    }
    throw std::invalid_argument("unknown circuit '" + kind + "' (use real_amplitudes, ghz or qft)");
}

// A concrete, runnable circuit (real_amplitudes bound to fixed random angles:
// the VQE ansatz of bench_vqe).
Circuit make_circuit(const std::string& kind, std::size_t n, std::size_t reps) {
    const Circuit c = make_template(kind, n, reps);
    return c.num_params() ? c.bind(ansatz::random_parameters(c.num_params(), 7)) : c;
}

// --------------------------------------------------------------- stats ----

struct Stats {
    double median = 0, min = 0, mean = 0, stddev = 0;
};

Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size();
    s.median = m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
    s.min = v.front();
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(m);
    double acc = 0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.stddev = m > 1 ? std::sqrt(acc / static_cast<double>(m - 1)) : 0.0;
    return s;
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

class CsvSink {
public:
    CsvSink(const std::string& path, const char* header) {
        if (path.empty()) return;
        std::ifstream probe(path);
        const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
        out_.open(path, std::ios::app);
        if (!out_) throw std::runtime_error("cannot open CSV file '" + path + "'");
        if (fresh) out_ << header << "\n";
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

std::string quoted(std::string s) {
    std::replace(s.begin(), s.end(), '"', '\'');
    std::replace(s.begin(), s.end(), '\n', ' ');
    return "\"" + s + "\"";
}

std::unique_ptr<QuantumRuntime> make_runtime(const std::string& dev_name, DeviceType dev) {
    try {
        return std::make_unique<QuantumRuntime>(dev);
    } catch (const std::exception& e) {
        std::printf("%-6s skipped: %s\n", dev_name.c_str(), e.what());
        return nullptr;
    }
}

// "CUNQA (3 vQPU(s): ...)" -> 3; a fallback/other name -> 0.
std::size_t vqpus_in_use(const std::string& device_name) {
    std::size_t n = 0;
    if (std::sscanf(device_name.c_str(), "CUNQA (%zu vQPU", &n) == 1) return n;
    return 0;
}

// For CUNQA: "" if the runtime still has `expected` vQPUs, otherwise why not.
// A vQPU that dies mid-measurement (e.g. OOM-killed) is dropped and the work
// re-runs on the others (or on the CPU fallback): that point is then not a
// measurement of the configuration it is labelled with.
std::string vqpu_loss(DeviceType dev, const QuantumRuntime& rt, std::size_t expected) {
    if (dev != DeviceType::CUNQA) return "";
    const std::size_t now = vqpus_in_use(rt.device_name());
    if (now >= expected) return "";
    return "vqpu_lost: started with " + std::to_string(expected) + " vQPU(s), " +
           (now ? "ended with " + std::to_string(now) : std::string("ended on the CPU fallback"));
}

double state_mb(DeviceType dev, std::size_t n) {
    return is_exact_simulator(dev) ? std::ldexp(16.0, static_cast<int>(n)) / (1024.0 * 1024.0) : 0.0;
}

// ------------------------------------------------------------- latency ----

void run_latency(const Config& cfg, const std::string& dev_name, DeviceType dev, CsvSink& csv) {
    if (dev == DeviceType::CUNQA) setenv("CUNQA_MAX_VQPUS", "0", 1); // one circuit = one vQPU anyway
    auto rt = make_runtime(dev_name, dev);
    if (!rt) return;
    const std::string backend_name = rt->device_name();
    const std::size_t k_start = vqpus_in_use(backend_name);
    std::printf("-- latency on %s\n", backend_name.c_str());
    std::printf("%-6s %-16s %6s %7s | %11s %11s %11s %11s | %9s %s\n", "device", "circuit", "qubits", "gates",
                "exec ms", "sample ms", "total ms", "min ms", "state MB", "status");

    for (const auto& kind : cfg.circuits) {
        for (std::size_t n = cfg.min_q; n <= cfg.max_q; n += cfg.step) {
            // Exact simulators hold a complex<double> state of 2^n amplitudes (the
            // CPU backend also keeps a host mirror, hence the factor 2 in the
            // guard). Shot-based backends hold nothing locally.
            const double mb = state_mb(dev, n);
            const Circuit circuit = make_circuit(kind, n, cfg.reps);
            const std::size_t gates = circuit.operations().size();

            auto report = [&](const Stats& ex, const Stats& sa, const Stats& tot, const std::string& status) {
                std::printf("%-6s %-16s %6zu %7zu | %11.3f %11.3f %11.3f %11.3f | %9.1f %s\n", dev_name.c_str(),
                            kind.c_str(), n, gates, ex.median, sa.median, tot.median, tot.min, mb, status.c_str());
                std::fflush(stdout);
                csv.row(dev_name, kind, n, gates, cfg.shots, cfg.repeats, ex.median, sa.median, tot.median, tot.min,
                        tot.mean, tot.stddev, mb, quoted(status), quoted(backend_name));
            };

            if (cfg.max_memory_gb > 0 && 2.0 * mb / 1024.0 > cfg.max_memory_gb) {
                report({}, {}, {}, "memory_limit");
                break;
            }

            std::vector<double> exec_ms, sample_ms, total_ms;
            std::string status = "ok";
            try {
                for (std::size_t r = 0; r < cfg.warmup + cfg.repeats; ++r) {
                    const auto t0 = Clock::now();
                    rt->run(circuit);
                    const double e = ms_since(t0);
                    const auto t1 = Clock::now();
                    const auto shots = rt->backend().sample({}, cfg.shots);
                    const double s = ms_since(t1);
                    if (shots.size() != cfg.shots)
                        throw std::runtime_error("backend returned " + std::to_string(shots.size()) +
                                                 " shots instead of " + std::to_string(cfg.shots));
                    const bool over = (e + s) / 1e3 > cfg.time_limit_s;
                    if (r >= cfg.warmup || over) {
                        exec_ms.push_back(e); sample_ms.push_back(s); total_ms.push_back(e + s);
                    }
                    if (over) break; // no point repeating a circuit already over the limit
                }
            } catch (const std::exception& e) {
                status = std::string("failed: ") + e.what();
            }

            const Stats tot = stats(total_ms);
            if (status == "ok" && tot.median / 1e3 > cfg.time_limit_s) status = "time_limit";
            const std::string lost = vqpu_loss(dev, *rt, k_start);
            if (!lost.empty()) status = status == "ok" ? lost : lost + " (" + status + ")";
            report(stats(exec_ms), stats(sample_ms), tot, status);
            if (!lost.empty()) {
                std::printf("%-6s stopping: %s\n", dev_name.c_str(), lost.c_str());
                return; // every later point would run on the wrong configuration
            }
            if (status != "ok") break; // limit reached for this device/circuit
        }
    }
}

// ---------------------------------------------------------- throughput ----

void run_throughput(const Config& cfg, const std::string& dev_name, DeviceType dev, CsvSink& csv) {
    // The vQPU count only matters for CUNQA.
    const std::vector<std::size_t> sweep = dev == DeviceType::CUNQA ? cfg.vqpus : std::vector<std::size_t>{0};

    for (std::size_t k : sweep) {
        if (dev == DeviceType::CUNQA) setenv("CUNQA_MAX_VQPUS", std::to_string(k).c_str(), 1);
        auto rt = make_runtime(dev_name, dev);
        if (!rt) return;
        const std::string backend_name = rt->device_name();
        const std::size_t k_used = dev == DeviceType::CUNQA ? vqpus_in_use(backend_name) : 0;
        std::printf("-- throughput on %s\n", backend_name.c_str());
        if (dev == DeviceType::CUNQA && k_used == 0)
            std::printf("%-6s WARNING: no vQPU found, CUNQA is running on its CPU fallback\n", dev_name.c_str());
        if (dev == DeviceType::CUNQA && k != 0 && k_used != 0 && k_used < k)
            std::printf("%-6s WARNING: asked for %zu vQPUs but only %zu are registered\n", dev_name.c_str(), k,
                        k_used);
        std::printf("%-6s %-16s %5s %6s %6s | %11s %11s %13s %13s | %s\n", "device", "circuit", "vqpus", "qubits",
                    "batch", "batch ms", "min ms", "circuits/s", "ms/circuit", "status");

        for (const auto& kind : cfg.circuits) {
            for (std::size_t n : cfg.tp_qubits) {
                const Circuit tmpl = make_template(kind, n, cfg.reps);
                Observable zs(n);
                zs.add_term(1.0, std::string(n, 'Z')); // one measurement group: one circuit per copy

                for (std::size_t B : cfg.batch_sizes) {
                    auto report = [&](const Stats& t, const std::string& status) {
                        const double cps = t.median > 0 ? 1e3 * static_cast<double>(B) / t.median : 0.0;
                        const double mspc = B ? t.median / static_cast<double>(B) : 0.0;
                        std::printf("%-6s %-16s %5zu %6zu %6zu | %11.3f %11.3f %13.1f %13.3f | %s\n",
                                    dev_name.c_str(), kind.c_str(), k_used, n, B, t.median, t.min, cps, mspc,
                                    status.c_str());
                        std::fflush(stdout);
                        csv.row(dev_name, kind, k_used, n, B, cfg.shots, cfg.repeats, t.median, t.min, t.mean,
                                t.stddev, cps, mspc, quoted(status), quoted(backend_name));
                    };

                    if (cfg.max_memory_gb > 0 && 2.0 * static_cast<double>(B) * state_mb(dev, n) / 1024.0 >
                                                     cfg.max_memory_gb) {
                        report({}, "memory_limit");
                        break;
                    }

                    EvalBatch batch;
                    batch.shots = cfg.shots;
                    for (std::size_t b = 0; b < B; ++b)
                        batch.add(tmpl.num_params() ? ansatz::random_parameters(tmpl.num_params(), 100 + b)
                                                    : std::vector<double>{});

                    std::vector<double> ms;
                    std::string status = "ok";
                    try {
                        for (std::size_t r = 0; r < cfg.warmup + cfg.repeats; ++r) {
                            const auto t0 = Clock::now();
                            const auto e = rt->expectation_batch(tmpl, zs, batch);
                            const double t = ms_since(t0);
                            if (e.size() != B)
                                throw std::runtime_error("backend returned " + std::to_string(e.size()) +
                                                         " results for a batch of " + std::to_string(B));
                            const bool over = t / 1e3 > cfg.time_limit_s;
                            if (r >= cfg.warmup || over) ms.push_back(t);
                            if (over) break;
                        }
                    } catch (const std::exception& e) {
                        status = std::string("failed: ") + e.what();
                    }
                    const Stats t = stats(ms);
                    if (status == "ok" && t.median / 1e3 > cfg.time_limit_s) status = "time_limit";
                    const std::string lost = vqpu_loss(dev, *rt, k_used);
                    if (!lost.empty()) status = status == "ok" ? lost : lost + " (" + status + ")";
                    report(t, status);
                    if (!lost.empty()) {
                        std::printf("%-6s stopping this vQPU configuration: %s\n", dev_name.c_str(), lost.c_str());
                        goto next_configuration;
                    }
                    if (status != "ok") break; // larger batches would only be worse
                }
            }
        }
    next_configuration:;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        CsvSink lat_csv(cfg.has_mode("latency") ? cfg.csv : std::string(),
                        "device,circuit,qubits,gates,shots,repeats,exec_ms_median,sample_ms_median,"
                        "total_ms_median,total_ms_min,total_ms_mean,total_ms_std,state_mem_mb,status,backend_name");
        CsvSink tp_csv(cfg.has_mode("throughput") ? cfg.tp_csv : std::string(),
                       "device,circuit,vqpus,qubits,batch,shots,repeats,batch_ms_median,batch_ms_min,"
                       "batch_ms_mean,batch_ms_std,circuits_per_s,ms_per_circuit,status,backend_name");

        std::printf("Circuit benchmark | modes:");
        for (auto& m : cfg.modes) std::printf(" %s", m.c_str());
        std::printf(" | shots=%zu | repeats=%zu (+%zu warm-up) | time limit %.0f s | memory guard %s\n\n",
                    cfg.shots, cfg.repeats, cfg.warmup, cfg.time_limit_s,
                    cfg.max_memory_gb > 0 ? (std::to_string(cfg.max_memory_gb) + " GB").c_str() : "off");

        for (const auto& dev_name : cfg.devices) {
            const DeviceType dev = parse_device(dev_name);
            if (cfg.has_mode("latency")) run_latency(cfg, dev_name, dev, lat_csv);
            if (cfg.has_mode("throughput")) run_throughput(cfg, dev_name, dev, tp_csv);
        }
        std::puts("\nlatency   : exec = gate application (CPU/GPU) or local gate buffering only (CUNQA/QMIO);\n"
                  "            sample = drawing shots (CPU/GPU) or the full remote round trip (CUNQA/QMIO).\n"
                  "throughput: B independent circuits in one expectation_batch() call (CUNQA: spread over vQPUs).");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
