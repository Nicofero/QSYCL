#include "cunqa_backend.hpp"
#include "quantum/shot_expectation.hpp"
#include "quantum/native_vqa.hpp"
#include "quantum/device_selector.hpp"

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------
// Everything in this file talks to cunqa purely through its on-disk
// registry (qpus.json) and its ZMQ wire protocol -- reverse-engineered
// from cunqa's Python layer (cunqa/qpu.py, cunqa/qjob.py, cunqa/result.py)
// and its ZMQ client (src/comm/comm_impl/zmq/zmq_client.cpp), not from a
// linked cunqa library. None of this is a documented/public interface,
// so re-verify against your cunqa version if anything looks wrong --
// especially the bit-order note in run_on_cunqa() below, which is the
// one detail we couldn't confirm from source alone.
// ---------------------------------------------------------------------

namespace quantum {
namespace backends {

using json = nlohmann::json;

namespace {

// cunqa's own install guide requires $STORE to be set and always uses
// $STORE/.cunqa as its registry location, so there's no need for this
// to be configurable -- if $STORE isn't set, cunqa itself isn't
// properly installed/configured on this system either.
std::string cunqa_registry_path() {
    const char* store = std::getenv("STORE");
    if (!store) {
        throw std::runtime_error(
            "CUNQABackend: $STORE is not set, cannot locate "
            "$STORE/.cunqa/qpus.json");
    }
    return std::string(store) + "/.cunqa/qpus.json";
}

// One entry from qpus.json that we care about.
struct QpuRegistryEntry {
    std::string id;
    std::string endpoint;
    json device;             // net.device -- passed back verbatim in config.device
    std::size_t max_qubits;  // backend.n_qubits
};

// Reads every vQPU listed in the registry -- no co-located filtering.
// cunqa's own get_QPUs(co_located=True) restricts to same-node vQPUs,
// but that's not a real requirement here: we talk to vQPUs over a plain
// TCP/ZMQ endpoint regardless of where they're running, so a
// co-location check would only needlessly exclude perfectly reachable
// QPUs on other nodes.
std::vector<QpuRegistryEntry> read_registry() {
    std::ifstream file(cunqa_registry_path());
    if (!file) {
        return {};
    }
    json registry;
    file >> registry;

    std::vector<QpuRegistryEntry> found;
    for (auto& [id, info] : registry.items()) {
        const auto& net = info.at("net");

        QpuRegistryEntry entry;
        entry.id = id;
        entry.endpoint = net.at("endpoint").get<std::string>();
        entry.device = net.at("device");
        entry.max_qubits = info.at("backend").at("n_qubits").get<std::size_t>();
        found.push_back(std::move(entry));
    }
    return found;
}

std::string generate_circuit_id() {
    static std::atomic<std::uint64_t> counter{0};
    std::ostringstream oss;
    oss << "sycl_cunqa_" << counter.fetch_add(1);
    return oss.str();
}

// Translates one GateOp into cunqa's instruction JSON.
// ASSUMED: GateOp has .label (string), .qubits (vector<size_t>), and
// .params (vector<double>) -- rename further if these don't match
// quantum/types.hpp. Gate names are matched case-insensitively (label
// is lowercased before comparison/serialization).
// Gate lists below are NOT exhaustive of what cunqa's CunqaCircuit
// supports (it has many more, see cunqa/circuit/core.py) -- extend as
// your GateOp set grows.
json gate_to_instruction(const GateOp& op) {
    static const std::vector<std::string> single_qubit_no_param = {
        "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg"
    };
    static const std::vector<std::string> two_qubit_no_param = {
        "cx", "cy", "cz", "swap", "ch"
    };
    static const std::vector<std::string> single_qubit_one_param = {
        "rx", "ry", "rz", "p", "u1"
    };

    auto contains = [](const std::vector<std::string>& v, const std::string& s) {
        return std::find(v.begin(), v.end(), s) != v.end();
    };

    json instr;
    std::string label = op.label;
    std::transform(label.begin(), label.end(), label.begin(),
               [](unsigned char c) { return std::tolower(c); });
    instr["name"] = label;

    if (contains(single_qubit_no_param, label)) {
        instr["qubits"] = op.qubits;
    } else if (contains(two_qubit_no_param, label)) {
        instr["qubits"] = op.qubits; // [control, target]
    } else if (contains(single_qubit_one_param, label)) {
        instr["qubits"] = op.qubits;
        instr["params"] = op.params; // e.g. [theta]
    } else {
        throw std::runtime_error(
            "CUNQABackend: gate '" + op.label +
            "' has no known cunqa translation -- extend gate_to_instruction()");
    }
    return instr;
}

std::vector<GateOp> deconstruct_gate(const GateOp& op) {
    std::string label = op.label;

    std::transform(
        label.begin(), label.end(), label.begin(),
        [](unsigned char c) { return std::tolower(c); }
    );

    if (label != "crx" && label != "cry" && label != "crz") {
        return {op};
    }

    if (op.qubits.size() != 2) {
        throw std::runtime_error("Malformed " + label + ": expected 2 qubits");
    }
    if (op.params.size() != 1) {
        throw std::runtime_error("Malformed " + label + ": expected 1 parameter");
    }

    const std::size_t control = op.qubits[0];
    const std::size_t target  = op.qubits[1];
    const double theta = op.params[0];

    // CRZ/CRY: X anticommutes with both Z and Y, which is exactly what
    // CNOT exploits on the target -> R(theta/2), CNOT, R(-theta/2), CNOT
    // implements the controlled rotation directly.
    if (label == "crz") {
        return {
            GateOp{GateType::RZ,   {target},          {theta / 2.0},  "rz", false},
            GateOp{GateType::CNOT, {control, target},  {},             "cx", false},
            GateOp{GateType::RZ,   {target},          {-theta / 2.0}, "rz", false},
            GateOp{GateType::CNOT, {control, target},  {},             "cx", false}
        };
    }

    if (label == "cry") {
        return {
            GateOp{GateType::RY,   {target},          {theta / 2.0},  "ry", false},
            GateOp{GateType::CNOT, {control, target},  {},             "cx", false},
            GateOp{GateType::RY,   {target},          {-theta / 2.0}, "ry", false},
            GateOp{GateType::CNOT, {control, target},  {},             "cx", false}
        };
    }

    // label == "crx": X commutes with itself, so the CNOT sandwich above
    // does NOT implement CRX. Use Rx(theta) = H * Rz(theta) * H to reuse
    // the CRZ pattern under a Hadamard basis change on the target.
    return {
        GateOp{GateType::H,    {target},          {},              "h",  false},
        GateOp{GateType::RZ,   {target},          {theta / 2.0},   "rz", false},
        GateOp{GateType::CNOT, {control, target},  {},              "cx", false},
        GateOp{GateType::RZ,   {target},          {-theta / 2.0},  "rz", false},
        GateOp{GateType::CNOT, {control, target},  {},              "cx", false},
        GateOp{GateType::H,    {target},          {},              "h",  false}
    };
}

// Serializes one circuit (gates + an explicit measure per qubit in
// `measured`) into cunqa's task JSON for a vQPU with registry `device`.
std::string build_task(const std::vector<GateOp>& gates, std::size_t num_qubits,
                       const std::vector<std::size_t>& measured, std::size_t shots,
                       const json& device) {
    // Buffered gates, then one explicit measure per requested qubit --
    // cunqa's CunqaCircuit::measure(qubit, clbit) supports arbitrary
    // subsets directly, so no marginalization needed. Any MEASURE ops
    // already in the circuit are dropped: the measurements are fully
    // described by `measured`.
    // CONFIRMED against a real client dump: "qubits"/"clbits" are
    // single-element arrays here too, and there's no "save" key.
    json instructions = json::array();
    for (const auto& op : gates) {
        if (op.type == GateType::MEASURE) continue;
        for (const auto& decomposed_op : deconstruct_gate(op)) {    // Atomic deconstruction of gates
            instructions.push_back(gate_to_instruction(decomposed_op));
        }
    }
    for (std::size_t clbit = 0; clbit < measured.size(); ++clbit) {
        instructions.push_back({
            {"name", "measure"},
            {"qubits", json::array({measured[clbit]})},
            {"clbits", json::array({clbit})}
        });
    }

    // CONFIRMED structure against a real client dump: "sending_to",
    // "is_dynamic" and "id" are top-level siblings of "config" and
    // "instructions", NOT nested inside "config" -- and there is no
    // "qpu_id" field at all (this deployment's no_comm backend doesn't
    // need it; the ZMQ endpoint alone identifies the target vQPU).
    // "num_qubits" is a plain scalar (total qubit count), not the
    // [data, comm] pair the main-branch source suggested.
    json task;
    task["config"] = {
        {"shots", shots},
        {"method", "automatic"},
        {"avoid_parallelization", false},
        {"num_clbits", measured.size()},
        {"num_qubits", num_qubits},
        {"device", device}
    };
    task["instructions"] = instructions;
    task["sending_to"] = json::array();
    task["is_dynamic"] = false;
    task["id"] = generate_circuit_id();
    return task.dump();
}

// A vQPU answered, but with an error about the circuit itself. Retrying
// the same circuit on another vQPU would fail the same way, so the
// dispatcher propagates this one instead of dropping the vQPU.
struct VqpuReportedError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Parses a vQPU reply into one outcome (bitstring-as-int) per shot.
std::vector<unsigned long long> parse_reply(const std::string& response_str,
                                            std::size_t shots) {
    json response = json::parse(response_str);

    if (response.contains("ERROR")) {
        throw VqpuReportedError(
            "CUNQABackend: vQPU reported an error: " +
            response.at("ERROR").get<std::string>());
    }

    // Aer nests counts under results[0].data.counts; cunqa/Munich-style
    // simulators put counts directly at the top level (mirrors
    // cunqa.result.Result.counts's own branching logic).
    json counts_json;
    if (response.contains("results")) {
        counts_json = response.at("results").at(0).at("data").at("counts");
    } else if (response.contains("counts")) {
        counts_json = response.at("counts");
    } else {
        throw std::runtime_error(
            "CUNQABackend: unrecognized result format from vQPU (no "
            "'results' or 'counts' key)");
    }

    std::vector<unsigned long long> outcomes;
    outcomes.reserve(shots);
    for (auto& [bitstring, count] : counts_json.items()) {
        unsigned long long value = std::stoull(bitstring, nullptr, 2);
        std::size_t n = count.get<std::size_t>();
        for (std::size_t i = 0; i < n; ++i) {
            outcomes.push_back(value);
        }
    }
    return outcomes;
}

void send_string(zmq::socket_t& socket, const std::string& payload) {
    zmq::message_t request(payload.begin(), payload.end());
    if (!socket.send(request, zmq::send_flags::none)) {
        throw std::runtime_error("CUNQABackend: could not send circuit to vQPU");
    }
}

std::string recv_string(zmq::socket_t& socket) {
    zmq::message_t reply;
    auto recv_size = socket.recv(reply, zmq::recv_flags::none);
    if (!recv_size) {
        throw std::runtime_error("CUNQABackend: no reply received from vQPU");
    }
    return std::string(static_cast<char*>(reply.data()), recv_size.value());
}

// How long to wait for a vQPU's reply before treating it as dead. 0 (the
// default) = wait indefinitely. A vQPU that is killed (e.g. OOM-killed by
// SLURM) never answers -- ZMQ doesn't report a dead peer -- so without this a
// lost vQPU blocks its caller forever. Applies to the batched dispatcher (the
// circuit is re-queued on the other vQPUs) and to single-circuit sample()
// (the call fails over to another vQPU). Set it above the longest legitimate
// circuit time.
std::chrono::milliseconds dispatch_timeout() {
    const char* v = std::getenv("CUNQA_RECV_TIMEOUT_MS");
    if (!v) return std::chrono::milliseconds(0);
    try {
        return std::chrono::milliseconds(std::stoll(v));
    } catch (...) {
        return std::chrono::milliseconds(0);
    }
}

} // namespace

struct CUNQABackend::QpuHandle {
    QpuRegistryEntry entry;
    zmq::context_t& context;
    zmq::socket_t socket;
    bool healthy = true; // false once a submission to it has failed

    QpuHandle(zmq::context_t& ctx, QpuRegistryEntry e)
        : entry(std::move(e)), context(ctx) {
        reconnect();
    }

    // Replaces the socket with a fresh one (new ZMQ identity). Any reply
    // still owed to the old socket is then discarded by the vQPU's ROUTER
    // instead of being read later as the answer to a different circuit.
    void reconnect() {
        socket = zmq::socket_t(context, zmq::socket_type::dealer);
        // Don't block destruction on unsent/unanswered messages to a vQPU
        // that has gone away.
        socket.set(zmq::sockopt::linger, 0);
        // Bounds the blocking recv() of the single-circuit path; the batched
        // dispatcher polls with the same timeout instead.
        const auto timeout = dispatch_timeout();
        if (timeout.count() > 0) socket.set(zmq::sockopt::rcvtimeo, static_cast<int>(timeout.count()));
        socket.connect(entry.endpoint);
    }
};

struct CUNQABackend::QpuPool {
    zmq::context_t context{1};
    // Declared after `context`, so every socket is closed before the
    // context is terminated.
    std::vector<std::unique_ptr<QpuHandle>> qpus;
};

CUNQABackend::CUNQABackend()
    : fallback_(std::make_unique<CPUBackend>()) {
    discover_qpus();
}

CUNQABackend::~CUNQABackend() = default;

void CUNQABackend::discover_qpus() {
    pool_.reset();
    using_cunqa_ = false;
    try {
        auto candidates = read_registry();
        // A killed vQPU can stay listed in the registry: never reconnect to
        // one this backend has already seen die.
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&](const QpuRegistryEntry& e) { return dead_endpoints_.count(e.endpoint) > 0; }),
                         candidates.end());
        if (candidates.empty()) {
            std::cerr << "[CUNQABackend] no qraised QPU found in "
                      << cunqa_registry_path()
                      << ", falling back to CPU backend\n";
            return;
        }
        // Optional cap on how many of the registered vQPUs to use (e.g. to
        // benchmark scaling with the vQPU count). Registry entries are
        // iterated in sorted-id order, so the same subset is always chosen.
        if (const char* cap_env = std::getenv("CUNQA_MAX_VQPUS")) {
            try {
                const std::size_t cap = std::stoul(cap_env);
                if (cap > 0 && candidates.size() > cap) candidates.resize(cap);
            } catch (...) {
                std::cerr << "[CUNQABackend] ignoring invalid CUNQA_MAX_VQPUS='" << cap_env << "'\n";
            }
        }
        auto pool = std::make_unique<QpuPool>();
        for (auto& entry : candidates) {
            pool->qpus.push_back(std::make_unique<QpuHandle>(pool->context, std::move(entry)));
        }
        pool_ = std::move(pool);
        next_qpu_ = 0;
        using_cunqa_ = true;
    } catch (const std::exception& e) {
        pool_.reset();
        using_cunqa_ = false;
        std::cerr << "[CUNQABackend] QPU discovery failed (" << e.what()
                  << "), falling back to CPU backend\n";
    }
}

std::vector<CUNQABackend::QpuHandle*> CUNQABackend::eligible_qpus(std::size_t n) const {
    std::vector<QpuHandle*> out;
    if (!pool_) return out;
    for (auto& q : pool_->qpus) {
        if (q->healthy && q->entry.max_qubits >= n) out.push_back(q.get());
    }
    return out;
}

void CUNQABackend::mark_dead(QpuHandle& qpu) {
    qpu.healthy = false;
    dead_endpoints_.insert(qpu.entry.endpoint);
}

std::size_t CUNQABackend::num_vqpus() const {
    if (!pool_) return 0;
    std::size_t n = 0;
    for (auto& q : pool_->qpus) n += q->healthy ? 1 : 0;
    return n;
}

void CUNQABackend::initialize(std::size_t num_qubits) {
    num_qubits_ = num_qubits;
    gate_buffer_.clear();
    // Deliberately not re-running discover_qpus() here -- circuit-buffer
    // lifecycle and QPU discovery are on separate cadences (see header).
    if (!using_cunqa_) {
        fallback_->initialize(num_qubits);
    } else if (eligible_qpus(num_qubits).empty()) {
        std::size_t largest = 0;
        for (auto& q : pool_->qpus) largest = std::max(largest, q->entry.max_qubits);
        throw std::runtime_error(
            "CUNQABackend: circuit needs " + std::to_string(num_qubits) +
            " qubits but the largest available vQPU only supports " +
            std::to_string(largest));
    }
}

void CUNQABackend::apply_gate(const GateOp& op) {
    if (!using_cunqa_) {
        fallback_->apply_gate(op);
        return;
    }
    gate_buffer_.push_back(op); // buffered, not executed -- see header
}

std::vector<Complex> CUNQABackend::get_state() const {
    if (!using_cunqa_) {
        return fallback_->get_state();
    }
    throw std::runtime_error(
        "CUNQABackend::get_state() is not supported on a real vQPU: "
        "cunqa emulates real QPU behavior, and real QPUs cannot report "
        "exact statevector amplitudes. Use sample() instead.");
}

std::vector<double> CUNQABackend::probabilities() const {
    if (!using_cunqa_) {
        return fallback_->probabilities();
    }
    throw std::runtime_error(
        "CUNQABackend::probabilities() is not supported on a real vQPU: "
        "cunqa emulates real QPU behavior, and real QPUs cannot report "
        "exact probabilities. Use sample() instead.");
}

std::vector<unsigned long long> CUNQABackend::sample(
    const std::vector<std::size_t>& qubits, std::size_t shots) {
    if (!using_cunqa_) {
        return fallback_->sample(qubits, shots);
    }

    auto pick = [&]() -> QpuHandle* {
        auto eligible = eligible_qpus(num_qubits_);
        if (eligible.empty()) return nullptr;
        return eligible[next_qpu_++ % eligible.size()];
    };

    // Try every healthy vQPU in turn. One that fails (no reply within
    // CUNQA_RECV_TIMEOUT_MS, connection error, ...) is taken out for good.
    QpuHandle* qpu = pick();
    if (!qpu) {
        throw std::runtime_error(
            "CUNQABackend: no healthy vQPU can run a " + std::to_string(num_qubits_) +
            "-qubit circuit");
    }
    for (; qpu; qpu = pick()) {
        try {
            return run_on_cunqa(*qpu, gate_buffer_, qubits, shots);
        } catch (const VqpuReportedError&) {
            throw; // the circuit is at fault, not the vQPU
        } catch (const std::exception& e) {
            // Submission failed at run time -- e.g. the vQPU was OOM-killed or
            // its time-boxed qraise allocation expired mid-session.
            std::cerr << "[CUNQABackend] cunqa submission to vQPU '" << qpu->entry.id
                      << "' failed (" << e.what() << "), dropping it\n";
            mark_dead(*qpu);
        }
    }

    // Re-probe once, then fall back if no vQPU is really left, replaying
    // the buffered circuit.
    discover_qpus();
    if (!using_cunqa_) {
        fallback_->initialize(num_qubits_);
        for (const auto& op : gate_buffer_) {
            fallback_->apply_gate(op);
        }
        return fallback_->sample(qubits, shots);
    }
    qpu = pick();
    if (!qpu) {
        throw std::runtime_error(
            "CUNQABackend: no rediscovered vQPU can run a " + std::to_string(num_qubits_) +
            "-qubit circuit");
    }
    try {
        return run_on_cunqa(*qpu, gate_buffer_, qubits, shots);
    } catch (const VqpuReportedError&) {
        throw;
    } catch (...) {
        mark_dead(*qpu);
        throw;
    }
}

std::vector<unsigned long long> CUNQABackend::run_on_cunqa(
    QpuHandle& qpu, const std::vector<GateOp>& gates,
    const std::vector<std::size_t>& qubits, std::size_t shots) {

    std::vector<std::size_t> measured = qubits;
    if (measured.empty()) {
        measured.resize(num_qubits_);
        for (std::size_t i = 0; i < num_qubits_; ++i) measured[i] = i;
    }

    send_string(qpu.socket, build_task(gates, num_qubits_, measured, shots, qpu.entry.device));
    return parse_reply(recv_string(qpu.socket), shots);
}

std::vector<std::vector<unsigned long long>> CUNQABackend::dispatch(
    const std::vector<Circuit>& circuits, std::size_t shots) {
    using clock = std::chrono::steady_clock;

    std::vector<std::vector<unsigned long long>> results(circuits.size());
    if (circuits.empty()) return results;

    const std::size_t n = circuits.front().num_qubits();
    std::vector<std::size_t> all_qubits(n);
    for (std::size_t q = 0; q < n; ++q) all_qubits[q] = q;

    const auto timeout = dispatch_timeout();

    // Circuits not yet handed to any vQPU. A circuit taken back from a
    // vQPU that failed goes to the front, so it isn't starved.
    std::deque<std::size_t> pending;
    for (std::size_t i = 0; i < circuits.size(); ++i) pending.push_back(i);

    struct Worker {
        QpuHandle* qpu;
        long job = -1;              // circuit index in flight, -1 = idle
        clock::time_point sent_at;
    };
    std::vector<Worker> workers;
    for (QpuHandle* q : eligible_qpus(n)) workers.push_back(Worker{q, -1, clock::time_point{}});

    auto drop = [&](Worker& w, const std::string& why) {
        std::cerr << "[CUNQABackend] dropping vQPU '" << w.qpu->entry.id << "' (" << why
                  << ")" << (w.job >= 0 ? ", re-queueing its circuit" : "") << "\n";
        mark_dead(*w.qpu);
        if (w.job >= 0) pending.push_front(static_cast<std::size_t>(w.job));
        w.job = -1;
    };

    std::size_t done = 0;
    try {
        while (done < circuits.size()) {
            // Keep every healthy, idle vQPU busy.
            for (auto& w : workers) {
                if (!w.qpu->healthy || w.job >= 0 || pending.empty()) continue;
                const std::size_t job = pending.front();
                pending.pop_front();
                w.job = static_cast<long>(job);
                try {
                    send_string(w.qpu->socket, build_task(circuits[job].operations(), n,
                                                          all_qubits, shots, w.qpu->entry.device));
                    w.sent_at = clock::now();
                } catch (const std::exception& e) {
                    drop(w, e.what());
                }
            }

            std::vector<zmq::pollitem_t> items;
            std::vector<Worker*> busy;
            for (auto& w : workers) {
                if (w.job < 0) continue;
                items.push_back({w.qpu->socket.handle(), 0, ZMQ_POLLIN, 0});
                busy.push_back(&w);
            }
            if (busy.empty()) {
                if (!pending.empty()) {
                    throw std::runtime_error(
                        "CUNQABackend: every vQPU able to run a " + std::to_string(n) +
                        "-qubit circuit has failed; " + std::to_string(circuits.size() - done) +
                        " circuit(s) of this batch could not be run");
                }
                break;
            }

            // Wait for the first reply, or for the earliest in-flight circuit
            // to hit the timeout (if one is configured).
            std::chrono::milliseconds wait(-1);
            if (timeout.count() > 0) {
                auto earliest = busy.front()->sent_at;
                for (auto* w : busy) earliest = std::min(earliest, w->sent_at);
                wait = std::max(std::chrono::milliseconds(0),
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    earliest + timeout - clock::now()));
            }
            zmq::poll(items, wait);

            for (std::size_t k = 0; k < busy.size(); ++k) {
                Worker& w = *busy[k];
                if (items[k].revents & ZMQ_POLLIN) {
                    std::string reply;
                    try {
                        reply = recv_string(w.qpu->socket);
                    } catch (const std::exception& e) {
                        drop(w, e.what());
                        continue;
                    }
                    // A VqpuReportedError (or a malformed reply) propagates:
                    // it is about the circuit/protocol, not this vQPU.
                    results[static_cast<std::size_t>(w.job)] = parse_reply(reply, shots);
                    w.job = -1;
                    ++done;
                } else if (timeout.count() > 0 && clock::now() - w.sent_at >= timeout) {
                    drop(w, "no reply within " + std::to_string(timeout.count()) + " ms");
                }
            }
        }
    } catch (...) {
        // Aborting with circuits still in flight: their replies would sit in
        // those sockets and be read later as the answer to a different
        // circuit. Give those vQPUs fresh sockets so the replies are dropped.
        for (auto& w : workers) {
            if (w.job >= 0 && w.qpu->healthy) w.qpu->reconnect();
        }
        throw;
    }
    return results;
}

std::string CUNQABackend::device_name() const {
    if (using_cunqa_) {
        std::string ids;
        for (auto& q : pool_->qpus) {
            if (!q->healthy) continue;
            ids += (ids.empty() ? "" : ", ") + q->entry.id;
        }
        return "CUNQA (" + std::to_string(num_vqpus()) + " vQPU(s): " + ids + ")";
    }
    return "CUNQA (fallback: " + fallback_->device_name() + ")";
}

std::vector<double> CUNQABackend::expectation_batch(const Circuit& ansatz, const Observable& obs,
                                                     const EvalBatch& batch) {
    if (!using_cunqa_) {
        return fallback_->expectation_batch(ansatz, obs, batch);
    }
    if (obs.num_qubits() != ansatz.num_qubits()) {
        throw std::invalid_argument("CUNQABackend::expectation_batch: observable and ansatz qubit counts differ");
    }
    if (batch.count != 0 && batch.num_params != ansatz.num_params()) {
        throw std::invalid_argument("CUNQABackend::expectation_batch: batch parameter count does not match ansatz");
    }

    std::vector<double> out(batch.count, 0.0);
    if (batch.count == 0 || batch.shots == 0) return out;

    const std::size_t n = ansatz.num_qubits();
    if (eligible_qpus(n).empty()) {
        // Every vQPU was dropped by an earlier batch (or none was ever big
        // enough): re-read the registry once -- qraise may have brought new
        // ones up -- but never switch to the CPU fallback mid-run.
        discover_qpus();
        if (eligible_qpus(n).empty()) {
            throw std::runtime_error(
                "CUNQABackend: no vQPU available that can run a " + std::to_string(n) +
                "-qubit circuit");
        }
    }

    // Build the whole batch up front -- one complete circuit per (request,
    // QWC group) -- so it can be spread over every vQPU at once instead of
    // run one circuit at a time.
    const auto groups = native::measurement_groups(obs);
    const std::size_t G = groups.size();
    std::vector<Circuit> circuits;
    circuits.reserve(batch.count * G);
    for (std::size_t i = 0; i < batch.count; ++i) {
        const int op = batch.shift_op.empty() ? -1 : batch.shift_op[i];
        const double sh = batch.shift.empty() ? 0.0 : batch.shift[i];
        const auto theta = batch.theta_of(i);
        for (const auto& g : groups) {
            circuits.push_back(native::measurement_circuit(ansatz, theta, op, sh, g));
        }
    }

    // Deliberately no CPU fallback from here on: a batch that loses every
    // vQPU partway through fails loudly rather than silently mixing
    // emulated and locally simulated energies into one VQA trajectory.
    const auto outcomes = dispatch(circuits, batch.shots);

    for (std::size_t i = 0; i < batch.count; ++i) {
        for (std::size_t g = 0; g < G; ++g) {
            out[i] += native::group_contribution(obs, groups[g], outcomes[i * G + g]);
        }
    }
    return out;
}

bool CUNQABackend::run_native_vqa(const Circuit& ansatz, const Observable& obs,
                                  const VQAOptions& options,
                                  std::vector<double>& theta, VQAResult& result) {
    if (!using_cunqa_) {
        return fallback_->run_native_vqa(ansatz, obs, options, theta, result);
    }
    sycl::queue optimizer_q = DeviceSelector::make_queue(options.optimizer_device);
    return native::run_native_vqa(*this, optimizer_q, ansatz, obs, options, theta, result);
}

} // namespace backends
} // namespace quantum
