#include "cunqa_backend.hpp"

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
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
std::vector<QpuRegistryEntry> discover_qpus() {
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

} // namespace

struct CUNQABackend::QpuHandle {
    QpuRegistryEntry entry;
    zmq::context_t context{1};
    zmq::socket_t socket{context, zmq::socket_type::dealer};

    explicit QpuHandle(QpuRegistryEntry e) : entry(std::move(e)) {
        socket.connect(entry.endpoint);
    }
};

CUNQABackend::CUNQABackend()
    : fallback_(std::make_unique<CPUBackend>()) {
    discover_qpu();
}

CUNQABackend::~CUNQABackend() = default;

void CUNQABackend::discover_qpu() {
    try {
        auto candidates = discover_qpus();
        if (candidates.empty()) {
            using_cunqa_ = false;
            qpu_.reset();
            std::cerr << "[CUNQABackend] no qraised QPU found in "
                      << cunqa_registry_path()
                      << ", falling back to CPU backend\n";
            return;
        }
        // ASSUMED: taking the first available match is good enough --
        // add real selection logic (least loaded, matching family, etc.)
        // once you have more than one raised vQPU to choose between.
        qpu_ = std::make_unique<QpuHandle>(candidates.front());
        using_cunqa_ = true;
    } catch (const std::exception& e) {
        using_cunqa_ = false;
        qpu_.reset();
        std::cerr << "[CUNQABackend] QPU discovery failed (" << e.what()
                  << "), falling back to CPU backend\n";
    }
}

void CUNQABackend::initialize(std::size_t num_qubits) {
    num_qubits_ = num_qubits;
    gate_buffer_.clear();
    // Deliberately not re-running discover_qpu() here -- circuit-buffer
    // lifecycle and QPU discovery are on separate cadences (see header).
    if (!using_cunqa_) {
        fallback_->initialize(num_qubits);
    } else if (qpu_ && num_qubits > qpu_->entry.max_qubits) {
        throw std::runtime_error(
            "CUNQABackend: circuit needs " + std::to_string(num_qubits) +
            " qubits but the discovered vQPU only supports " +
            std::to_string(qpu_->entry.max_qubits));
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

    SampleResult result;
    try {
        result = run_on_cunqa(qubits, shots);
    } catch (const std::exception& e) {
        // Submission failed at run time -- e.g. the qraised allocation's
        // time box expired mid-session. Re-probe once, then fall back
        // if the QPU is really gone, replaying the buffered circuit.
        std::cerr << "[CUNQABackend] cunqa submission failed (" << e.what()
                  << "), re-checking QPU availability\n";
        discover_qpu();
        if (!using_cunqa_) {
            fallback_->initialize(num_qubits_);
            for (const auto& op : gate_buffer_) {
                fallback_->apply_gate(op);
            }
            return fallback_->sample(qubits, shots);
        }
        result = run_on_cunqa(qubits, shots);
    }
    return result.outcomes;
}

CUNQABackend::SampleResult CUNQABackend::run_on_cunqa(
    const std::vector<std::size_t>& qubits, std::size_t shots) {

    std::vector<std::size_t> measured = qubits;
    if (measured.empty()) {
        measured.resize(num_qubits_);
        for (std::size_t i = 0; i < num_qubits_; ++i) measured[i] = i;
    }

    // Build instructions: buffered gates, then one explicit measure per
    // requested qubit -- cunqa's CunqaCircuit::measure(qubit, clbit)
    // supports arbitrary subsets directly, so no marginalization needed.
    // CONFIRMED against a real client dump: "qubits"/"clbits" are
    // single-element arrays here too, and there's no "save" key.
    json instructions = json::array();
    for (const auto& op : gate_buffer_) {
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
        {"num_qubits", num_qubits_},
        {"device", qpu_->entry.device}
    };
    task["instructions"] = instructions;
    task["sending_to"] = json::array();
    task["is_dynamic"] = false;
    task["id"] = generate_circuit_id();

    const std::string task_str = task.dump();
    zmq::message_t request(task_str.begin(), task_str.end());
    qpu_->socket.send(request, zmq::send_flags::none);

    zmq::message_t reply;
    auto recv_size = qpu_->socket.recv(reply, zmq::recv_flags::none);
    if (!recv_size) {
        throw std::runtime_error("CUNQABackend: no reply received from vQPU");
    }
    std::string response_str(static_cast<char*>(reply.data()), recv_size.value());
    json response = json::parse(response_str);

    if (response.contains("ERROR")) {
        throw std::runtime_error(
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

    SampleResult out;
    out.outcomes.reserve(shots);
    for (auto& [bitstring, count] : counts_json.items()) {
        unsigned long long value = std::stoull(bitstring, nullptr, 2);
        std::size_t n = count.get<std::size_t>();
        for (std::size_t i = 0; i < n; ++i) {
            out.outcomes.push_back(value);
        }
    }
    return out;
}

std::string CUNQABackend::device_name() const {
    if (using_cunqa_) {
        return "CUNQA (vQPU: " + qpu_->entry.id + ")";
    }
    return "CUNQA (fallback: " + fallback_->device_name() + ")";
}

} // namespace backends
} // namespace quantum