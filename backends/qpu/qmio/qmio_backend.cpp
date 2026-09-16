#include "qmio_backend.hpp"
#include "qasm2_export.hpp"
#include "safe_pickle.hpp"

#include "cpu_backend.hpp"

#include <zmq.hpp>

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <utility>

namespace quantum {
namespace backends {

using json = nlohmann::json;

namespace {

// qpu registry path: $STORE/.qpu/zmq_server
std::string qpu_registry_path() {
    const char* store = std::getenv("STORE");
    if (!store) {
        throw std::runtime_error(
            "QPUBackend: $STORE is not set, cannot locate "
            "$STORE/.qpu/zmq_server.txt");
    }
    return std::string(store) + "/.qpu/zmq_server.txt";
}

std::uint64_t env_u64(const char* name, std::uint64_t default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    try {
        return std::stoull(v);
    } catch (...) {
        return default_value;
    }
}

// Map a results-format name to its (InlineResultsProcessing,
// ResultsFormatting) $value pair. Only "binary_count" is implemented
// end-to-end; the rest of the table is reproduced from the reference
// client for forward compatibility, matching its own scoping note.
std::pair<long long, long long> results_format_values(const std::string& fmt) {
    if (fmt == "binary_count") return {1, 3};
    if (fmt == "raw") return {1, 2};
    if (fmt == "binary") return {2, 2};
    if (fmt == "squash_binary_result_arrays") return {2, 6};
    throw std::runtime_error(
        "QmioBackend: unsupported results format '" + fmt +
        "'; only 'binary_count' is implemented end-to-end");
}

// Map an optimization level to the CompilerConfig TketOptimizations
// $value enum integer.
long long tket_opt_value(unsigned level) {
    switch (level) {
        case 0: return 0;  // TketOptimizations::Empty -- no compilation/routing
        case 1: return 1;  // DefaultMappingPass only
        case 2: return 18; // + circuit simplifications
        // case 3: return 30; // full optimization including SWAP routing
        default:
            throw std::runtime_error(
                "QmioBackend: unsupported optimization level " + std::to_string(level) +
                "; expected 0, 1 or 2");
    }
}

std::string normalize_bitstring(const std::string& key) {
    std::string out;
    out.reserve(key.size());
    for (char c : key) {
        if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
    }
    return out;
}

// A {bitstring: count} object, or std::nullopt if `v` isn't one (a
// non-empty object whose values are all non-negative integers).
std::optional<json> as_counts_object(const json& v) {
    if (!v.is_object() || v.empty()) return std::nullopt;
    json out = json::object();
    for (auto& [key, val] : v.items()) {
        if (!val.is_number_integer() || val.get<long long>() < 0) return std::nullopt;
        out[normalize_bitstring(key)] = val;
    }
    return out;
}

// A map of register -> {bitstring: count}, merged into one counts
// object -- the QMIO/qat reply shape, e.g. {"c": {"00": 1000}}.
std::optional<json> as_register_counts(const json& v) {
    if (!v.is_object() || v.empty()) return std::nullopt;
    json merged = json::object();
    for (auto& [register_name, val] : v.items()) {
        auto counts = as_counts_object(val);
        if (!counts) return std::nullopt; // not every entry is a counts object
        for (auto& [bitstring, count] : counts->items()) {
            std::uint64_t prev = merged.contains(bitstring) ? merged[bitstring].get<std::uint64_t>() : 0;
            merged[bitstring] = prev + count.get<std::uint64_t>();
        }
    }
    return merged;
}

// Locate the bitstring->count object inside the reply JSON. Mirrors the
// reference client's counts_from_json: tries a handful of known
// container keys, then the QMIO/qat register-grouped shape, then falls
// back to the first nested counts-like object.
json counts_from_json(const json& value) {
    if (auto direct = as_counts_object(value)) return *direct;

    if (value.is_object()) {
        for (const char* key : {"counts", "result", "results", "data", "c", "register", "measurements"}) {
            if (value.contains(key)) {
                if (auto c = as_counts_object(value.at(key))) return *c;
            }
        }
        for (const char* key : {"results", "result", "data"}) {
            if (value.contains(key)) {
                if (auto c = as_register_counts(value.at(key))) return *c;
            }
        }
        for (auto& [key, inner] : value.items()) {
            if (auto c = as_counts_object(inner)) return *c;
        }
    }
    throw std::runtime_error(
        "QmioBackend: could not locate a bitstring->count object in the QPU JSON reply");
}

std::unique_ptr<zmq::socket_t> make_socket(zmq::context_t& ctx, const std::string& endpoint,
                                            std::chrono::milliseconds timeout) {
    auto socket = std::make_unique<zmq::socket_t>(ctx, zmq::socket_type::req);
    socket->set(zmq::sockopt::linger, 0);
    socket->set(zmq::sockopt::rcvtimeo, static_cast<int>(timeout.count()));
    socket->set(zmq::sockopt::sndtimeo, static_cast<int>(timeout.count()));
    socket->connect(endpoint); // ZMQ TCP connect is lazy/async; failures
                                // surface at send/recv time, not here
    return socket;
}

} // namespace

struct QmioBackend::Connection {
    zmq::context_t context{1};
    std::unique_ptr<zmq::socket_t> socket; // null until first use or after a fault
};

QmioBackend::QmioBackend(std::string endpoint, std::unique_ptr<Backend> fallback)
    : QPUBackend(fallback ? std::move(fallback) : std::make_unique<CPUBackend>()),
      conn_(std::make_unique<Connection>()),
      recv_timeout_(std::chrono::milliseconds(env_u64("QMIO_RECV_TIMEOUT_MS", 30'000))),
      retry_backoff_(std::chrono::milliseconds(env_u64("QMIO_RETRY_BACKOFF_MS", 100))),
      max_retries_(static_cast<std::size_t>(env_u64("QMIO_MAX_RETRIES", 3))) {
    endpoint_ = endpoint.empty() ? std::string() : std::move(endpoint);
    if (endpoint_.empty()) {
        if (const char* env_endpoint = std::getenv("ZMQ_SERVER")) {
            endpoint_ = env_endpoint;
        }else{
            std::cout<< "[QmioBackend] ZMQ_SERVER env var not set, trying $STORE/.qpu/zmq_server.txt\n";
            std::ifstream file(qpu_registry_path());
            if (file) {
                std::getline(file, endpoint_);
            }else{
                endpoint_ = "tcp://10.255.3.70:5556";
            }
        }
    }

    std::cout << "[QmioBackend] endpoint: " << (endpoint_.empty() ? "(none)" : endpoint_) << "\n";

    // Global deadline default: generous enough to never cut a
    // legitimate reconnect/retry sequence short, only a pathological
    // hang. Mirrors the reference client's own default formula.
    std::uint64_t attempts = static_cast<std::uint64_t>(max_retries_) + 1;
    std::uint64_t backoff_sum = static_cast<std::uint64_t>(retry_backoff_.count()) *
        (static_cast<std::uint64_t>(max_retries_) * (static_cast<std::uint64_t>(max_retries_) + 1) / 2);
    std::uint64_t default_global =
        attempts * 3 * static_cast<std::uint64_t>(recv_timeout_.count()) + backoff_sum;
    global_timeout_ = std::chrono::milliseconds(env_u64("QMIO_GLOBAL_TIMEOUT_MS", default_global));

    // device_available_ is set purely on whether we HAVE an endpoint to
    // try -- ZMQ TCP connect is lazy, so we can't (and don't try to)
    // verify reachability here. An unreachable-but-configured endpoint
    // surfaces its failure at submit time, as a real error -- see the
    // class doc's "out of scope" note on why that does NOT fall back to
    // CPU automatically.
    device_available_ = !endpoint_.empty();
}

QmioBackend::~QmioBackend() = default;

std::string QmioBackend::device_name() const {
    if (device_available_) {
        return "QMIO (" + endpoint_ + ")";
    }
    return "QMIO (fallback: " + fallback().device_name() + ")";
}

std::string QmioBackend::build_config_json(std::size_t shots) const {
    // Reproduces qmio's own request-config shape: mimics
    // qat.purr.compiler.config.CompilerConfig serialized via the
    // $type/$data/$value tagging scheme (see the class doc for the
    // provenance/verification caveat).
    auto [format_value, transforms_value] = results_format_values(results_format_);
    long long optimization_value = tket_opt_value(optimization_level_);

    json j;
    j["$type"] = "<class 'qat.purr.compiler.config.CompilerConfig'>";
    j["$data"]["repeats"] = shots;
    j["$data"]["repetition_period"] =
        repetition_period_.has_value() ? json(*repetition_period_) : json(nullptr);
    j["$data"]["results_format"] = {
        {"$type", "<class 'qat.purr.compiler.config.QuantumResultsFormat'>"},
        {"$data", {
            {"format", {
                {"$type", "<enum 'qat.purr.compiler.config.InlineResultsProcessing'>"},
                {"$value", format_value}
            }},
            {"transforms", {
                {"$type", "<enum 'qat.purr.compiler.config.ResultsFormatting'>"},
                {"$value", transforms_value}
            }}
        }}
    };
    j["$data"]["metrics"] = {
        {"$type", "<enum 'qat.purr.compiler.config.MetricsType'>"},
        {"$value", 6}
    };
    j["$data"]["active_calibrations"] = json::array();
    j["$data"]["optimizations"] = {
        {"$type", "<enum 'qat.purr.compiler.config.TketOptimizations'>"},
        {"$value", optimization_value}
    };
    return j.dump();
}

std::vector<std::uint8_t> QmioBackend::send_and_receive(
    const std::vector<std::uint8_t>& request) {
    auto deadline = std::chrono::steady_clock::now() + global_timeout_;
    std::string last_error;

    for (std::size_t attempt = 0; attempt <= max_retries_; ++attempt) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error(
                "QmioBackend: global timeout (" + std::to_string(global_timeout_.count()) +
                " ms) exceeded after " + std::to_string(attempt) + " attempt(s)" +
                (last_error.empty() ? "" : ("; last error: " + last_error)));
        }
        if (attempt > 0) {
            std::this_thread::sleep_for(retry_backoff_ * attempt);
        }

        // (Re)connect if we have no live socket -- a prior fault
        // discarded it (Lazy Pirate: a REQ socket is unusable after a
        // failed send/recv and must be recreated, not reused).
        if (!conn_->socket) {
            try {
                conn_->socket = make_socket(conn_->context, endpoint_, recv_timeout_);
            } catch (const std::exception& e) {
                last_error = std::string("connect failed: ") + e.what();
                conn_->socket.reset();
                continue; // safe to retry -- nothing was sent yet
            }
        }

        // Send, bounded by ZMQ_SNDTIMEO. A failure/timeout here means
        // the request never reached the peer (or we simply can't tell
        // whether it did before anything was sent), so retrying is
        // safe.
        zmq::message_t msg(request.begin(), request.end());
        try {
            // std::cout << "[QmioBackend] sending request of " << msg.size() << " bytes to " << endpoint_ << "\n";
            // std::cout << "[QmioBackend] request content: " << std::string(request.begin(), request.end()) << "\n";
            zmq::send_result_t sent = conn_->socket->send(msg, zmq::send_flags::none);
            if (!sent) {
                last_error = "send timed out after " + std::to_string(recv_timeout_.count()) + " ms";
                conn_->socket.reset();
                continue;
            }
        } catch (const std::exception& e) {
            last_error = std::string("send failed: ") + e.what();
            conn_->socket.reset();
            continue;
        }

        // Receive, bounded by ZMQ_RCVTIMEO. From here on, a failure
        // means the QPU MAY already have executed the request -- do
        // NOT retry (a blind resend risks a double execution on real
        // hardware). Discard the faulted socket and surface a terminal,
        // non-retried error instead.
        zmq::message_t reply;
        try {
            zmq::recv_result_t received = conn_->socket->recv(reply, zmq::recv_flags::none);
            if (!received) {
                conn_->socket.reset();
                throw std::runtime_error(
                    "QmioBackend: request was sent but no reply arrived within " +
                    std::to_string(recv_timeout_.count()) +
                    " ms; the QPU may have already executed it, so it was NOT resent "
                    "(to avoid a double execution). Treat this run's result as unknown "
                    "before resubmitting.");
            }
        } catch (const std::exception& e) {
            conn_->socket.reset();
            // Re-throw as-is if we already built the "no reply" message
            // above; otherwise this is a genuine recv-level fault.
            throw std::runtime_error(
                std::string(
                    "QmioBackend: request was sent but the reply could not be received (") +
                e.what() +
                "); the QPU may have already executed it, so it was NOT resent (to avoid "
                "a double execution). Treat this run's result as unknown before "
                "resubmitting.");
        }

        if (reply.size() > max_reply_bytes_) {
            throw std::runtime_error(
                "QmioBackend: reply of " + std::to_string(reply.size()) +
                " bytes exceeds the " + std::to_string(max_reply_bytes_) + "-byte safety limit");
        }
        const auto* bytes = static_cast<const std::uint8_t*>(reply.data());
        return std::vector<std::uint8_t>(bytes, bytes + reply.size());
    }

    throw std::runtime_error(
        "QmioBackend: exhausted " + std::to_string(max_retries_ + 1) +
        " attempt(s) against " + endpoint_ +
        (last_error.empty() ? "" : ("; last error: " + last_error)));
}

QPUBackend::SampleResult QmioBackend::submit_circuit(
    const std::vector<std::size_t>& qubits, std::size_t shots) {
    std::string program = to_qasm2(num_qubits_, gate_buffer_, qubits);
    std::string config_json = build_config_json(shots);

    // std::cout << "[QmioBackend] program:\n" << program << "\n";
    // std::cout << "[QmioBackend] config_json:\n" << config_json << "\n";

    auto request = safe_pickle::encode_str_tuple(program, config_json);
    auto reply_bytes = send_and_receive(request);

    json decoded = safe_pickle::decode(reply_bytes);
    // The reply is typically a pickled JSON string; a server that
    // pickles a dict directly instead is accepted as a defensive
    // fallback, matching the reference client.
    json reply_json = decoded.is_string() ? json::parse(decoded.get<std::string>()) : decoded;

    json counts = counts_from_json(reply_json);

    SampleResult result;
    result.outcomes.reserve(shots);
    for (auto& [bitstring, count] : counts.items()) {
        unsigned long long value = std::stoull(bitstring, nullptr, 2);
        std::uint64_t n = count.get<std::uint64_t>();
        for (std::uint64_t i = 0; i < n; ++i) {
            result.outcomes.push_back(value);
        }
    }
    return result;
}

} // namespace backends
} // namespace quantum
