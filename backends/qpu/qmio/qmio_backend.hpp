#pragma once

#include "qpu_backend.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace quantum {
namespace backends {

// QmioBackend talks to CESGA's QMIO real QPU directly over its ZeroMQ
// endpoint -- the QMIO-specific ("Layer 2") implementation sitting on
// top of QPUBackend's shared dispatcher logic ("Layer 1").
//
// # Wire protocol
//
// Ported from the reference client (polypus,
// github.com/Bahia-Software/polypus, crates/polypus-infrastructure/src/
// qmio.rs), whose header comment states this was verified against a
// live QMIO endpoint -- NOT independently re-verified here, so treat
// the specifics below the same way the CUNQA backend's ported schema
// was treated: trust but confirm against a real run before relying on
// it broadly.
//
//   - Transport: a ZeroMQ REQ socket connected to a fixed endpoint
//     (`tcp://host:port`). REQ is strict lock-step -- one request in
//     flight, one reply, repeat -- which is also why QMIO's
//     capabilities() below caps concurrency at 1.
//   - Request payload: a Python pickle (protocol 2) of the 2-tuple
//     (program, config_json) -- exactly what `socket.send_pyobj((circuit,
//     config))` produces on the Python side. `program` is an OpenQASM
//     2.0 text string; `config_json` is a JSON string mimicking
//     `qat.purr.compiler.config.CompilerConfig` via `$type`/`$data`/
//     `$value` tags (see build_config_json()).
//   - Reply: a pickle of the results -- typically a pickled JSON
//     string, with a pickled dict accepted as a defensive fallback.
//
// Encoding/decoding uses safe_pickle (safe_pickle.hpp): it only ever
// builds data, never executes anything, unlike Python's own
// pickle.loads().
//
// # Deliberately out of scope (documented gaps, not oversights)
//
//   - QIR text/bitcode program formats: only OpenQASM 2.0 is
//     implemented (see qasm2_export.hpp). Adding QIR means adding
//     another ProgramPayload variant and a pickle BYTES (not
//     BINUNICODE) encoding path for the bitcode case.
//   - Bit/qubit order of returned bitstrings vs. this project's own
//     convention: UNVERIFIED, exactly as the reference client flags it.
//     If results look reversed, that's the first thing to check.
//   - No automatic CPU fallback on a submission failure once the device
//     is considered available (see recheck_device() in the .cpp): a
//     failed real-hardware run surfaces as an exception rather than
//     silently substituting a local simulation -- a deliberate choice,
//     not an accident. Only a genuinely unconfigured endpoint (no
//     `$ZMQ_SERVER` and none passed explicitly) falls back to CPU.
class QmioBackend : public QPUBackend {
public:
    // `endpoint`: a `tcp://host:port` ZMQ address. If empty, read from
    // the `$ZMQ_SERVER` environment variable (the convention the
    // official qmio client itself uses). If neither is set, this
    // backend falls back to `fallback` immediately -- there is no
    // hardcoded default endpoint here, since baking in one site's
    // address would silently point at the wrong server everywhere else.
    explicit QmioBackend(std::string endpoint = "",
                         std::unique_ptr<Backend> fallback = nullptr);
    ~QmioBackend() override;

    std::string device_name() const override;

protected:
    SampleResult submit_circuit(const std::vector<std::size_t>& qubits,
                                 std::size_t shots) override;
    // Deliberately left at the base class's no-op default -- see the
    // class doc's "out of scope" note and the .cpp for the reasoning.

private:
    struct Connection; // owns the ZMQ REQ socket; defined in the .cpp
    std::unique_ptr<Connection> conn_;
    std::string endpoint_;

    // Timeouts/retry knobs. Defaults and env var names match the
    // reference client (QMIO_RECV_TIMEOUT_MS, QMIO_MAX_RETRIES,
    // QMIO_RETRY_BACKOFF_MS, QMIO_GLOBAL_TIMEOUT_MS) so an existing
    // QMIO deployment's tuning carries over unchanged.
    std::chrono::milliseconds recv_timeout_;
    std::chrono::milliseconds retry_backoff_;
    std::size_t max_retries_;
    std::chrono::milliseconds global_timeout_;
    std::size_t max_reply_bytes_ = 64 * 1024 * 1024; // 64 MiB safety cap

    // CompilerConfig knobs -- see build_config_json() in the .cpp for
    // what these map to on the wire.
    unsigned optimization_level_ = 1; // 0-3, see tket_opt_value() in .cpp
    std::string results_format_ = "binary_count"; // only this one is implemented
    std::optional<double> repetition_period_;

    std::string build_config_json(std::size_t shots) const;

    // Sends `request` and returns the raw reply bytes, implementing the
    // Lazy-Pirate retry pattern: connect/send failures are retried (the
    // request never left our socket, so retrying is safe); a receive
    // failure or timeout AFTER a successful send is NEVER retried and
    // throws immediately, since the device may already have executed
    // the request. Bounded by recv_timeout_ per operation and
    // global_timeout_ across all attempts.
    std::vector<std::uint8_t> send_and_receive(
        const std::vector<std::uint8_t>& request);
};

} // namespace backends
} // namespace quantum
