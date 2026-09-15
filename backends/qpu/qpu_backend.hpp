#pragma once

#include "quantum/backend.hpp"

#include <memory>
#include <vector>

namespace quantum {
namespace backends {

// Shared orchestration layer for any backend that delegates circuit
// execution to a remote QPU -- real hardware or an emulator standing in
// for one -- rather than computing a state vector locally. This is the
// "dispatcher" layer: it owns everything that is true of *any* such
// device, so a concrete backend (QmioBackend, CunqaBackend, a future
// IBM/IonQ backend, ...) only has to implement the vendor-specific
// serialization/transport/parsing in submit_circuit().
//
// What's shared, and why it belongs here rather than in each backend:
//   - Gate buffering: no remote QPU executes gates one at a time: a
//     circuit is always compiled and submitted as a whole unit, so
//     apply_gate() always just appends to gate_buffer_, and submission
//     only happens lazily, inside sample().
//   - get_state()/probabilities() are unsupported whenever a real
//     device is in use: no real QPU can report exact amplitudes, so
//     there is nothing vendor-specific to differ here -- every remote
//     backend should refuse the same way.
//   - CPU fallback: if no device is available (never discovered, or a
//     submission fails at run time), every such backend should behave
//     the same way -- fall back to a local CPUBackend rather than
//     failing outright, and stay consistent about it (sample() must not
//     silently answer from a different simulation than probabilities()
//     would have, which is why get_state()/probabilities() ALSO check
//     device_available_ rather than always throwing).
//   - Retry ownership: sample() calls submit_circuit() exactly ONCE per
//     invocation and never retries it itself. A subclass whose protocol
//     makes some failures safe to retry (e.g. a connect/send failure)
//     and others definitely NOT safe to retry (QMIO: a receive failure
//     *after* a successful send, where the device may already have
//     executed the request -- a blind resend risks double-executing on
//     real hardware) must do that retrying internally, inside its own
//     submit_circuit(), before it ever throws out to this base class.
//     After a failure, this dispatcher only decides whether to fall
//     back to `fallback_` (via recheck_device()) or propagate the
//     error as-is -- it never second-guesses a subclass's own retry
//     decision by trying again itself.
//
// A subclass is responsible for its own device discovery/connection
// (setting device_available_ from its constructor) and for
// submit_circuit() -- the only vendor-specific piece of the contract.
class QPUBackend : public Backend {
public:
    // `fallback` is used whenever no device is available. Ownership is
    // taken so the fallback's lifetime matches this backend's.
    explicit QPUBackend(std::unique_ptr<Backend> fallback);
    ~QPUBackend() override;

    void initialize(std::size_t num_qubits) override;
    void apply_gate(const GateOp& op) override;
    std::vector<Complex> get_state() const override;
    std::vector<double> probabilities() const override;
    std::vector<unsigned long long> sample(
        const std::vector<std::size_t>& qubits, std::size_t shots) override;
    // device_name() is left pure virtual: every backend must say what
    // it actually is (and, by convention established in CUNQABackend,
    // reflect fallback state too -- see cunqa_backend.cpp for the
    // pattern: "<vendor> (fallback: <fallback device_name()>)").

protected:
    // Set by a subclass's own discovery/connection logic. Read here to
    // decide whether to delegate to the device or to `fallback_`.
    bool device_available_ = false;
    std::size_t num_qubits_ = 0;
    std::vector<GateOp> gate_buffer_;

    struct SampleResult {
        std::vector<unsigned long long> outcomes; // one entry per shot
    };

    // Vendor-specific core: serialize gate_buffer_ plus the qubits to
    // measure (already resolved to "every qubit" if the caller passed
    // an empty set -- see sample()) into that device's wire format,
    // submit it, and return the parsed measurement outcomes. Throw on
    // any failure (network, protocol, device-reported error) -- but see
    // the class-level "Retry ownership" note: do any retrying that is
    // safe for this device's protocol INSIDE this method before
    // throwing, since sample() will not retry it for you.
    virtual SampleResult submit_circuit(
        const std::vector<std::size_t>& qubits, std::size_t shots) = 0;

    // Called by sample() after submit_circuit() throws, to decide
    // whether the device should now be considered unavailable (causing
    // sample() to fall back to `fallback_`) or whether the failure
    // should simply be re-thrown to the caller as a real error.
    //
    // Default: no-op, i.e. device_available_ is left unchanged (true),
    // so by default a submission failure is surfaced as a real
    // exception rather than silently substituted with a different
    // simulation -- appropriate for a backend with no meaningful way to
    // "re-check" a single fixed real device (see QmioBackend, which
    // deliberately keeps this default: a QMIO run failing should be
    // visible, never silently answered by a local simulator instead).
    // A subclass that DOES have a real re-discovery mechanism (e.g.
    // CUNQA re-probing its vQPU registry, since "no longer found" is a
    // legitimate, common outcome there) overrides this to flip
    // device_available_ to false when appropriate.
    virtual void recheck_device() {}

    // Exposed so a subclass's device_name() can mention the fallback
    // when device_available_ is false, matching the established
    // CUNQABackend convention.
    const Backend& fallback() const { return *fallback_; }

private:
    std::unique_ptr<Backend> fallback_;
};

} // namespace backends
} // namespace quantum