#pragma once

#include "quantum/backend.hpp"
#include "cpu_backend.hpp"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace quantum {
namespace backends {

// CUNQABackend talks directly to already-qraised cunqa vQPUs over their
// ZeroMQ wire protocol (a DEALER socket per vQPU, one JSON message per
// circuit, one JSON message back). It does NOT link against cunqa's C++
// sources: cunqa's own CMake install only exports its CLI tools (qraise/
// qdrop/qinfo/...) and its Python pybind extension -- none of its internal
// C++ headers or libraries (comm/client.hpp, sim/backend.hpp, etc.) are
// installed or exported via find_package(), so there is nothing stable
// to link against there. Instead this backend:
//
//   1. Reads $STORE/.cunqa/qpus.json directly -- the same registry file
//      cunqa's own get_QPUs() reads -- and connects to EVERY vQPU listed
//      there (one DEALER socket each, sharing one ZMQ context).
//   2. Speaks cunqa's wire protocol itself via libzmq, reproducing what
//      cunqa.qclient.QClient does, without depending on cunqa's build.
//   3. Builds the same JSON circuit-task schema cunqa.qjob.QJob builds.
//
// Only depends on libzmq + a JSON library (nlohmann::json here) --
// both ordinary, independently versioned dependencies, decoupled from
// cunqa's internal source layout and commit history.
//
// Two execution paths:
//   - Single circuit (initialize()/apply_gate()/sample()): apply_gate()
//     buffers ops (cunqa only runs whole circuits as a unit) and sample()
//     submits them to ONE vQPU, chosen round-robin among those large
//     enough for the circuit.
//   - Batched (expectation_batch(), and therefore run_native_vqa()): every
//     (parameter set x QWC measurement group) circuit of the batch is
//     built up front and the whole set is load-balanced across ALL
//     eligible vQPUs at once -- one circuit in flight per vQPU, the next
//     one sent to whichever vQPU replies first. See dispatch().
//
// get_state()/probabilities() are intentionally unsupported when
// running on a real vQPU (see .cpp for why) and fall through to
// CPUBackend when no vQPU was found at all.
class CUNQABackend : public Backend {
public:
    CUNQABackend();
    ~CUNQABackend() override;

    void initialize(std::size_t num_qubits) override;
    void apply_gate(const GateOp& op) override;
    std::vector<Complex> get_state() const override;
    std::vector<double> probabilities() const override;
    std::vector<unsigned long long> sample(
        const std::vector<std::size_t>& qubits, std::size_t shots) override;
    std::string device_name() const override;

    // get_state() is unsupported on a real vQPU, so the default
    // Backend::expectation_batch (which reads the state back) won't work
    // there either. Shot-based estimate instead (quantum/shot_expectation.hpp),
    // with all of the batch's circuits distributed across every eligible
    // vQPU concurrently. Falls through to `fallback_` when no vQPU was
    // found, same as every other method here.
    std::vector<double> expectation_batch(const Circuit& ansatz, const Observable& obs,
                                          const EvalBatch& batch) override;

    // Optimizer arithmetic (Adam/GD/SPSA) runs as SYCL kernels on
    // options.optimizer_device; circuits still go through this backend's
    // own expectation_batch() (shots spread over the vQPUs). Falls through
    // to `fallback_` when no vQPU was found.
    bool run_native_vqa(const Circuit& ansatz, const Observable& obs,
                        const VQAOptions& options,
                        std::vector<double>& theta, VQAResult& result) override;

    // Number of vQPUs currently connected and considered healthy.
    std::size_t num_vqpus() const;

private:
    struct QpuHandle;   // registry entry + open ZMQ DEALER socket, in the .cpp
    struct QpuPool;     // shared ZMQ context + every QpuHandle, in the .cpp

    // Reads $STORE/.cunqa/qpus.json and connects to every vQPU found,
    // replacing any previous pool. Sets using_cunqa_. Called at
    // construction, and again -- lazily -- if submissions later fail
    // (e.g. a time-boxed qraise allocation expired mid-session).
    void discover_qpus();

    // Takes a vQPU out for the rest of this backend's life, including across
    // registry re-reads (a killed vQPU may stay listed there).
    void mark_dead(QpuHandle& qpu);

    // Healthy vQPUs whose qubit count can hold an n-qubit circuit.
    std::vector<QpuHandle*> eligible_qpus(std::size_t n) const;

    // Submits `gates` (+ explicit measurements of `qubits`, all qubits if
    // empty) to `qpu` and blocks for the reply. Throws on failure.
    std::vector<unsigned long long> run_on_cunqa(QpuHandle& qpu,
                                                 const std::vector<GateOp>& gates,
                                                 const std::vector<std::size_t>& qubits,
                                                 std::size_t shots);

    // Runs every circuit in `circuits` (all qubits measured, `shots` each)
    // across all eligible vQPUs concurrently and returns their outcomes in
    // the same order. A vQPU that fails to accept a circuit, or (if
    // $CUNQA_RECV_TIMEOUT_MS is set) doesn't answer in time, is dropped
    // and its circuit is re-queued for the others; a vQPU-reported ERROR
    // is a property of the circuit, not the vQPU, and is thrown as-is.
    // Never falls back to the CPU: mixing emulated and locally simulated
    // energies inside one batch would be silently wrong.
    std::vector<std::vector<unsigned long long>> dispatch(
        const std::vector<Circuit>& circuits, std::size_t shots);

    std::size_t num_qubits_ = 0;
    std::vector<GateOp> gate_buffer_;

    bool using_cunqa_ = false;
    std::unique_ptr<QpuPool> pool_;
    std::size_t next_qpu_ = 0; // round-robin cursor for single-circuit sample()
    std::set<std::string> dead_endpoints_; // vQPUs seen dying; never reconnected

    // Used whenever no qraised QPU is found, or a cunqa submission fails.
    std::unique_ptr<Backend> fallback_;
};

} // namespace backends
} // namespace quantum
