#pragma once

#include "quantum/backend.hpp"
#include "backends/cpu/cpu_backend.hpp"

#include <memory>
#include <string>
#include <vector>

namespace quantum {

// CUNQABackend talks directly to an already-qraised cunqa vQPU over its
// ZeroMQ wire protocol (a DEALER socket, one JSON message per circuit,
// one JSON message back). It does NOT link against cunqa's C++ sources:
// cunqa's own CMake install only exports its CLI tools (qraise/qdrop/
// qinfo/...) and its Python pybind extension -- none of its internal
// C++ headers or libraries (comm/client.hpp, sim/backend.hpp, etc.) are
// installed or exported via find_package(), so there is nothing stable
// to link against there. Instead this backend:
//
//   1. Reads $STORE/.cunqa/qpus.json directly -- the same registry file
//      cunqa's own get_QPUs() reads -- to discover an already-raised,
//      co-located vQPU.
//   2. Speaks cunqa's wire protocol itself via libzmq, reproducing what
//      cunqa.qclient.QClient does, without depending on cunqa's build.
//   3. Builds the same JSON circuit-task schema cunqa.qjob.QJob builds.
//
// Only depends on libzmq + a JSON library (nlohmann::json here) --
// both ordinary, independently versioned dependencies, decoupled from
// cunqa's internal source layout and commit history.
//
// apply_gate() buffers ops rather than executing them: cunqa only runs
// whole circuits as a unit. The circuit is built and submitted lazily,
// the first time sample() is called.
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

private:
    struct QpuHandle;   // registry entry + open ZMQ DEALER socket, in the .cpp
    struct SampleResult { std::vector<unsigned long long> outcomes; };

    // Reads $STORE/.cunqa/qpus.json and connects to the first co-located
    // vQPU found. Sets using_cunqa_. Called at construction, and again
    // -- lazily -- if a submission later fails (e.g. a time-boxed
    // qraise allocation expired mid-session).
    void discover_qpu();

    // Flushes gate_buffer_ plus explicit measurements for `qubits`
    // (all qubits if empty) into cunqa's JSON schema, sends it over the
    // vQPU's ZMQ socket, and blocks for the reply. Throws on failure.
    SampleResult run_on_cunqa(const std::vector<std::size_t>& qubits,
                               std::size_t shots);

    std::size_t num_qubits_ = 0;
    std::vector<GateOp> gate_buffer_;

    bool using_cunqa_ = false;
    std::unique_ptr<QpuHandle> qpu_;

    // Used whenever no qraised QPU is found, or a cunqa submission fails.
    std::unique_ptr<Backend> fallback_;
};

} // namespace quantum
