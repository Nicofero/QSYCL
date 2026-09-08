#pragma once

#include "quantum/backend.hpp"
#include "cpu_backend.hpp"

#include <memory>
#include <string>
#include <vector>

namespace quantum {
namespace backends {

// QPUBackend talks directly to an already-qraised qpu vQPU over its
// ZeroMQ wire protocol (a DEALER socket, one JSON message per circuit,
// one JSON message back). It does NOT link against qpu's C++ sources:
// qpu's own CMake install only exports its CLI tools (qraise/qdrop/
// qinfo/...) and its Python pybind extension -- none of its internal
// C++ headers or libraries (comm/client.hpp, sim/backend.hpp, etc.) are
// installed or exported via find_package(), so there is nothing stable
// to link against there. Instead this backend:
//
//   1. Reads $STORE/.qpu/qpus.json directly -- the same registry file
//      qpu's own get_QPUs() reads -- to discover an already-raised,
//      co-located vQPU.
//   2. Speaks qpu's wire protocol itself via libzmq, reproducing what
//      qpu.qclient.QClient does, without depending on qpu's build.
//   3. Builds the same JSON circuit-task schema qpu.qjob.QJob builds.
//
// Only depends on libzmq + a JSON library (nlohmann::json here) --
// both ordinary, independently versioned dependencies, decoupled from
// qpu's internal source layout and commit history.
//
// apply_gate() buffers ops rather than executing them: qpu only runs
// whole circuits as a unit. The circuit is built and submitted lazily,
// the first time sample() is called.
//
// get_state()/probabilities() are intentionally unsupported when
// running on a real vQPU (see .cpp for why) and fall through to
// CPUBackend when no vQPU was found at all.
class QPUBackend : public Backend {
public:
    QPUBackend();
    ~QPUBackend() override;

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

    // Reads $STORE/.qpu/qpus.json and connects to the first
    // vQPU found. Sets using_qpu_. Called at construction, and again
    // -- lazily -- if a submission later fails (e.g. a time-boxed
    // qraise allocation expired mid-session).
    void discover_qpu();

    // Flushes gate_buffer_ plus explicit measurements for `qubits`
    // (all qubits if empty) into qpu's JSON schema, sends it over the
    // vQPU's ZMQ socket, and blocks for the reply. Throws on failure.
    SampleResult run_on_qpu(const std::vector<std::size_t>& qubits,
                               std::size_t shots);

    std::size_t num_qubits_ = 0;
    std::vector<GateOp> gate_buffer_;

    bool using_qpu_ = false;
    std::unique_ptr<QpuHandle> qpu_;

    // Used whenever no qraised QPU is found, or a qpu submission fails.
    std::unique_ptr<Backend> fallback_;
};

} // namespace backends
} // namespace quantum