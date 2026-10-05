#pragma once

#include "quantum/types.hpp"
#include "quantum/circuit.hpp"
#include "quantum/observable.hpp"
#include "quantum/vqa_types.hpp"
#include <vector>
#include <memory>
#include <string>

namespace quantum {

// Backend is the boundary between the device-agnostic runtime and a
// specific SYCL device implementation (CPU, GPU, FPGA, custom...).
// Every backend owns its own state vector and knows how to apply gates
// to it on its target device. The runtime never touches SYCL directly --
// only backends do.
class Backend {
public:
    virtual ~Backend() = default;

    // Allocate/reset the state vector to |0...0> for `num_qubits` qubits.
    virtual void initialize(std::size_t num_qubits) = 0;

    // Apply a single gate operation to the current state.
    virtual void apply_gate(const GateOp& op) = 0;

    // Pull the full state vector back to host memory (2^n amplitudes).
    // Intended for small circuits / debugging, not production-scale sims.
    virtual std::vector<Complex> get_state() const = 0;

    // Per-basis-state probabilities |amplitude|^2, length 2^n.
    virtual std::vector<double> probabilities() const = 0;

    // Sample `shots` measurements in the computational basis, restricted
    // to `qubits` (empty = all qubits). Returns one bitstring-as-int per shot.
    virtual std::vector<unsigned long long> sample(
        const std::vector<std::size_t>& qubits, std::size_t shots) = 0;

    // Human-readable identifier, e.g. "CPU (SYCL: Intel(R) Xeon(R) ...)".
    virtual std::string device_name() const = 0;

    // ------------------------------------------------------------------
    // Variational (VQA) hooks. ALL have working defaults built on the pure
    // virtuals above, so an existing/user-defined backend supports VQAs with
    // no changes. Override them progressively to go native:
    //   1. expectation()        -> reduce <H> on-device, skip get_state()
    //   2. expectation_batch()  -> evaluate many parameter sets in parallel
    //   3. run_native_vqa()     -> run the optimizer itself on the device
    // ------------------------------------------------------------------

    // <psi|H|psi> for the CURRENT state (after initialize()/apply_gate()).
    // Default: pull the state to host and evaluate there. Backends without a
    // state vector (QPUs) should override this with a shot-based estimate.
    virtual double expectation(const Observable& obs);

    // Evaluate `ansatz` for every request in `batch`, returning one <H> per
    // request. Default: run them one after another via initialize()/apply_gate()
    // /expectation(). Native backends should override with a batched kernel
    // (e.g. a [batch][2^n] USM state). Result order matches batch order.
    virtual std::vector<double> expectation_batch(const Circuit& ansatz,
                                                  const Observable& obs,
                                                  const EvalBatch& batch);

    // Full native optimization loop on the backend's own device. Return
    // false (default) if unsupported; VQA then runs the host loop instead.
    // If it returns true it must have filled `theta` and `result`.
    virtual bool run_native_vqa(const Circuit& ansatz, const Observable& obs,
                                const VQAOptions& options,
                                std::vector<double>& theta, VQAResult& result) {
        (void)ansatz; (void)obs; (void)options; (void)theta; (void)result;
        return false;
    }
};

} // namespace quantum
