#pragma once

#include "quantum/backend.hpp"
#include "quantum/circuit.hpp"
#include "quantum/observable.hpp"
#include "quantum/vqa_types.hpp"
#include <sycl/sycl.hpp>
#include <vector>

namespace quantum::native {

// Shared SYCL implementation of the batched-VQA backend hooks. Usable by
// ANY backend that can hand it a sycl::queue: it never touches a
// backend's persistent single-circuit state (state_dev_ / state_), it
// allocates its own [B][2^n] USM state for the lifetime of one call.
// CPUBackend and GPUBackend both delegate to these two functions from
// their Backend overrides -- only the queues passed in differ.
//
// run_native_vqa() takes the circuit-execution queue and the optimizer
// queue SEPARATELY on purpose: circuit_q runs expectation_batch()'s
// per-gate kernels (the expensive, O(2^n) part), optimizer_q runs the
// O(num_params) Adam/gradient-descent/SPSA update kernels. They are
// independent SYCL queues and may sit on entirely different devices
// (e.g. circuits on a GPU, the optimizer loop on a CPU).

// Evaluates `ansatz` for every request in `batch` (see EvalBatch), all in
// parallel on `q`. Throws std::invalid_argument on a size mismatch and
// std::invalid_argument for a gate type this kernel doesn't implement.
std::vector<double> expectation_batch(sycl::queue& q, const Circuit& ansatz,
                                      const Observable& obs, const EvalBatch& batch);

// Runs the full optimization loop with circuits evaluated as batched SYCL
// kernels on circuit_q (CPUBackend/GPUBackend's own [B][2^n] state-vector
// path from expectation_batch() above) and the optimizer's parameter-update
// kernels on optimizer_q. Always returns true (unlike Backend::run_native_vqa's
// default) -- callers only reach this once they've already decided to go
// native.
bool run_native_vqa(sycl::queue& circuit_q, sycl::queue& optimizer_q,
                    const Circuit& ansatz, const Observable& obs,
                    const VQAOptions& options,
                    std::vector<double>& theta, VQAResult& result);

// Same optimizer loop, but circuits are evaluated through `backend`'s OWN
// expectation_batch() instead of the SYCL state-vector kernel above. This
// is what lets the optimizer run on a SYCL device (optimizer_q -- CPU, GPU,
// whatever options.optimizer_device names) while circuits run wherever
// `backend` actually executes them: a real/remote QPU's shot-based
// expectation_batch() (see quantum/shot_expectation.hpp) for
// QPUBackend/CUNQABackend, or still the batched kernel above if `backend`
// happens to be a CPUBackend/GPUBackend. Always returns true.
bool run_native_vqa(Backend& backend, sycl::queue& optimizer_q,
                    const Circuit& ansatz, const Observable& obs,
                    const VQAOptions& options,
                    std::vector<double>& theta, VQAResult& result);

} // namespace quantum::native