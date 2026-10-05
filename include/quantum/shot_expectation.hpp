#pragma once

#include "quantum/backend.hpp"
#include "quantum/circuit.hpp"
#include "quantum/observable.hpp"
#include "quantum/vqa_types.hpp"
#include <cstddef>
#include <vector>

namespace quantum::native {

// Partitions obs.masks() into qubit-wise-commuting (QWC) groups: within one
// group, every pair of terms either doesn't touch a given qubit, or touches
// it with the IDENTICAL Pauli (same X, Y or Z there), so one measurement
// circuit (one basis rotation + one sample() call) is enough for the whole
// group. Greedy, not a globally-optimal grouping -- fine for the handful of
// terms a typical VQA Hamiltonian has; a smarter grouping (max-clique over
// the QWC compatibility graph) is a reasonable follow-up if that ever stops
// being true.
std::vector<std::vector<std::size_t>> group_qwc(const Observable& obs);

// Evaluates <ansatz(theta)| H |ansatz(theta)> (optionally with the
// parameter-shift rule's extra `shift` on gate `shift_op`'s first angle,
// matching Circuit::bind's convention) by SAMPLING, using nothing but
// Backend's public interface (initialize/apply_gate/sample). Works for any
// backend, but exists for the ones where get_state() isn't available
// (QPUBackend and CUNQABackend on real/remote hardware).
//
// Re-runs the full circuit once per QWC group: a real or remote device
// can't rewind a mid-circuit state to try a different measurement basis,
// so this is necessary, not just simple -- see the comment in
// qpu_backend.cpp / cunqa_backend.cpp for why expectation_batch() is the
// right hook to put this behind rather than Backend::expectation().
double shot_based_expectation(Backend& backend, const Circuit& ansatz,
                              const std::vector<double>& theta, int shift_op, double shift,
                              const Observable& obs, std::size_t shots);

// Backend::expectation_batch() built on shot_based_expectation(). What
// QPUBackend/CUNQABackend delegate their override to.
std::vector<double> shot_based_expectation_batch(Backend& backend, const Circuit& ansatz,
                                                  const Observable& obs, const EvalBatch& batch,
                                                  std::size_t shots);

} // namespace quantum::native