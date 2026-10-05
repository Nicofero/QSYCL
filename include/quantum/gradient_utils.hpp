#pragma once

#include "quantum/circuit.hpp"
#include "quantum/vqa_types.hpp"
#include <vector>

namespace quantum {

struct GradientTerm { std::size_t param; double weight; };

// Builds the evaluation batch [plain energy, then shifted-pair requests...]
// for one gradient of `ansatz` at `theta`, plus the terms needed to turn
// the resulting energies into that gradient:
//   grad[terms[t].param] += terms[t].weight * (energies[1+2t] - energies[2+2t])
// energies[0] is the unshifted energy. Shared by the host VQA loop
// (vqa.cpp) and any native backend's on-device optimizer (native_vqa.cpp),
// so both use the identical shift/weight convention and agree on results.
EvalBatch build_gradient_batch(const Circuit& ansatz, const std::vector<double>& theta,
                               GradientMethod method, double fd_epsilon,
                               std::vector<GradientTerm>& terms);

} // namespace quantum
