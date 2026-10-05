#pragma once

#include "quantum/runtime.hpp"
#include "quantum/observable.hpp"
#include "quantum/vqa_types.hpp"

namespace quantum {

// Variational driver: minimizes <ansatz(theta)| H |ansatz(theta)> over theta.
//
//   Circuit ansatz(2); auto th = ansatz.new_params(2);
//   ansatz.ry(0, th[0]).ry(1, th[1]).cnot(0, 1);
//   Observable H(2); H.add_term(1.0, "ZZ").add_term(0.5, "XI");
//
//   QuantumRuntime rt(DeviceType::GPU);
//   VQA vqa(rt, ansatz, H);
//   auto res = vqa.minimize({0.1, 0.1});
//
// Execution strategy (chosen automatically, same user code everywhere):
//   1. backend.run_native_vqa()      -- whole loop on the backend's device
//   2. host loop + expectation_batch -- optimizer on host, evaluations batched
//   3. (default expectation_batch)   -- sequential evaluations, any backend
class VQA {
public:
    VQA(QuantumRuntime& runtime, Circuit ansatz, Observable observable);

    double cost(const std::vector<double>& theta);

    std::vector<double> gradient(const std::vector<double>& theta,
                                 GradientMethod method = GradientMethod::ParameterShift);

    VQAResult minimize(std::vector<double> theta0, const VQAOptions& options = {});

    const Circuit& ansatz() const { return ansatz_; }

private:
    // Builds the batch [plain evaluation, shifted evaluations...] and turns
    // the results into (energy, gradient) with ONE backend call.
    double energy_and_gradient(const std::vector<double>& theta, GradientMethod method,
                               double fd_eps, std::vector<double>& grad, std::size_t& evals);

    QuantumRuntime& runtime_;
    Circuit ansatz_;
    Observable observable_;
};

} // namespace quantum
