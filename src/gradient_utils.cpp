#include "quantum/gradient_utils.hpp"
#include <stdexcept>

namespace quantum {

namespace { constexpr double kHalfPi = 1.57079632679489661923; }

EvalBatch build_gradient_batch(const Circuit& ansatz, const std::vector<double>& theta,
                               GradientMethod method, double fd_epsilon,
                               std::vector<GradientTerm>& terms) {
    terms.clear();
    EvalBatch batch;
    batch.add(theta); // request 0: the plain energy

    if (method == GradientMethod::ParameterShift) {
        const auto& ops = ansatz.operations();
        for (std::size_t k = 0; k < ops.size(); ++k) {
            const auto& op = ops[k];
            if (!op.is_parametric()) continue;
            if (op.type != GateType::RX && op.type != GateType::RY && op.type != GateType::RZ) {
                throw std::invalid_argument(
                    "parameter-shift supports only RX/RY/RZ; use GradientMethod::FiniteDifference "
                    "for gate '" + op.label + "'");
            }
            const Param& a = op.symbolic[0];
            if (!a.is_symbolic()) continue;
            batch.add(theta, static_cast<int>(k), +kHalfPi);
            batch.add(theta, static_cast<int>(k), -kHalfPi);
            terms.push_back({static_cast<std::size_t>(a.index), 0.5 * a.scale});
        }
    } else {
        for (std::size_t i = 0; i < theta.size(); ++i) {
            std::vector<double> tp = theta, tm = theta;
            tp[i] += fd_epsilon; tm[i] -= fd_epsilon;
            batch.add(tp); batch.add(tm);
            terms.push_back({i, 0.5 / fd_epsilon});
        }
    }
    return batch;
}

} // namespace quantum
