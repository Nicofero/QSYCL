#include "quantum/vqa.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace quantum {

namespace {
constexpr double kHalfPi = 1.57079632679489661923;

bool is_shiftable(GateType t) {
    return t == GateType::RX || t == GateType::RY || t == GateType::RZ;
}
} // namespace

VQA::VQA(QuantumRuntime& runtime, Circuit ansatz, Observable observable)
    : runtime_(runtime), ansatz_(std::move(ansatz)), observable_(std::move(observable)) {
    if (observable_.num_qubits() != ansatz_.num_qubits()) {
        throw std::invalid_argument("VQA: observable and ansatz must act on the same number of qubits");
    }
    if (ansatz_.num_params() == 0) {
        throw std::invalid_argument("VQA: ansatz has no parameters (use Circuit::new_param())");
    }
}

double VQA::cost(const std::vector<double>& theta) {
    EvalBatch b;
    b.add(theta);
    return runtime_.expectation_batch(ansatz_, observable_, b).at(0);
}

double VQA::energy_and_gradient(const std::vector<double>& theta, GradientMethod method,
                                double fd_eps, std::vector<double>& grad, std::size_t& evals) {
    const std::size_t P = ansatz_.num_params();
    grad.assign(P, 0.0);

    EvalBatch batch;
    batch.add(theta); // request 0: the plain energy

    struct Term { std::size_t param; double weight; }; // grad[param] += weight * (E+ - E-)
    std::vector<Term> terms;

    if (method == GradientMethod::ParameterShift) {
        const auto& ops = ansatz_.operations();
        for (std::size_t k = 0; k < ops.size(); ++k) {
            const auto& op = ops[k];
            if (!op.is_parametric()) continue;
            if (!is_shiftable(op.type)) {
                throw std::invalid_argument(
                    "VQA: parameter-shift supports only RX/RY/RZ; use GradientMethod::FiniteDifference "
                    "for gate '" + op.label + "'");
            }
            const Param& a = op.symbolic[0];
            if (!a.is_symbolic()) continue;
            // one gate occurrence contributes  scale * (E(phi+pi/2) - E(phi-pi/2)) / 2
            batch.add(theta, static_cast<int>(k), +kHalfPi);
            batch.add(theta, static_cast<int>(k), -kHalfPi);
            terms.push_back({static_cast<std::size_t>(a.index), 0.5 * a.scale});
        }
    } else {
        for (std::size_t i = 0; i < P; ++i) {
            std::vector<double> tp = theta, tm = theta;
            tp[i] += fd_eps; tm[i] -= fd_eps;
            batch.add(tp); batch.add(tm);
            terms.push_back({i, 0.5 / fd_eps});
        }
    }

    const auto e = runtime_.expectation_batch(ansatz_, observable_, batch);
    evals += batch.count;
    for (std::size_t t = 0; t < terms.size(); ++t) {
        grad[terms[t].param] += terms[t].weight * (e[1 + 2 * t] - e[2 + 2 * t]);
    }
    return e[0];
}

std::vector<double> VQA::gradient(const std::vector<double>& theta, GradientMethod method) {
    std::vector<double> g;
    std::size_t evals = 0;
    energy_and_gradient(theta, method, 1e-4, g, evals);
    return g;
}

VQAResult VQA::minimize(std::vector<double> theta, const VQAOptions& opt) {
    const std::size_t P = ansatz_.num_params();
    if (theta.size() != P) throw std::invalid_argument("VQA::minimize: theta0 size != number of ansatz parameters");

    VQAResult res;

    // 1) Backend-native optimization, if the backend offers it.
    if (opt.prefer_native && runtime_.backend().run_native_vqa(ansatz_, observable_, opt, theta, res)) {
        res.used_native = true;
        return res;
    }

    // 2) Host-driven loop with batched evaluations.
    std::mt19937_64 rng(opt.seed);
    std::vector<double> m(P, 0.0), v(P, 0.0), g;
    double prev = 0.0;

    for (std::size_t k = 0; k < opt.max_iterations; ++k) {
        double e = 0.0;
        if (opt.optimizer == OptimizerKind::SPSA) {
            const double ak = opt.learning_rate / std::pow(k + 1 + 0.1 * opt.max_iterations, 0.602);
            const double ck = opt.spsa_c / std::pow(k + 1.0, 0.101);
            std::vector<double> delta(P), tp = theta, tm = theta;
            for (std::size_t i = 0; i < P; ++i) {
                delta[i] = (rng() & 1ULL) ? 1.0 : -1.0;
                tp[i] += ck * delta[i]; tm[i] -= ck * delta[i];
            }
            EvalBatch b; b.add(tp); b.add(tm);
            const auto r = runtime_.expectation_batch(ansatz_, observable_, b);
            res.evaluations += 2;
            e = 0.5 * (r[0] + r[1]);
            for (std::size_t i = 0; i < P; ++i)
                theta[i] -= ak * (r[0] - r[1]) / (2.0 * ck * delta[i]);
        } else {
            e = energy_and_gradient(theta, opt.gradient, opt.fd_epsilon, g, res.evaluations);
            if (opt.optimizer == OptimizerKind::Adam) {
                const double b1t = 1.0 - std::pow(opt.beta1, k + 1.0);
                const double b2t = 1.0 - std::pow(opt.beta2, k + 1.0);
                for (std::size_t i = 0; i < P; ++i) {
                    m[i] = opt.beta1 * m[i] + (1.0 - opt.beta1) * g[i];
                    v[i] = opt.beta2 * v[i] + (1.0 - opt.beta2) * g[i] * g[i];
                    theta[i] -= opt.learning_rate * (m[i] / b1t) / (std::sqrt(v[i] / b2t) + opt.epsilon);
                }
            } else {
                for (std::size_t i = 0; i < P; ++i) theta[i] -= opt.learning_rate * g[i];
            }
        }
        res.history.push_back(e);
        res.iterations = k + 1;
        if (k > 0 && std::fabs(e - prev) < opt.tolerance) { res.converged = true; break; }
        prev = e;
    }

    res.theta = theta;
    res.energy = cost(theta);
    ++res.evaluations;
    return res;
}

} // namespace quantum
