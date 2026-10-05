#include "quantum/vqa.hpp"
#include "quantum/gradient_utils.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace quantum {

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
                                double fd_eps, std::size_t shots,
                                std::vector<double>& grad, std::size_t& evals) {
    grad.assign(ansatz_.num_params(), 0.0);

    std::vector<GradientTerm> terms;
    EvalBatch batch = build_gradient_batch(ansatz_, theta, method, fd_eps, terms);
    batch.shots = shots;

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
    energy_and_gradient(theta, method, 1e-4, EvalBatch{}.shots /* default shots */, g, evals);
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
            EvalBatch b; b.add(tp); b.add(tm); b.shots = opt.shots;
            const auto r = runtime_.expectation_batch(ansatz_, observable_, b);
            res.evaluations += 2;
            e = 0.5 * (r[0] + r[1]);
            for (std::size_t i = 0; i < P; ++i)
                theta[i] -= ak * (r[0] - r[1]) / (2.0 * ck * delta[i]);
        } else {
            e = energy_and_gradient(theta, opt.gradient, opt.fd_epsilon, opt.shots, g, res.evaluations);
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
    { EvalBatch final_batch; final_batch.add(theta); final_batch.shots = opt.shots;
      res.energy = runtime_.expectation_batch(ansatz_, observable_, final_batch).at(0); }
    ++res.evaluations;
    return res;
}

} // namespace quantum