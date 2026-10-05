#pragma once

#include <cstddef>
#include <vector>

namespace quantum {

enum class OptimizerKind {
    GradientDescent,
    Adam,
    SPSA   // 2 evaluations per step regardless of parameter count
};

enum class GradientMethod {
    ParameterShift,   // exact; RX/RY/RZ only. 2 evals per gate occurrence
    FiniteDifference  // approximate; works for any gate. 2 evals per parameter
};

struct VQAOptions {
    OptimizerKind  optimizer = OptimizerKind::Adam;
    GradientMethod gradient  = GradientMethod::ParameterShift;

    double learning_rate = 0.05;
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8; // Adam
    double spsa_c = 0.1;                               // SPSA perturbation size
    double fd_epsilon = 1e-4;                          // finite-difference step

    std::size_t max_iterations = 200;
    double tolerance = 1e-9;   // stop when |E_k - E_{k-1}| < tolerance
    unsigned long long seed = 12345;

    // If true, VQA first asks the backend to run the whole optimization
    // natively (on its own device). Backends that don't implement that hook
    // transparently fall back to the host-driven loop, which still batches
    // circuit evaluations through Backend::expectation_batch().
    bool prefer_native = true;
};

struct VQAResult {
    std::vector<double> theta;    // final parameters
    double energy = 0.0;          // cost at `theta`
    std::size_t iterations = 0;
    std::size_t evaluations = 0;  // total circuit executions
    std::vector<double> history;  // cost per iteration
    bool converged = false;
    bool used_native = false;     // true if the backend ran the optimizer itself
};

// A batch of circuit evaluations of ONE ansatz, in a flat, device-friendly
// layout: request i uses thetas[i*num_params ... (i+1)*num_params) and,
// optionally, adds shift[i] to the first angle of gate shift_op[i].
// (Exactly the semantics of Circuit::bind(theta, shift_op, shift).)
struct EvalBatch {
    std::size_t count = 0;
    std::size_t num_params = 0;
    std::vector<double> thetas;   // count * num_params, row-major
    std::vector<int>    shift_op; // empty, or one entry per request (-1 = none)
    std::vector<double> shift;    // empty, or one entry per request

    void add(const std::vector<double>& theta, int op = -1, double delta = 0.0) {
        if (count == 0) num_params = theta.size();
        thetas.insert(thetas.end(), theta.begin(), theta.end());
        shift_op.push_back(op);
        shift.push_back(delta);
        ++count;
    }
    std::vector<double> theta_of(std::size_t i) const {
        return {thetas.begin() + i * num_params, thetas.begin() + (i + 1) * num_params};
    }
};

} // namespace quantum
