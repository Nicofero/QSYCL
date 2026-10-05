#include "quantum/native_vqa.hpp"
#include "quantum/gradient_utils.hpp"
#include "quantum/device_selector.hpp"
#include "quantum/gates.hpp"
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>

namespace quantum::native {

namespace {

// --- Compiled, device-friendly gate program -----------------------------
// One op covers every 1- or 2-qubit gate except SWAP: a (possibly
// controlled) 2x2 matrix applied to `target`. For FIXED gates (H, X, Y, Z,
// S, Sdg, T, Tdg, CNOT, CZ) the matrix is baked in on the host, once, since
// it does not depend on theta. For ROTATING gates (RX, RY, RZ, U, CRX, CRY,
// CRZ) the matrix is rebuilt per batch row from up to 3 angle descriptors,
// each of which is either a constant or `scale * theta[idx] + offset`
// (mirrors Param::resolve, plus the parameter-shift rule's extra `+ shift`
// on the first angle of exactly one gate occurrence per shifted request).
enum class Shape : int { Fixed = 0, RX = 1, RY = 2, RZ = 3, U3 = 4 };

struct AngleRef {
    int index = -1;   // -1 = constant (use offset only)
    double scale = 1.0;
    double offset = 0.0;
};

struct DevOp {
    bool is_swap = false;
    int target = -1;      // matrix ops
    int control = -1;      // -1 = uncontrolled
    int qa = -1, qb = -1;  // swap qubits
    Shape shape = Shape::Fixed;
    Complex m0, m1, m2, m3;  // used when shape == Fixed
    AngleRef a0, a1, a2;     // used otherwise (a1/a2 only for U3)
};

AngleRef to_angle_ref(const Param& p) {
    return AngleRef{p.index, p.scale, p.offset};
}

std::vector<DevOp> compile(const Circuit& ansatz) {
    using namespace quantum::gates;
    std::vector<DevOp> prog;
    prog.reserve(ansatz.operations().size());

    for (const auto& op : ansatz.operations()) {
        DevOp d;
        auto fixed = [&](const Matrix2x2& m, int ctrl = -1) {
            d.shape = Shape::Fixed; d.m0 = m[0]; d.m1 = m[1]; d.m2 = m[2]; d.m3 = m[3];
            d.control = ctrl;
        };
        switch (op.type) {
            case GateType::H:  d.target = op.qubits[0]; fixed(H()); break;
            case GateType::X:  d.target = op.qubits[0]; fixed(X()); break;
            case GateType::Y:  d.target = op.qubits[0]; fixed(Y()); break;
            case GateType::Z:  d.target = op.qubits[0]; fixed(Z()); break;
            case GateType::S:  d.target = op.qubits[0]; fixed(op.dagger ? Sdg() : S()); break;
            case GateType::T:  d.target = op.qubits[0]; fixed(op.dagger ? Tdg() : T()); break;
            case GateType::CNOT: d.target = op.qubits[1]; fixed(X(), static_cast<int>(op.qubits[0])); break;
            case GateType::CZ:   d.target = op.qubits[1]; fixed(Z(), static_cast<int>(op.qubits[0])); break;

            case GateType::RX: d.target = op.qubits[0]; d.shape = Shape::RX;
                                d.a0 = op.is_parametric() ? to_angle_ref(op.symbolic[0]) : AngleRef{-1, 0, op.params[0]}; break;
            case GateType::RY: d.target = op.qubits[0]; d.shape = Shape::RY;
                                d.a0 = op.is_parametric() ? to_angle_ref(op.symbolic[0]) : AngleRef{-1, 0, op.params[0]}; break;
            case GateType::RZ: d.target = op.qubits[0]; d.shape = Shape::RZ;
                                d.a0 = op.is_parametric() ? to_angle_ref(op.symbolic[0]) : AngleRef{-1, 0, op.params[0]}; break;
            case GateType::U:  d.target = op.qubits[0]; d.shape = Shape::U3;
                                for (int i = 0; i < 3; ++i) {
                                    AngleRef ar = (i < static_cast<int>(op.symbolic.size()) && op.symbolic[i].is_symbolic())
                                                  ? to_angle_ref(op.symbolic[i]) : AngleRef{-1, 0, op.params[i]};
                                    (i == 0 ? d.a0 : i == 1 ? d.a1 : d.a2) = ar;
                                } break;
            case GateType::CRX: d.target = op.qubits[1]; d.control = static_cast<int>(op.qubits[0]); d.shape = Shape::RX;
                                d.a0 = op.is_parametric() ? to_angle_ref(op.symbolic[0]) : AngleRef{-1, 0, op.params[0]}; break;
            case GateType::CRY: d.target = op.qubits[1]; d.control = static_cast<int>(op.qubits[0]); d.shape = Shape::RY;
                                d.a0 = op.is_parametric() ? to_angle_ref(op.symbolic[0]) : AngleRef{-1, 0, op.params[0]}; break;
            case GateType::CRZ: d.target = op.qubits[1]; d.control = static_cast<int>(op.qubits[0]); d.shape = Shape::RZ;
                                d.a0 = op.is_parametric() ? to_angle_ref(op.symbolic[0]) : AngleRef{-1, 0, op.params[0]}; break;

            case GateType::SWAP: d.is_swap = true; d.qa = static_cast<int>(op.qubits[0]); d.qb = static_cast<int>(op.qubits[1]); break;
            case GateType::MEASURE: continue; // no-op for a state-vector expectation value
            default:
                throw std::invalid_argument("native::expectation_batch: unsupported gate '" + op.label + "'");
        }
        prog.push_back(d);
    }
    return prog;
}

inline int popcount64(std::uint64_t v) {
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return static_cast<int>((v * 0x0101010101010101ULL) >> 56);
}

} // namespace

std::vector<double> expectation_batch(sycl::queue& q, const Circuit& ansatz,
                                      const Observable& obs, const EvalBatch& batch) {
    if (obs.num_qubits() != ansatz.num_qubits())
        throw std::invalid_argument("native::expectation_batch: observable/ansatz qubit mismatch");
    if (batch.count != 0 && batch.num_params != ansatz.num_params())
        throw std::invalid_argument("native::expectation_batch: batch parameter count mismatch");

    const std::size_t n = ansatz.num_qubits();
    const std::size_t dim = std::size_t(1) << n;
    const std::size_t B = batch.count;
    const std::size_t P = batch.num_params;
    std::vector<double> out(B, 0.0);
    if (B == 0) return out;

    const auto prog = compile(ansatz);
    const auto& masks = obs.masks();
    const std::size_t T = masks.size();

    // A dedicated in-order queue on the SAME device/context as `q`: the
    // per-gate kernels below have a true data dependency on one another
    // (each reads the state the previous one wrote), and USM operations on
    // an out-of-order queue carry no implicit ordering. This sidesteps
    // manual event-chaining while `q` itself stays free to be out-of-order
    // for whatever else the backend uses it for.
    sycl::queue eq(q.get_context(), q.get_device(), sycl::property::queue::in_order());

    Complex* state = sycl::malloc_device<Complex>(B * dim, eq);
    double* theta_dev = P ? sycl::malloc_device<double>(B * P, eq) : nullptr;
    int* shift_op_dev = sycl::malloc_device<int>(B, eq);
    double* shift_dev = sycl::malloc_device<double>(B, eq);
    PauliMask* mask_dev = T ? sycl::malloc_device<PauliMask>(T, eq) : nullptr;
    double* energy_dev = sycl::malloc_device<double>(B, eq);
    if (!state || (P && !theta_dev) || !shift_op_dev || !shift_dev || (T && !mask_dev) || !energy_dev) {
        auto cleanup = [&] {
            sycl::free(state, eq); sycl::free(theta_dev, eq); sycl::free(shift_op_dev, eq);
            sycl::free(shift_dev, eq); sycl::free(mask_dev, eq); sycl::free(energy_dev, eq);
        };
        cleanup();
        throw std::runtime_error("native::expectation_batch: sycl::malloc_device failed (out of device memory?)");
    }

    // NOTE: these two host-side staging buffers must stay alive until the
    // async memcpy below has actually completed (the final .wait() at the
    // end of this function), not just until it's submitted -- an in-order
    // queue only orders device-side execution, it says nothing about when
    // the host-memory source of a memcpy is actually read. Destroying them
    // right after submission (as an earlier version of this function did,
    // by scoping them to a nested block) is a use-after-free: the copy can
    // read freed/reused host memory, silently corrupting shift_op/shift
    // with garbage for some batch rows. That garbage then makes the
    // parameter-shift rule skip the shift for a random subset of requests,
    // which is exactly what produced the irreproducible, sometimes-zero
    // gradients. Keep them at function scope so they outlive eq.wait().
    std::vector<int> sop(B, -1);
    std::vector<double> sval(B, 0.0);
    for (std::size_t b = 0; b < B; ++b) {
        if (!batch.shift_op.empty()) sop[b] = batch.shift_op[b];
        if (!batch.shift.empty())    sval[b] = batch.shift[b];
    }
    eq.memcpy(shift_op_dev, sop.data(), B * sizeof(int));
    eq.memcpy(shift_dev, sval.data(), B * sizeof(double));
    if (T) eq.memcpy(mask_dev, masks.data(), T * sizeof(PauliMask));

    // |0...0> for every row of the batch.
    eq.parallel_for(sycl::range<1>(B * dim), [=](sycl::id<1> idx) { state[idx[0]] = Complex(0.0, 0.0); });
    eq.parallel_for(sycl::range<1>(B), [=](sycl::id<1> b) { state[b[0] * dim] = Complex(1.0, 0.0); });

    auto eval_angle = [](const AngleRef& a, const double* th, std::size_t row, std::size_t P) -> double {
        return a.index >= 0 ? a.scale * th[row * P + static_cast<std::size_t>(a.index)] + a.offset : a.offset;
    };

    for (std::size_t k = 0; k < prog.size(); ++k) {
        const DevOp d = prog[k];
        const int op_index = static_cast<int>(k);

        if (d.is_swap) {
            const std::uint64_t mask_a = std::uint64_t(1) << d.qa, mask_b = std::uint64_t(1) << d.qb;
            eq.parallel_for(sycl::range<2>(B, dim), [=](sycl::id<2> id) {
                const std::size_t b = id[0], i = id[1];
                if ((i & mask_a) == 0 && (i & mask_b) != 0) {
                    const std::size_t j = (i | mask_a) & ~mask_b;
                    Complex* row = state + b * dim;
                    Complex tmp = row[i]; row[i] = row[j]; row[j] = tmp;
                }
            });
            continue;
        }

        const std::uint64_t mask_t = std::uint64_t(1) << d.target;
        const std::uint64_t mask_c = d.control >= 0 ? (std::uint64_t(1) << d.control) : 0;
        const bool controlled = d.control >= 0;

        eq.parallel_for(sycl::range<2>(B, dim / 2), [=](sycl::id<2> id) {
            const std::size_t b = id[0], k2 = id[1];
            const std::size_t low = k2 & (mask_t - 1);
            const std::size_t high = k2 & ~(mask_t - 1);
            const std::size_t i0 = (high << 1) | low;
            const std::size_t i1 = i0 | mask_t;
            if (controlled && !(i0 & mask_c)) return;

            Complex m0, m1, m2, m3;
            if (d.shape == Shape::Fixed) {
                m0 = d.m0; m1 = d.m1; m2 = d.m2; m3 = d.m3;
            } else {
                double v0 = eval_angle(d.a0, theta_dev, b, P);
                if (shift_op_dev[b] == op_index) v0 += shift_dev[b];
                if (d.shape == Shape::RX) {
                    const double c = sycl::cos(v0 / 2), s = sycl::sin(v0 / 2);
                    m0 = Complex(c, 0); m1 = Complex(0, -s); m2 = Complex(0, -s); m3 = Complex(c, 0);
                } else if (d.shape == Shape::RY) {
                    const double c = sycl::cos(v0 / 2), s = sycl::sin(v0 / 2);
                    m0 = Complex(c, 0); m1 = Complex(-s, 0); m2 = Complex(s, 0); m3 = Complex(c, 0);
                } else if (d.shape == Shape::RZ) {
                    const double c = sycl::cos(v0 / 2), s = sycl::sin(v0 / 2);
                    m0 = Complex(c, -s); m1 = Complex(0, 0); m2 = Complex(0, 0); m3 = Complex(c, s);
                } else { // U3
                    const double v1 = eval_angle(d.a1, theta_dev, b, P);
                    const double v2 = eval_angle(d.a2, theta_dev, b, P);
                    const double c = sycl::cos(v0 / 2), s = sycl::sin(v0 / 2);
                    const Complex ephi(sycl::cos(v1), sycl::sin(v1));
                    const Complex elam(sycl::cos(v2), sycl::sin(v2));
                    const Complex esum(sycl::cos(v1 + v2), sycl::sin(v1 + v2));
                    m0 = Complex(c, 0); m1 = Complex(-s, 0) * elam; m2 = Complex(s, 0) * ephi; m3 = Complex(c, 0) * esum;
                }
            }
            Complex* row = state + b * dim;
            const Complex a0v = row[i0], a1v = row[i1];
            row[i0] = m0 * a0v + m1 * a1v;
            row[i1] = m2 * a0v + m3 * a1v;
        });
    }

    // One work-item per batch row: sequential over dim * T. Parallel across
    // the batch (the main axis of interest for VQAs); a further work-group
    // reduction across `dim` within a row is a natural follow-up if a
    // single row's cost ever dominates.
    eq.parallel_for(sycl::range<1>(B), [=](sycl::id<1> id) {
        const std::size_t b = id[0];
        const Complex* row = state + b * dim;
        double total = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            const PauliMask term = mask_dev[t];
            double re = 0.0, im = 0.0;
            for (std::size_t j = 0; j < dim; ++j) {
                const bool neg = popcount64(j & term.z) & 1;
                const Complex conj_psi_x = Complex(row[j ^ term.x].re, -row[j ^ term.x].im);
                const Complex prod = conj_psi_x * row[j];
                re += neg ? -prod.re : prod.re;
                im += neg ? -prod.im : prod.im;
            }
            switch (term.ny & 3) {
                case 0: break;
                case 1: { const double r = -im; im = re; re = r; break; }
                case 2: re = -re; im = -im; break;
                case 3: { const double r = im; im = -re; re = r; break; }
            }
            total += term.coeff * re;
        }
        energy_dev[b] = total;
    });

    eq.memcpy(out.data(), energy_dev, B * sizeof(double)).wait();

    sycl::free(state, eq); sycl::free(theta_dev, eq); sycl::free(shift_op_dev, eq);
    sycl::free(shift_dev, eq); sycl::free(mask_dev, eq); sycl::free(energy_dev, eq);
    return out;
}

namespace {

// theta[i] -= lr * grad[i]  (plain gradient descent, and reused for SPSA
// with lr = a_k and grad already holding the SPSA gradient estimate).
void sgd_step(sycl::queue& oq, double* theta, const double* grad, std::size_t P, double lr) {
    oq.parallel_for(sycl::range<1>(P), [=](sycl::id<1> i) { theta[i] -= lr * grad[i]; }).wait();
}

// Adam, bias-corrected, state (m, v) kept resident on `oq` across iterations.
void adam_step(sycl::queue& oq, double* theta, double* m, double* v, const double* grad,
               std::size_t P, double lr, double b1, double b2, double eps, std::size_t k) {
    const double b1t = 1.0 - std::pow(b1, static_cast<double>(k + 1));
    const double b2t = 1.0 - std::pow(b2, static_cast<double>(k + 1));
    oq.parallel_for(sycl::range<1>(P), [=](sycl::id<1> i) {
        m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
        v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
        const double mhat = m[i] / b1t;
        const double vhat = v[i] / b2t;
        theta[i] -= lr * mhat / (sycl::sqrt(vhat) + eps);
    }).wait();
}

} // namespace

namespace {

// Shared by both public run_native_vqa overloads below. `evaluate` is the
// ONLY thing that differs between them: how a batch of circuits gets
// turned into a batch of energies. Everything else -- the optimizer state
// living on optimizer_q, the Adam/GD/SPSA math, the loop/tolerance/history
// bookkeeping -- is identical regardless of where or how circuits run.
bool run_native_vqa_core(const std::function<std::vector<double>(const EvalBatch&)>& evaluate,
                         sycl::queue& optimizer_q,
                         const Circuit& ansatz, const VQAOptions& opt,
                         std::vector<double>& theta, VQAResult& result) {
    const std::size_t P = ansatz.num_params();
    if (theta.size() != P) throw std::invalid_argument("native::run_native_vqa: theta0 size != ansatz.num_params()");

    // Optimizer state lives on optimizer_q for the whole loop; only theta
    // and the (tiny, size-P) gradient cross the host boundary each
    // iteration -- the O(2^n) (or O(shots)) circuit work never does.
    double* theta_dev = sycl::malloc_device<double>(P, optimizer_q);
    double* grad_dev  = sycl::malloc_device<double>(P, optimizer_q);
    double* m_dev = (opt.optimizer == OptimizerKind::Adam) ? sycl::malloc_device<double>(P, optimizer_q) : nullptr;
    double* v_dev = (opt.optimizer == OptimizerKind::Adam) ? sycl::malloc_device<double>(P, optimizer_q) : nullptr;
    if (!theta_dev || !grad_dev || (opt.optimizer == OptimizerKind::Adam && (!m_dev || !v_dev))) {
        sycl::free(theta_dev, optimizer_q); sycl::free(grad_dev, optimizer_q);
        sycl::free(m_dev, optimizer_q); sycl::free(v_dev, optimizer_q);
        throw std::runtime_error("native::run_native_vqa: sycl::malloc_device failed (out of device memory?)");
    }
    optimizer_q.memcpy(theta_dev, theta.data(), P * sizeof(double));
    if (m_dev) optimizer_q.memset(m_dev, 0, P * sizeof(double));
    if (v_dev) optimizer_q.memset(v_dev, 0, P * sizeof(double));
    optimizer_q.wait();

    std::mt19937_64 rng(opt.seed);
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
            const auto r = evaluate(b);
            result.evaluations += 2;
            e = 0.5 * (r[0] + r[1]);
            std::vector<double> ghat(P);
            for (std::size_t i = 0; i < P; ++i) ghat[i] = (r[0] - r[1]) / (2.0 * ck * delta[i]);
            optimizer_q.memcpy(grad_dev, ghat.data(), P * sizeof(double)).wait();
            sgd_step(optimizer_q, theta_dev, grad_dev, P, ak);
        } else {
            std::vector<GradientTerm> terms;
            EvalBatch batch = build_gradient_batch(ansatz, theta, opt.gradient, opt.fd_epsilon, terms);
            batch.shots = opt.shots;
            const auto en = evaluate(batch);
            result.evaluations += batch.count;
            e = en[0];
            std::vector<double> grad(P, 0.0);
            for (std::size_t t = 0; t < terms.size(); ++t)
                grad[terms[t].param] += terms[t].weight * (en[1 + 2 * t] - en[2 + 2 * t]);
            optimizer_q.memcpy(grad_dev, grad.data(), P * sizeof(double)).wait();
            if (opt.optimizer == OptimizerKind::Adam)
                adam_step(optimizer_q, theta_dev, m_dev, v_dev, grad_dev, P,
                         opt.learning_rate, opt.beta1, opt.beta2, opt.epsilon, k);
            else
                sgd_step(optimizer_q, theta_dev, grad_dev, P, opt.learning_rate);
        }

        optimizer_q.memcpy(theta.data(), theta_dev, P * sizeof(double)).wait();
        result.history.push_back(e);
        result.iterations = k + 1;
        if (k > 0 && std::fabs(e - prev) < opt.tolerance) { result.converged = true; break; }
        prev = e;
    }

    sycl::free(theta_dev, optimizer_q); sycl::free(grad_dev, optimizer_q);
    sycl::free(m_dev, optimizer_q); sycl::free(v_dev, optimizer_q);

    result.theta = theta;
    EvalBatch final_batch; final_batch.add(theta); final_batch.shots = opt.shots;
    result.energy = evaluate(final_batch)[0];
    ++result.evaluations;
    return true;
}

} // namespace

bool run_native_vqa(sycl::queue& circuit_q, sycl::queue& optimizer_q,
                    const Circuit& ansatz, const Observable& obs,
                    const VQAOptions& opt,
                    std::vector<double>& theta, VQAResult& result) {
    auto evaluate = [&](const EvalBatch& b) { return expectation_batch(circuit_q, ansatz, obs, b); };
    return run_native_vqa_core(evaluate, optimizer_q, ansatz, opt, theta, result);
}

bool run_native_vqa(Backend& backend, sycl::queue& optimizer_q,
                    const Circuit& ansatz, const Observable& obs,
                    const VQAOptions& opt,
                    std::vector<double>& theta, VQAResult& result) {
    auto evaluate = [&](const EvalBatch& b) { return backend.expectation_batch(ansatz, obs, b); };
    return run_native_vqa_core(evaluate, optimizer_q, ansatz, opt, theta, result);
}

} // namespace quantum::native