// Host-only tests for Param / bind / Observable / VQA.#include "quantum/vqa.hpp"
#include "quantum/gates.hpp"
#include "quantum/backend_factory.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>

using namespace quantum;

#ifdef QSYCL_TEST_STUB_FACTORY   // only for the SYCL-free g++ build (see top of file)
namespace quantum {
std::unique_ptr<Backend> BackendFactory::create(DeviceType) { throw std::runtime_error("stub"); }
}
#endif

// A "user-defined backend" implementing ONLY the original pure-virtual interface.
class RefBackend : public Backend {
public:
    void initialize(std::size_t n) override { n_ = n; s_.assign(std::size_t(1) << n, Complex(0)); s_[0] = Complex(1); }
    void apply_gate(const GateOp& op) override {
        using namespace gates;
        auto q0 = op.qubits.empty() ? 0 : op.qubits[0];
        switch (op.type) {
            case GateType::H: one(q0, H()); break;
            case GateType::X: one(q0, X()); break;
            case GateType::Y: one(q0, Y()); break;
            case GateType::Z: one(q0, Z()); break;
            case GateType::S: one(q0, op.dagger ? Sdg() : S()); break;
            case GateType::RX: one(q0, RX(op.params[0])); break;
            case GateType::RY: one(q0, RY(op.params[0])); break;
            case GateType::RZ: one(q0, RZ(op.params[0])); break;
            case GateType::CNOT: ctrl(q0, op.qubits[1], X()); break;
            case GateType::CRY: ctrl(q0, op.qubits[1], RY(op.params[0])); break;
            default: throw std::runtime_error("RefBackend: gate not implemented in test");
        }
    }
    std::vector<Complex> get_state() const override { return s_; }
    std::vector<double> probabilities() const override { std::vector<double> p; for (auto& a : s_) p.push_back(a.norm()); return p; }
    std::vector<unsigned long long> sample(const std::vector<std::size_t>&, std::size_t) override { return {}; }
    std::string device_name() const override { return "RefBackend"; }
private:
    void one(std::size_t q, const Matrix2x2& m) { apply(q, ~0ULL, m); }
    void ctrl(std::size_t c, std::size_t t, const Matrix2x2& m) { apply(t, 1ULL << c, m); }
    void apply(std::size_t t, unsigned long long cmask, const Matrix2x2& m) {
        for (std::size_t i = 0; i < s_.size(); ++i) {
            if (i & (1ULL << t)) continue;
            if (cmask != ~0ULL && !(i & cmask)) continue;
            std::size_t j = i | (1ULL << t);
            Complex a = s_[i], b = s_[j];
            s_[i] = m[0] * a + m[1] * b; s_[j] = m[2] * a + m[3] * b;
        }
    }
    std::size_t n_ = 0; std::vector<Complex> s_;
};

// Counts batch calls to prove the VQA loop batches (1 call / iteration).
class CountingBackend : public RefBackend {
public:
    std::vector<double> expectation_batch(const Circuit& a, const Observable& o, const EvalBatch& b) override {
        ++calls; return RefBackend::expectation_batch(a, o, b);
    }
    int calls = 0;
};

// Backend that claims to run the optimizer natively.
class NativeBackend : public RefBackend {
public:
    bool run_native_vqa(const Circuit&, const Observable&, const VQAOptions&,
                        std::vector<double>& theta, VQAResult& r) override { r.theta = theta; r.energy = -42; return true; }
};

#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)
static bool near(double a, double b, double t = 1e-9) { return std::fabs(a - b) < t; }

int main() {
    // --- Param arithmetic & bind ---
    Circuit c(2);
    auto th = c.new_params(2);
    c.ry(0, th[0]).rz(1, 2.0 * th[1] + 0.5).rx(0, 0.25).cnot(0, 1);
    CHECK(c.num_params() == 2 && c.is_parametric());
    Circuit b = c.bind({0.3, 0.4});
    CHECK(!b.is_parametric() && b.num_params() == 0);
    CHECK(near(b.operations()[0].params[0], 0.3));
    CHECK(near(b.operations()[1].params[0], 2 * 0.4 + 0.5));
    CHECK(near(b.operations()[2].params[0], 0.25));
    Circuit sh = c.bind({0.3, 0.4}, 0, 0.5);
    CHECK(near(sh.operations()[0].params[0], 0.8));
    bool threw = false;
    try { c.bind({0.1}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    RefBackend* raw = new RefBackend; QuantumRuntime rt{std::unique_ptr<Backend>(raw)};
    try { rt.run(c); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw); // unbound circuits are refused

    // --- Observable conventions (character k acts on qubit k) ---
    { Circuit p(1); p.h(0).s(0);   Observable Y(1); Y.add_term(1, "Y");  CHECK(near(rt.expectation(p, Y),  1.0)); }
    { Circuit p(2); p.x(0);        Observable a(2), z(2); a.add_term(1, "ZI"); z.add_term(1, "IZ");
      CHECK(near(rt.expectation(p, a), -1.0)); CHECK(near(rt.expectation(p, z), 1.0)); }
    { Circuit p(2); p.h(0).cnot(0, 1);
      Observable zz(2), xx(2), yy(2), sp(2);
      zz.add_term(1, "ZZ"); xx.add_term(1, "XX"); yy.add_term(1, "YY"); sp.add_term(1, {{'Z', 0}, {'Z', 1}});
      CHECK(near(rt.expectation(p, zz), 1)); CHECK(near(rt.expectation(p, xx), 1));
      CHECK(near(rt.expectation(p, yy), -1)); CHECK(near(rt.expectation(p, sp), 1)); }
    { Observable d(2); d.add_term(1, "ZI").add_term(0.5, "ZZ"); CHECK(d.is_diagonal());
      auto h = d.diagonal(); CHECK(near(h[0], 1.5)); CHECK(near(h[1], -1.5)); CHECK(near(h[2], 0.5)); CHECK(near(h[3], -0.5)); }

    // --- Parameter-shift == finite difference (shared param + scale + 2 gates on one param) ---
    {
        Circuit a(3); auto p = a.new_params(3);
        a.ry(0, p[0]).ry(1, 0.5 * p[1] + 0.2).rx(2, p[2]).cnot(0, 1).cnot(1, 2)
         .rz(0, p[0]).ry(2, 3.0 * p[1]).rx(1, -p[2]).cnot(2, 0).rx(0, p[0]);
        Observable H(3); H.add_term(1.0, "ZZI").add_term(0.7, "IXX").add_term(-0.4, "YIZ").add_term(0.3, "XYX");
        std::mt19937 g(1); std::uniform_real_distribution<double> u(-3, 3);
        std::vector<double> t = {u(g), u(g), u(g)};
        QuantumRuntime r2{std::make_unique<RefBackend>()};
        VQA vqa(r2, a, H);
        auto gps = vqa.gradient(t, GradientMethod::ParameterShift);
        auto gfd = vqa.gradient(t, GradientMethod::FiniteDifference);
        for (int i = 0; i < 3; ++i) { std::printf("grad[%d]: shift=% .8f fd=% .8f\n", i, gps[i], gfd[i]); CHECK(near(gps[i], gfd[i], 1e-6)); }
    }

    // --- Convergence: H = Z + X on 1 qubit, ground = -sqrt(2) ---
    {
        Circuit a(1); auto p = a.new_param(); a.ry(0, p);
        Observable H(1); H.add_term(1, "Z").add_term(1, "X");
        for (auto kind : {OptimizerKind::Adam, OptimizerKind::GradientDescent, OptimizerKind::SPSA}) {
            auto cb = std::make_unique<CountingBackend>(); auto* cbp = cb.get();
            QuantumRuntime r3{std::move(cb)};
            VQA vqa(r3, a, H);
            VQAOptions o; o.optimizer = kind; o.max_iterations = 300; o.learning_rate = kind == OptimizerKind::SPSA ? 0.6 : 0.1;
            auto res = vqa.minimize({0.3}, o);
            std::printf("opt=%d E=%.6f iters=%zu evals=%zu batch_calls=%d\n", (int)kind, res.energy, res.iterations, res.evaluations, cbp->calls);
            CHECK(std::fabs(res.energy + std::sqrt(2.0)) < 5e-2);
            if (kind != OptimizerKind::SPSA) CHECK(cbp->calls == (int)res.iterations + 1); // 1 batch / iter (+ final cost)
        }
    }
    // --- 2-qubit VQE: H = ZZ + XX, ground = -2 ---
    {
        Circuit a(2); auto p = a.new_params(2);
        a.ry(0, p[0]).cnot(0, 1).x(1).ry(1, p[1]).cry(0, 1, 0.0 * p[1]);
        Observable H(2); H.add_term(1, "ZZ").add_term(1, "XX");
        QuantumRuntime r4{std::make_unique<RefBackend>()};
        VQA vqa(r4, a, H);
        VQAOptions o; o.gradient = GradientMethod::FiniteDifference; o.max_iterations = 400; o.learning_rate = 0.1;
        auto res = vqa.minimize({0.4, 0.4}, o);
        std::printf("2q VQE E=%.6f (exact -2)\n", res.energy);
        CHECK(res.energy < -1.99);
    }
    // --- native hook dispatch ---
    {
        Circuit a(1); auto p = a.new_param(); a.ry(0, p);
        Observable H(1); H.add_term(1, "Z");
        QuantumRuntime r5{std::make_unique<NativeBackend>()};
        VQA vqa(r5, a, H);
        auto res = vqa.minimize({0.1});
        CHECK(res.used_native && res.energy == -42);
        VQAOptions o; o.prefer_native = false;
        CHECK(!vqa.minimize({0.1}, o).used_native);
    }
    std::puts("ALL TESTS PASSED");
    return 0;
}
