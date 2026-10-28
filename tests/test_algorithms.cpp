// Tests for the ansatz library, Hamiltonian builders and the vqe()/qaoa() drivers.
//
// Reference numbers (exact ground energies, exact QAOA expectation values) were
// produced INDEPENDENTLY of this code with dense linear algebra in numpy
// (Pauli matrices -> eigvalsh / scipy expm); they are quoted next to each use.
//
// Usage: ./test_algorithms            host reference backend (no SYCL device needed)
//        ./test_algorithms cpu|gpu    the real CPU/GPU backend (native batched kernels)
#include "quantum/algorithms/algorithms.hpp"
#include "quantum/backend_factory.hpp"
#include "quantum/algorithms/hamiltonians.hpp"
#include "sim_support.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace quantum;

#ifdef QSYCL_TEST_STUB_FACTORY   // only for the SYCL-free g++ build
namespace quantum {
std::unique_ptr<Backend> BackendFactory::create(DeviceType) { throw std::runtime_error("no real backend in the stub build"); }
}
#endif

// A "user-defined backend" implementing only the original pure virtuals, so the
// algorithms are tested through the generic (default) hooks too.
class RefBackend : public Backend {
public:
    void initialize(std::size_t n) override { n_ = n; s_ = simsup::zero_state(n); }
    void apply_gate(const GateOp& op) override { simsup::apply_op(s_, op); }
    std::vector<Complex> get_state() const override { return s_; }
    std::vector<double> probabilities() const override {
        std::vector<double> p(s_.size());
        for (std::size_t i = 0; i < s_.size(); ++i) p[i] = s_[i].norm();
        return p;
    }
    std::vector<unsigned long long> sample(const std::vector<std::size_t>& qubits, std::size_t shots) override {
        std::vector<std::size_t> which = qubits;
        if (which.empty()) for (std::size_t q = 0; q < n_; ++q) which.push_back(q);
        const auto p = probabilities();
        std::discrete_distribution<std::size_t> dist(p.begin(), p.end());
        std::vector<unsigned long long> out;
        for (std::size_t s = 0; s < shots; ++s) {
            const std::size_t full = dist(rng_);
            unsigned long long r = 0;
            for (std::size_t k = 0; k < which.size(); ++k)
                if ((full >> which[k]) & 1ULL) r |= 1ULL << k;
            out.push_back(r);
        }
        return out;
    }
    std::string device_name() const override { return "RefBackend"; }
private:
    std::size_t n_ = 0;
    std::vector<Complex> s_;
    std::mt19937_64 rng_{99};
};

static QuantumRuntime make_runtime(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "cpu")) return QuantumRuntime(DeviceType::CPU);
    if (argc > 1 && !std::strcmp(argv[1], "gpu")) return QuantumRuntime(DeviceType::GPU);
    return QuantumRuntime(std::make_unique<RefBackend>());
}

#define CHECK(cond)                                                              \
    do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

static std::size_t count_gates(const Circuit& c, GateType t) {
    std::size_t n = 0;
    for (const auto& op : c.operations()) n += (op.type == t);
    return n;
}

int main(int argc, char** argv) {
    QuantumRuntime rt = make_runtime(argc, argv);
    std::printf("backend: %s\n", rt.device_name().c_str());
    namespace A = ansatz;
    namespace Ham = hamiltonians;

    // ---------------- 1. ansatz structure ----------------
    {
        CHECK(A::entanglement_pairs(4, A::Entanglement::Linear).size() == 3);
        CHECK(A::entanglement_pairs(4, A::Entanglement::Circular).size() == 4);
        CHECK(A::entanglement_pairs(2, A::Entanglement::Circular).size() == 1);   // no doubled bond
        CHECK(A::entanglement_pairs(4, A::Entanglement::Full).size() == 6);
        const auto pw = A::entanglement_pairs(5, A::Entanglement::Pairwise);       // (0,1)(2,3) then (1,2)(3,4)
        const std::pair<std::size_t, std::size_t> p01{0, 1}, p12{1, 2};
        CHECK(pw.size() == 4 && pw[0] == p01 && pw[2] == p12);
        CHECK(A::entanglement_pairs(1, A::Entanglement::Full).empty());

        const Circuit ra = A::real_amplitudes(4, 2);
        CHECK(ra.num_params() == 4 * 3);                      // n * (reps + 1)
        CHECK(count_gates(ra, GateType::RY) == 12 && count_gates(ra, GateType::CNOT) == 2 * 3);
        const Circuit su2 = A::efficient_su2(4, 2, A::Entanglement::Circular);
        CHECK(su2.num_params() == 2 * 4 * 3);                 // 2n * (reps + 1)
        CHECK(count_gates(su2, GateType::RY) == 12 && count_gates(su2, GateType::RZ) == 12);
        CHECK(count_gates(su2, GateType::CNOT) == 2 * 4);
        CHECK(A::hardware_efficient(3, 2).num_params() == A::efficient_su2(3, 2).num_params());
        CHECK(A::two_local(3, "xyz", 1, A::Entanglement::Full, false).num_params() == 3 * 3 * 1);  // no final layer
        CHECK(A::two_local(3, "Y", 0).num_params() == 3);     // reps == 0 still gives one rotation layer

        // Parameter order: layer-major, then rotation, then qubit; indices are fresh and sequential.
        const Circuit c = A::two_local(2, "YZ", 1);
        std::vector<int> idx;
        for (const auto& op : c.operations()) if (op.is_parametric()) idx.push_back(op.symbolic[0].index);
        const std::vector<int> expect = {0, 1, 2, 3, 4, 5, 6, 7};
        CHECK(idx == expect);
        CHECK(c.operations()[0].type == GateType::RY && c.operations()[2].type == GateType::RZ);

        // Appending after existing parameters/gates allocates AFTER them.
        Circuit pre(3);
        pre.new_params(2);
        A::append_basis_state(pre, 0b101);
        A::append_real_amplitudes(pre, 1);
        CHECK(pre.num_params() == 2 + 3 * 2);
        CHECK(pre.operations()[0].type == GateType::X && pre.operations()[0].qubits[0] == 0);
        CHECK(pre.operations()[1].qubits[0] == 2);
        bool first_new = false;
        for (const auto& op : pre.operations()) if (op.is_parametric()) { first_new = op.symbolic[0].index == 2; break; }
        CHECK(first_new);
        bool threw = false;
        try { A::append_basis_state(pre, 0b1000); } catch (const std::out_of_range&) { threw = true; }
        CHECK(threw);
        threw = false;
        try { A::two_local(2, "Q", 1); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        std::puts("[1] ansatz structure OK");
    }

    // ---------------- 2. MaxCut Hamiltonian ----------------
    const std::vector<Ham::Edge> ring4 = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    const std::vector<Ham::Edge> weighted5 = {{0, 1, 1.5}, {1, 2, 0.7}, {2, 3, 2.0}, {3, 4, 1.0}, {4, 0, 0.5}, {0, 2, 1.2}};
    {
        for (const auto& edges : {ring4, weighted5}) {
            const std::size_t n = (&edges == &ring4) ? 4 : 5;
            const Observable H = Ham::maxcut(n, edges);
            const Observable Hn = Ham::maxcut(n, edges, false);
            double total_w = 0;
            for (const auto& e : edges) total_w += e.weight;
            for (unsigned long long x = 0; x < (1ULL << n); ++x) {
                CHECK(near(diagonal_cost(H, x), -Ham::cut_value(edges, x), 1e-12));       // energy == -cut
                CHECK(near(diagonal_cost(Hn, x) - diagonal_cost(H, x), total_w / 2, 1e-12)); // constant dropped
            }
        }
        const auto bf4 = Ham::maxcut_brute_force(4, ring4);
        const auto bf5 = Ham::maxcut_brute_force(5, weighted5);
        CHECK(near(bf4.first, 4.0, 1e-12));       // even ring: all 4 edges cut
        CHECK(near(bf5.first, 6.2, 1e-12));       // numpy: min eigenvalue of C = -6.2
        CHECK(near(Ham::cut_value(weighted5, bf5.second), bf5.first, 1e-12));
        bool threw = false;
        try { Ham::maxcut(3, {{0, 5}}); } catch (const std::out_of_range&) { threw = true; }
        CHECK(threw);
        std::puts("[2] MaxCut Hamiltonian OK");
    }

    // ---------------- 3. QAOA circuit == exact exp(-i gamma C) exp(-i beta X) ----------------
    {
        // numpy/scipy expm reference: ring4 p=2, theta=[0.3,-0.2,0.5,-0.4] -> <C> = -3.0671850538
        const Observable C4 = Ham::maxcut(4, ring4);
        const Circuit q4 = A::qaoa(C4, 2);
        CHECK(q4.num_params() == 4);
        const double e4 = rt.expectation(q4.bind({0.3, -0.2, 0.5, -0.4}), C4);
        std::printf("[3] ring4 p=2: got %.10f  numpy -3.0671850538\n", e4);
        CHECK(near(e4, -3.0671850538, 1e-8));
        // weighted (non-uniform edge weights, a chord): p=1, theta=[0.4,-0.3] -> -4.7799196417
        const Observable C5 = Ham::maxcut(5, weighted5);
        const double e5 = rt.expectation(A::qaoa(C5, 1).bind({0.4, -0.3}), C5);
        std::printf("[3] weighted5 p=1: got %.10f  numpy -4.7799196417\n", e5);
        CHECK(near(e5, -4.7799196417, 1e-8));
        // theta = 0 leaves |+>^n: <C> = -(total weight)/2 for MaxCut
        CHECK(near(rt.expectation(q4.bind({0, 0, 0, 0}), C4), -2.0, 1e-9));

        // 3-body Z term (a PUBO-style cost): gadget must handle |support| > 2 and 1
        Observable P3(3);
        P3.add_term(0.8, "ZZZ").add_term(-0.5, "ZII").add_term(0.3, "IZZ").add_term(0.7, "III");
        const Circuit q3 = A::qaoa(P3, 1);
        // exact reference computed here from the definition, with a tiny dense simulation of
        // exp(-i gamma C) exp(-i beta X) |+>: apply it gate-free via diagonal phases + RX.
        const double gamma = 0.45, beta = -0.35;
        std::vector<Complex> psi(8, Complex(1.0 / std::sqrt(8.0)));
        for (unsigned long long x = 0; x < 8; ++x) {
            const double phase = -gamma * diagonal_cost(P3, x);
            psi[x] = psi[x] * Complex(std::cos(phase), std::sin(phase));
        }
        for (std::size_t q = 0; q < 3; ++q) simsup::apply(psi, q, -1, gates::RX(2 * beta));
        const double exact = expectation_value(psi, P3);
        const double got = rt.expectation(q3.bind({gamma, beta}), P3);
        std::printf("[3] 3-body cost p=1: got %.10f exact %.10f\n", got, exact);
        CHECK(near(got, exact, 1e-9));

        bool threw = false;
        Observable bad(2); bad.add_term(1.0, "XZ");
        try { A::qaoa(bad, 1); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        // ramp: the sign convention is checked numerically (numpy, ring6 p=3, delta=.75):
        // beta<0 -> -4.34164 (better than |+>'s -3), beta>0 -> -1.65836 (worse).
        const std::vector<Ham::Edge> ring6 = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}};
        const Observable C6 = Ham::maxcut(6, ring6);
        const auto ramp = A::qaoa_linear_ramp(3, 0.75);
        CHECK(ramp.size() == 6 && ramp[0] > 0 && ramp[1] < 0 && ramp[0] < ramp[2] && ramp[1] < ramp[3] + 1e-12);
        const double er = rt.expectation(A::qaoa(C6, 3).bind(ramp), C6);
        std::printf("[3] ring6 p=3 ramp energy %.5f  numpy -4.34164\n", er);
        CHECK(near(er, -4.34164, 1e-4));
        std::puts("[3] QAOA circuit OK");
    }

    // ---------------- 4. VQE on spin chains vs exact diagonalization ----------------
    {
        struct Case { const char* name; Observable H; double exact; };
        std::vector<Case> cases;
        cases.push_back({"ising open n=4 J=1 h=0.8",     Ham::ising_chain(4, 1.0, 0.8, false),    -4.1455609133});
        cases.push_back({"ising periodic n=4 J=1 h=0.5", Ham::ising_chain(4, 1.0, 0.5, true),     -4.2715584101});
        cases.push_back({"heisenberg XXX open n=4",      Ham::heisenberg_chain(4, 1.0, 1.0, false), -6.4641016151});
        cases.push_back({"heisenberg XXZ periodic d=.5", Ham::heisenberg_chain(4, 1.0, 0.5, true),  -6.7445626465});
        for (const auto& cs : cases) {
            VQEOptions o;
            o.restarts = 3;
            o.init_range = 0.6;
            o.vqa.optimizer = OptimizerKind::Adam;
            o.vqa.learning_rate = 0.08;
            o.vqa.max_iterations = 400;
            o.vqa.tolerance = 1e-10;
            // the XXZ ground states of an even chain are real, so RealAmplitudes can reach them
            const Circuit ans = A::real_amplitudes(4, 4, A::Entanglement::Full);
            const VQEResult r = vqe(rt, ans, cs.H, o);
            std::printf("[4] %-32s E=%.8f exact=%.8f  (restarts:", cs.name, r.best.energy, cs.exact);
            for (double e : r.restart_energies) std::printf(" %.5f", e);
            std::printf(")\n");
            CHECK(r.best.energy >= cs.exact - 1e-8);          // variational principle: never below E0
            CHECK(near(r.best.energy, cs.exact, 2e-3));       // and it gets there
            CHECK(r.restart_energies.size() == 3 && r.best_restart < 3);
            CHECK(near(r.restart_energies[r.best_restart], r.best.energy, 1e-12));
            for (double e : r.restart_energies) CHECK(e >= r.best.energy - 1e-12);   // best really is the min
        }
        std::puts("[4] VQE OK");
    }

    // ---------------- 5. QAOA end to end on MaxCut ----------------
    {
        QAOAOptions o;
        o.p = 2;
        o.restarts = 3;
        o.vqa.optimizer = OptimizerKind::Adam;
        o.vqa.learning_rate = 0.05;
        o.vqa.max_iterations = 300;
        o.vqa.tolerance = 1e-10;
        o.sample_shots = 4096;

        const Observable C4 = Ham::maxcut(4, ring4);
        const QAOAResult r4 = qaoa(rt, C4, o);
        std::printf("[5] ring4 p=2: <C>=%.5f best_cost=%.1f cut(best)=%.1f\n", r4.best.energy, r4.best_cost,
                    Ham::cut_value(ring4, r4.best_bitstring));
        CHECK(r4.best.theta.size() == 4);
        CHECK(r4.best.energy < -3.0);                                   // beats |+>
        CHECK(near(r4.best_cost, -4.0, 1e-9));                          // found the optimum
        CHECK(near(Ham::cut_value(ring4, r4.best_bitstring), 4.0, 1e-9));
        CHECK(near(diagonal_cost(C4, r4.best_bitstring), r4.best_cost, 1e-12));
        std::size_t total = 0;
        for (const auto& [bits, n] : r4.counts) { (void)bits; total += n; }
        CHECK(total == 4096);

        const Observable C5 = Ham::maxcut(5, weighted5);
        const QAOAResult r5 = qaoa(rt, C5, o);
        const auto bf = Ham::maxcut_brute_force(5, weighted5);
        std::printf("[5] weighted5 p=2: <C>=%.5f best cut=%.2f (optimum %.2f)\n", r5.best.energy,
                    Ham::cut_value(weighted5, r5.best_bitstring), bf.first);
        CHECK(r5.best.energy >= -bf.first - 1e-8);                      // can't beat the true optimum
        CHECK(near(Ham::cut_value(weighted5, r5.best_bitstring), bf.first, 1e-9));

        // explicit start, p validation, bad inputs
        QAOAOptions bad = o; bad.p = 0;
        bool threw = false;
        try { qaoa(rt, C4, bad); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        bad = o; bad.initial_point = {0.1, 0.2, 0.3};               // needs 2p = 4
        threw = false;
        try { qaoa(rt, C4, bad); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        threw = false;
        try { qaoa(rt, Ham::ising_chain(4, 1.0, 0.5), o); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);                                                // non-diagonal cost refused
        VQEOptions vb; vb.initial_point = {0.0};
        threw = false;
        try { vqe(rt, A::real_amplitudes(2, 1), Ham::ising_chain(2, 1.0, 0.5), vb); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        std::puts("[5] QAOA OK");
    }

    std::puts("ALL ALGORITHM TESTS PASSED");
    return 0;
}
