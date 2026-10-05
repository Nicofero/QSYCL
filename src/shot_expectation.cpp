#include "quantum/shot_expectation.hpp"
#include <algorithm>
#include <unordered_map>

namespace quantum::native {

namespace {
inline int popcount64(unsigned long long v) {
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return static_cast<int>((v * 0x0101010101010101ULL) >> 56);
}

// 0 = I, 1 = X, 2 = Z, 3 = Y, matching how the group needs to rotate qubit q.
int term_basis(const PauliMask& m, std::size_t q) {
    const bool xb = (m.x >> q) & 1ULL, zb = (m.z >> q) & 1ULL;
    if (xb && zb) return 3;
    if (xb) return 1;
    if (zb) return 2;
    return 0;
}
} // namespace

std::vector<std::vector<std::size_t>> group_qwc(const Observable& obs) {
    const auto& masks = obs.masks();
    const std::size_t n = obs.num_qubits();
    std::vector<std::vector<std::size_t>> groups;
    std::vector<std::unordered_map<std::size_t, int>> group_basis;

    for (std::size_t t = 0; t < masks.size(); ++t) {
        const auto& m = masks[t];
        const std::uint64_t support = m.x | m.z;
        bool placed = false;
        for (std::size_t g = 0; g < groups.size() && !placed; ++g) {
            bool compatible = true;
            for (std::size_t q = 0; q < n && compatible; ++q) {
                if (!((support >> q) & 1ULL)) continue;
                auto it = group_basis[g].find(q);
                if (it != group_basis[g].end() && it->second != term_basis(m, q)) compatible = false;
            }
            if (compatible) {
                for (std::size_t q = 0; q < n; ++q)
                    if ((support >> q) & 1ULL) group_basis[g][q] = term_basis(m, q);
                groups[g].push_back(t);
                placed = true;
            }
        }
        if (!placed) {
            groups.push_back({t});
            std::unordered_map<std::size_t, int> b;
            for (std::size_t q = 0; q < n; ++q) if ((support >> q) & 1ULL) b[q] = term_basis(m, q);
            group_basis.push_back(std::move(b));
        }
    }
    return groups;
}

double shot_based_expectation(Backend& backend, const Circuit& ansatz,
                              const std::vector<double>& theta, int shift_op, double shift,
                              const Observable& obs, std::size_t shots) {
    const auto groups = group_qwc(obs);
    const auto& masks = obs.masks();
    const std::size_t n = ansatz.num_qubits();
    double total = 0.0;

    for (const auto& group : groups) {
        // Every term in a QWC group agrees on each qubit's basis by
        // construction, so OR-ing is safe: it just finds which qubits need
        // which single rotation for the whole group.
        std::uint64_t needs_x = 0, needs_y = 0;
        for (auto t : group) {
            const auto& m = masks[t];
            needs_y |= (m.x & m.z);
            needs_x |= (m.x & ~m.z);
        }

        // Rebuild+rebind from scratch for each group (not just reuse a
        // single buffered circuit): a real/remote device executes and
        // measures a circuit as one atomic unit, so trying a second basis
        // means submitting a second, independently complete circuit.
        Circuit bound = ansatz.bind(theta, shift_op, shift);
        for (std::size_t q = 0; q < n; ++q) {
            if ((needs_y >> q) & 1ULL) bound.sdg(q).h(q);       // Y -> computational basis
            else if ((needs_x >> q) & 1ULL) bound.h(q);          // X -> computational basis
        }                                                        // Z (or I): no rotation needed

        backend.initialize(n);
        for (const auto& op : bound.operations()) backend.apply_gate(op);
        const auto outcomes = backend.sample({}, shots); // {} = every qubit
        if (outcomes.empty()) continue; // nothing to estimate from (shots == 0)

        for (auto t : group) {
            const auto& m = masks[t];
            const unsigned long long contributing = m.x | m.z;
            long long sum_sign = 0;
            for (auto outcome : outcomes) {
                sum_sign += (popcount64(outcome & contributing) & 1) ? -1 : 1;
            }
            total += m.coeff * (static_cast<double>(sum_sign) / static_cast<double>(outcomes.size()));
        }
    }
    return total;
}

std::vector<double> shot_based_expectation_batch(Backend& backend, const Circuit& ansatz,
                                                  const Observable& obs, const EvalBatch& batch,
                                                  std::size_t shots) {
    std::vector<double> out;
    out.reserve(batch.count);
    for (std::size_t i = 0; i < batch.count; ++i) {
        const int op = batch.shift_op.empty() ? -1 : batch.shift_op[i];
        const double sh = batch.shift.empty() ? 0.0 : batch.shift[i];
        out.push_back(shot_based_expectation(backend, ansatz, batch.theta_of(i), op, sh, obs, shots));
    }
    return out;
}

} // namespace quantum::native