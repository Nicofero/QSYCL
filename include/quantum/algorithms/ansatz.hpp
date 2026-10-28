#pragma once

#include "quantum/circuit.hpp"
#include "quantum/observable.hpp"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace quantum::ansatz {

// Library of commonly used parameterized circuits ("ansatzes").
//
// Every builder comes in two forms:
//   * append_xxx(Circuit&, ...)  adds the ansatz to an existing circuit, so you
//     can put your own state preparation first (append_basis_state, a custom
//     initial layer, ...). New parameters are allocated AFTER any the circuit
//     already has, in the order documented on each function.
//   * xxx(num_qubits, ...)       returns a fresh Circuit containing just it.
//
// Only RX/RY/RZ/H/X/CNOT are used, so every ansatz here
//   * has exact parameter-shift gradients (VQA's default GradientMethod), and
//   * runs on every backend (CPU, GPU, CUNQA, QPU).

// How two-qubit (CNOT) gates connect the qubits inside an entangling block.
enum class Entanglement {
    Linear,    // (0,1) (1,2) ... (n-2,n-1)
    Circular,  // Linear plus (n-1,0); for n == 2 identical to Linear
    Full,      // every pair (i,j), i<j: n(n-1)/2 CNOTs
    Pairwise   // (0,1) (2,3) ... then (1,2) (3,4) ...: depth-2 version of Linear
};

// The (control, target) pairs of one entangling block over `n` qubits.
std::vector<std::pair<std::size_t, std::size_t>> entanglement_pairs(std::size_t n, Entanglement e);

// ---------------------------------------------------------------------------
// Generic two-local ansatz:
//     [rotation layer, entangling block] x reps, then a final rotation layer
// A rotation layer applies, for each character of `rotations` in order, one
// independent rotation to every qubit. Characters are 'X', 'Y' or 'Z' (any
// case): "Y" = RY on every qubit, "YZ" = RY on all qubits then RZ on all qubits.
//
// Parameter order: layer by layer; within a layer, rotation by rotation; within
// a rotation, qubit 0..n-1. Total: n * len(rotations) * (reps + final ? 1 : 0).
// ---------------------------------------------------------------------------
Circuit& append_two_local(Circuit& circuit, const std::string& rotations, std::size_t reps,
                          Entanglement entanglement = Entanglement::Linear,
                          bool final_rotation_layer = true);
Circuit two_local(std::size_t num_qubits, const std::string& rotations, std::size_t reps,
                  Entanglement entanglement = Entanglement::Linear,
                  bool final_rotation_layer = true);

// RY rotations + CNOTs. All amplitudes stay real, which is enough for
// real-symmetric Hamiltonians (Ising, Heisenberg, MaxCut...) and has the
// fewest parameters: n * (reps + 1).
Circuit& append_real_amplitudes(Circuit& circuit, std::size_t reps,
                                Entanglement entanglement = Entanglement::Linear);
Circuit real_amplitudes(std::size_t num_qubits, std::size_t reps = 2,
                        Entanglement entanglement = Entanglement::Linear);

// RY+RZ rotations + CNOTs. Can reach any single-qubit state per layer, so it
// also handles complex-valued ground states: 2n * (reps + 1) parameters.
Circuit& append_efficient_su2(Circuit& circuit, std::size_t reps,
                              Entanglement entanglement = Entanglement::Linear);
Circuit efficient_su2(std::size_t num_qubits, std::size_t reps = 2,
                      Entanglement entanglement = Entanglement::Linear);

// The usual "hardware-efficient ansatz": alternating RY/RZ rotation layers and
// CNOT entanglers. Same circuit as efficient_su2, under its common name.
Circuit hardware_efficient(std::size_t num_qubits, std::size_t layers = 2,
                           Entanglement entanglement = Entanglement::Linear);

// ---------------------------------------------------------------------------
// Basis-state preparation: X on every qubit whose bit is set in `bits`
// (bit q <-> qubit q). Handy as a starting state, e.g. a Neel state for a
// spin chain, or a known good bitstring for a warm start.
// ---------------------------------------------------------------------------
Circuit& append_basis_state(Circuit& circuit, unsigned long long bits);

// ---------------------------------------------------------------------------
// QAOA. For a DIAGONAL cost Hamiltonian C = sum_k c_k Z-string_k (any Observable
// with only Z/I terms: MaxCut, Ising, PUBO ...), p layers of
//     cost layer   exp(-i gamma_k C)
//     mixer layer  exp(-i beta_k sum_q X_q)
// are appended, each cost term as a CNOT-fan-in / RZ / CNOT-fan-out gadget.
// Identity terms only contribute a global phase and are skipped.
//
// Parameter order is layer-major and INTERLEAVED:
//     [gamma_0, beta_0, gamma_1, beta_1, ..., gamma_{p-1}, beta_{p-1}]
// (2p parameters). gamma_k is shared by every cost term of layer k, beta_k by
// every qubit of layer k; VQA's parameter-shift handles the shared use.
//
// append_qaoa_layers() does NOT prepare |+>^n; use qaoa() for the standard
// start, or add your own initial state first (warm starts, constrained problems).
// ---------------------------------------------------------------------------
Circuit& append_qaoa_layers(Circuit& circuit, const Observable& cost, std::size_t p);
Circuit qaoa(const Observable& cost, std::size_t p);  // H on every qubit, then p layers

// ---------------------------------------------------------------------------
// Initial-parameter helpers
// ---------------------------------------------------------------------------

// `count` values drawn uniformly from [lo, hi] (deterministic for a given seed).
std::vector<double> random_parameters(std::size_t count, unsigned long long seed,
                                      double lo = -3.14159265358979323846,
                                      double hi = 3.14159265358979323846);

// A "linear ramp" initial point for QAOA, in this library's parameter order and
// sign convention (cost minimized, mixer exp(-i beta X)): it discretizes a
// quantum-annealing schedule, with gamma growing and |beta| shrinking over the
// layers. `delta` is the total ramp size. beta is NEGATIVE because, with this
// convention, the annealing path starts in the ground state of -sum X, which is
// |+>^n (checked numerically: MaxCut on a 6-ring, p=3 gives <C> = -4.34 with
// negative beta but -1.66 with positive beta). A heuristic start, not a guarantee.
std::vector<double> qaoa_linear_ramp(std::size_t p, double delta = 0.75);

} // namespace quantum::ansatz
