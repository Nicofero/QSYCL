# QSYCL roadmap

Planned work on QSYCL after the current benchmarking campaign. Contributions to any of these items
are welcome; see "Adding a new device backend" in the [README](README.md) for how backends plug in.

## Other simulators as backends

Wrap existing simulators behind QSYCL's `Backend` interface, so a circuit written once can be
delegated to them like to any other device:

- [ ] Qiskit Aer
- [ ] Qulacs
- [ ] PennyLane-Lightning

These backends also serve as reference points: the benchmarks in `examples/benchmarking/` can then
compare QSYCL's own simulators with them on identical circuits.

## Adjoint differentiation

A new gradient method for the exact simulators (CPU and GPU). Parameter shift needs 2P+1 circuit
evaluations per gradient for P parameters; adjoint differentiation computes the exact gradient with a
few passes over the state vector, whatever the number of parameters.

- [ ] `GradientMethod::Adjoint` in the batched SYCL kernels
- [ ] Keeping the state and the gradient on the device across the whole native VQA loop
- [ ] Shot-based backends (CUNQA, QPUs) keep the parameter-shift rule

## Gate fusion

Faster simulation by applying several gates per pass over the state vector:

- [ ] Small circuits: the whole circuit in a single kernel, with the state in local memory, to remove
  per-gate kernel-launch overhead
- [ ] Large circuits: fusing consecutive gates into small multi-qubit blocks, to cut the number of
  memory passes
- [ ] Support for parametric circuits, with fused matrices built per parameter set on the device

## Circuit optimization passes

Shorter circuits for every backend, which on real hardware also means less error:

- [ ] Cancelling inverse gate pairs
- [ ] Merging consecutive rotations
- [ ] Commuting gates to expose more cancellations
- [ ] Running the passes once on a parametric ansatz, before binding parameters (needs parameters
  that can combine several indices)

## Beyond the state-vector memory limit

State-vector simulation is limited to roughly 30 qubits on a single node. Planned approaches to go
further:

- [ ] Matrix product state (MPS) backend, well suited to low-depth, low-entanglement ansatzes
- [ ] Distributed state vector across nodes (MPI + SYCL)

## Shot-based backends (CUNQA and QPUs)

- [ ] Better measurement grouping, to need fewer circuits per expectation value
- [ ] Adaptive shot allocation during optimization
- [ ] Several circuits in flight per vQPU
- [ ] More QPUs behind the abstract QPU interface, alongside QMIO
