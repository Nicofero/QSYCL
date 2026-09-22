# CPU backend

`cpu_backend.cpp` holds a `2^num_qubits`-length state vector on the host and mutates it
via SYCL kernels submitted to a CPU-selected `sycl::queue`.

- **Single-qubit gates** run a kernel over `dim/2` work-items, each responsible for one
  amplitude pair `(i0, i1)` that differ only in the target qubit's bit, and apply the
  gate's 2x2 matrix to that pair.
- **Controlled gates** (CNOT, CZ) reuse the same pairing but only touch amplitudes
  where the control bit is set.
- **SWAP** flips amplitude pairs whose two qubit bits are `(0,1)` and `(1,0)`.
- **Measurement** is handled by the runtime, not the backend's state update:
  `probabilities()` derives `|amplitude|^2` per basis state, and `sample()` draws from
  that distribution with `std::discrete_distribution`.

This is a reference implementation (readability over performance): it rebuilds a
`sycl::buffer` per gate call rather than keeping the state resident on-device across a
whole circuit. That's the natural place to optimize before scaling up, along with using
USM `device` allocations instead of buffers.
