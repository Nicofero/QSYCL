# CUNQA backend

`cunqa_backend.cpp` talks directly to an already-qraised CUNQA vQPU over its ZeroMQ
wire protocol, rather than linking against CUNQA's own C++ sources — CUNQA's CMake
install only exports its CLI tools (`qraise`/`qdrop`/...) and its Python extension, not
a public C++ library, so there's nothing stable to `find_package()` there.

- **Discovery** reads `$STORE/.cunqa/qpus.json` directly — the same registry file
  CUNQA's own `get_QPUs()` reads — and connects to **every** vQPU listed there (one ZMQ
  `DEALER` socket each). Unlike `get_QPUs(co_located=True)`, this isn't restricted to
  same-node vQPUs: we talk to them over a plain ZMQ/TCP endpoint regardless of where
  they're running. vQPUs whose `n_qubits` is smaller than the circuit are skipped.
- **`apply_gate()` buffers, it doesn't execute.** Unlike the CPU backend's per-gate
  kernel, CUNQA only runs whole circuits as a unit, so each call just appends to an
  in-memory instruction list.
- **`sample()` triggers the actual submission**: it flushes the buffered gates plus one
  explicit `measure` instruction per requested qubit into CUNQA's JSON circuit schema,
  sends it to one vQPU (round-robin among the eligible ones), and blocks for the reply.
- **Batched evaluation is spread over all vQPUs.** `expectation_batch()` (and therefore
  `VQA::minimize()` / `run_native_vqa()`) builds every circuit of the batch up front —
  one per (parameter set × qubit-wise-commuting measurement group) — and load-balances
  them over all eligible vQPUs: one circuit in flight per vQPU, the next one sent to
  whichever vQPU answers first, so faster vQPUs naturally take more work. With `k`
  vQPUs a parameter-shift gradient step takes roughly `1/k` of the sequential time.
  - A vQPU that fails to accept a circuit is dropped for the rest of the session and its
    circuit is re-queued on the others. Set `CUNQA_RECV_TIMEOUT_MS` to also drop vQPUs
    that don't answer within that time (default: wait indefinitely).
  - A vQPU-reported `ERROR` is about the circuit, not the vQPU, and is thrown as-is.
  - If every vQPU is lost, the registry is re-read once at the next batch; if still none,
    the batch throws. A batch **never** falls back to the CPU partway through, so a VQA
    trajectory can't silently mix emulated and locally simulated energies.
- **`get_state()`/`probabilities()` are intentionally unsupported** on a real vQPU and
  throw rather than approximate: real QPUs can't report exact amplitudes, and since the
  point of CUNQA is emulating that hardware model, silently answering from a different
  simulation underneath would be more misleading than an explicit error.
- **Fallback to `CPUBackend`** happens transparently whenever no qraised vQPU is found,
  or a submission fails at run time (e.g. a time-boxed `qraise` allocation expired
  mid-session) — `device_name()` reflects this, and a message is written to stderr, so a
  caller never silently gets a different simulator than the one they asked for.