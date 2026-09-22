# CUNQA backend

`cunqa_backend.cpp` talks directly to an already-qraised CUNQA vQPU over its ZeroMQ
wire protocol, rather than linking against CUNQA's own C++ sources — CUNQA's CMake
install only exports its CLI tools (`qraise`/`qdrop`/...) and its Python extension, not
a public C++ library, so there's nothing stable to `find_package()` there.

- **Discovery** reads `$STORE/.cunqa/qpus.json` directly — the same registry file
  CUNQA's own `get_QPUs()` reads — and connects to the first available vQPU found.
  Unlike `get_QPUs(co_located=True)`, this isn't restricted to same-node vQPUs: we talk
  to them over a plain ZMQ/TCP endpoint regardless of where they're running.
- **`apply_gate()` buffers, it doesn't execute.** Unlike the CPU backend's per-gate
  kernel, CUNQA only runs whole circuits as a unit, so each call just appends to an
  in-memory instruction list.
- **`sample()` triggers the actual submission**: it flushes the buffered gates plus one
  explicit `measure` instruction per requested qubit into CUNQA's JSON circuit schema,
  sends it over a ZMQ `DEALER` socket to the vQPU's endpoint, and blocks for the reply.
- **`get_state()`/`probabilities()` are intentionally unsupported** on a real vQPU and
  throw rather than approximate: real QPUs can't report exact amplitudes, and since the
  point of CUNQA is emulating that hardware model, silently answering from a different
  simulation underneath would be more misleading than an explicit error.
- **Fallback to `CPUBackend`** happens transparently whenever no qraised vQPU is found,
  or a submission fails at run time (e.g. a time-boxed `qraise` allocation expired
  mid-session) — `device_name()` reflects this, and a message is written to stderr, so a
  caller never silently gets a different simulator than the one they asked for.