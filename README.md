# quantum_sycl

A layered SYCL quantum circuit simulator:

```
Circuit API        include/quantum/circuit.hpp, src/circuit.cpp
                    (gates + measurement, no device knowledge)
       |
Quantum Runtime     include/quantum/runtime.hpp, src/runtime.cpp
                    (state lifecycle, scheduling gates onto a Backend)
       |
Device Selector     include/quantum/device_selector.hpp, src/device_selector.cpp
                    (wraps SYCL device selection: CPU/GPU/FPGA/custom)
       |
Backend             include/quantum/backend.hpp (abstract interface)
Implementation      backends/cpu/cpu_backend.{hpp,cpp}  <- implemented
                    backends/gpu/...                     <- add later
                    backends/fpga/...                     <- add later
```

Only `Backend` is device-specific. `Circuit`, `QuantumRuntime`, and
`DeviceSelector` never include anything from `backends/`.

## Build

Requires the Intel oneAPI DPC++/C++ Compiler (`icpx`), which provides SYCL.

```bash
source /opt/intel/oneapi/setvars.sh   # sets up icpx on PATH
cmake -B build -DCMAKE_CXX_COMPILER=ICPX # -DENABLE_GPU_BACKEND=ON -DENABLE_CUNQA_BACKEND=ON
cmake --build build
./build/bell_state
```

If you're on a machine without Intel oneAPI, you can substitute another
SYCL implementation (AdaptiveCpp/hipSYCL, ComputeCpp) by changing the
compiler and, if needed, the `-fsycl` flags in `CMakeLists.txt` to that
implementation's equivalents.

## Usage

```cpp
#include "quantum/circuit.hpp"
#include "quantum/runtime.hpp"

using namespace quantum;

Circuit bell(2);
bell.h(0).cnot(0, 1);   // or bell.h(0); bell.cnot(0,1);



QuantumRuntime runtime(DeviceType::CPU);
runtime.run(bell);

auto state = runtime.state_vector();          // exact amplitudes
auto counts = runtime.sample_counts(bell, 1000); // simulated measurement shots
```

Chainable gate calls: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`,`cnot`, `cz`, `crx`, `cry`, `crz`, `swap`, `measure`.

## How the CPU backend works

`backends/cpu/cpu_backend.cpp` holds a `2^num_qubits`-length state vector
on the host and mutates it via SYCL kernels submitted to a CPU-selected
`sycl::queue`:

- **Single-qubit gates** run a kernel over `dim/2` work-items, each
  responsible for one amplitude pair `(i0, i1)` that differ only in the
  target qubit's bit, and apply the gate's 2x2 matrix to that pair.
- **Controlled gates** (CNOT, CZ) reuse the same pairing but only touch
  amplitudes where the control bit is set.
- **SWAP** flips amplitude pairs whose two qubit bits are `(0,1)` and
  `(1,0)`.
- **Measurement** is handled by the runtime, not the backend's state
  update: `probabilities()` derives `|amplitude|^2` per basis state, and
  `sample()` draws from that distribution with `std::discrete_distribution`.

This is a reference implementation (readability over performance): it
rebuilds a `sycl::buffer` per gate call rather than keeping the state
resident on-device across a whole circuit. That's the natural place to
optimize before scaling up, along with using USM `device` allocations
instead of buffers.

## How the GPU backend works

`backends/gpu/gpu_backend.cpp` holds the same `2^num_qubits`-length state vector, but as a USM (Unified Shared Memory) allocation living directly on the GPU, mutated via SYCL kernels submitted to a GPU-selected `sycl::queue`:

- **Single-qubit gates** run a kernel over `dim/2` work-items, each
  responsible for one amplitude pair `(i0, i1)` that differ only in the
  target qubit's bit, and apply the gate's 2x2 matrix to that pair --
  same indexing scheme as the CPU backend.
- **Controlled gates** (CNOT, CZ) reuse the same pairing but only touch
  amplitudes where the control bit is set.
- **SWAP** flips amplitude pairs whose two qubit bits are `(0,1)` and
  `(1,0)`.
- **Measurement** is handled by the runtime, not the backend's state
  update: `probabilities()` derives `|amplitude|^2` per basis state, and
  `sample()` draws from that distribution with `std::discrete_distribution`.

Unlike the CPU backend, the state vector is allocated once with
`sycl::malloc_device<Complex>` in `initialize()` and stays resident on
the GPU for the whole circuit -- gates never round-trip to the host,
only `get_state()`/`probabilities()` do (via an explicit
`queue_.memcpy()` back to a host `std::vector`). This is the main reason
a GPU backend is worth having at all: PCIe transfer would dominate
runtime if every gate synced back to host like the CPU reference
implementation does.

Two things to be careful of if you extend this file, since USM bugs tend
to surface as silent crashes rather than compile errors:

- **Never dereference `state_dev_` from host code.** It's a device
  pointer; the only host-legal operations on it are passing it to a
  kernel (captured by value) or `queue_.memcpy()`/`sycl::free()`, both of
  which go through the SYCL runtime rather than touching the address
  directly.
- **The queue that allocates must be the queue that frees.**
  `free_state()` calls `sycl::free(state_dev_, queue_)` using the same
  `queue_` member that allocated it in `initialize()` -- mixing queues
  (or contexts) here is a common source of GPU segfaults.

## How the CUNQA backend works

`backends/cunqa/cunqa_backend.cpp` talks directly to an already-qraised
cunqa vQPU over its ZeroMQ wire protocol, rather than linking against
cunqa's own C++ sources — cunqa's CMake install only exports its CLI
tools (`qraise`/`qdrop`/...) and its Python extension, not a public
C++ library, so there's nothing stable to `find_package()` there:

- **Discovery** reads `$STORE/.cunqa/qpus.json` directly — the same
  registry file cunqa's own `get_QPUs()` reads — and connects to the
  first available vQPU found. Unlike `get_QPUs(co_located=True)`, this
  isn't restricted to same-node vQPUs: we talk to them over a plain
  ZMQ/TCP endpoint regardless of where they're running.
- **`apply_gate()` buffers, it doesn't execute.** Unlike the CPU
  backend's per-gate kernel, cunqa only runs whole circuits as a unit,
  so each call just appends to an in-memory instruction list.
- **`sample()` triggers the actual submission**: it flushes the
  buffered gates plus one explicit `measure` instruction per requested
  qubit into cunqa's JSON circuit schema, sends it over a ZMQ `DEALER`
  socket to the vQPU's endpoint, and blocks for the reply.
- **`get_state()`/`probabilities()` are intentionally unsupported** on
  a real vQPU and throw rather than approximate: real QPUs can't report
  exact amplitudes, and since the point of cunqa is emulating that
  hardware model, silently answering from a different simulation
  underneath would be more misleading than an explicit error.
- **Fallback to `CPUBackend`** happens transparently whenever no
  qraised vQPU is found, or a submission fails at run time (e.g. a
  time-boxed `qraise` allocation expired mid-session) — `device_name()`
  reflects this, and a message is written to stderr, so a caller never
  silently gets a different simulator than the one they asked for.

## Adding a new device backend (e.g. QPU)

1. Create `backends/qpu/qpu_backend.{hpp,cpp}` implementing the `Backend`
   interface from `include/quantum/backend.hpp` (mirror `backends/cpu`).
2. Use `DeviceSelector::make_queue(DeviceType::QPU)` to get your queue.
3. Register it in `src/backend_factory.cpp`'s `BackendFactory::create()`.
4. Add the new source file and include dir to `CMakeLists.txt`.

Nothing in `Circuit`, `QuantumRuntime`, or user code needs to change: callers just pass `DeviceType::QPU` instead of `DeviceType::CPU`.

For hardware SYCL can't auto-detect with a built-in selector (a custom
ASIC, a specific FPGA image, picking among several GPUs by memory size),
use `DeviceSelector::make_queue_custom()` with your own scoring function
and construct that backend directly instead of going through
`BackendFactory`.

## Known limitations of this reference implementation

- State-vector simulation only: memory is `O(2^n)`, so it's practical up
  to roughly 26-28 qubits depending on available RAM.
- No gate fusion or circuit optimization pass: gates are applied one at
  a time, exactly as scheduled.
- `CPUBackend::sample()` regenerates the full probability distribution
  each call rather than caching it across repeated sampling.
