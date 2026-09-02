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
cmake -B build -DCMAKE_CXX_COMPILER= # -DENABLE_GPU_BACKEND=ON -DENABLE_CUNQA_BACKEND=ON
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
bell.h(0).cnot(0, 1);

QuantumRuntime runtime(DeviceType::CPU);
runtime.run(bell);

auto state = runtime.state_vector();          // exact amplitudes
auto counts = runtime.sample_counts(bell, 1000); // simulated measurement shots
```

Chainable gate calls: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`,
`cnot`, `cz`, `swap`, `measure`.

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

## Adding a new device backend (e.g. GPU)

1. Create `backends/gpu/gpu_backend.{hpp,cpp}` implementing the `Backend`
   interface from `include/quantum/backend.hpp` (mirror `backends/cpu`).
2. Use `DeviceSelector::make_queue(DeviceType::GPU)` to get your queue.
3. Register it in `src/backend_factory.cpp`'s `BackendFactory::create()`.
4. Add the new source file and include dir to `CMakeLists.txt`.

Nothing in `Circuit`, `QuantumRuntime`, or user code needs to change --
callers just pass `DeviceType::GPU` instead of `DeviceType::CPU`.

For hardware SYCL can't auto-detect with a built-in selector (a custom
ASIC, a specific FPGA image, picking among several GPUs by memory size),
use `DeviceSelector::make_queue_custom()` with your own scoring function
and construct that backend directly instead of going through
`BackendFactory`.

## Known limitations of this reference implementation

- State-vector simulation only: memory is `O(2^n)`, so it's practical up
  to roughly 26-28 qubits depending on available RAM.
- No gate fusion or circuit optimization pass -- gates are applied one at
  a time, exactly as scheduled.
- `CPUBackend::sample()` regenerates the full probability distribution
  each call rather than caching it across repeated sampling.
