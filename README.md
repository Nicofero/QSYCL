# QSYCL

QSYCL is an **open-source** C++ framework that extends the SYCL heterogeneous-computing standard to quantum computing. It exposes quantum simulators, emulators, and real QPUs as first-class SYCL devices, so a single circuit-construction API runs unchanged across backends. Thus, switching hardware is just a matter of changing a device type, with no change to circuit-level code.

Contributions and new backends are welcome; see [Adding a new device backend](#adding-a-new-device-backend) below.

## Architecture

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
Implementation      backends/cpu/     <- state-vector simulator, host
                    backends/gpu/     <- state-vector simulator, NVIDIA GPU
                    backends/cunqa/   <- delegates to the CUNQA emulator
                    backends/qpu/     <- abstract QPU interface + QMIO implementation
```

Only `Backend` is device-specific. `Circuit`, `QuantumRuntime`, and `DeviceSelector`
are fully device-agnostic.

## Build

Requires the Intel oneAPI DPC++/C++ Compiler (`icpx`), which provides SYCL.

```bash
source /path/to/intel/oneapi/setvars.sh   # sets up icpx on PATH
cmake -B build # -DENABLE_GPU_BACKEND=ON -DENABLE_CUNQA_BACKEND=ON -DENABLE_QPU_BACKEND=ON
cmake --build build
./build/bell_state
```

If you're on a machine without Intel oneAPI, you can substitute another SYCL
implementation (AdaptiveCpp/hipSYCL, ComputeCpp) by changing the compiler and,
if needed, the `-fsycl` flags in `CMakeLists.txt` to that implementation's equivalents.

## Usage

```cpp
#include "quantum/circuit.hpp"
#include "quantum/runtime.hpp"

using namespace quantum;

Circuit bell(2);
bell.h(0).cnot(0, 1);   // or bell.h(0); bell.cnot(0,1);

QuantumRuntime runtime(DeviceType::CPU);
runtime.run(bell);

auto state = runtime.state_vector();             // exact amplitudes (only simulated backends)
auto counts = runtime.sample_counts(bell, 1000);  // simulated measurement shots
```

Chainable gate calls: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `cnot`, `cz`, `crx`, `cry`, `crz`, `swap`, `measure`.

## Backends

| Backend | Device type | What it does | Details |
|---|---|---|---|
| CPU | `DeviceType::CPU` | State-vector simulator, SYCL kernels on host | [backends/cpu/README.md](backends/cpu/README.md) |
| GPU | `DeviceType::GPU` | State-vector simulator, SYCL kernels on NVIDIA GPU, state resident in USM | [backends/gpu/README.md](backends/gpu/README.md) |
| CUNQA | `DeviceType::CUNQA` | Delegates to a qraised CUNQA vQPU over ZeroMQ; falls back to CPU | [backends/cunqa/README.md](backends/cunqa/README.md) |
| QMIO | `DeviceType::QMIO` | Concrete implementation for CESGA's QMIO, using an abstract QPU interface | [backends/qpu/README.md](backends/qpu/README.md) |

The CPU and GPU backends compute locally via SYCL kernels; CUNQA and QPU use SYCL only for device selection and delegate execution to an external system.

## Adding a new device backend

1. Create `backends/<name>/` implementing the `Backend` interface from
   `include/quantum/backend.hpp` (mirror `backends/cpu` for simulation or `backends/cunqa` for dispatchers).
2. Define the new `DeviceType` in `include/quantum/device_selector.hpp`.
3. Define `DeviceSelector::make_queue(DeviceType::<Name>)` in `src/device_selector.cpp`.
4. Register it in `src/backend_factory.cpp`'s `BackendFactory::create()`.
5. Add the new source file and include dir to `CMakeLists.txt`.
6. Add a `backends/<name>/README.md` documenting it, and link it from the table above.

Nothing in `Circuit`, `QuantumRuntime`, or user code needs to change: callers just pass
the new `DeviceType` instead of `DeviceType::CPU`.

For hardware SYCL can't auto-detect with a built-in selector (a custom ASIC, a specific
FPGA image, picking among several GPUs by memory size), use
`DeviceSelector::make_queue_custom()` with your own scoring function and construct that
backend directly instead of going through `BackendFactory`.

## Known limitations of this reference implementation

- State-vector simulation only: memory is `O(2^n)`, so it's practical up to roughly
  26–28 qubits depending on available RAM.
- No gate fusion or circuit optimization pass: gates are applied one at a time, exactly
  as scheduled.
- `CPUBackend::sample()` of simulator backends regenerates the full probability distribution each call rather
  than caching it across repeated sampling.

## Next steps

- [ ] Additional simulator backends beyond the current state-vector CPU/GPU implementations.
- [ ] Additional QPU hardware behind the existing abstract QPU interface, alongside QMIO.
- [ ] Native execution of variational quantum algorithms (VQAs) within the SYCL execution model.
- [ ] Gate fusion / circuit optimization passes ahead of execution, by using a backend (first idea)

## License

MIT License — Copyright (c) 2026 Nicolás Fernández Otero

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
