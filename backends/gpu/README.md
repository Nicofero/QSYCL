# GPU backend

`gpu_backend.cpp` holds the same `2^num_qubits`-length state vector as the
[CPU backend](../cpu/README.md), but as a USM (Unified Shared Memory) allocation living
directly on the GPU, mutated via SYCL kernels submitted to a GPU-selected `sycl::queue`.
Gate indexing (single-qubit, controlled, SWAP) and measurement follow the same scheme
as the CPU backend.

The state vector is allocated once with `sycl::malloc_device<Complex>` in
`initialize()` and stays resident on the GPU for the whole circuit — gates never
round-trip to the host, only `get_state()`/`probabilities()` do (via an explicit
`queue_.memcpy()` back to a host `std::vector`). This is the main reason a GPU backend
is worth having at all: PCIe transfer would dominate runtime if every gate synced back
to host like the CPU reference implementation does.

Two things to be careful of if you extend this file, since USM bugs tend to surface as
silent crashes rather than compile errors:

- **Never dereference `state_dev_` from host code.** It's a device pointer; the only
  host-legal operations on it are passing it to a kernel (captured by value) or
  `queue_.memcpy()`/`sycl::free()`, both of which go through the SYCL runtime rather
  than touching the address directly.
- **The queue that allocates must be the queue that frees.** `free_state()` calls
  `sycl::free(state_dev_, queue_)` using the same `queue_` member that allocated it in
  `initialize()` — mixing queues (or contexts) here is a common source of GPU segfaults.

## GPU adapter note

For Intel oneAPI versions 2025.3 or later, you need to build the GPU adapter from
source. The instructions are the following ([see this issue](https://github.com/intel/llvm/issues/20945)):

1. After setting the environment, make sure you have an appropriate CUDA toolkit
   version (e.g. 12.3.0) in your environment.
2. Clone the [llvm repo](https://github.com/intel/llvm):
   ```bash
   git clone https://github.com/intel/llvm.git
   ```
3. Checkout commit [`5c82df75db7d`](https://github.com/intel/llvm/commit/5c82df75db7d1619a1aebafc85b7c2d384415aaa):
   ```bash
   git checkout 5c82df75db7d
   ```
4. `cd /path/to/llvm/unified-runtime`
5. Configure the build:
   ```bash
   cmake -S . -B build -DUR_BUILD_TESTS=OFF -DUR_BUILD_ADAPTER_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCMAKE_INSTALL_PREFIX=/path/to/intel-unified-runtime-6.3.0-rc1
   ```
6. Build:
   ```bash
   cmake --build build -j
   ```
7. Install:
   ```bash
   cmake --install build
   ```
8. Set paths appropriately:
   ```bash
   export LD_LIBRARY_PATH=${ONEAPI_ROOT}/compiler/latest/lib:$LD_LIBRARY_PATH
   export UR_ADAPTERS_SEARCH_PATH=/path/to/install/lib
   ```
9. Check `sycl-ls` to confirm your GPU is being detected.