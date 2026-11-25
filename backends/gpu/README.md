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

Since Intel oneAPI 2025.3, the CUDA adapter for SYCL is no longer shipped in the
binary packages and has to be built from source
([see this issue](https://github.com/intel/llvm/issues/20945)). The adapter **must be
built from the intel/llvm release that matches your oneAPI version**: the oneAPI
Unified Runtime (UR) loader and the adapter have to agree on the same API, otherwise
`sycl-ls` segfaults.

| oneAPI version | intel/llvm tag |
| -------------- | -------------- |
| 2025.3+         | `v6.3.1`       | 
| 2026.0+         | `v7.2.0`       |

For other versions, check the [intel/llvm releases](https://github.com/intel/llvm/releases):
each release notes which oneAPI DPC++/C++ Compiler version it is closest to. Don't rely
on the tag given in Intel's oneAPI release notes, which can be out of date.

### Requirements

- Intel oneAPI (the instructions below assume 2026.2).
- **CUDA Toolkit 12.2 or newer**, including `nvcc`. The v7.x adapter uses
  `cuMemAdvise_v2` and `CU_MEM_LOCATION_TYPE_HOST`, which don't exist in older
  toolkits.
- An NVIDIA driver that supports that CUDA version (check "CUDA Version" in
  `nvidia-smi`).

### Build

1. Make sure the right CUDA toolkit is found first. Many distros ship an older
   `nvidia-cuda-toolkit` package in `/usr` (e.g. CUDA 12.0), and CMake will pick it
   up if `/usr/bin/nvcc` comes first in your `PATH`:
   ```bash
   export CUDA_HOME=/usr/local/cuda-13.5   # adjust to your install
   export PATH=$CUDA_HOME/bin:$PATH
   which nvcc                              # must point inside $CUDA_HOME
   ```
   If `$CUDA_HOME/bin/nvcc` doesn't exist, install the compiler for that toolkit
   (e.g. `sudo apt install cuda-nvcc-13-5`).
2. Clone the matching intel/llvm tag:
   ```bash
   git clone https://github.com/intel/llvm -b v7.2.0 --depth=1
   cd llvm
   ```
3. Configure the build, pointing CMake explicitly at your CUDA toolkit:
   ```bash
   cmake -S unified-runtime -B build \
     -DUR_BUILD_TESTS=OFF \
     -DUR_ENABLE_TRACING=ON \
     -DUR_BUILD_ADAPTER_CUDA=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDAToolkit_ROOT=$CUDA_HOME \
     -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
     -DCMAKE_INSTALL_PREFIX=/path/to/unified-runtime/install
   ```
4. Check that CMake found the right toolkit. This line must show paths inside
   `$CUDA_HOME` and its version, not `/usr/include`:
   ```bash
   grep FIND_PACKAGE_MESSAGE_DETAILS_CUDAToolkit build/CMakeCache.txt
   ```
   If it's wrong, fix your `PATH` and run `rm -rf build` before configuring again:
   CMake caches the toolkit it found and won't search again otherwise.
5. Build and install:
   ```bash
   cmake --build build -j
   cmake --install build
   ```

### Use the adapter

Make the oneAPI runtime find the new adapter, either through the search path:

```bash
source ${ONEAPI_ROOT}/setvars.sh
export UR_ADAPTERS_SEARCH_PATH=/path/to/unified-runtime/install/lib
```

or by copying **only the CUDA adapter** into the oneAPI lib directory:

```bash
cp -P /path/to/unified-runtime/install/lib/libur_adapter_cuda.so* \
  ${ONEAPI_ROOT}/compiler/latest/lib/
```

Don't copy the whole `libur*` set: that overwrites oneAPI's own UR loader and adapters
with the versions you built.

Then check that the GPU is detected:

```bash
sycl-ls
```

You should see a `[cuda:gpu]` entry. To compile for it:

```bash
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
  -Xsycl-target-backend --cuda-gpu-arch=sm_XX app.cpp -o app
```

### Troubleshooting

- **`sycl-ls` segfaults after `dlsym failed to load function 'urGet...ProcAddrTable'`
  errors:** the adapter was built from an intel/llvm version that doesn't match your
  oneAPI version. Rebuild from the matching tag. You can compare the exported tables
  with `nm -D libur_adapter_cuda.so.0 | grep ProcAddrTable` against oneAPI's
  `libur_adapter_level_zero.so.0`.
- **`'cuMemAdvise_v2' was not declared in this scope` or
  `'CU_MEM_LOCATION_TYPE_HOST' was not declared`:** CMake is using a CUDA toolkit
  older than 12.2. Go back to steps 1 and 4.
- **To see which loader and adapters are actually loaded:**
  `sycl-ls --verbose` or `LD_DEBUG=libs sycl-ls 2>&1 | grep libur`.