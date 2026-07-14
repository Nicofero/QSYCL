cmake_minimum_required(VERSION 3.20)
project(quantum_sycl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This project targets Intel oneAPI DPC++ (icpx). Configure with:
#   cmake -B build -DCMAKE_CXX_COMPILER=icpx
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx")
    message(WARNING
        "CMAKE_CXX_COMPILER is not icpx. Configure with "
        "-DCMAKE_CXX_COMPILER=icpx to build with SYCL support.")
endif()

add_compile_options(-fsycl)
add_link_options(-fsycl)

# --- Core library: Circuit API + Runtime + device selector (device-agnostic) ---
add_library(quantum_core
    src/circuit.cpp
    src/runtime.cpp
    src/device_selector.cpp
    src/backend_factory.cpp
)
target_include_directories(quantum_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- CPU backend ---
add_library(quantum_backend_cpu
    backends/cpu/cpu_backend.cpp
)
target_include_directories(quantum_backend_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu
)
target_link_libraries(quantum_core PUBLIC quantum_backend_cpu)
target_link_libraries(quantum_backend_cpu PUBLIC quantum_core)
target_compile_options(quantum_core PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
target_link_options(quantum_core PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)

# --- GPU backend ---
add_library(quantum_backend_gpu
    backends/gpu/gpu_backend.cpp
)
target_include_directories(quantum_backend_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/gpu
)
target_link_libraries(quantum_core PUBLIC quantum_backend_gpu)
target_link_libraries(quantum_backend_gpu PUBLIC quantum_core)

target_compile_options(quantum_backend_gpu PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
target_link_options(quantum_backend_gpu PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)

# --- Example ---
add_executable(bell_state_gpu examples/bell_state_gpu.cpp)
target_link_libraries(bell_state_gpu PRIVATE quantum_core)

target_compile_options(bell_state_gpu PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
target_link_options(bell_state_gpu PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
