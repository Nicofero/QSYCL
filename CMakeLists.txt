cmake_minimum_required(VERSION 3.20)
project(quantum_sycl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ENABLE_GPU_BACKEND "Build the SYCL GPU backend" OFF)
option(ENABLE_CUNQA_BACKEND "Build the CUNQA backend" OFF)


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

if(ENABLE_GPU_BACKEND)
    target_compile_definitions(quantum_core PRIVATE QUANTUM_ENABLE_GPU)
endif()

if(ENABLE_CUNQA_BACKEND)
    target_compile_definitions(quantum_core PRIVATE QUANTUM_ENABLE_CUNQA)
endif()

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

if(ENABLE_GPU_BACKEND)
    target_compile_options(quantum_core PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
    target_link_options(quantum_core PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
endif()


# --- GPU backend ---
if(ENABLE_GPU_BACKEND)
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
endif()

# --- CUNQA backend ---
if(ENABLE_CUNQA_BACKEND)
    # CUNQABackend talks to cunqa directly over ZeroMQ + a JSON registry
    # file (see backends/cunqa/cunqa_backend.cpp for why -- cunqa's own
    # CMake install doesn't export a linkable C++ library). That means
    # two new, ordinary dependencies, unrelated to cunqa's own build:
    #   - nlohmann::json  (header-only)
    #   - libzmq + cppzmq (libzmq is a compiled lib; cppzmq is the
    #     header-only zmq.hpp C++ wrapper cunqa's own client uses)
    #
    # We prefer whatever's already on the system (likely on an HPC
    # cluster via a module or system package) and only fall back to
    # fetching sources if nothing is found. The libzmq wire protocol is
    # stable across versions, so this does NOT need to match whatever
    # version cunqa itself was built against.
    include(FetchContent)

    find_package(nlohmann_json 3.11 QUIET)
    if(NOT nlohmann_json_FOUND)
        message(STATUS "nlohmann_json not found on system, fetching it")
        FetchContent_Declare(nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.11.3
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()

    # libzmq: try a CMake package config first (common if installed via
    # a package manager or built+installed manually), then pkg-config
    # (common for a plain ./configure && make install or system package).
    find_package(ZeroMQ QUIET)
    if(NOT ZeroMQ_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(ZMQ QUIET libzmq)
        endif()
    endif()

    if(NOT ZeroMQ_FOUND AND NOT ZMQ_FOUND)
        message(WARNING
            "libzmq not found via find_package(ZeroMQ) or pkg-config. "
            "Install it (or 'module load' it on an HPC cluster) and "
            "re-configure, or point CMAKE_PREFIX_PATH/PKG_CONFIG_PATH "
            "at it -- ZeroMQ is not fetched/built automatically here "
            "since it's a compiled system library, not header-only.")
    endif()

    # cppzmq (zmq.hpp): header-only, rarely packaged on its own, so we
    # fetch it directly rather than requiring the system to have it.
    find_package(cppzmq QUIET)
    if(NOT cppzmq_FOUND)
        message(STATUS "cppzmq not found on system, fetching it")
        set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(cppzmq
            GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
            GIT_TAG v4.10.0
        )
        FetchContent_MakeAvailable(cppzmq)
    endif()

    add_library(quantum_backend_cunqa
        backends/cunqa/cunqa_backend.cpp
    )
    target_include_directories(quantum_backend_cunqa PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/cunqa
    )

    # CUNQABackend falls back to CPUBackend when no vQPU is found (see
    # cunqa_backend.cpp), so it needs cpu_backend.hpp and the compiled
    # quantum_backend_cpu target -- quantum_core links quantum_backend_cpu
    # PUBLIC already, but we depend on it directly too since we #include
    # its header, not just transitively use it.
    target_link_libraries(quantum_backend_cunqa PUBLIC
        quantum_core
        quantum_backend_cpu
        nlohmann_json::nlohmann_json
    )

    if(TARGET cppzmq)
        target_link_libraries(quantum_backend_cunqa PUBLIC cppzmq)
    else()
        # find_package(cppzmq) succeeded but didn't define the usual
        # imported target name on this system -- link libzmq directly.
        target_include_directories(quantum_backend_cunqa PUBLIC ${cppzmq_SOURCE_DIR})
    endif()

    if(ZeroMQ_FOUND)
        target_link_libraries(quantum_backend_cunqa PUBLIC libzmq)
    elseif(ZMQ_FOUND)
        target_include_directories(quantum_backend_cunqa PUBLIC ${ZMQ_INCLUDE_DIRS})
        target_link_libraries(quantum_backend_cunqa PUBLIC ${ZMQ_LIBRARIES})
    endif()

    target_link_libraries(quantum_core PUBLIC quantum_backend_cunqa)
    target_link_libraries(quantum_backend_cunqa PUBLIC quantum_core)
endif()

# --- Example GPU ---
# add_executable(bell_state_gpu examples/bell_state_gpu.cpp)
# target_link_libraries(bell_state_gpu PRIVATE quantum_core)

# target_compile_options(bell_state_gpu PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)
# target_link_options(bell_state_gpu PRIVATE -fsycl-targets=nvptx64-nvidia-cuda)

# --- Example CPU --- 
# add_executable(bell_state examples/bell_state.cpp)
# target_link_libraries(bell_state PRIVATE quantum_core)


# --- Example CUNQA ---
add_executable(bell_state_cunqa examples/bell_state_cunqa.cpp)
target_link_libraries(bell_state_cunqa PRIVATE quantum_core)