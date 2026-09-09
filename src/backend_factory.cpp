#include "quantum/backend_factory.hpp"
#include "../backends/cpu/cpu_backend.hpp"

#ifdef QUANTUM_ENABLE_GPU
#include "../backends/gpu/gpu_backend.hpp"
#endif

#ifdef QUANTUM_ENABLE_CUNQA
#include "../backends/cunqa/cunqa_backend.hpp"
#endif

#ifdef QUANTUM_ENABLE_QPU
#include "../backends/qpu/qpu_backend.hpp"
#endif

#include <stdexcept>

namespace quantum {

std::unique_ptr<Backend> BackendFactory::create(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return std::make_unique<backends::CPUBackend>();
        case DeviceType::GPU:
            #ifdef QUANTUM_ENABLE_GPU
                return std::make_unique<backends::GPUBackend>();
            #else
                throw std::runtime_error("GPU backend was not built.");
            #endif
        case DeviceType::FPGA:
        case DeviceType::FPGA_EMULATOR:
            throw std::runtime_error("FPGA backend not implemented yet.");
        case DeviceType::CUNQA:
            #ifdef QUANTUM_ENABLE_CUNQA
                return std::make_unique<backends::CUNQABackend>();
            #else
                throw std::runtime_error("CUNQA backend was not built.");
            #endif
        case DeviceType::QPU:
            #ifdef QUANTUM_ENABLE_QPU
                return std::make_unique<backends::QPUBackend>();
            #else
                throw std::runtime_error("QPU backend was not built.");
            #endif
        case DeviceType::CUSTOM:
            throw std::runtime_error("CUSTOM backend not implemented yet.");
    }
    throw std::runtime_error("Unknown DeviceType");
}

} // namespace quantum
