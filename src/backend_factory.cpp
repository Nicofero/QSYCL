#include "quantum/backend_factory.hpp"
#include "../backends/cpu/cpu_backend.hpp"

#ifdef QUANTUM_ENABLE_GPU
#include "../backends/gpu/gpu_backend.hpp"
#endif

#ifdef QUANTUM_ENABLE_CUNQA
#include "../backends/cunqa/cunqa_backend.hpp"
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
        case DeviceType::CUSTOM:
            #ifdef QUANTUM_ENABLE_CUNQA
                return std::make_unique<backends::CUNQABackend>();
            #else
                throw std::runtime_error("CUNQA backend was not built.");
            #endif
    }
    throw std::runtime_error("Unknown DeviceType");
}

} // namespace quantum
