#include "quantum/backend_factory.hpp"
#include "../backends/cpu/cpu_backend.hpp"
#include "../backends/gpu/gpu_backend.hpp"
#include <stdexcept>

namespace quantum {

std::unique_ptr<Backend> BackendFactory::create(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return std::make_unique<backends::CPUBackend>();
        case DeviceType::GPU:
            return std::make_unique<backends::GPUBackend>();
        case DeviceType::FPGA:
        case DeviceType::FPGA_EMULATOR:
            throw std::runtime_error("FPGA backend not implemented yet.");
        case DeviceType::CUSTOM:
            throw std::runtime_error(
                "DeviceType::CUSTOM has no default backend -- construct your Backend "
                "subclass directly instead of going through BackendFactory.");
    }
    throw std::runtime_error("Unknown DeviceType");
}

} // namespace quantum
