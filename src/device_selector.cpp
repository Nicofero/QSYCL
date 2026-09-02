#include "quantum/device_selector.hpp"
#include <stdexcept>

namespace quantum {

sycl::queue DeviceSelector::make_queue(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return sycl::queue(sycl::cpu_selector_v);
        case DeviceType::GPU:
            return sycl::queue(sycl::gpu_selector_v);
        case DeviceType::FPGA:
            // Requires an FPGA board support package / ahead-of-time image.
            return sycl::queue([](const sycl::device& d) -> int {
                return (d.is_accelerator() &&
                        d.get_info<sycl::info::device::name>().find("FPGA") != std::string::npos)
                       ? 1 : -1;
            });
        case DeviceType::FPGA_EMULATOR:
            return sycl::queue([](const sycl::device& d) -> int {
                return (d.is_accelerator() &&
                        d.get_info<sycl::info::device::name>().find("Emulation") != std::string::npos)
                       ? 1 : -1;
            });
        
        case DeviceType::CUNQA:
            return sycl::queue(sycl::cpu_selector_v);
        case DeviceType::CUSTOM:
            return sycl::queue(sycl::cpu_selector_v); // current idea, CUSTOM device represents CUNQA backend, which is CPU-based for now. This may change in the future.
    }
    throw std::invalid_argument("Unknown DeviceType");
}

sycl::queue DeviceSelector::make_queue_custom(
    const std::function<int(const sycl::device&)>& scoring_fn) {
    return sycl::queue(scoring_fn);
}

std::string DeviceSelector::to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::GPU: return "GPU";
        case DeviceType::FPGA: return "FPGA";
        case DeviceType::FPGA_EMULATOR: return "FPGA_EMULATOR";
        case DeviceType::CUSTOM: return "CUSTOM";
    }
    return "UNKNOWN";
}

} // namespace quantum
