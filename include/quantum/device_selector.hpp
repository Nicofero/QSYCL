#pragma once

#include <sycl/sycl.hpp>
#include <functional>
#include <string>

namespace quantum {

enum class DeviceType {
    CPU,
    GPU,
    FPGA,
    FPGA_EMULATOR,
    CUSTOM
};

// Wraps SYCL's device-selection machinery so the rest of the codebase
// deals with a small enum instead of sycl::device_selector boilerplate.
class DeviceSelector {
public:
    // Returns a sycl::queue bound to the requested device type.
    // Throws sycl::exception if no matching device is found.
    static sycl::queue make_queue(DeviceType type);

    // For DeviceType::CUSTOM: pass your own scoring function, same
    // contract as a SYCL custom selector (higher score wins, negative
    // score = reject). Useful for e.g. "pick the GPU with the most memory".
    static sycl::queue make_queue_custom(
        const std::function<int(const sycl::device&)>& scoring_fn);

    static std::string to_string(DeviceType type);
};

} // namespace quantum
