#pragma once

#include "quantum/backend.hpp"
#include "quantum/device_selector.hpp"
#include <memory>

namespace quantum {

// Central place that knows which Backend subclass to instantiate for a
// given DeviceType. To add a new device (GPU, FPGA, custom ASIC...):
//   1. Implement the Backend interface in backends/<device>/...
//   2. Add a case here that constructs it.
// Nothing else in the runtime or Circuit API needs to change.
class BackendFactory {
public:
    static std::unique_ptr<Backend> create(DeviceType type);
};

} // namespace quantum
