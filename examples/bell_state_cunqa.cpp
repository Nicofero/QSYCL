#include "quantum/circuit.hpp"
#include "quantum/runtime.hpp"
#include <iostream>

int main() {
    using namespace quantum;

    // 1. Build the circuit with the user-facing API. No device concepts here.
    Circuit ghz(1);

    ghz.h(0);
    ghz.s(0);

    // 2. Pick a device and get a runtime for it. Swapping DeviceType::CPU
    //    for DeviceType::GPU (once implemented) is the only line that changes.
    QuantumRuntime runtime(DeviceType::CUSTOM);

    std::cout << "Running on: " << runtime.device_name() << "\n\n";

    // 3. Inspect the exact state vector (fine for small circuits).
    runtime.run(ghz);
    std::cout << "State vector:\n";
    auto state = runtime.state_vector();
    for (std::size_t i = 0; i < state.size(); ++i) {
        std::cout << "  |" << i << "> : " << state[i] << "\n";
    }

    // 4. Or sample it like real hardware would.
    std::cout << "\nMeasurement counts (1000 shots):\n";
    auto counts = runtime.sample_counts(ghz, 1000);
    for (const auto& [bitstring, count] : counts) {
        std::cout << "  " << bitstring << " : " << count << "\n";
    }

    return 0;
}
