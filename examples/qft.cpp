#include "quantum/circuit.hpp"
#include "quantum/runtime.hpp"
#include <iostream>

using namespace quantum;

// Helper function to apply a phase gate to qubit q with angle phi.
inline void p(Circuit& c, std::size_t q, double phi) {
    c.u(q, 0.0, 0.0, phi);
}

// Helper function to apply a controlled phase gate with control qubit and target qubit.
void cp(Circuit& c, std::size_t control,
        std::size_t target, double theta)
{
    p(c, control, theta/2);
    c.crz(control, target, theta);
}

// Quantum Fourier Transform (QFT) on n qubits. The function is not necessary, you can just inline the code in main() if you want.
void qft(Circuit& c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        c.h(i);

        for (std::size_t j = i + 1; j < n; ++j) {
            cp(c,i,j, M_PI / (1 << (j - i)));
        }
    }
}

int main() {    

    // Build the circuit with 3 qubits and apply the QFT to it.
    Circuit c(3);
    qft(c, 3);

    // Pick a device and get a runtime for it. Swapping DeviceType::CPU with other device types (e.g., DeviceType::GPU) will run the same circuit on different backends.
    QuantumRuntime runtime(DeviceType::CPU);

    std::cout << "Running on: " << runtime.device_name() << "\n\n";

    // Inspect the exact state vector (only for GPU and CPU).
    runtime.run(c);
    std::cout << "State vector:\n";
    auto state = runtime.state_vector();
    for (std::size_t i = 0; i < state.size(); ++i) {
        std::cout << "  |" << i << "> : " << state[i] << "\n";
    }

    // Sampling for real hardware
    std::cout << "\nMeasurement counts (1000 shots):\n";
    auto counts = runtime.sample_counts(c, 1000);
    for (const auto& [bitstring, count] : counts) {
        std::cout << "  " << bitstring << " : " << count << "\n";
    }

    return 0;
}
