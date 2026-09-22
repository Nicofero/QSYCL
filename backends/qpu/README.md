# QPU backend

The QPU backend follows the same delegation pattern as the
[CUNQA backend](../cunqa/README.md): SYCL is used purely for device selection, and
execution is handed off to an external system rather than computed on-device.

The backend is split into two parts:

- **An abstract QPU interface** that captures what any real-hardware backend needs
  (submitting a circuit, retrieving measurement counts, reporting device availability),
  independent of any specific machine.
- **A concrete implementation targeting CESGA's QMIO quantum processor**, which
  implements that interface for QMIO's transport and job-submission model. Adding
  support for another QPU means implementing the same abstract interface against that
  machine's own transport, without touching the circuit API, the runtime, or any other
  backend.

As with CUNQA, `get_state()`/`probabilities()` are not available on real hardware —
only measurement statistics are — so calling them throws rather than silently
approximating from a simulator underneath.