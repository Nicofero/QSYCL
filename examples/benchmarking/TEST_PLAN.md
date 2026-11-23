# QSYCL benchmark test plan

A step-by-step plan for benchmarking QSYCL end to end: every backend, the raw single-circuit cost and limits, every VQA
execution method, and CUNQA's multi-vQPU scaling. Work through the phases in
order: each one checks something the next one relies on. The single-circuit limits come first
because they set the problem sizes that are worth running as VQEs.

## What is being measured

| Question | Tool | Output |
|---|---|---|
| What does one circuit cost on each backend, and how many qubits can each reach? | `bench_circuit --modes latency` | `circuit.csv` |
| How many independent circuits per second can each backend sustain, and does it scale with vQPUs? | `bench_circuit --modes throughput` | `circuit_throughput.csv` |
| How much faster are batched circuits than one circuit at a time, inside a VQE? | `bench_vqe`, methods `host` vs `sequential` | `<backend>.csv` |
| Does running the optimizer on a SYCL device (native loop) pay off? | `bench_vqe`, methods `native` vs `host` | `<backend>.csv` |
| How does a VQE on CUNQA scale with the number of vQPUs? | `bench_vqe --vqpus ...` | `cunqa.csv` |

All runs go through `scripts/benchmarking.sh`. Results land in `results/benchmarking/<jobid>/`.
Plot them with `results/benchmarking/analysis.ipynb`, a local notebook that is not tracked by git
(set `JOBS = [...]` in it to pick runs).

## Phase 0: preparation

- [ ] **Pin down the code version.** Commit or stash all local changes, then write down `git rev-parse --short HEAD`. Every job in this plan must use the same build.
- [ ] **Build everything you will benchmark:**
  ```bash
  source $STORE/intel/oneapi/setvars.sh
  module load qmio/hpc gcc/12.3.0 cunqa/2.4.0-python-3.11.9-mpi
  cmake -B build -DENABLE_CUNQA_BACKEND=ON -DENABLE_QPU_BACKEND=ON   # add -DENABLE_GPU_BACKEND=ON on a GPU build
  cmake --build build --target bench_vqe bench_circuit bell_state bell_state_cunqa bell_state_qpu
  ```
- [ ] **Check the CUNQA commands for your version** (`qraise --help`, `qdrop --help`). The script runs `qraise -n N -t T --co-located` and `qdrop --all`. If yours differ, override with `QRAISE_EXTRA` / `QRAISE_TIME`.
  - `qdrop --all` drops **every** vQPU you own. Don't keep other vQPUs running during the benchmark. If you need to, raise the benchmark's vQPUs yourself and pass `CUNQA_RAISE=0`.
- [ ] **Know your resources.** Check the cores and memory of the `ilk` nodes (`sinfo -N -l -p ilk`) and how many cores one vQPU takes. N vQPUs need N times that, so this caps the vQPU sweep.
- [ ] **Keep timings comparable.** Use the same node type for every CPU/CUNQA job. Add `--exclusive` to `sbatch` when you can, so other jobs on the node don't distort the timings.
- [ ] **Fill in the run log** at the end of this file as you go.

## Phase 1: smoke tests (correctness before timing)

Small and fast. Run them before anything long, because a timing from a wrong result is worthless.

- [ ] **Each backend runs a circuit:** `./build/bell_state` (CPU). Then raise one vQPU and run `./build/bell_state_cunqa`; counts should be about 50/50 on `00`/`11`, and the device name must say `CUNQA (1 vQPU(s): ...)`, not `fallback`.
- [ ] **VQE benchmark, tiny run on CPU:**
  ```bash
  ./build/bench_vqe --devices cpu --min-qubits 4 --max-qubits 6 --iters 3 --seq-iters 2
  ```
  `native` and `host` must report **identical** final energies.
- [ ] **CUNQA uses every vQPU:** raise 2 vQPUs, then run
  ```bash
  ./build/bench_vqe --devices cunqa --methods host --vqpus 1,2 --min-qubits 4 --max-qubits 4 --iters 2
  ```
  The `vqpus` column must show 1, then 2, and the 2-vQPU row should take roughly half the time.
- [ ] **Circuit benchmark, tiny run:**
  ```bash
  ./build/bench_circuit --devices cpu,cunqa --modes latency,throughput --max-qubits 6 --batch-sizes 1,4 --throughput-qubits 4 --vqpus 1,2 --repeats 2
  ```
  No `failed:` rows.
- [ ] Drop the vQPUs (`qdrop --all`).

## Phase 2: single circuit, latency and limits

```bash
sbatch --exclusive --export=ALL,CIRCUIT_MODES=latency scripts/benchmarking.sh circuit
```
- [ ] **CPU:** the sweep stops at `memory_limit` or `time_limit`. Record the largest qubit count that ran and its time.
- [ ] **CUNQA:** with `exec` ≈ 0, all of the cost is in `sample` (the remote round trip). Record the fixed overhead at small n (the flat part of the curve) and the limit (the vQPU's maximum qubit count, or `time_limit`).
- [ ] **Crossover:** the qubit count where a single CUNQA circuit becomes cheaper than a local CPU simulation, if it exists.
- [ ] **GPU:** rerun with `--export=ALL,CIRCUIT_DEVICES=gpu` on the GPU partition.
- [ ] If the CPU sweep ends at `time_limit` before `memory_limit`, rerun with `CIRCUIT_TIME_LIMIT=300` to find the true memory limit.
- [ ] **Use these limits to size Phases 4 and 5:** keep `MAX_Q` / `CUNQA_MAX_Q` below the largest qubit count that ran here. A VQE iteration runs dozens to hundreds of circuits, so set it well below the point where a single circuit takes seconds.

## Phase 3: circuit throughput, with no VQE

```bash
sbatch --exclusive --export=ALL,CIRCUIT_MODES=throughput,CIRCUIT_VQPUS="1 2 4 8",CIRCUIT_BATCH_SIZES="1,8,32,128",CIRCUIT_TP_QUBITS="4,8,12,16" scripts/benchmarking.sh circuit
```
- [ ] **CUNQA:** circuits/s should grow roughly in proportion to the vQPU count once `batch ≥ vQPUs`. With `batch = 1` every vQPU count gives the same rate (only one circuit to run).
- [ ] **Baseline for Phase 5:** this is the best scaling CUNQA can reach on raw circuits. Phase 5 compares the VQE against it; any gap is overhead the VQE adds on top of circuit execution.
- [ ] **CPU:** throughput from the batched state-vector kernel. Compare with 1 / latency from Phase 2 to see what batching buys on the CPU itself.

## Phase 4: VQE on the exact simulators

**4a. CPU, all methods.** About 15 min with the defaults (the 20-qubit sequential run dominates).
```bash
sbatch --exclusive scripts/benchmarking.sh cpu
```
- [ ] Check: `sequential / host` is much larger than 1. In job 618219 it was 60–100× up to 12 qubits, falling to about 8× at 20 qubits.
- [ ] Check: `host / native` is about 1. Native is slightly slower at small sizes, where the overhead of launching SYCL kernels dominates.

**4b. Larger problem sizes,** up to the CPU limit found in Phase 2.
```bash
sbatch --exclusive --export=ALL,MIN_Q=20,MAX_Q=26,STEP_Q=2,ITERS=3,SEQ_ITERS=1 scripts/benchmarking.sh cpu
```

**4c. GPU, with the optimizer on CPU and on GPU.** Needs the GPU build.
```bash
sbatch -p <gpu-partition> --gres=gpu:1 scripts/benchmarking.sh gpu
```
- [ ] Compare against 4a with the same qubit counts. Look for the size where the GPU overtakes the CPU.
- [ ] Optimizer on CPU vs GPU should make little difference: the update only touches O(params) numbers.

**4d. Sensitivity.** Rerun 4a, once per variant, so every result isn't tied to a single problem shape.
- [ ] More measurement groups: `--export=ALL,HAMILTONIAN=heisenberg` (3 groups instead of 2).
- [ ] More parameters: `--export=ALL,REPS=4` (gradient batches about 1.7× larger).

## Phase 5: VQE on CUNQA, scaling with vQPUs

**5a. Main sweep.**
```bash
sbatch --exclusive --export=ALL,CUNQA_VQPUS="1 2 4 8",CUNQA_MIN_Q=4,CUNQA_MAX_Q=16,CUNQA_STEP_Q=4 scripts/benchmarking.sh cunqa
```
- [ ] Check: parallel efficiency at 8 vQPUs. Job 618219 gave 80% at 4 qubits and 90% at 12 qubits; bigger circuits should scale better.
- [ ] Compare with Phase 3: the VQE's efficiency at each vQPU count should be close to the raw-circuit throughput scaling. A clear gap is per-iteration overhead (building circuits, the optimizer step).
- [ ] Check: with 1 vQPU, `sequential` ≈ `host` ≈ `native`. That is expected, because one vQPU runs one circuit at a time either way.

**5b. Past the knee.** Keep adding vQPUs until the speed-up stops growing, within your allocation (Phase 0).
```bash
sbatch --exclusive --export=ALL,CUNQA_VQPUS="1 8 16 32",CUNQA_MIN_Q=8,CUNQA_MAX_Q=16,CUNQA_STEP_Q=8 scripts/benchmarking.sh cunqa
```
- [ ] Note where the speed-up flattens. With 2 groups × (1 + 2 × params) circuits per iteration, a small problem can run out of circuits to hand out.

**5c. Shots.** Shots change how much work each circuit is relative to the per-circuit overhead.
- [ ] Run 5a with `SHOTS=256` and with `SHOTS=8192` (`--export=ALL,SHOTS=...`), at a single vQPU count of 8.

**5d. More measurement groups:** `--export=ALL,HAMILTONIAN=heisenberg`. That is 1.5× more circuits per iteration, so the vQPUs have more to share.

## Phase 6: QMIO (real hardware, run last)

Every circuit uses real QPU time, so keep runs minimal and check the queue first.

- [ ] Connectivity: `sbatch scripts/test_comm.sh` (runs `bell_state_qpu`).
- [ ] Latency: `sbatch -p qpu -c 1 -t 01:00:00 --export=ALL,CIRCUIT_DEVICES=qmio,CIRCUIT_MODES=latency,CIRCUIT_MIN_Q=2,CIRCUIT_MAX_Q=6,CIRCUIT_REPEATS=3 scripts/benchmarking.sh circuit`
- [ ] VQE: `sbatch -p qpu -c 1 -t 01:00:00 scripts/benchmarking.sh qmio` (2–4 qubits, 3 iterations).
- [ ] Compare QMIO per-circuit latency with CUNQA's. This is how far the emulator's timing is from the real device.

## Phase 7: repeatability

- [ ] Rerun the key configurations (2, 3, 4a, 5a) **3 times as separate jobs**, ideally on different days or nodes.
- [ ] Load the jobs together in the notebook (`JOBS = [...]`) and check how much they vary. Treat any differences smaller than that variation as noise.

## Phase 8: analysis and report

- [ ] Open `results/benchmarking/analysis.ipynb`, set `JOBS`, and run all cells. Figures are saved to `<job>/figures/`.
- [ ] **Headline numbers to report:**
  - largest qubit count per backend and what stopped it
  - CUNQA per-circuit overhead
  - CUNQA raw throughput scaling with vQPUs
  - batching speed-up inside the VQE on the CPU (range over qubits)
  - native vs host
  - CUNQA VQE speed-up and efficiency at the largest vQPU count, per qubit count
  - QMIO vs CUNQA latency
- [ ] **Record next to the numbers:** commit hash, node type, cores and memory per job, CUNQA version, shots, Hamiltonian, ansatz depth.

## Run log

| Phase | Job ID | Date | Commit | Settings (`--export`) | Notes |
|---|---|---|---|---|---|
| 4a | 618219 | | | defaults (cpu) | first run |
| 5a | 618219 | | | defaults (cunqa, 1 2 4 8 vQPUs) | first run |
| | | | | | |
