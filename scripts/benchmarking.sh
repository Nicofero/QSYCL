#! /bin/bash
#SBATCH -J qsycl_bench
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -p ilk
#SBATCH -t 06:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH -o bench_vqe_%j.out

# General VQE benchmark over every execution method (native / host / sequential)
# for each backend, plus the CUNQA vQPU-count sweep. Drives ./build/bench_vqe
# (examples/benchmarking/bench_vqe.cpp).
#
# Usage:
#   sbatch scripts/benchmarking.sh [cpu|gpu|cunqa|qmio|all]      (default: all = cpu + cunqa)
#
#   The SBATCH header above targets the CPU partition. GPU and QMIO runs need
#   their own partition, so override it on the command line, e.g.:
#     sbatch -p <gpu-partition> --gres=gpu:1 scripts/benchmarking.sh gpu
#     sbatch -p qpu -c 1 -t 01:00:00          scripts/benchmarking.sh qmio
#
#   Every knob below can be overridden without editing this file, e.g.:
#     sbatch --export=ALL,MAX_Q=16,CUNQA_VQPUS="1 2 4 8 16" scripts/benchmarking.sh cunqa
#
# Results: results/benchmarking/<jobid>/{<mode>.csv,<mode>.log}

set -uo pipefail

MODE="${1:-all}"

# ---------------------------------------------------------------- knobs ----
BIN="${BIN:-./build/bench_vqe}"
HAMILTONIAN="${HAMILTONIAN:-ising}"     # ising (2 measurement groups) | heisenberg (3)
REPS="${REPS:-2}"                       # RealAmplitudes depth
ITERS="${ITERS:-10}"                    # iterations per native/host run
SEQ_ITERS="${SEQ_ITERS:-2}"             # iterations per sequential run
SHOTS="${SHOTS:-1024}"                  # shots per circuit (CUNQA / QMIO)

# Exact simulators (CPU / GPU)
MIN_Q="${MIN_Q:-4}"
MAX_Q="${MAX_Q:-20}"
STEP_Q="${STEP_Q:-2}"
OPTIMIZERS_CPU="${OPTIMIZERS_CPU:-cpu}"
OPTIMIZERS_GPU="${OPTIMIZERS_GPU:-cpu,gpu}"

# CUNQA: qubit range and the vQPU counts to compare
CUNQA_MIN_Q="${CUNQA_MIN_Q:-4}"
CUNQA_MAX_Q="${CUNQA_MAX_Q:-12}"
CUNQA_STEP_Q="${CUNQA_STEP_Q:-4}"
CUNQA_VQPUS="${CUNQA_VQPUS:-1 2 4 8}"
# Raise the vQPUs from this script (1) or use ones already raised (0).
CUNQA_RAISE="${CUNQA_RAISE:-1}"
QRAISE_TIME="${QRAISE_TIME:-05:00:00}"
# Extra qraise flags. --co-located makes the vQPUs reachable over TCP from this
# job's node. Check them against `qraise --help` for your CUNQA version.
QRAISE_EXTRA="${QRAISE_EXTRA:---co-located}"
QRAISE_WAIT_S="${QRAISE_WAIT_S:-600}"   # max time to wait for the vQPUs to register

# QMIO: real hardware -- keep it small
QMIO_MIN_Q="${QMIO_MIN_Q:-2}"
QMIO_MAX_Q="${QMIO_MAX_Q:-4}"
QMIO_STEP_Q="${QMIO_STEP_Q:-2}"
QMIO_ITERS="${QMIO_ITERS:-3}"

# ---------------------------------------------------------- environment ----
source "$STORE/intel/oneapi/setvars.sh" > /dev/null # Start SYCL interface

# For CUNQA loading in qmio
module load qmio/hpc gcc/12.3.0 cunqa/2.4.0-python-3.11.9-mpi

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found -- build first (cmake --build build --target bench_vqe)" >&2
    exit 1
fi

OUT_DIR="results/benchmarking/${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
echo "Results in $OUT_DIR"

COMMON=(--hamiltonian "$HAMILTONIAN" --reps "$REPS" --iters "$ITERS"
        --seq-iters "$SEQ_ITERS" --shots "$SHOTS")

section() {
    echo
    echo "=========================================================="
    echo " $*"
    echo "=========================================================="
}

# ------------------------------------------------------------- CPU / GPU ----
run_cpu() {
    section "CPU state-vector simulator"
    "$BIN" "${COMMON[@]}" --devices cpu --optimizers "$OPTIMIZERS_CPU" \
        --methods native,host,sequential \
        --min-qubits "$MIN_Q" --max-qubits "$MAX_Q" --step "$STEP_Q" \
        --csv "$OUT_DIR/cpu.csv" 2>&1 | tee "$OUT_DIR/cpu.log"
}

run_gpu() {
    section "GPU state-vector simulator"
    "$BIN" "${COMMON[@]}" --devices gpu --optimizers "$OPTIMIZERS_GPU" \
        --methods native,host,sequential \
        --min-qubits "$MIN_Q" --max-qubits "$MAX_Q" --step "$STEP_Q" \
        --csv "$OUT_DIR/gpu.csv" 2>&1 | tee "$OUT_DIR/gpu.log"
}

# ----------------------------------------------------------------- CUNQA ----
CUNQA_REGISTRY="$STORE/.cunqa/qpus.json"

registered_vqpus() {
    python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$CUNQA_REGISTRY" 2>/dev/null || echo 0
}

drop_vqpus() {
    if [[ "$CUNQA_RAISE" == "1" ]]; then
        echo "Dropping the vQPUs raised by this benchmark"
        qdrop --all || true
    fi
}

raise_vqpus() {
    local wanted="$1"
    if [[ "$CUNQA_RAISE" != "1" ]]; then
        echo "CUNQA_RAISE=0: using the $(registered_vqpus) vQPU(s) already in $CUNQA_REGISTRY"
        return 0
    fi
    echo "Raising $wanted vQPU(s): qraise -n $wanted -t $QRAISE_TIME $QRAISE_EXTRA"
    # shellcheck disable=SC2086
    qraise -n "$wanted" -t "$QRAISE_TIME" $QRAISE_EXTRA || return 1
    trap drop_vqpus EXIT

    local waited=0
    while (( $(registered_vqpus) < wanted )); do
        if (( waited >= QRAISE_WAIT_S )); then
            echo "error: only $(registered_vqpus)/$wanted vQPUs registered after ${QRAISE_WAIT_S}s" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "$(registered_vqpus) vQPU(s) registered after ${waited}s"
}

run_cunqa() {
    section "CUNQA emulator -- vQPU sweep: $CUNQA_VQPUS"
    local max_vqpus=0 v
    for v in $CUNQA_VQPUS; do (( v > max_vqpus )) && max_vqpus=$v; done
    raise_vqpus "$max_vqpus" || { echo "skipping CUNQA" >&2; return 1; }

    local vqpu_list
    vqpu_list=$(echo "$CUNQA_VQPUS" | tr ' ' ',')

    # native + host for every vQPU count: this is where the vQPU count matters.
    "$BIN" "${COMMON[@]}" --devices cunqa --optimizers cpu \
        --methods native,host --vqpus "$vqpu_list" \
        --min-qubits "$CUNQA_MIN_Q" --max-qubits "$CUNQA_MAX_Q" --step "$CUNQA_STEP_Q" \
        --csv "$OUT_DIR/cunqa.csv" 2>&1 | tee "$OUT_DIR/cunqa.log"

    # Sequential baseline: one blocking circuit at a time, so it can't use more
    # than one vQPU at once -- run it once rather than per vQPU count.
    "$BIN" "${COMMON[@]}" --devices cunqa --optimizers cpu \
        --methods sequential --vqpus 1 \
        --min-qubits "$CUNQA_MIN_Q" --max-qubits "$CUNQA_MAX_Q" --step "$CUNQA_STEP_Q" \
        --csv "$OUT_DIR/cunqa.csv" 2>&1 | tee -a "$OUT_DIR/cunqa.log"

    drop_vqpus
    trap - EXIT
}

# ------------------------------------------------------------------ QMIO ----
run_qmio() {
    section "QMIO real QPU"
    mkdir -p "$STORE/.qpu"
    echo "${ZMQ_SERVER:-}" > "$STORE/.qpu/zmq_server.txt"

    # native and host only: for QMIO the batched path already submits one
    # circuit at a time, so "sequential" would just repeat "host". No warm-up:
    # every iteration is real QPU time.
    "$BIN" "${COMMON[@]}" --devices qmio --optimizers cpu \
        --methods native,host --iters "$QMIO_ITERS" --warmup 0 \
        --min-qubits "$QMIO_MIN_Q" --max-qubits "$QMIO_MAX_Q" --step "$QMIO_STEP_Q" \
        --csv "$OUT_DIR/qmio.csv" 2>&1 | tee "$OUT_DIR/qmio.log"

    echo "" > "$STORE/.qpu/zmq_server.txt"
}

# ------------------------------------------------------------------ main ----
case "$MODE" in
    cpu)   run_cpu ;;
    gpu)   run_gpu ;;
    cunqa) run_cunqa ;;
    qmio)  run_qmio ;;
    all)   run_cpu; run_cunqa ;;
    *)     echo "usage: sbatch scripts/benchmarking.sh [cpu|gpu|cunqa|qmio|all]" >&2; exit 1 ;;
esac

section "Benchmarking done -- results in $OUT_DIR"
